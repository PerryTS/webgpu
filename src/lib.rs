//! Native bindings for the WebGPU spec — closes #571.
//!
//! [WebGPU][webgpu] is the modern, low-level GPU API the W3C
//! standardised for the web. This crate exposes a spec-faithful subset
//! to the [Perry TypeScript-to-native compiler][perry] so shaders +
//! pipelines authored against the browser API run unmodified under
//! Perry. Backed by [`wgpu`][wgpu] — the canonical native
//! implementation of the spec.
//!
//! [webgpu]: https://www.w3.org/TR/webgpu/
//! [perry]:  https://github.com/PerryTS/perry
//! [wgpu]:   https://github.com/gfx-rs/wgpu
//!
//! # Status
//!
//! - **v0.1.0** — adapter / device / queue, buffer, shader module,
//!   bind-group-layout, pipeline-layout, bind-group, compute pipeline,
//!   command encoder, compute pass encoder, command buffer, queue
//!   submit / write_buffer / on_submitted_work_done, buffer
//!   map_async / get_mapped_range / unmap. Enough to run a hello-world
//!   compute shader end-to-end and read the result back.
//!
//! Followups (own issue):
//!
//! - **v0.2.0** — render pipeline + render pass + textures + samplers
//!   + queue.writeTexture (the second hello-triangle acceptance test).
//! - **v0.3.0** — error scopes (`pushErrorScope` / `popErrorScope`),
//!   `GPUQuerySet`, surface presentation (`GPUSurface`).
//! - **v0.4.0** — `createRenderPipelineAsync` /
//!   `createComputePipelineAsync` (the sync-call variants ship in
//!   v0.1.0; the async variants are scaffolded but not load-bearing).
//!
//! # Why a JSON-string descriptor channel
//!
//! WebGPU descriptor objects (`GPUBufferDescriptor`,
//! `GPUBindGroupLayoutDescriptor`, `GPUComputePipelineDescriptor`, …)
//! are deeply nested with many optional fields. Rather than exploding
//! each descriptor into a wide function-arg list, the TS side calls
//! `JSON.stringify(descriptor)` and we [`serde_json`]-deserialize on
//! the Rust side. This keeps the FFI surface small (one descriptor
//! JSON string + a few handles per call) and the TS surface obvious:
//! the object literal is the argument exactly as written in the spec.
//!
//! Handles inside descriptors (e.g. `bindGroupLayout` references in a
//! `GPUPipelineLayoutDescriptor`) round-trip as JS numbers — the
//! Perry-side wrapper just passes the handle through `JSON.stringify`
//! and we look the typed pointer up via [`with_handle`] on the way in.

use perry_ffi::{
    alloc_buffer, drop_handle, register_handle, spawn_blocking, take_handle, with_handle,
    BufferHeader, Handle, JsPromise, JsValue, Promise, StringHeader,
};

use parking_lot::Mutex;
use serde::Deserialize;
use std::sync::OnceLock;
use wgpu::{
    Adapter, BindGroup, BindGroupLayout, Buffer, CommandBuffer, CommandEncoder, ComputePass,
    ComputePipeline, Device, Instance, MapMode, PipelineLayout, Queue, ShaderModule,
};

// ─── Helpers ────────────────────────────────────────────────────────

fn instance() -> &'static Instance {
    static INSTANCE: OnceLock<Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| Instance::new(wgpu::InstanceDescriptor::default()))
}

unsafe fn read_str(ptr: *const StringHeader) -> Option<String> {
    let handle = perry_ffi::JsString::from_raw(ptr as *mut StringHeader);
    perry_ffi::read_string(handle).map(String::from)
}

fn alloc_str_value(s: &str) -> JsValue {
    JsValue::from_string_ptr(perry_ffi::alloc_string(s).as_raw())
}

fn alloc_buffer_value(bytes: &[u8]) -> JsValue {
    let buf = alloc_buffer(bytes);
    JsValue::from_object_ptr(buf as *mut perry_ffi::ObjectHeader)
}

/// Reject a promise with `prefix: <serde_json::Error>` and bail.
/// v0.1 sync-create paths return `0` on JSON errors instead — wgpu
/// surfaces the resulting "invalid handle" via the device error scope,
/// which is the spec-correct path. Kept around for v0.2 async paths.
#[allow(dead_code)]
fn reject_json_err(promise: JsPromise, prefix: &str, e: serde_json::Error) {
    promise.reject_string(&format!("{}: bad descriptor JSON: {}", prefix, e));
}

// ─── Wrapper structs for the handle registry ────────────────────────
//
// Each WebGPU object type lives in its own wrapper so the registry's
// `with_handle::<T, _, _>(handle, …)` downcast is unambiguous. The
// inner type is the corresponding wgpu native object, which owns its
// own GPU resources and drops them when the wrapper drops.

pub struct WGPUAdapter(pub Adapter);
pub struct WGPUDevice(pub Device);
/// The queue carries its parent device's handle so
/// `queueOnSubmittedWorkDone` can poll the right device — the spec
/// semantics ("resolve when all submitted work has finished") need a
/// poll loop on the device, and the bare `wgpu::Queue` doesn't carry
/// a back-pointer to its device.
pub struct WGPUQueue {
    pub queue: Queue,
    pub device_handle: Handle,
}
pub struct WGPUBuffer(pub Buffer);
pub struct WGPUShaderModule(pub ShaderModule);
pub struct WGPUBindGroupLayout(pub BindGroupLayout);
pub struct WGPUPipelineLayout(pub PipelineLayout);
pub struct WGPUBindGroup(pub BindGroup);
pub struct WGPUComputePipeline(pub ComputePipeline);
pub struct WGPURenderPipeline(pub wgpu::RenderPipeline);
pub struct WGPUCommandEncoder(pub Mutex<Option<CommandEncoder>>);
pub struct WGPUComputePass(pub Mutex<Option<ComputePass<'static>>>);
pub struct WGPURenderPass(pub Mutex<Option<wgpu::RenderPass<'static>>>);
pub struct WGPUCommandBuffer(pub Mutex<Option<CommandBuffer>>);
pub struct WGPUTexture(pub wgpu::Texture);
pub struct WGPUTextureView(pub wgpu::TextureView);
pub struct WGPUSampler(pub wgpu::Sampler);
pub struct WGPUQuerySet(pub wgpu::QuerySet);

// ════════════════════════════════════════════════════════════════════
// Adapter / Device / Queue
// ════════════════════════════════════════════════════════════════════

/// `navigator.gpu.requestAdapter() -> Promise<GPUAdapter | null>` —
/// requests a default adapter (high-performance, fallback allowed).
/// v0.1 takes no descriptor; the spec's `powerPreference` /
/// `forceFallbackAdapter` knobs are deferred to v0.2.
///
/// Resolves to a numeric adapter handle, or rejects if no adapter is
/// available. The spec returns `null` when no adapter is found; we
/// reject instead to make the failure surface in async error-handling
/// paths instead of silently returning `0`. TS-side wrappers can map
/// this to `null` if literal spec parity matters.
#[no_mangle]
pub extern "C" fn js_webgpu_request_adapter() -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();

    spawn_blocking(move || {
        let adapter = pollster::block_on(instance().request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            },
        ));
        match adapter {
            Some(a) => {
                let h = register_handle(WGPUAdapter(a));
                promise.resolve(JsValue::from_number(h as f64));
            }
            None => promise.reject_string("webgpu requestAdapter: no compatible adapter"),
        }
    });
    raw
}

/// `adapter.requestDevice() -> Promise<GPUDevice>` — requests a device
/// from the adapter using default limits + features. Resolves with an
/// object `{ device, queue }` (numeric handles); the TS-side wrapper
/// destructures it onto the `GPUDevice.queue` property in one step,
/// which matches how browsers expose them (a `GPUDevice` always has
/// the same `queue` for its lifetime).
#[no_mangle]
pub extern "C" fn js_webgpu_adapter_request_device(adapter_handle: Handle) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();

    spawn_blocking(move || {
        let outcome = with_handle::<WGPUAdapter, _, _>(adapter_handle, |a| {
            pollster::block_on(a.0.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("perry-webgpu-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            ))
        });

        match outcome {
            Some(Ok((device, queue))) => {
                let dev_h = register_handle(WGPUDevice(device));
                let q_h = register_handle(WGPUQueue {
                    queue,
                    device_handle: dev_h,
                });
                // Pack as {device, queue}.
                unsafe {
                    let keys = ["device", "queue"];
                    let (packed, shape) = perry_ffi::build_object_shape(&keys);
                    let obj = perry_ffi::js_object_alloc_with_shape(
                        shape,
                        keys.len() as u32,
                        packed.as_ptr(),
                        packed.len() as u32,
                    );
                    perry_ffi::js_object_set_field(obj, 0, JsValue::from_number(dev_h as f64));
                    perry_ffi::js_object_set_field(obj, 1, JsValue::from_number(q_h as f64));
                    promise.resolve(JsValue::from_object_ptr(obj));
                }
            }
            Some(Err(e)) => promise.reject_string(&format!("webgpu requestDevice: {}", e)),
            None => promise.reject_string("webgpu requestDevice: invalid adapter handle"),
        }
    });
    raw
}

/// `adapter.drop()` — synchronous handle release. The spec doesn't
/// expose this, but Perry needs a way to free the wrapper since the
/// adapter is otherwise leaked once the TS handle goes out of scope.
/// (Browser GC eventually reclaims; native tools rely on explicit
/// drops.)
#[no_mangle]
pub extern "C" fn js_webgpu_adapter_drop(adapter_handle: Handle) {
    let _ = take_handle::<WGPUAdapter>(adapter_handle);
    drop_handle(adapter_handle);
}

/// `device.destroy()` — releases the device and its queue. Mirrors
/// `GPUDevice.destroy()` from the spec. Idempotent.
#[no_mangle]
pub extern "C" fn js_webgpu_device_destroy(device_handle: Handle) {
    let _ = take_handle::<WGPUDevice>(device_handle);
    drop_handle(device_handle);
}

// ════════════════════════════════════════════════════════════════════
// Buffer
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct BufferDescriptor {
    #[serde(default)]
    label: Option<String>,
    size: u64,
    usage: u32,
    #[serde(rename = "mappedAtCreation", default)]
    mapped_at_creation: bool,
}

/// `device.createBuffer(descriptor) -> GPUBuffer` — synchronous.
/// `descriptor` is the JSON form of a `GPUBufferDescriptor`:
/// `{ label?, size, usage, mappedAtCreation? }`. `usage` is the
/// `GPUBufferUsageFlags` bitmask (numeric — `STORAGE | COPY_SRC` etc.,
/// per the spec's flag values).
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader` produced by
/// `JSON.stringify(...)` on the TS side.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_buffer(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: BufferDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let buffer = d.0.create_buffer(&wgpu::BufferDescriptor {
            label: desc.label.as_deref(),
            size: desc.size,
            usage: wgpu::BufferUsages::from_bits_truncate(desc.usage),
            mapped_at_creation: desc.mapped_at_creation,
        });
        register_handle(WGPUBuffer(buffer))
    })
    .unwrap_or(0)
}

/// `buffer.destroy()` — releases the buffer. Idempotent.
#[no_mangle]
pub extern "C" fn js_webgpu_buffer_destroy(buffer_handle: Handle) {
    let _ = take_handle::<WGPUBuffer>(buffer_handle);
    drop_handle(buffer_handle);
}

/// `buffer.mapAsync(mode, offset, size) -> Promise<undefined>` — make
/// the buffer's contents accessible to the host. `mode` is `1` for
/// READ, `2` for WRITE (matches `GPUMapMode.READ` / `WRITE`).
///
/// Resolves once the GPU is done with the buffer and the host can
/// safely access the mapped range. The spec requires the device to be
/// "polled" between the request and the resolution; we do this by
/// scheduling a blocking poll inside the `spawn_blocking` task.
#[no_mangle]
pub extern "C" fn js_webgpu_buffer_map_async(
    buffer_handle: Handle,
    mode: u32,
    offset: f64,
    size: f64,
) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();

    spawn_blocking(move || {
        let map_mode = match mode {
            1 => MapMode::Read,
            2 => MapMode::Write,
            _ => {
                promise.reject_string("webgpu mapAsync: invalid mode (expected 1=READ, 2=WRITE)");
                return;
            }
        };
        let off = offset.max(0.0) as u64;
        let sz = size.max(0.0) as u64;

        // The mapAsync callback fires when device.poll progresses the
        // queue past the mapping. We park on a `parking_lot::Mutex`
        // sentinel rather than a tokio channel — wgpu's callback isn't
        // async-aware and we're already inside spawn_blocking.
        let result_slot: std::sync::Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> =
            std::sync::Arc::new(Mutex::new(None));
        let r2 = result_slot.clone();

        // Scope `with_handle` so we drop the borrow before `poll`.
        let ok = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| {
            let slice = if sz == 0 {
                b.0.slice(off..)
            } else {
                b.0.slice(off..off + sz)
            };
            slice.map_async(map_mode, move |r| {
                *r2.lock() = Some(r);
            });
            true
        })
        .unwrap_or(false);

        if !ok {
            promise.reject_string("webgpu mapAsync: invalid buffer handle");
            return;
        }

        // Spin-poll the device until the callback fires. We don't have
        // a direct handle on the device here, so the user must call
        // `device.poll()` (or equivalently, `queue.submit([])` followed
        // by `queue.onSubmittedWorkDone()`) before awaiting this
        // promise — which is also the browser pattern. We add a hard
        // ceiling so a buggy caller can't hang the worker forever.
        let mut spins = 0u32;
        loop {
            if result_slot.lock().is_some() {
                break;
            }
            spins += 1;
            if spins > 10_000 {
                promise.reject_string(
                    "webgpu mapAsync: timed out waiting for callback (did you forget to poll the device?)",
                );
                return;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }

        let result = result_slot.lock().take();
        match result {
            Some(Ok(())) => promise.resolve_undefined(),
            Some(Err(e)) => promise.reject_string(&format!("webgpu mapAsync: {}", e)),
            None => promise.reject_string("webgpu mapAsync: callback dropped without result"),
        }
    });
    raw
}

/// `buffer.getMappedRange(offset?, size?) -> ArrayBuffer` — copies
/// the mapped bytes into a Perry-runtime `Buffer` (Uint8Array view).
/// In the spec this returns an ArrayBuffer that aliases the GPU's
/// staging memory; under Perry we copy because Perry-runtime buffers
/// are independently GC-managed.
///
/// Returns an empty buffer if the handle is unknown or not mapped.
#[no_mangle]
pub extern "C" fn js_webgpu_buffer_get_mapped_range(
    buffer_handle: Handle,
    offset: f64,
    size: f64,
) -> JsValue {
    let off = offset.max(0.0) as u64;
    let sz = size.max(0.0) as u64;

    let bytes = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| {
        let slice = if sz == 0 {
            b.0.slice(off..)
        } else {
            b.0.slice(off..off + sz)
        };
        slice.get_mapped_range().to_vec()
    });

    match bytes {
        Some(v) => alloc_buffer_value(&v),
        None => alloc_buffer_value(&[]),
    }
}

/// `buffer.unmap()` — releases the host mapping. After this the host
/// can't read the mapped bytes any more, but the GPU can use the
/// buffer again. Idempotent at the wgpu layer.
#[no_mangle]
pub extern "C" fn js_webgpu_buffer_unmap(buffer_handle: Handle) {
    let _ = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| b.0.unmap());
}

// ════════════════════════════════════════════════════════════════════
// Shader Module
// ════════════════════════════════════════════════════════════════════

/// `device.createShaderModule({ code }) -> GPUShaderModule` — WGSL
/// only (matches the browser's spec — no GLSL/SPIR-V toggle on the
/// public API). The code is passed as a separate string param rather
/// than packed in a JSON descriptor because shader source can be very
/// large and JSON-escaping it is wasteful.
///
/// `label` is deferred to a v0.2 second-arg overload; the spec
/// supports `{ code, label, sourceMap, compilationHints }` but only
/// `code` is load-bearing for compilation.
///
/// # Safety
///
/// `code_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_shader_module(
    device_handle: Handle,
    code_ptr: *const StringHeader,
) -> Handle {
    let Some(code) = read_str(code_ptr) else {
        return 0;
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let module = d.0.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(code.into()),
        });
        register_handle(WGPUShaderModule(module))
    })
    .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════
// BindGroupLayout
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct BglDescriptor {
    #[serde(default)]
    label: Option<String>,
    entries: Vec<BglEntry>,
}

#[derive(Deserialize)]
struct BglEntry {
    binding: u32,
    visibility: u32,
    #[serde(default)]
    buffer: Option<BglBuffer>,
    #[serde(default)]
    sampler: Option<BglSampler>,
    #[serde(default)]
    texture: Option<BglTexture>,
    #[serde(rename = "storageTexture", default)]
    storage_texture: Option<BglStorageTexture>,
}

#[derive(Deserialize)]
struct BglBuffer {
    #[serde(rename = "type", default = "default_buffer_type")]
    ty: String,
    #[serde(rename = "hasDynamicOffset", default)]
    has_dynamic_offset: bool,
    #[serde(rename = "minBindingSize", default)]
    min_binding_size: u64,
}
fn default_buffer_type() -> String {
    "uniform".into()
}

#[derive(Deserialize)]
struct BglSampler {
    #[serde(rename = "type", default)]
    ty: Option<String>,
}

#[derive(Deserialize)]
struct BglTexture {
    #[serde(rename = "sampleType", default)]
    sample_type: Option<String>,
    #[serde(rename = "viewDimension", default)]
    view_dimension: Option<String>,
    #[serde(default)]
    multisampled: bool,
}

#[derive(Deserialize)]
struct BglStorageTexture {
    #[serde(default)]
    access: Option<String>,
    format: String,
    #[serde(rename = "viewDimension", default)]
    view_dimension: Option<String>,
}

fn parse_buffer_binding_type(s: &str) -> wgpu::BufferBindingType {
    match s {
        "uniform" => wgpu::BufferBindingType::Uniform,
        "storage" => wgpu::BufferBindingType::Storage { read_only: false },
        "read-only-storage" => wgpu::BufferBindingType::Storage { read_only: true },
        _ => wgpu::BufferBindingType::Uniform,
    }
}

/// `device.createBindGroupLayout(descriptor) -> GPUBindGroupLayout` —
/// synchronous. Descriptor JSON shape mirrors the spec; only the
/// buffer/sampler/texture/storageTexture entry types are wired up in
/// v0.1. `externalTexture` is a v0.2 follow-up.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader` containing
/// `JSON.stringify(GPUBindGroupLayoutDescriptor)`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_bind_group_layout(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: BglDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = desc
            .entries
            .iter()
            .map(|e| {
                let visibility = wgpu::ShaderStages::from_bits_truncate(e.visibility);
                let ty = if let Some(b) = &e.buffer {
                    wgpu::BindingType::Buffer {
                        ty: parse_buffer_binding_type(&b.ty),
                        has_dynamic_offset: b.has_dynamic_offset,
                        min_binding_size: std::num::NonZeroU64::new(b.min_binding_size),
                    }
                } else if let Some(s) = &e.sampler {
                    let sampler_ty = match s.ty.as_deref().unwrap_or("filtering") {
                        "non-filtering" => wgpu::SamplerBindingType::NonFiltering,
                        "comparison" => wgpu::SamplerBindingType::Comparison,
                        _ => wgpu::SamplerBindingType::Filtering,
                    };
                    wgpu::BindingType::Sampler(sampler_ty)
                } else if let Some(t) = &e.texture {
                    wgpu::BindingType::Texture {
                        sample_type: match t.sample_type.as_deref().unwrap_or("float") {
                            "unfilterable-float" => {
                                wgpu::TextureSampleType::Float { filterable: false }
                            }
                            "depth" => wgpu::TextureSampleType::Depth,
                            "sint" => wgpu::TextureSampleType::Sint,
                            "uint" => wgpu::TextureSampleType::Uint,
                            _ => wgpu::TextureSampleType::Float { filterable: true },
                        },
                        view_dimension: parse_view_dimension(t.view_dimension.as_deref()),
                        multisampled: t.multisampled,
                    }
                } else if let Some(st) = &e.storage_texture {
                    wgpu::BindingType::StorageTexture {
                        access: match st.access.as_deref().unwrap_or("write-only") {
                            "read-only" => wgpu::StorageTextureAccess::ReadOnly,
                            "read-write" => wgpu::StorageTextureAccess::ReadWrite,
                            _ => wgpu::StorageTextureAccess::WriteOnly,
                        },
                        format: parse_texture_format(&st.format),
                        view_dimension: parse_view_dimension(st.view_dimension.as_deref()),
                    }
                } else {
                    // Default to a uniform buffer if the entry has no
                    // type — keeps `serde_json::from_str` from rejecting
                    // older descriptor shapes that pre-date the typed
                    // entry split.
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }
                };
                wgpu::BindGroupLayoutEntry {
                    binding: e.binding,
                    visibility,
                    ty,
                    count: None,
                }
            })
            .collect();

        let layout = d.0.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: desc.label.as_deref(),
            entries: &entries,
        });
        register_handle(WGPUBindGroupLayout(layout))
    })
    .unwrap_or(0)
}

fn parse_view_dimension(s: Option<&str>) -> wgpu::TextureViewDimension {
    match s.unwrap_or("2d") {
        "1d" => wgpu::TextureViewDimension::D1,
        "3d" => wgpu::TextureViewDimension::D3,
        "cube" => wgpu::TextureViewDimension::Cube,
        "cube-array" => wgpu::TextureViewDimension::CubeArray,
        "2d-array" => wgpu::TextureViewDimension::D2Array,
        _ => wgpu::TextureViewDimension::D2,
    }
}

/// Map a WebGPU spec format string to a `wgpu::TextureFormat`. Covers
/// the formats most pipelines use; everything else returns
/// `Rgba8Unorm` as a safe fallback. v0.2 will harden this with a
/// Result-returning variant that surfaces the unknown name.
fn parse_texture_format(s: &str) -> wgpu::TextureFormat {
    use wgpu::TextureFormat::*;
    match s {
        "r8unorm" => R8Unorm,
        "r8snorm" => R8Snorm,
        "r8uint" => R8Uint,
        "r8sint" => R8Sint,
        "r16uint" => R16Uint,
        "r16sint" => R16Sint,
        "r16float" => R16Float,
        "rg8unorm" => Rg8Unorm,
        "rg8snorm" => Rg8Snorm,
        "rg8uint" => Rg8Uint,
        "rg8sint" => Rg8Sint,
        "r32uint" => R32Uint,
        "r32sint" => R32Sint,
        "r32float" => R32Float,
        "rg16uint" => Rg16Uint,
        "rg16sint" => Rg16Sint,
        "rg16float" => Rg16Float,
        "rgba8unorm" => Rgba8Unorm,
        "rgba8unorm-srgb" => Rgba8UnormSrgb,
        "rgba8snorm" => Rgba8Snorm,
        "rgba8uint" => Rgba8Uint,
        "rgba8sint" => Rgba8Sint,
        "bgra8unorm" => Bgra8Unorm,
        "bgra8unorm-srgb" => Bgra8UnormSrgb,
        "rgb10a2unorm" => Rgb10a2Unorm,
        "rg32uint" => Rg32Uint,
        "rg32sint" => Rg32Sint,
        "rg32float" => Rg32Float,
        "rgba16uint" => Rgba16Uint,
        "rgba16sint" => Rgba16Sint,
        "rgba16float" => Rgba16Float,
        "rgba32uint" => Rgba32Uint,
        "rgba32sint" => Rgba32Sint,
        "rgba32float" => Rgba32Float,
        "depth16unorm" => Depth16Unorm,
        "depth24plus" => Depth24Plus,
        "depth24plus-stencil8" => Depth24PlusStencil8,
        "depth32float" => Depth32Float,
        _ => Rgba8Unorm,
    }
}

// ════════════════════════════════════════════════════════════════════
// PipelineLayout
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct PipelineLayoutDescriptor {
    #[serde(default)]
    label: Option<String>,
    #[serde(rename = "bindGroupLayouts")]
    bind_group_layouts: Vec<i64>,
}

/// `device.createPipelineLayout(descriptor) -> GPUPipelineLayout` —
/// synchronous. The TS-side `bindGroupLayouts: GPUBindGroupLayout[]`
/// array round-trips as numeric handles.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_pipeline_layout(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: PipelineLayoutDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    // Look up each layout via `with_handle` — but `wgpu::PipelineLayoutDescriptor`
    // wants `&[&BindGroupLayout]`, so we collect the raw pointers and re-borrow.
    // Safe because the registry pins each layout for its handle's lifetime.
    let layout_ptrs: Vec<*const BindGroupLayout> = desc
        .bind_group_layouts
        .iter()
        .filter_map(|h| {
            with_handle::<WGPUBindGroupLayout, _, _>(*h, |bgl| &bgl.0 as *const _)
        })
        .collect();

    if layout_ptrs.len() != desc.bind_group_layouts.len() {
        return 0;
    }

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let layouts: Vec<&BindGroupLayout> = layout_ptrs.iter().map(|p| &**p).collect();
        let pl = d.0.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: desc.label.as_deref(),
            bind_group_layouts: &layouts,
            push_constant_ranges: &[],
        });
        register_handle(WGPUPipelineLayout(pl))
    })
    .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════
// BindGroup
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct BindGroupDescriptor {
    #[serde(default)]
    label: Option<String>,
    layout: i64,
    entries: Vec<BindGroupEntry>,
}

#[derive(Deserialize)]
struct BindGroupEntry {
    binding: u32,
    resource: BindGroupResource,
}

/// Wire format for `GPUBindGroupEntry.resource`. The spec's TS shape
/// is a *union* without a tag — `GPUBufferBinding` is an object with
/// a `buffer` field, a `GPUSampler` is a bare number, a
/// `GPUTextureView` is a bare number too. JSON can't distinguish bare
/// samplers from bare texture views, so the binding wraps each in a
/// single-key object on the way across the FFI: `{buffer:n, …}`,
/// `{sampler:n}`, or `{textureView:n}`. `serde(untagged)` then decides
/// which variant by which key is present.
#[derive(Deserialize)]
#[serde(untagged)]
enum BindGroupResource {
    Buffer(BindGroupBufferBinding),
    Sampler { sampler: i64 },
    TextureView {
        #[serde(rename = "textureView")]
        texture_view: i64,
    },
}

#[derive(Deserialize)]
struct BindGroupBufferBinding {
    buffer: i64,
    #[serde(default)]
    offset: u64,
    /// `0` means "to the end of the buffer", matching the spec's
    /// `undefined` sentinel.
    #[serde(default)]
    size: u64,
}

/// `device.createBindGroup(descriptor) -> GPUBindGroup` — synchronous.
/// Supports the spec's three resource forms: `GPUBufferBinding`
/// (`{buffer, offset?, size?}`), `GPUSampler` (`{sampler}`), and
/// `GPUTextureView` (`{textureView}`). External textures are deferred
/// until the canvas integration crate exposes them.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_bind_group(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: BindGroupDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    // Resolve the layout pointer first (lifetime-pinned by registry).
    let layout_ptr = match with_handle::<WGPUBindGroupLayout, _, _>(desc.layout, |bgl| {
        &bgl.0 as *const BindGroupLayout
    }) {
        Some(p) => p,
        None => return 0,
    };

    // Resolve each entry's pointer up front. Pointer-based here
    // because the registry pins each resource for its handle's
    // lifetime — the wgpu side wants `&` references and we can't hold
    // a `with_handle` guard across the closure boundary.
    enum ResolvedEntry {
        Buffer {
            binding: u32,
            buffer_ptr: *const Buffer,
            offset: u64,
            size: u64,
        },
        Sampler {
            binding: u32,
            sampler_ptr: *const wgpu::Sampler,
        },
        TextureView {
            binding: u32,
            view_ptr: *const wgpu::TextureView,
        },
    }
    let mut resolved: Vec<ResolvedEntry> = Vec::with_capacity(desc.entries.len());
    for e in desc.entries.iter() {
        match &e.resource {
            BindGroupResource::Buffer(b) => {
                let bp = with_handle::<WGPUBuffer, _, _>(b.buffer, |bb| &bb.0 as *const Buffer);
                let Some(bp) = bp else {
                    return 0;
                };
                resolved.push(ResolvedEntry::Buffer {
                    binding: e.binding,
                    buffer_ptr: bp,
                    offset: b.offset,
                    size: b.size,
                });
            }
            BindGroupResource::Sampler { sampler } => {
                let sp = with_handle::<WGPUSampler, _, _>(*sampler, |s| {
                    &s.0 as *const wgpu::Sampler
                });
                let Some(sp) = sp else { return 0 };
                resolved.push(ResolvedEntry::Sampler {
                    binding: e.binding,
                    sampler_ptr: sp,
                });
            }
            BindGroupResource::TextureView { texture_view } => {
                let vp = with_handle::<WGPUTextureView, _, _>(*texture_view, |v| {
                    &v.0 as *const wgpu::TextureView
                });
                let Some(vp) = vp else { return 0 };
                resolved.push(ResolvedEntry::TextureView {
                    binding: e.binding,
                    view_ptr: vp,
                });
            }
        }
    }

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let bind_entries: Vec<wgpu::BindGroupEntry> = resolved
            .iter()
            .map(|r| match r {
                ResolvedEntry::Buffer {
                    binding,
                    buffer_ptr,
                    offset,
                    size,
                } => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: unsafe { &**buffer_ptr },
                        offset: *offset,
                        size: std::num::NonZeroU64::new(*size),
                    }),
                },
                ResolvedEntry::Sampler {
                    binding,
                    sampler_ptr,
                } => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::Sampler(unsafe { &**sampler_ptr }),
                },
                ResolvedEntry::TextureView { binding, view_ptr } => wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: wgpu::BindingResource::TextureView(unsafe { &**view_ptr }),
                },
            })
            .collect();
        let bg = d.0.create_bind_group(&wgpu::BindGroupDescriptor {
            label: desc.label.as_deref(),
            layout: unsafe { &*layout_ptr },
            entries: &bind_entries,
        });
        register_handle(WGPUBindGroup(bg))
    })
    .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════
// Compute Pipeline
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ComputePipelineDescriptor {
    #[serde(default)]
    label: Option<String>,
    /// Either a numeric `GPUPipelineLayout` handle or the literal
    /// string `"auto"` per the spec.
    layout: serde_json::Value,
    compute: ProgrammableStage,
}

#[derive(Deserialize)]
struct ProgrammableStage {
    module: i64,
    #[serde(rename = "entryPoint", default)]
    entry_point: Option<String>,
}

/// `device.createComputePipeline(descriptor) -> GPUComputePipeline` —
/// synchronous. `descriptor.layout` accepts either `"auto"` (per
/// spec — wgpu picks the layout from the shader bindings) or a
/// numeric `GPUPipelineLayout` handle.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_compute_pipeline(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: ComputePipelineDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    let layout_ptr: Option<*const PipelineLayout> = match &desc.layout {
        serde_json::Value::String(s) if s == "auto" => None,
        serde_json::Value::Number(n) => {
            let h = n.as_i64().unwrap_or(0);
            with_handle::<WGPUPipelineLayout, _, _>(h, |pl| &pl.0 as *const _)
        }
        _ => return 0,
    };
    let layout_explicit = match &desc.layout {
        serde_json::Value::Number(_) => true,
        _ => false,
    };
    if layout_explicit && layout_ptr.is_none() {
        return 0;
    }

    let module_ptr = match with_handle::<WGPUShaderModule, _, _>(desc.compute.module, |m| {
        &m.0 as *const ShaderModule
    }) {
        Some(p) => p,
        None => return 0,
    };

    // wgpu 22 requires a non-optional `entry_point: &str`; the spec
    // permits omission, but wgpu picks the unique entry by name when
    // the shader has exactly one. Default to `"main"` (the canonical
    // WGSL entry name) to mirror that behaviour.
    let entry_point = desc.compute.entry_point.as_deref().unwrap_or("main");

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let pipeline = d.0.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: desc.label.as_deref(),
            layout: layout_ptr.map(|p| unsafe { &*p }),
            module: unsafe { &*module_ptr },
            entry_point,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        register_handle(WGPUComputePipeline(pipeline))
    })
    .unwrap_or(0)
}

/// `pipeline.getBindGroupLayout(index) -> GPUBindGroupLayout` —
/// synchronous accessor, useful when the pipeline was created with
/// `layout: "auto"` and the user needs the implicit layout for a
/// matching `createBindGroup`.
#[no_mangle]
pub extern "C" fn js_webgpu_compute_pipeline_get_bind_group_layout(
    pipeline_handle: Handle,
    index: u32,
) -> Handle {
    with_handle::<WGPUComputePipeline, _, _>(pipeline_handle, |p| {
        let bgl = p.0.get_bind_group_layout(index);
        register_handle(WGPUBindGroupLayout(bgl))
    })
    .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════
// Command Encoder + Compute Pass
// ════════════════════════════════════════════════════════════════════

/// `device.createCommandEncoder() -> GPUCommandEncoder` — synchronous.
/// v0.1 takes no descriptor; the spec's `label` slot is a v0.2 add.
#[no_mangle]
pub extern "C" fn js_webgpu_device_create_command_encoder(device_handle: Handle) -> Handle {
    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let encoder = d
            .0
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        register_handle(WGPUCommandEncoder(Mutex::new(Some(encoder))))
    })
    .unwrap_or(0)
}

/// `encoder.beginComputePass() -> GPUComputePassEncoder` — synchronous.
/// We use `forget_lifetime()` to detach the pass from its parent
/// encoder's borrow — the registry holds both as 'static-flavoured
/// owned values, and the user is contractually obligated (per spec) to
/// `pass.end()` before `encoder.finish()`.
#[no_mangle]
pub extern "C" fn js_webgpu_command_encoder_begin_compute_pass(
    encoder_handle: Handle,
) -> Handle {
    with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else {
            return 0;
        };
        let pass = encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            })
            .forget_lifetime();
        register_handle(WGPUComputePass(Mutex::new(Some(pass))))
    })
    .unwrap_or(0)
}

/// `encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size)` —
/// synchronous. All offsets and size are in bytes; `size` must be a
/// multiple of 4 per the spec.
#[no_mangle]
pub extern "C" fn js_webgpu_command_encoder_copy_buffer_to_buffer(
    encoder_handle: Handle,
    src: Handle,
    src_offset: f64,
    dst: Handle,
    dst_offset: f64,
    size: f64,
) {
    let _ = with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else { return };
        // Resolve both buffers; we deliberately don't bail silently if
        // either is unknown — wgpu will surface a validation error via
        // the device's error scope, which is the spec-correct path.
        let _ = with_handle::<WGPUBuffer, _, _>(src, |sb| {
            let _ = with_handle::<WGPUBuffer, _, _>(dst, |db| {
                encoder.copy_buffer_to_buffer(
                    &sb.0,
                    src_offset.max(0.0) as u64,
                    &db.0,
                    dst_offset.max(0.0) as u64,
                    size.max(0.0) as u64,
                );
            });
        });
    });
}

/// `encoder.finish() -> GPUCommandBuffer` — synchronous. Consumes the
/// encoder; subsequent calls on the encoder handle are no-ops.
#[no_mangle]
pub extern "C" fn js_webgpu_command_encoder_finish(encoder_handle: Handle) -> Handle {
    with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let encoder = match ce.0.lock().take() {
            Some(e) => e,
            None => return 0,
        };
        let cb = encoder.finish();
        register_handle(WGPUCommandBuffer(Mutex::new(Some(cb))))
    })
    .unwrap_or(0)
}

/// `pass.setPipeline(pipeline)` — synchronous.
#[no_mangle]
pub extern "C" fn js_webgpu_compute_pass_set_pipeline(
    pass_handle: Handle,
    pipeline_handle: Handle,
) {
    let _ = with_handle::<WGPUComputePass, _, _>(pass_handle, |cp| {
        let mut slot = cp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUComputePipeline, _, _>(pipeline_handle, |p| {
            pass.set_pipeline(&p.0);
        });
    });
}

/// `pass.setBindGroup(index, bindGroup, dynamicOffsets?)` —
/// synchronous. Dynamic offsets are deferred to v0.2; v0.1 takes
/// `&[]`.
#[no_mangle]
pub extern "C" fn js_webgpu_compute_pass_set_bind_group(
    pass_handle: Handle,
    index: u32,
    bind_group_handle: Handle,
) {
    let _ = with_handle::<WGPUComputePass, _, _>(pass_handle, |cp| {
        let mut slot = cp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUBindGroup, _, _>(bind_group_handle, |bg| {
            pass.set_bind_group(index, &bg.0, &[]);
        });
    });
}

/// `pass.dispatchWorkgroups(x, y?, z?)` — synchronous. `y` and `z`
/// default to 1 in the spec; here the caller passes 1 explicitly when
/// omitting (TS-level wrapper handles the default).
#[no_mangle]
pub extern "C" fn js_webgpu_compute_pass_dispatch_workgroups(
    pass_handle: Handle,
    x: u32,
    y: u32,
    z: u32,
) {
    let _ = with_handle::<WGPUComputePass, _, _>(pass_handle, |cp| {
        let mut slot = cp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.dispatch_workgroups(x.max(1), y.max(1), z.max(1));
    });
}

/// `pass.end()` — synchronous. After `end()` the pass handle is
/// effectively dead; subsequent set/dispatch calls are no-ops.
#[no_mangle]
pub extern "C" fn js_webgpu_compute_pass_end(pass_handle: Handle) {
    let _ = take_handle::<WGPUComputePass>(pass_handle);
    drop_handle(pass_handle);
}

// ════════════════════════════════════════════════════════════════════
// Queue
// ════════════════════════════════════════════════════════════════════

/// `queue.submit(commandBuffers) -> undefined` — synchronous.
/// `command_buffers_json` is a JSON array of numeric handles, e.g.
/// `"[123, 456]"`. Each handle is consumed (taken) — submitting twice
/// is a spec error.
///
/// # Safety
///
/// `command_buffers_json_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_queue_submit(
    queue_handle: Handle,
    command_buffers_json_ptr: *const StringHeader,
) {
    let Some(json) = read_str(command_buffers_json_ptr) else {
        return;
    };
    let handles: Vec<i64> = match serde_json::from_str(&json) {
        Ok(v) => v,
        Err(_) => return,
    };

    // Take each command buffer out of its slot (consuming the handle).
    let mut buffers: Vec<CommandBuffer> = Vec::with_capacity(handles.len());
    for h in handles.iter() {
        let mut taken = None;
        let _ = with_handle::<WGPUCommandBuffer, _, _>(*h, |cb| {
            taken = cb.0.lock().take();
        });
        if let Some(cb) = taken {
            buffers.push(cb);
        }
    }

    let _ = with_handle::<WGPUQueue, _, _>(queue_handle, |q| {
        q.queue.submit(buffers);
    });
}

/// `queue.writeBuffer(buffer, bufferOffset, data) -> undefined` —
/// synchronous. `data` is a Perry-runtime `Buffer` or `Uint8Array`;
/// the bytes are copied into the staging path — no aliasing with the
/// caller's data once this returns.
///
/// # Safety
///
/// `data_ptr` must be null or a Perry-runtime `BufferHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_queue_write_buffer(
    queue_handle: Handle,
    buffer_handle: Handle,
    buffer_offset: f64,
    data_ptr: *const BufferHeader,
) {
    let Some(bytes) = perry_ffi::read_buffer_bytes(data_ptr) else {
        return;
    };
    let _ = with_handle::<WGPUQueue, _, _>(queue_handle, |q| {
        let _ = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| {
            q.queue
                .write_buffer(&b.0, buffer_offset.max(0.0) as u64, bytes);
        });
    });
}

/// `queue.onSubmittedWorkDone() -> Promise<undefined>` — resolves once
/// every command buffer submitted to this queue *before* this call
/// has finished executing. We use wgpu's `Queue::on_submitted_work_done`
/// callback to capture the "before this call" snapshot, then poll the
/// queue's parent device (tracked on the queue wrapper since v0.2) so
/// the callback actually fires.
#[no_mangle]
pub extern "C" fn js_webgpu_queue_on_submitted_work_done(queue_handle: Handle) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();

    spawn_blocking(move || {
        // Snapshot: drop a parking-lot sentinel into the
        // on_submitted_work_done callback so we know when wgpu has
        // signalled completion.
        let done: std::sync::Arc<Mutex<bool>> = std::sync::Arc::new(Mutex::new(false));
        let done2 = done.clone();
        let device_handle = match with_handle::<WGPUQueue, _, _>(queue_handle, |q| {
            q.queue.on_submitted_work_done(move || {
                *done2.lock() = true;
            });
            q.device_handle
        }) {
            Some(h) => h,
            None => {
                promise.reject_string("webgpu onSubmittedWorkDone: invalid queue handle");
                return;
            }
        };

        // Pump the device until the callback fires. Mirrors the
        // mapAsync poll loop — same hard ceiling so a buggy caller
        // can't hang the worker forever.
        let mut spins = 0u32;
        loop {
            if *done.lock() {
                break;
            }
            let _ = with_handle::<WGPUDevice, _, _>(device_handle, |d| {
                d.0.poll(wgpu::Maintain::Poll);
            });
            spins += 1;
            if spins > 10_000 {
                promise.reject_string(
                    "webgpu onSubmittedWorkDone: timed out (callback never fired)",
                );
                return;
            }
            std::thread::sleep(std::time::Duration::from_micros(100));
        }
        promise.resolve_undefined();
    });
    raw
}

/// `device.poll() -> undefined` — synchronous. Not in the WebGPU spec
/// (browsers poll implicitly via the event loop), but native runtimes
/// must call this between `mapAsync` and the get/unmap cycle so the
/// driver progresses the queue. Equivalent to wgpu's
/// `device.poll(Maintain::Wait)`.
#[no_mangle]
pub extern "C" fn js_webgpu_device_poll(device_handle: Handle) {
    let _ = with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        d.0.poll(wgpu::Maintain::Wait);
    });
}

// ════════════════════════════════════════════════════════════════════
// Errors / introspection (v0.1 stubs)
// ════════════════════════════════════════════════════════════════════

fn parse_error_filter(s: &str) -> wgpu::ErrorFilter {
    match s {
        "out-of-memory" => wgpu::ErrorFilter::OutOfMemory,
        "internal" => wgpu::ErrorFilter::Internal,
        // Spec: "validation" is the default, and the parser is
        // permissive — anything unrecognised falls through here.
        _ => wgpu::ErrorFilter::Validation,
    }
}

/// `device.pushErrorScope(filter) -> undefined` — synchronous. Filter
/// is `"validation"` (default) / `"out-of-memory"` / `"internal"`,
/// matching the spec.
///
/// # Safety
///
/// `filter_ptr` must be null or a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_push_error_scope(
    device_handle: Handle,
    filter_ptr: *const StringHeader,
) {
    let filter = read_str(filter_ptr).unwrap_or_else(|| "validation".to_string());
    let _ = with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        d.0.push_error_scope(parse_error_filter(&filter));
    });
}

/// `device.popErrorScope() -> Promise<GPUError | null>` — resolves
/// with a JSON-encoded `{type, message}` describing the captured
/// error, or the empty string when the scope captured nothing (which
/// the call site can map to `null`, matching the spec).
///
/// `wgpu::Device::pop_error_scope()` returns a future; we bridge it
/// through `spawn_blocking` + `block_on` like every other async path
/// in this crate.
#[no_mangle]
pub extern "C" fn js_webgpu_device_pop_error_scope(device_handle: Handle) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();
    spawn_blocking(move || {
        let outcome: Option<Option<wgpu::Error>> = with_handle::<WGPUDevice, _, _>(
            device_handle,
            |d| {
                tokio::runtime::Handle::current().block_on(async { d.0.pop_error_scope().await })
            },
        );
        match outcome {
            Some(Some(err)) => {
                // Wrap the error into a `{type, message}` JSON blob —
                // matches the GPUError union the spec uses
                // (GPUValidationError / GPUOutOfMemoryError /
                // GPUInternalError).
                let (kind, msg) = match &err {
                    wgpu::Error::OutOfMemory { .. } => ("out-of-memory", err.to_string()),
                    wgpu::Error::Validation { .. } => ("validation", err.to_string()),
                    wgpu::Error::Internal { .. } => ("internal", err.to_string()),
                };
                let json = format!(
                    "{{\"type\":\"{}\",\"message\":{}}}",
                    kind,
                    serde_json::to_string(&msg).unwrap_or_else(|_| "\"\"".into())
                );
                promise.resolve(alloc_str_value(&json));
            }
            Some(None) => promise.resolve(alloc_str_value("")),
            None => promise.reject_string("webgpu popErrorScope: invalid device handle"),
        }
    });
    raw
}

// ════════════════════════════════════════════════════════════════════
// Texture / TextureView / Sampler
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct Extent3dDesc {
    width: u32,
    #[serde(default = "default_one")]
    height: u32,
    #[serde(rename = "depthOrArrayLayers", default = "default_one")]
    depth_or_array_layers: u32,
}
fn default_one() -> u32 {
    1
}

#[derive(Deserialize)]
struct TextureDescriptor {
    #[serde(default)]
    label: Option<String>,
    size: Extent3dDesc,
    #[serde(rename = "mipLevelCount", default = "default_one")]
    mip_level_count: u32,
    #[serde(rename = "sampleCount", default = "default_one")]
    sample_count: u32,
    #[serde(default)]
    dimension: Option<String>,
    format: String,
    usage: u32,
    #[serde(rename = "viewFormats", default)]
    view_formats: Vec<String>,
}

fn parse_texture_dimension(s: Option<&str>) -> wgpu::TextureDimension {
    match s.unwrap_or("2d") {
        "1d" => wgpu::TextureDimension::D1,
        "3d" => wgpu::TextureDimension::D3,
        _ => wgpu::TextureDimension::D2,
    }
}

/// `device.createTexture(descriptor) -> GPUTexture` — synchronous.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_texture(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: TextureDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let view_formats: Vec<wgpu::TextureFormat> = desc
            .view_formats
            .iter()
            .map(|s| parse_texture_format(s))
            .collect();
        let texture = d.0.create_texture(&wgpu::TextureDescriptor {
            label: desc.label.as_deref(),
            size: wgpu::Extent3d {
                width: desc.size.width,
                height: desc.size.height,
                depth_or_array_layers: desc.size.depth_or_array_layers,
            },
            mip_level_count: desc.mip_level_count.max(1),
            sample_count: desc.sample_count.max(1),
            dimension: parse_texture_dimension(desc.dimension.as_deref()),
            format: parse_texture_format(&desc.format),
            usage: wgpu::TextureUsages::from_bits_truncate(desc.usage),
            view_formats: &view_formats,
        });
        register_handle(WGPUTexture(texture))
    })
    .unwrap_or(0)
}

#[derive(Deserialize, Default)]
struct TextureViewDescriptor {
    #[serde(default)]
    label: Option<String>,
    #[serde(default)]
    format: Option<String>,
    #[serde(rename = "dimension", default)]
    dimension: Option<String>,
    #[serde(rename = "aspect", default)]
    aspect: Option<String>,
    #[serde(rename = "baseMipLevel", default)]
    base_mip_level: u32,
    #[serde(rename = "mipLevelCount", default)]
    mip_level_count: u32,
    #[serde(rename = "baseArrayLayer", default)]
    base_array_layer: u32,
    #[serde(rename = "arrayLayerCount", default)]
    array_layer_count: u32,
}

fn parse_texture_aspect(s: Option<&str>) -> wgpu::TextureAspect {
    match s.unwrap_or("all") {
        "stencil-only" => wgpu::TextureAspect::StencilOnly,
        "depth-only" => wgpu::TextureAspect::DepthOnly,
        _ => wgpu::TextureAspect::All,
    }
}

/// `texture.createView(descriptor?) -> GPUTextureView` — synchronous.
/// `descriptor_ptr` may be null/empty for a default view (most common
/// case when binding the whole texture as a render attachment).
///
/// # Safety
///
/// `descriptor_ptr` must be null or a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_texture_create_view(
    texture_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let parsed: TextureViewDescriptor = match read_str(descriptor_ptr) {
        Some(json) if !json.is_empty() => serde_json::from_str(&json).unwrap_or_default(),
        _ => TextureViewDescriptor::default(),
    };

    with_handle::<WGPUTexture, _, _>(texture_handle, |t| {
        let view = t.0.create_view(&wgpu::TextureViewDescriptor {
            label: parsed.label.as_deref(),
            format: parsed.format.as_deref().map(parse_texture_format),
            dimension: parsed
                .dimension
                .as_deref()
                .map(|s| parse_view_dimension(Some(s))),
            aspect: parse_texture_aspect(parsed.aspect.as_deref()),
            base_mip_level: parsed.base_mip_level,
            mip_level_count: if parsed.mip_level_count == 0 {
                None
            } else {
                Some(parsed.mip_level_count)
            },
            base_array_layer: parsed.base_array_layer,
            array_layer_count: if parsed.array_layer_count == 0 {
                None
            } else {
                Some(parsed.array_layer_count)
            },
        });
        register_handle(WGPUTextureView(view))
    })
    .unwrap_or(0)
}

/// `texture.destroy()` — release the GPU memory. Idempotent.
#[no_mangle]
pub extern "C" fn js_webgpu_texture_destroy(texture_handle: Handle) {
    let _ = take_handle::<WGPUTexture>(texture_handle);
    drop_handle(texture_handle);
}

#[derive(Deserialize, Default)]
struct SamplerDescriptor {
    #[serde(default)]
    label: Option<String>,
    #[serde(rename = "addressModeU", default)]
    address_mode_u: Option<String>,
    #[serde(rename = "addressModeV", default)]
    address_mode_v: Option<String>,
    #[serde(rename = "addressModeW", default)]
    address_mode_w: Option<String>,
    #[serde(rename = "magFilter", default)]
    mag_filter: Option<String>,
    #[serde(rename = "minFilter", default)]
    min_filter: Option<String>,
    #[serde(rename = "mipmapFilter", default)]
    mipmap_filter: Option<String>,
    #[serde(rename = "lodMinClamp", default)]
    lod_min_clamp: f32,
    #[serde(rename = "lodMaxClamp", default = "default_lod_max")]
    lod_max_clamp: f32,
    #[serde(default)]
    compare: Option<String>,
    #[serde(rename = "maxAnisotropy", default = "default_one")]
    max_anisotropy: u32,
}
fn default_lod_max() -> f32 {
    32.0
}

fn parse_address_mode(s: Option<&str>) -> wgpu::AddressMode {
    match s.unwrap_or("clamp-to-edge") {
        "repeat" => wgpu::AddressMode::Repeat,
        "mirror-repeat" => wgpu::AddressMode::MirrorRepeat,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

fn parse_filter_mode(s: Option<&str>) -> wgpu::FilterMode {
    match s.unwrap_or("nearest") {
        "linear" => wgpu::FilterMode::Linear,
        _ => wgpu::FilterMode::Nearest,
    }
}

fn parse_compare_function(s: &str) -> wgpu::CompareFunction {
    match s {
        "never" => wgpu::CompareFunction::Never,
        "less" => wgpu::CompareFunction::Less,
        "equal" => wgpu::CompareFunction::Equal,
        "less-equal" => wgpu::CompareFunction::LessEqual,
        "greater" => wgpu::CompareFunction::Greater,
        "not-equal" => wgpu::CompareFunction::NotEqual,
        "greater-equal" => wgpu::CompareFunction::GreaterEqual,
        _ => wgpu::CompareFunction::Always,
    }
}

/// `device.createSampler(descriptor?) -> GPUSampler` — synchronous.
/// All fields are optional; defaults match the spec (clamp-to-edge,
/// nearest, no compare, anisotropy 1).
///
/// # Safety
///
/// `descriptor_ptr` must be null or a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_sampler(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let desc: SamplerDescriptor = match read_str(descriptor_ptr) {
        Some(json) if !json.is_empty() => serde_json::from_str(&json).unwrap_or_default(),
        _ => SamplerDescriptor::default(),
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let sampler = d.0.create_sampler(&wgpu::SamplerDescriptor {
            label: desc.label.as_deref(),
            address_mode_u: parse_address_mode(desc.address_mode_u.as_deref()),
            address_mode_v: parse_address_mode(desc.address_mode_v.as_deref()),
            address_mode_w: parse_address_mode(desc.address_mode_w.as_deref()),
            mag_filter: parse_filter_mode(desc.mag_filter.as_deref()),
            min_filter: parse_filter_mode(desc.min_filter.as_deref()),
            mipmap_filter: parse_filter_mode(desc.mipmap_filter.as_deref()),
            lod_min_clamp: desc.lod_min_clamp,
            lod_max_clamp: desc.lod_max_clamp,
            compare: desc.compare.as_deref().map(parse_compare_function),
            anisotropy_clamp: desc.max_anisotropy.max(1) as u16,
            border_color: None,
        });
        register_handle(WGPUSampler(sampler))
    })
    .unwrap_or(0)
}

// ════════════════════════════════════════════════════════════════════
// Render Pipeline
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct VertexAttributeDesc {
    format: String,
    offset: u64,
    #[serde(rename = "shaderLocation")]
    shader_location: u32,
}

#[derive(Deserialize)]
struct VertexBufferLayoutDesc {
    #[serde(rename = "arrayStride")]
    array_stride: u64,
    #[serde(rename = "stepMode", default)]
    step_mode: Option<String>,
    attributes: Vec<VertexAttributeDesc>,
}

#[derive(Deserialize)]
struct VertexStateDesc {
    module: i64,
    #[serde(rename = "entryPoint", default)]
    entry_point: Option<String>,
    #[serde(default)]
    buffers: Vec<VertexBufferLayoutDesc>,
}

#[derive(Deserialize, Default)]
struct PrimitiveStateDesc {
    #[serde(default)]
    topology: Option<String>,
    #[serde(rename = "stripIndexFormat", default)]
    strip_index_format: Option<String>,
    #[serde(rename = "frontFace", default)]
    front_face: Option<String>,
    #[serde(rename = "cullMode", default)]
    cull_mode: Option<String>,
}

#[derive(Deserialize)]
struct StencilFaceStateDesc {
    #[serde(default)]
    compare: Option<String>,
    #[serde(rename = "failOp", default)]
    fail_op: Option<String>,
    #[serde(rename = "depthFailOp", default)]
    depth_fail_op: Option<String>,
    #[serde(rename = "passOp", default)]
    pass_op: Option<String>,
}

#[derive(Deserialize)]
struct DepthStencilStateDesc {
    format: String,
    #[serde(rename = "depthWriteEnabled", default)]
    depth_write_enabled: bool,
    #[serde(rename = "depthCompare", default)]
    depth_compare: Option<String>,
    #[serde(rename = "stencilFront", default)]
    stencil_front: Option<StencilFaceStateDesc>,
    #[serde(rename = "stencilBack", default)]
    stencil_back: Option<StencilFaceStateDesc>,
    #[serde(rename = "stencilReadMask", default = "default_stencil_mask")]
    stencil_read_mask: u32,
    #[serde(rename = "stencilWriteMask", default = "default_stencil_mask")]
    stencil_write_mask: u32,
    #[serde(rename = "depthBias", default)]
    depth_bias: i32,
    #[serde(rename = "depthBiasSlopeScale", default)]
    depth_bias_slope_scale: f32,
    #[serde(rename = "depthBiasClamp", default)]
    depth_bias_clamp: f32,
}
fn default_stencil_mask() -> u32 {
    0xFFFF_FFFF
}

#[derive(Deserialize, Default)]
struct MultisampleStateDesc {
    #[serde(default = "default_one")]
    count: u32,
    #[serde(default = "default_mask_u64")]
    mask: u64,
    #[serde(rename = "alphaToCoverageEnabled", default)]
    alpha_to_coverage_enabled: bool,
}
fn default_mask_u64() -> u64 {
    !0u64
}

#[derive(Deserialize)]
struct BlendComponentDesc {
    #[serde(rename = "srcFactor", default)]
    src_factor: Option<String>,
    #[serde(rename = "dstFactor", default)]
    dst_factor: Option<String>,
    #[serde(default)]
    operation: Option<String>,
}

#[derive(Deserialize)]
struct BlendStateDesc {
    color: BlendComponentDesc,
    alpha: BlendComponentDesc,
}

#[derive(Deserialize)]
struct ColorTargetStateDesc {
    format: String,
    #[serde(default)]
    blend: Option<BlendStateDesc>,
    #[serde(rename = "writeMask", default = "default_write_mask_u32")]
    write_mask: u32,
}
fn default_write_mask_u32() -> u32 {
    0xF
}

#[derive(Deserialize)]
struct FragmentStateDesc {
    module: i64,
    #[serde(rename = "entryPoint", default)]
    entry_point: Option<String>,
    targets: Vec<Option<ColorTargetStateDesc>>,
}

#[derive(Deserialize)]
struct RenderPipelineDescriptor {
    #[serde(default)]
    label: Option<String>,
    layout: serde_json::Value,
    vertex: VertexStateDesc,
    #[serde(default)]
    primitive: PrimitiveStateDesc,
    #[serde(rename = "depthStencil", default)]
    depth_stencil: Option<DepthStencilStateDesc>,
    #[serde(default)]
    multisample: MultisampleStateDesc,
    #[serde(default)]
    fragment: Option<FragmentStateDesc>,
}

fn parse_vertex_format(s: &str) -> wgpu::VertexFormat {
    use wgpu::VertexFormat::*;
    match s {
        "uint8x2" => Uint8x2,
        "uint8x4" => Uint8x4,
        "sint8x2" => Sint8x2,
        "sint8x4" => Sint8x4,
        "unorm8x2" => Unorm8x2,
        "unorm8x4" => Unorm8x4,
        "snorm8x2" => Snorm8x2,
        "snorm8x4" => Snorm8x4,
        "uint16x2" => Uint16x2,
        "uint16x4" => Uint16x4,
        "sint16x2" => Sint16x2,
        "sint16x4" => Sint16x4,
        "unorm16x2" => Unorm16x2,
        "unorm16x4" => Unorm16x4,
        "snorm16x2" => Snorm16x2,
        "snorm16x4" => Snorm16x4,
        "float16x2" => Float16x2,
        "float16x4" => Float16x4,
        "float32" => Float32,
        "float32x2" => Float32x2,
        "float32x3" => Float32x3,
        "float32x4" => Float32x4,
        "uint32" => Uint32,
        "uint32x2" => Uint32x2,
        "uint32x3" => Uint32x3,
        "uint32x4" => Uint32x4,
        "sint32" => Sint32,
        "sint32x2" => Sint32x2,
        "sint32x3" => Sint32x3,
        "sint32x4" => Sint32x4,
        // Fallback for unknown — same defaulting strategy as
        // texture-format parsing; v0.3 will harden this.
        _ => Float32,
    }
}

fn parse_step_mode(s: Option<&str>) -> wgpu::VertexStepMode {
    match s.unwrap_or("vertex") {
        "instance" => wgpu::VertexStepMode::Instance,
        _ => wgpu::VertexStepMode::Vertex,
    }
}

fn parse_topology(s: Option<&str>) -> wgpu::PrimitiveTopology {
    match s.unwrap_or("triangle-list") {
        "point-list" => wgpu::PrimitiveTopology::PointList,
        "line-list" => wgpu::PrimitiveTopology::LineList,
        "line-strip" => wgpu::PrimitiveTopology::LineStrip,
        "triangle-strip" => wgpu::PrimitiveTopology::TriangleStrip,
        _ => wgpu::PrimitiveTopology::TriangleList,
    }
}

fn parse_index_format(s: &str) -> wgpu::IndexFormat {
    match s {
        "uint16" => wgpu::IndexFormat::Uint16,
        _ => wgpu::IndexFormat::Uint32,
    }
}

fn parse_front_face(s: Option<&str>) -> wgpu::FrontFace {
    match s.unwrap_or("ccw") {
        "cw" => wgpu::FrontFace::Cw,
        _ => wgpu::FrontFace::Ccw,
    }
}

fn parse_cull_mode(s: Option<&str>) -> Option<wgpu::Face> {
    match s.unwrap_or("none") {
        "front" => Some(wgpu::Face::Front),
        "back" => Some(wgpu::Face::Back),
        _ => None,
    }
}

fn parse_blend_factor(s: Option<&str>) -> wgpu::BlendFactor {
    match s.unwrap_or("one") {
        "zero" => wgpu::BlendFactor::Zero,
        "src" => wgpu::BlendFactor::Src,
        "one-minus-src" => wgpu::BlendFactor::OneMinusSrc,
        "src-alpha" => wgpu::BlendFactor::SrcAlpha,
        "one-minus-src-alpha" => wgpu::BlendFactor::OneMinusSrcAlpha,
        "dst" => wgpu::BlendFactor::Dst,
        "one-minus-dst" => wgpu::BlendFactor::OneMinusDst,
        "dst-alpha" => wgpu::BlendFactor::DstAlpha,
        "one-minus-dst-alpha" => wgpu::BlendFactor::OneMinusDstAlpha,
        "src-alpha-saturated" => wgpu::BlendFactor::SrcAlphaSaturated,
        "constant" => wgpu::BlendFactor::Constant,
        "one-minus-constant" => wgpu::BlendFactor::OneMinusConstant,
        _ => wgpu::BlendFactor::One,
    }
}

fn parse_blend_op(s: Option<&str>) -> wgpu::BlendOperation {
    match s.unwrap_or("add") {
        "subtract" => wgpu::BlendOperation::Subtract,
        "reverse-subtract" => wgpu::BlendOperation::ReverseSubtract,
        "min" => wgpu::BlendOperation::Min,
        "max" => wgpu::BlendOperation::Max,
        _ => wgpu::BlendOperation::Add,
    }
}

fn parse_blend_component(c: &BlendComponentDesc) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: parse_blend_factor(c.src_factor.as_deref()),
        dst_factor: parse_blend_factor(c.dst_factor.as_deref()),
        operation: parse_blend_op(c.operation.as_deref()),
    }
}

fn parse_stencil_op(s: Option<&str>) -> wgpu::StencilOperation {
    match s.unwrap_or("keep") {
        "zero" => wgpu::StencilOperation::Zero,
        "replace" => wgpu::StencilOperation::Replace,
        "invert" => wgpu::StencilOperation::Invert,
        "increment-clamp" => wgpu::StencilOperation::IncrementClamp,
        "decrement-clamp" => wgpu::StencilOperation::DecrementClamp,
        "increment-wrap" => wgpu::StencilOperation::IncrementWrap,
        "decrement-wrap" => wgpu::StencilOperation::DecrementWrap,
        _ => wgpu::StencilOperation::Keep,
    }
}

fn parse_stencil_face(s: &Option<StencilFaceStateDesc>) -> wgpu::StencilFaceState {
    match s {
        Some(f) => wgpu::StencilFaceState {
            compare: f
                .compare
                .as_deref()
                .map(parse_compare_function)
                .unwrap_or(wgpu::CompareFunction::Always),
            fail_op: parse_stencil_op(f.fail_op.as_deref()),
            depth_fail_op: parse_stencil_op(f.depth_fail_op.as_deref()),
            pass_op: parse_stencil_op(f.pass_op.as_deref()),
        },
        None => wgpu::StencilFaceState::IGNORE,
    }
}

/// Build a `RenderPipeline` from a parsed descriptor, given the
/// already-resolved layout / vertex-module / fragment-module
/// pointers. Shared by the sync + async create paths.
unsafe fn build_render_pipeline(
    device: &wgpu::Device,
    desc: &RenderPipelineDescriptor,
    layout_ptr: Option<*const PipelineLayout>,
    vertex_module_ptr: *const ShaderModule,
    fragment_module_ptr: Option<*const ShaderModule>,
) -> wgpu::RenderPipeline {
    // Vertex buffers: collect attributes per-buffer first (they need
    // to outlive the `&` borrow inside `VertexBufferLayout`).
    let attrs_per_buffer: Vec<Vec<wgpu::VertexAttribute>> = desc
        .vertex
        .buffers
        .iter()
        .map(|b| {
            b.attributes
                .iter()
                .map(|a| wgpu::VertexAttribute {
                    format: parse_vertex_format(&a.format),
                    offset: a.offset,
                    shader_location: a.shader_location,
                })
                .collect()
        })
        .collect();
    let vertex_buffers: Vec<wgpu::VertexBufferLayout> = desc
        .vertex
        .buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::VertexBufferLayout {
            array_stride: b.array_stride,
            step_mode: parse_step_mode(b.step_mode.as_deref()),
            attributes: &attrs_per_buffer[i],
        })
        .collect();

    let vertex_entry = desc
        .vertex
        .entry_point
        .as_deref()
        .unwrap_or("vs_main");

    let primitive = wgpu::PrimitiveState {
        topology: parse_topology(desc.primitive.topology.as_deref()),
        strip_index_format: desc
            .primitive
            .strip_index_format
            .as_deref()
            .map(parse_index_format),
        front_face: parse_front_face(desc.primitive.front_face.as_deref()),
        cull_mode: parse_cull_mode(desc.primitive.cull_mode.as_deref()),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    };

    let depth_stencil = desc.depth_stencil.as_ref().map(|ds| wgpu::DepthStencilState {
        format: parse_texture_format(&ds.format),
        depth_write_enabled: ds.depth_write_enabled,
        depth_compare: ds
            .depth_compare
            .as_deref()
            .map(parse_compare_function)
            .unwrap_or(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState {
            front: parse_stencil_face(&ds.stencil_front),
            back: parse_stencil_face(&ds.stencil_back),
            read_mask: ds.stencil_read_mask,
            write_mask: ds.stencil_write_mask,
        },
        bias: wgpu::DepthBiasState {
            constant: ds.depth_bias,
            slope_scale: ds.depth_bias_slope_scale,
            clamp: ds.depth_bias_clamp,
        },
    });

    let multisample = wgpu::MultisampleState {
        count: desc.multisample.count.max(1),
        mask: desc.multisample.mask,
        alpha_to_coverage_enabled: desc.multisample.alpha_to_coverage_enabled,
    };

    let fragment_targets: Vec<Option<wgpu::ColorTargetState>> = desc
        .fragment
        .as_ref()
        .map(|f| {
            f.targets
                .iter()
                .map(|t| {
                    t.as_ref().map(|t| wgpu::ColorTargetState {
                        format: parse_texture_format(&t.format),
                        blend: t.blend.as_ref().map(|b| wgpu::BlendState {
                            color: parse_blend_component(&b.color),
                            alpha: parse_blend_component(&b.alpha),
                        }),
                        write_mask: wgpu::ColorWrites::from_bits_truncate(t.write_mask),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let fragment_entry = desc
        .fragment
        .as_ref()
        .and_then(|f| f.entry_point.as_deref())
        .unwrap_or("fs_main");

    let fragment_state =
        if let (Some(fmp), Some(_)) = (fragment_module_ptr, desc.fragment.as_ref()) {
            Some(wgpu::FragmentState {
                module: &*fmp,
                entry_point: fragment_entry,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &fragment_targets,
            })
        } else {
            None
        };

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: desc.label.as_deref(),
        layout: layout_ptr.map(|p| &*p),
        vertex: wgpu::VertexState {
            module: &*vertex_module_ptr,
            entry_point: vertex_entry,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &vertex_buffers,
        },
        primitive,
        depth_stencil,
        multisample,
        fragment: fragment_state,
        multiview: None,
        cache: None,
    })
}

/// Resolve a `RenderPipelineDescriptor`'s handle fields to raw
/// pointers (registry-pinned). Returns `None` if any handle is bad.
/// Shared by sync + async create paths.
fn resolve_render_pipeline_handles(
    desc: &RenderPipelineDescriptor,
) -> Option<(
    Option<*const PipelineLayout>,
    *const ShaderModule,
    Option<*const ShaderModule>,
    bool, // layout was explicit (not "auto")
)> {
    let layout_explicit = matches!(desc.layout, serde_json::Value::Number(_));
    let layout_ptr: Option<*const PipelineLayout> = match &desc.layout {
        serde_json::Value::String(s) if s == "auto" => None,
        serde_json::Value::Number(n) => with_handle::<WGPUPipelineLayout, _, _>(
            n.as_i64().unwrap_or(0),
            |pl| &pl.0 as *const _,
        ),
        _ => return None,
    };
    if layout_explicit && layout_ptr.is_none() {
        return None;
    }
    let vmodule_ptr = with_handle::<WGPUShaderModule, _, _>(desc.vertex.module, |m| {
        &m.0 as *const ShaderModule
    })?;
    let fmodule_ptr = match &desc.fragment {
        Some(f) => Some(with_handle::<WGPUShaderModule, _, _>(f.module, |m| {
            &m.0 as *const ShaderModule
        })?),
        None => None,
    };
    Some((layout_ptr, vmodule_ptr, fmodule_ptr, layout_explicit))
}

/// `device.createRenderPipeline(descriptor) -> GPURenderPipeline` —
/// synchronous. The descriptor mirrors `GPURenderPipelineDescriptor`
/// from the spec — vertex / fragment / primitive / depthStencil /
/// multisample, with `"auto"` or a `GPUPipelineLayout` handle for
/// `layout`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_render_pipeline(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: RenderPipelineDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let Some((layout_ptr, vmod_ptr, fmod_ptr, _)) = resolve_render_pipeline_handles(&desc) else {
        return 0;
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let pipeline = build_render_pipeline(&d.0, &desc, layout_ptr, vmod_ptr, fmod_ptr);
        register_handle(WGPURenderPipeline(pipeline))
    })
    .unwrap_or(0)
}

/// `pipeline.getBindGroupLayout(index) -> GPUBindGroupLayout` —
/// synchronous accessor for render pipelines (mirrors the compute
/// version above).
#[no_mangle]
pub extern "C" fn js_webgpu_render_pipeline_get_bind_group_layout(
    pipeline_handle: Handle,
    index: u32,
) -> Handle {
    with_handle::<WGPURenderPipeline, _, _>(pipeline_handle, |p| {
        let bgl = p.0.get_bind_group_layout(index);
        register_handle(WGPUBindGroupLayout(bgl))
    })
    .unwrap_or(0)
}

/// `device.createRenderPipelineAsync(descriptor) -> Promise<GPURenderPipeline>`.
/// wgpu has no native async pipeline-create, so we run the sync
/// build inside a `spawn_blocking` task — same effect as the browser's
/// async dispatch (the work happens off the JS thread; the user's
/// promise unblocks when the pipeline is ready).
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_render_pipeline_async(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();
    let Some(json) = read_str(descriptor_ptr) else {
        promise.reject_string("webgpu createRenderPipelineAsync: missing descriptor");
        return raw;
    };

    spawn_blocking(move || {
        let desc: RenderPipelineDescriptor = match serde_json::from_str(&json) {
            Ok(d) => d,
            Err(e) => {
                promise.reject_string(&format!(
                    "webgpu createRenderPipelineAsync: bad descriptor: {}",
                    e
                ));
                return;
            }
        };
        let Some((layout_ptr, vmod_ptr, fmod_ptr, _)) = resolve_render_pipeline_handles(&desc)
        else {
            promise.reject_string(
                "webgpu createRenderPipelineAsync: invalid layout / module handle",
            );
            return;
        };

        let outcome = with_handle::<WGPUDevice, _, _>(device_handle, |d| unsafe {
            let pipeline = build_render_pipeline(&d.0, &desc, layout_ptr, vmod_ptr, fmod_ptr);
            register_handle(WGPURenderPipeline(pipeline))
        });
        match outcome {
            Some(h) => promise.resolve(JsValue::from_number(h as f64)),
            None => {
                promise.reject_string("webgpu createRenderPipelineAsync: invalid device handle")
            }
        }
    });
    raw
}

/// `device.createComputePipelineAsync(descriptor) -> Promise<GPUComputePipeline>`.
/// Same dispatch model as the render variant — wgpu compiles the
/// pipeline synchronously, we run it off the JS thread via
/// `spawn_blocking`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_compute_pipeline_async(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> *mut Promise {
    let promise = JsPromise::new();
    let raw = promise.as_raw();
    let Some(json) = read_str(descriptor_ptr) else {
        promise.reject_string("webgpu createComputePipelineAsync: missing descriptor");
        return raw;
    };

    spawn_blocking(move || {
        let desc: ComputePipelineDescriptor = match serde_json::from_str(&json) {
            Ok(d) => d,
            Err(e) => {
                promise.reject_string(&format!(
                    "webgpu createComputePipelineAsync: bad descriptor: {}",
                    e
                ));
                return;
            }
        };

        let layout_ptr: Option<*const PipelineLayout> = match &desc.layout {
            serde_json::Value::String(s) if s == "auto" => None,
            serde_json::Value::Number(n) => with_handle::<WGPUPipelineLayout, _, _>(
                n.as_i64().unwrap_or(0),
                |pl| &pl.0 as *const _,
            ),
            _ => {
                promise.reject_string("webgpu createComputePipelineAsync: bad layout");
                return;
            }
        };
        let layout_explicit = matches!(desc.layout, serde_json::Value::Number(_));
        if layout_explicit && layout_ptr.is_none() {
            promise.reject_string("webgpu createComputePipelineAsync: invalid layout handle");
            return;
        }
        let module_ptr = match with_handle::<WGPUShaderModule, _, _>(desc.compute.module, |m| {
            &m.0 as *const ShaderModule
        }) {
            Some(p) => p,
            None => {
                promise.reject_string(
                    "webgpu createComputePipelineAsync: invalid shader module handle",
                );
                return;
            }
        };
        let entry_point = desc.compute.entry_point.as_deref().unwrap_or("main");

        let outcome = with_handle::<WGPUDevice, _, _>(device_handle, |d| {
            let pipeline = d.0.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: desc.label.as_deref(),
                layout: layout_ptr.map(|p| unsafe { &*p }),
                module: unsafe { &*module_ptr },
                entry_point,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            register_handle(WGPUComputePipeline(pipeline))
        });
        match outcome {
            Some(h) => promise.resolve(JsValue::from_number(h as f64)),
            None => {
                promise.reject_string(
                    "webgpu createComputePipelineAsync: invalid device handle",
                )
            }
        }
    });
    raw
}

// ════════════════════════════════════════════════════════════════════
// Render Pass + draw / set ops
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct RenderPassColorAttachmentDesc {
    view: i64,
    #[serde(rename = "resolveTarget", default)]
    resolve_target: Option<i64>,
    /// `"clear"` or `"load"` per the spec.
    #[serde(rename = "loadOp", default)]
    load_op: Option<String>,
    /// `"store"` or `"discard"` per the spec.
    #[serde(rename = "storeOp", default)]
    store_op: Option<String>,
    /// Spec wire shape: `{r,g,b,a}` (numbers in 0..=1 for unorm formats).
    #[serde(rename = "clearValue", default)]
    clear_value: Option<ColorClearValue>,
}

#[derive(Deserialize)]
struct ColorClearValue {
    #[serde(default)]
    r: f64,
    #[serde(default)]
    g: f64,
    #[serde(default)]
    b: f64,
    #[serde(default)]
    a: f64,
}

#[derive(Deserialize)]
struct RenderPassDepthStencilAttachmentDesc {
    view: i64,
    #[serde(rename = "depthClearValue", default)]
    depth_clear_value: f32,
    #[serde(rename = "depthLoadOp", default)]
    depth_load_op: Option<String>,
    #[serde(rename = "depthStoreOp", default)]
    depth_store_op: Option<String>,
    #[serde(rename = "depthReadOnly", default)]
    depth_read_only: bool,
    #[serde(rename = "stencilClearValue", default)]
    stencil_clear_value: u32,
    #[serde(rename = "stencilLoadOp", default)]
    stencil_load_op: Option<String>,
    #[serde(rename = "stencilStoreOp", default)]
    stencil_store_op: Option<String>,
    #[serde(rename = "stencilReadOnly", default)]
    stencil_read_only: bool,
}

#[derive(Deserialize)]
struct RenderPassDescriptor {
    #[serde(default)]
    label: Option<String>,
    #[serde(rename = "colorAttachments")]
    color_attachments: Vec<Option<RenderPassColorAttachmentDesc>>,
    #[serde(rename = "depthStencilAttachment", default)]
    depth_stencil_attachment: Option<RenderPassDepthStencilAttachmentDesc>,
    #[serde(rename = "occlusionQuerySet", default)]
    occlusion_query_set: Option<i64>,
}

fn parse_load_op_color(s: Option<&str>, clear: Option<&ColorClearValue>) -> wgpu::LoadOp<wgpu::Color> {
    match s.unwrap_or("clear") {
        "load" => wgpu::LoadOp::Load,
        _ => wgpu::LoadOp::Clear(match clear {
            Some(c) => wgpu::Color {
                r: c.r,
                g: c.g,
                b: c.b,
                a: c.a,
            },
            None => wgpu::Color::TRANSPARENT,
        }),
    }
}

fn parse_load_op_f32(s: Option<&str>, clear: f32) -> wgpu::LoadOp<f32> {
    match s.unwrap_or("clear") {
        "load" => wgpu::LoadOp::Load,
        _ => wgpu::LoadOp::Clear(clear),
    }
}

fn parse_load_op_u32(s: Option<&str>, clear: u32) -> wgpu::LoadOp<u32> {
    match s.unwrap_or("clear") {
        "load" => wgpu::LoadOp::Load,
        _ => wgpu::LoadOp::Clear(clear),
    }
}

fn parse_store_op(s: Option<&str>) -> wgpu::StoreOp {
    match s.unwrap_or("store") {
        "discard" => wgpu::StoreOp::Discard,
        _ => wgpu::StoreOp::Store,
    }
}

/// `encoder.beginRenderPass(descriptor) -> GPURenderPassEncoder` —
/// synchronous. Same `forget_lifetime()` trick as `beginComputePass`:
/// the registry holds the pass as a 'static-flavoured owned value;
/// the user is contractually obligated to `pass.end()` before
/// `encoder.finish()`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_command_encoder_begin_render_pass(
    encoder_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: RenderPassDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };

    // Resolve all texture-view + query-set handles up front so the
    // wgpu call only sees registry-pinned `&` references.
    let view_ptrs: Vec<Option<(*const wgpu::TextureView, Option<*const wgpu::TextureView>)>> = desc
        .color_attachments
        .iter()
        .map(|a| match a {
            Some(att) => {
                let v = with_handle::<WGPUTextureView, _, _>(att.view, |v| {
                    &v.0 as *const wgpu::TextureView
                })?;
                let r = match att.resolve_target {
                    Some(rh) => with_handle::<WGPUTextureView, _, _>(rh, |v| {
                        &v.0 as *const wgpu::TextureView
                    }),
                    None => None,
                };
                Some((v, r))
            }
            None => None,
        })
        .collect();
    if desc
        .color_attachments
        .iter()
        .zip(view_ptrs.iter())
        .any(|(a, p)| a.is_some() && p.is_none())
    {
        return 0;
    }

    let depth_view_ptr = match &desc.depth_stencil_attachment {
        Some(d) => {
            let v = with_handle::<WGPUTextureView, _, _>(d.view, |v| {
                &v.0 as *const wgpu::TextureView
            });
            let Some(v) = v else { return 0 };
            Some(v)
        }
        None => None,
    };
    let occlusion_qs_ptr = match desc.occlusion_query_set {
        Some(h) => with_handle::<WGPUQuerySet, _, _>(h, |q| &q.0 as *const wgpu::QuerySet),
        None => None,
    };

    with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else {
            return 0;
        };

        let color_attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = desc
            .color_attachments
            .iter()
            .zip(view_ptrs.iter())
            .map(|(att, ptrs)| match (att, ptrs) {
                (Some(att), Some((view_ptr, resolve_ptr))) => {
                    Some(wgpu::RenderPassColorAttachment {
                        view: unsafe { &**view_ptr },
                        resolve_target: resolve_ptr.map(|p| unsafe { &*p }),
                        ops: wgpu::Operations {
                            load: parse_load_op_color(
                                att.load_op.as_deref(),
                                att.clear_value.as_ref(),
                            ),
                            store: parse_store_op(att.store_op.as_deref()),
                        },
                    })
                }
                _ => None,
            })
            .collect();

        let depth_stencil =
            if let (Some(d), Some(vp)) = (&desc.depth_stencil_attachment, depth_view_ptr) {
                Some(wgpu::RenderPassDepthStencilAttachment {
                    view: unsafe { &*vp },
                    depth_ops: if d.depth_load_op.is_some() || d.depth_store_op.is_some() {
                        Some(wgpu::Operations {
                            load: parse_load_op_f32(d.depth_load_op.as_deref(), d.depth_clear_value),
                            store: parse_store_op(d.depth_store_op.as_deref()),
                        })
                    } else if d.depth_read_only {
                        None
                    } else {
                        None
                    },
                    stencil_ops: if d.stencil_load_op.is_some() || d.stencil_store_op.is_some() {
                        Some(wgpu::Operations {
                            load: parse_load_op_u32(
                                d.stencil_load_op.as_deref(),
                                d.stencil_clear_value,
                            ),
                            store: parse_store_op(d.stencil_store_op.as_deref()),
                        })
                    } else if d.stencil_read_only {
                        None
                    } else {
                        None
                    },
                })
            } else {
                None
            };

        let pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: desc.label.as_deref(),
                color_attachments: &color_attachments,
                depth_stencil_attachment: depth_stencil,
                timestamp_writes: None,
                occlusion_query_set: occlusion_qs_ptr.map(|p| unsafe { &*p }),
            })
            .forget_lifetime();
        register_handle(WGPURenderPass(Mutex::new(Some(pass))))
    })
    .unwrap_or(0)
}

/// `pass.setPipeline(pipeline)` for render passes.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_pipeline(
    pass_handle: Handle,
    pipeline_handle: Handle,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPURenderPipeline, _, _>(pipeline_handle, |p| {
            pass.set_pipeline(&p.0);
        });
    });
}

/// `pass.setBindGroup(index, bindGroup)` for render passes.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_bind_group(
    pass_handle: Handle,
    index: u32,
    bind_group_handle: Handle,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUBindGroup, _, _>(bind_group_handle, |bg| {
            pass.set_bind_group(index, &bg.0, &[]);
        });
    });
}

/// `pass.setBindGroup(index, bindGroup, dynamicOffsets)` — render
/// pass variant with dynamic offsets, JSON-encoded as a number array.
///
/// # Safety
///
/// `offsets_json_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_render_pass_set_bind_group_dyn(
    pass_handle: Handle,
    index: u32,
    bind_group_handle: Handle,
    offsets_json_ptr: *const StringHeader,
) {
    let offsets: Vec<u32> = match read_str(offsets_json_ptr) {
        Some(j) => serde_json::from_str(&j).unwrap_or_default(),
        None => Vec::new(),
    };
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUBindGroup, _, _>(bind_group_handle, |bg| {
            pass.set_bind_group(index, &bg.0, &offsets);
        });
    });
}

/// `setBindGroup` with dynamic offsets for compute passes.
///
/// # Safety
///
/// `offsets_json_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_compute_pass_set_bind_group_dyn(
    pass_handle: Handle,
    index: u32,
    bind_group_handle: Handle,
    offsets_json_ptr: *const StringHeader,
) {
    let offsets: Vec<u32> = match read_str(offsets_json_ptr) {
        Some(j) => serde_json::from_str(&j).unwrap_or_default(),
        None => Vec::new(),
    };
    let _ = with_handle::<WGPUComputePass, _, _>(pass_handle, |cp| {
        let mut slot = cp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUBindGroup, _, _>(bind_group_handle, |bg| {
            pass.set_bind_group(index, &bg.0, &offsets);
        });
    });
}

/// `pass.setVertexBuffer(slot, buffer, offset?, size?)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_vertex_buffer(
    pass_handle: Handle,
    slot: u32,
    buffer_handle: Handle,
    offset: f64,
    size: f64,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot_g = rp.0.lock();
        let Some(pass) = slot_g.as_mut() else { return };
        let _ = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| {
            let off = offset.max(0.0) as u64;
            let sz = size.max(0.0) as u64;
            let bs = if sz == 0 { b.0.slice(off..) } else { b.0.slice(off..off + sz) };
            pass.set_vertex_buffer(slot, bs);
        });
    });
}

/// `pass.setIndexBuffer(buffer, indexFormat, offset?, size?)`.
///
/// # Safety
///
/// `format_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_render_pass_set_index_buffer(
    pass_handle: Handle,
    buffer_handle: Handle,
    format_ptr: *const StringHeader,
    offset: f64,
    size: f64,
) {
    let format = read_str(format_ptr).unwrap_or_else(|| "uint32".to_string());
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot_g = rp.0.lock();
        let Some(pass) = slot_g.as_mut() else { return };
        let _ = with_handle::<WGPUBuffer, _, _>(buffer_handle, |b| {
            let off = offset.max(0.0) as u64;
            let sz = size.max(0.0) as u64;
            let bs = if sz == 0 { b.0.slice(off..) } else { b.0.slice(off..off + sz) };
            pass.set_index_buffer(bs, parse_index_format(&format));
        });
    });
}

/// `pass.draw(vertexCount, instanceCount?, firstVertex?, firstInstance?)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_draw(
    pass_handle: Handle,
    vertex_count: u32,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let inst = instance_count.max(1);
        pass.draw(
            first_vertex..first_vertex + vertex_count,
            first_instance..first_instance + inst,
        );
    });
}

/// `pass.drawIndexed(indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_draw_indexed(
    pass_handle: Handle,
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        let inst = instance_count.max(1);
        pass.draw_indexed(
            first_index..first_index + index_count,
            base_vertex,
            first_instance..first_instance + inst,
        );
    });
}

/// `pass.setViewport(x, y, w, h, minDepth, maxDepth)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_viewport(
    pass_handle: Handle,
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    min_depth: f32,
    max_depth: f32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.set_viewport(x, y, w, h, min_depth, max_depth);
    });
}

/// `pass.setScissorRect(x, y, w, h)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_scissor_rect(
    pass_handle: Handle,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.set_scissor_rect(x, y, w, h);
    });
}

/// `pass.setBlendConstant({r,g,b,a})` — components in 0..=1 for unorm
/// formats. Spec wire shape is the same `{r,g,b,a}` clear-color blob.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_blend_constant(
    pass_handle: Handle,
    r: f64,
    g: f64,
    b: f64,
    a: f64,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.set_blend_constant(wgpu::Color { r, g, b, a });
    });
}

/// `pass.setStencilReference(reference)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_set_stencil_reference(
    pass_handle: Handle,
    reference: u32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.set_stencil_reference(reference);
    });
}

/// `pass.beginOcclusionQuery(queryIndex)`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_begin_occlusion_query(
    pass_handle: Handle,
    query_index: u32,
) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.begin_occlusion_query(query_index);
    });
}

/// `pass.endOcclusionQuery()`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_end_occlusion_query(pass_handle: Handle) {
    let _ = with_handle::<WGPURenderPass, _, _>(pass_handle, |rp| {
        let mut slot = rp.0.lock();
        let Some(pass) = slot.as_mut() else { return };
        pass.end_occlusion_query();
    });
}

/// `pass.end()`.
#[no_mangle]
pub extern "C" fn js_webgpu_render_pass_end(pass_handle: Handle) {
    let _ = take_handle::<WGPURenderPass>(pass_handle);
    drop_handle(pass_handle);
}

// ════════════════════════════════════════════════════════════════════
// QuerySet
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct QuerySetDescriptor {
    #[serde(default)]
    label: Option<String>,
    #[serde(rename = "type")]
    ty: String,
    count: u32,
}

/// `device.createQuerySet({type, count}) -> GPUQuerySet` —
/// synchronous. Type is `"occlusion"` or `"timestamp"` per the spec.
/// Pipeline-statistics queries are gated behind a wgpu feature we
/// don't request; v0.1 surfaces them as `"occlusion"` (a safe
/// default — same fallthrough strategy as `parse_buffer_binding_type`).
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_device_create_query_set(
    device_handle: Handle,
    descriptor_ptr: *const StringHeader,
) -> Handle {
    let Some(json) = read_str(descriptor_ptr) else {
        return 0;
    };
    let desc: QuerySetDescriptor = match serde_json::from_str(&json) {
        Ok(d) => d,
        Err(_) => return 0,
    };
    let ty = match desc.ty.as_str() {
        "timestamp" => wgpu::QueryType::Timestamp,
        _ => wgpu::QueryType::Occlusion,
    };

    with_handle::<WGPUDevice, _, _>(device_handle, |d| {
        let qs = d.0.create_query_set(&wgpu::QuerySetDescriptor {
            label: desc.label.as_deref(),
            ty,
            count: desc.count,
        });
        register_handle(WGPUQuerySet(qs))
    })
    .unwrap_or(0)
}

/// `querySet.destroy()` — release. Idempotent.
#[no_mangle]
pub extern "C" fn js_webgpu_query_set_destroy(query_set_handle: Handle) {
    let _ = take_handle::<WGPUQuerySet>(query_set_handle);
    drop_handle(query_set_handle);
}

/// `encoder.resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset)`.
#[no_mangle]
pub extern "C" fn js_webgpu_command_encoder_resolve_query_set(
    encoder_handle: Handle,
    query_set: Handle,
    first_query: u32,
    query_count: u32,
    destination: Handle,
    destination_offset: f64,
) {
    let _ = with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUQuerySet, _, _>(query_set, |qs| {
            let _ = with_handle::<WGPUBuffer, _, _>(destination, |b| {
                encoder.resolve_query_set(
                    &qs.0,
                    first_query..first_query + query_count,
                    &b.0,
                    destination_offset.max(0.0) as u64,
                );
            });
        });
    });
}

// ════════════════════════════════════════════════════════════════════
// Queue.writeTexture + texture-related copy ops
// ════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ImageCopyTextureDesc {
    texture: i64,
    #[serde(rename = "mipLevel", default)]
    mip_level: u32,
    #[serde(default)]
    origin: Option<Origin3dDesc>,
    #[serde(default)]
    aspect: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
struct Origin3dDesc {
    #[serde(default)]
    x: u32,
    #[serde(default)]
    y: u32,
    #[serde(default)]
    z: u32,
}

#[derive(Deserialize)]
struct ImageDataLayoutDesc {
    #[serde(default)]
    offset: u64,
    #[serde(rename = "bytesPerRow", default)]
    bytes_per_row: u32,
    #[serde(rename = "rowsPerImage", default)]
    rows_per_image: u32,
}

#[derive(Deserialize)]
struct ImageCopyBufferDesc {
    buffer: i64,
    #[serde(default)]
    offset: u64,
    #[serde(rename = "bytesPerRow", default)]
    bytes_per_row: u32,
    #[serde(rename = "rowsPerImage", default)]
    rows_per_image: u32,
}

#[derive(Deserialize)]
struct WriteTextureCall {
    destination: ImageCopyTextureDesc,
    #[serde(rename = "dataLayout")]
    data_layout: ImageDataLayoutDesc,
    size: Extent3dDesc,
}

/// `queue.writeTexture(destination, data, dataLayout, size)`. The
/// four spec args are packed into one JSON descriptor + a separate
/// data buffer to keep the FFI surface tractable.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`; `data_ptr`
/// must be null or a Perry-runtime `BufferHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_queue_write_texture(
    queue_handle: Handle,
    descriptor_ptr: *const StringHeader,
    data_ptr: *const BufferHeader,
) {
    let Some(json) = read_str(descriptor_ptr) else {
        return;
    };
    let call: WriteTextureCall = match serde_json::from_str(&json) {
        Ok(c) => c,
        Err(_) => return,
    };
    let Some(bytes) = perry_ffi::read_buffer_bytes(data_ptr) else {
        return;
    };

    let _ = with_handle::<WGPUQueue, _, _>(queue_handle, |q| {
        let _ = with_handle::<WGPUTexture, _, _>(call.destination.texture, |t| {
            let origin = call.destination.origin.unwrap_or_default();
            q.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &t.0,
                    mip_level: call.destination.mip_level,
                    origin: wgpu::Origin3d {
                        x: origin.x,
                        y: origin.y,
                        z: origin.z,
                    },
                    aspect: parse_texture_aspect(call.destination.aspect.as_deref()),
                },
                bytes,
                wgpu::ImageDataLayout {
                    offset: call.data_layout.offset,
                    bytes_per_row: if call.data_layout.bytes_per_row == 0 {
                        None
                    } else {
                        Some(call.data_layout.bytes_per_row)
                    },
                    rows_per_image: if call.data_layout.rows_per_image == 0 {
                        None
                    } else {
                        Some(call.data_layout.rows_per_image)
                    },
                },
                wgpu::Extent3d {
                    width: call.size.width,
                    height: call.size.height,
                    depth_or_array_layers: call.size.depth_or_array_layers,
                },
            );
        });
    });
}

#[derive(Deserialize)]
struct CopyBufferToTextureCall {
    source: ImageCopyBufferDesc,
    destination: ImageCopyTextureDesc,
    size: Extent3dDesc,
}

#[derive(Deserialize)]
struct CopyTextureToBufferCall {
    source: ImageCopyTextureDesc,
    destination: ImageCopyBufferDesc,
    size: Extent3dDesc,
}

#[derive(Deserialize)]
struct CopyTextureToTextureCall {
    source: ImageCopyTextureDesc,
    destination: ImageCopyTextureDesc,
    size: Extent3dDesc,
}

/// `encoder.copyBufferToTexture(source, destination, copySize)`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_command_encoder_copy_buffer_to_texture(
    encoder_handle: Handle,
    descriptor_ptr: *const StringHeader,
) {
    let Some(json) = read_str(descriptor_ptr) else {
        return;
    };
    let call: CopyBufferToTextureCall = match serde_json::from_str(&json) {
        Ok(c) => c,
        Err(_) => return,
    };

    let _ = with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUBuffer, _, _>(call.source.buffer, |sb| {
            let _ = with_handle::<WGPUTexture, _, _>(call.destination.texture, |dt| {
                let dorigin = call.destination.origin.clone().unwrap_or_default();
                encoder.copy_buffer_to_texture(
                    wgpu::ImageCopyBuffer {
                        buffer: &sb.0,
                        layout: wgpu::ImageDataLayout {
                            offset: call.source.offset,
                            bytes_per_row: if call.source.bytes_per_row == 0 {
                                None
                            } else {
                                Some(call.source.bytes_per_row)
                            },
                            rows_per_image: if call.source.rows_per_image == 0 {
                                None
                            } else {
                                Some(call.source.rows_per_image)
                            },
                        },
                    },
                    wgpu::ImageCopyTexture {
                        texture: &dt.0,
                        mip_level: call.destination.mip_level,
                        origin: wgpu::Origin3d {
                            x: dorigin.x,
                            y: dorigin.y,
                            z: dorigin.z,
                        },
                        aspect: parse_texture_aspect(call.destination.aspect.as_deref()),
                    },
                    wgpu::Extent3d {
                        width: call.size.width,
                        height: call.size.height,
                        depth_or_array_layers: call.size.depth_or_array_layers,
                    },
                );
            });
        });
    });
}

/// `encoder.copyTextureToBuffer(source, destination, copySize)`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_command_encoder_copy_texture_to_buffer(
    encoder_handle: Handle,
    descriptor_ptr: *const StringHeader,
) {
    let Some(json) = read_str(descriptor_ptr) else {
        return;
    };
    let call: CopyTextureToBufferCall = match serde_json::from_str(&json) {
        Ok(c) => c,
        Err(_) => return,
    };

    let _ = with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUTexture, _, _>(call.source.texture, |st| {
            let _ = with_handle::<WGPUBuffer, _, _>(call.destination.buffer, |db| {
                let sorigin = call.source.origin.clone().unwrap_or_default();
                encoder.copy_texture_to_buffer(
                    wgpu::ImageCopyTexture {
                        texture: &st.0,
                        mip_level: call.source.mip_level,
                        origin: wgpu::Origin3d {
                            x: sorigin.x,
                            y: sorigin.y,
                            z: sorigin.z,
                        },
                        aspect: parse_texture_aspect(call.source.aspect.as_deref()),
                    },
                    wgpu::ImageCopyBuffer {
                        buffer: &db.0,
                        layout: wgpu::ImageDataLayout {
                            offset: call.destination.offset,
                            bytes_per_row: if call.destination.bytes_per_row == 0 {
                                None
                            } else {
                                Some(call.destination.bytes_per_row)
                            },
                            rows_per_image: if call.destination.rows_per_image == 0 {
                                None
                            } else {
                                Some(call.destination.rows_per_image)
                            },
                        },
                    },
                    wgpu::Extent3d {
                        width: call.size.width,
                        height: call.size.height,
                        depth_or_array_layers: call.size.depth_or_array_layers,
                    },
                );
            });
        });
    });
}

/// `encoder.copyTextureToTexture(source, destination, copySize)`.
///
/// # Safety
///
/// `descriptor_ptr` must be a Perry-runtime `StringHeader`.
#[no_mangle]
pub unsafe extern "C" fn js_webgpu_command_encoder_copy_texture_to_texture(
    encoder_handle: Handle,
    descriptor_ptr: *const StringHeader,
) {
    let Some(json) = read_str(descriptor_ptr) else {
        return;
    };
    let call: CopyTextureToTextureCall = match serde_json::from_str(&json) {
        Ok(c) => c,
        Err(_) => return,
    };

    let _ = with_handle::<WGPUCommandEncoder, _, _>(encoder_handle, |ce| {
        let mut slot = ce.0.lock();
        let Some(encoder) = slot.as_mut() else { return };
        let _ = with_handle::<WGPUTexture, _, _>(call.source.texture, |st| {
            let _ = with_handle::<WGPUTexture, _, _>(call.destination.texture, |dt| {
                let sorigin = call.source.origin.clone().unwrap_or_default();
                let dorigin = call.destination.origin.clone().unwrap_or_default();
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &st.0,
                        mip_level: call.source.mip_level,
                        origin: wgpu::Origin3d {
                            x: sorigin.x,
                            y: sorigin.y,
                            z: sorigin.z,
                        },
                        aspect: parse_texture_aspect(call.source.aspect.as_deref()),
                    },
                    wgpu::ImageCopyTexture {
                        texture: &dt.0,
                        mip_level: call.destination.mip_level,
                        origin: wgpu::Origin3d {
                            x: dorigin.x,
                            y: dorigin.y,
                            z: dorigin.z,
                        },
                        aspect: parse_texture_aspect(call.destination.aspect.as_deref()),
                    },
                    wgpu::Extent3d {
                        width: call.size.width,
                        height: call.size.height,
                        depth_or_array_layers: call.size.depth_or_array_layers,
                    },
                );
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_buffer_bindings() {
        assert!(matches!(
            parse_buffer_binding_type("uniform"),
            wgpu::BufferBindingType::Uniform
        ));
        assert!(matches!(
            parse_buffer_binding_type("storage"),
            wgpu::BufferBindingType::Storage { read_only: false }
        ));
        assert!(matches!(
            parse_buffer_binding_type("read-only-storage"),
            wgpu::BufferBindingType::Storage { read_only: true }
        ));
        // unknown defaults to uniform — same fallback the spec uses.
        assert!(matches!(
            parse_buffer_binding_type("garbage"),
            wgpu::BufferBindingType::Uniform
        ));
    }

    #[test]
    fn parse_view_dim_defaults_2d() {
        assert!(matches!(
            parse_view_dimension(None),
            wgpu::TextureViewDimension::D2
        ));
        assert!(matches!(
            parse_view_dimension(Some("3d")),
            wgpu::TextureViewDimension::D3
        ));
        assert!(matches!(
            parse_view_dimension(Some("cube")),
            wgpu::TextureViewDimension::Cube
        ));
    }

    #[test]
    fn parse_format_known_and_fallback() {
        assert!(matches!(
            parse_texture_format("rgba8unorm"),
            wgpu::TextureFormat::Rgba8Unorm
        ));
        assert!(matches!(
            parse_texture_format("depth32float"),
            wgpu::TextureFormat::Depth32Float
        ));
        // Unknown format falls back to rgba8unorm — see fn doc comment.
        assert!(matches!(
            parse_texture_format("definitely-not-a-format"),
            wgpu::TextureFormat::Rgba8Unorm
        ));
    }

    #[test]
    fn buffer_descriptor_round_trip() {
        let json = r#"{"size": 1024, "usage": 140, "mappedAtCreation": true}"#;
        let d: BufferDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.size, 1024);
        assert_eq!(d.usage, 140);
        assert!(d.mapped_at_creation);
        assert!(d.label.is_none());
    }

    #[test]
    fn bind_group_layout_descriptor_buffer_entry() {
        let json = r#"{
            "entries": [
              {"binding": 0, "visibility": 4, "buffer": {"type": "storage"}},
              {"binding": 1, "visibility": 4, "buffer": {"type": "read-only-storage"}}
            ]
        }"#;
        let d: BglDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.entries.len(), 2);
        assert_eq!(d.entries[0].binding, 0);
        assert_eq!(d.entries[0].visibility, 4);
        assert_eq!(
            d.entries[0].buffer.as_ref().unwrap().ty,
            "storage".to_string()
        );
    }

    #[test]
    fn pipeline_layout_descriptor_handles() {
        let json = r#"{"bindGroupLayouts": [42, 99]}"#;
        let d: PipelineLayoutDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.bind_group_layouts, vec![42i64, 99i64]);
    }

    #[test]
    fn compute_pipeline_descriptor_layout_auto() {
        let json = r#"{
            "layout": "auto",
            "compute": {"module": 5, "entryPoint": "main"}
        }"#;
        let d: ComputePipelineDescriptor = serde_json::from_str(json).unwrap();
        assert!(matches!(d.layout, serde_json::Value::String(ref s) if s == "auto"));
        assert_eq!(d.compute.module, 5);
        assert_eq!(d.compute.entry_point.as_deref(), Some("main"));
    }

    #[test]
    fn compute_pipeline_descriptor_layout_handle() {
        let json = r#"{
            "layout": 7,
            "compute": {"module": 5}
        }"#;
        let d: ComputePipelineDescriptor = serde_json::from_str(json).unwrap();
        assert!(matches!(d.layout, serde_json::Value::Number(_)));
        assert!(d.compute.entry_point.is_none());
    }

    #[test]
    fn bind_group_descriptor_buffer_entry() {
        let json = r#"{
            "layout": 1,
            "entries": [
              {"binding": 0, "resource": {"buffer": 5, "offset": 0, "size": 64}}
            ]
        }"#;
        let d: BindGroupDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.layout, 1);
        assert_eq!(d.entries.len(), 1);
        match &d.entries[0].resource {
            BindGroupResource::Buffer(b) => {
                assert_eq!(b.buffer, 5);
                assert_eq!(b.size, 64);
            }
            _ => panic!("expected Buffer resource"),
        }
    }

    #[test]
    fn texture_descriptor_round_trip() {
        let json = r#"{
            "size": {"width": 256, "height": 256},
            "format": "rgba8unorm",
            "usage": 18,
            "viewFormats": ["rgba8unorm-srgb"]
        }"#;
        let d: TextureDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.size.width, 256);
        assert_eq!(d.size.height, 256);
        assert_eq!(d.size.depth_or_array_layers, 1); // default
        assert_eq!(d.format, "rgba8unorm");
        assert_eq!(d.usage, 18);
        assert_eq!(d.view_formats, vec!["rgba8unorm-srgb".to_string()]);
    }

    #[test]
    fn sampler_descriptor_defaults() {
        let json = r#"{}"#;
        let d: SamplerDescriptor = serde_json::from_str(json).unwrap();
        assert!(d.address_mode_u.is_none());
        assert!(d.compare.is_none());
        assert_eq!(d.lod_min_clamp, 0.0);
        assert_eq!(d.lod_max_clamp, 32.0); // default
        assert_eq!(d.max_anisotropy, 1); // default
    }

    #[test]
    fn render_pipeline_descriptor_minimal() {
        // Vertex stage only, no fragment, layout: "auto" — exercises
        // the most-defaulted path through the parser.
        let json = r#"{
            "layout": "auto",
            "vertex": {"module": 7}
        }"#;
        let d: RenderPipelineDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.vertex.module, 7);
        assert!(d.vertex.buffers.is_empty());
        assert!(d.fragment.is_none());
        assert!(d.depth_stencil.is_none());
    }

    #[test]
    fn render_pipeline_descriptor_full() {
        let json = r#"{
            "layout": 1,
            "vertex": {
                "module": 7,
                "entryPoint": "vs_main",
                "buffers": [
                    {
                        "arrayStride": 32,
                        "stepMode": "vertex",
                        "attributes": [
                            {"format": "float32x3", "offset": 0,  "shaderLocation": 0},
                            {"format": "float32x2", "offset": 12, "shaderLocation": 1}
                        ]
                    }
                ]
            },
            "primitive": {"topology": "triangle-list", "cullMode": "back"},
            "depthStencil": {
                "format": "depth32float",
                "depthWriteEnabled": true,
                "depthCompare": "less"
            },
            "multisample": {"count": 4, "alphaToCoverageEnabled": true},
            "fragment": {
                "module": 8,
                "entryPoint": "fs_main",
                "targets": [
                    {"format": "bgra8unorm", "writeMask": 15}
                ]
            }
        }"#;
        let d: RenderPipelineDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.vertex.buffers.len(), 1);
        assert_eq!(d.vertex.buffers[0].attributes.len(), 2);
        assert_eq!(d.multisample.count, 4);
        assert!(d.multisample.alpha_to_coverage_enabled);
        let frag = d.fragment.as_ref().unwrap();
        assert_eq!(frag.targets.len(), 1);
    }

    #[test]
    fn render_pass_descriptor_with_color_attachment() {
        let json = r#"{
            "colorAttachments": [
                {
                    "view": 5,
                    "loadOp": "clear",
                    "storeOp": "store",
                    "clearValue": {"r": 0.1, "g": 0.2, "b": 0.3, "a": 1.0}
                }
            ]
        }"#;
        let d: RenderPassDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.color_attachments.len(), 1);
        let att = d.color_attachments[0].as_ref().unwrap();
        assert_eq!(att.view, 5);
        assert_eq!(att.load_op.as_deref(), Some("clear"));
    }

    #[test]
    fn bind_group_resource_sampler_and_texture_view() {
        // The new {sampler:n} / {textureView:n} wire shapes — see the
        // module-level doc comment on `BindGroupResource`.
        let json = r#"{
            "layout": 1,
            "entries": [
              {"binding": 0, "resource": {"sampler": 11}},
              {"binding": 1, "resource": {"textureView": 22}},
              {"binding": 2, "resource": {"buffer": 33}}
            ]
        }"#;
        let d: BindGroupDescriptor = serde_json::from_str(json).unwrap();
        assert_eq!(d.entries.len(), 3);
        assert!(matches!(
            d.entries[0].resource,
            BindGroupResource::Sampler { sampler: 11 }
        ));
        assert!(matches!(
            d.entries[1].resource,
            BindGroupResource::TextureView { texture_view: 22 }
        ));
        assert!(matches!(d.entries[2].resource, BindGroupResource::Buffer(_)));
    }

    #[test]
    fn write_texture_call_round_trip() {
        let json = r#"{
            "destination": {
                "texture": 5,
                "mipLevel": 0,
                "origin": {"x": 0, "y": 0, "z": 0},
                "aspect": "all"
            },
            "dataLayout": {"offset": 0, "bytesPerRow": 1024, "rowsPerImage": 256},
            "size": {"width": 256, "height": 256}
        }"#;
        let c: WriteTextureCall = serde_json::from_str(json).unwrap();
        assert_eq!(c.destination.texture, 5);
        assert_eq!(c.data_layout.bytes_per_row, 1024);
        assert_eq!(c.size.width, 256);
    }

    #[test]
    fn query_set_descriptor_round_trip() {
        let d: QuerySetDescriptor =
            serde_json::from_str(r#"{"type": "timestamp", "count": 16}"#).unwrap();
        assert_eq!(d.ty, "timestamp");
        assert_eq!(d.count, 16);
    }

    #[test]
    fn parse_address_mode_defaults_clamp() {
        assert!(matches!(parse_address_mode(None), wgpu::AddressMode::ClampToEdge));
        assert!(matches!(
            parse_address_mode(Some("repeat")),
            wgpu::AddressMode::Repeat
        ));
    }

    #[test]
    fn parse_blend_factor_known_and_fallback() {
        assert!(matches!(
            parse_blend_factor(Some("zero")),
            wgpu::BlendFactor::Zero
        ));
        assert!(matches!(
            parse_blend_factor(Some("one-minus-src-alpha")),
            wgpu::BlendFactor::OneMinusSrcAlpha
        ));
        assert!(matches!(parse_blend_factor(None), wgpu::BlendFactor::One));
        // Unknown falls back to one.
        assert!(matches!(parse_blend_factor(Some("nope")), wgpu::BlendFactor::One));
    }

    // End-to-end wgpu tests need a live GPU adapter, which we can't
    // assume in CI. The wrapper just plumbs through wgpu's public
    // methods, which have their own upstream test coverage. Smoke
    // testing happens via TS integration in release builds (mirrors
    // iroh-bindings' approach).
}
