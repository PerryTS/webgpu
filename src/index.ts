/**
 * @perryts/webgpu — native WebGPU bindings for the Perry
 * TypeScript-to-native compiler. Spec-faithful surface backed by
 * `wgpu`: shaders + pipelines authored against the browser API run
 * unmodified under perry.
 *
 * # Surface coverage
 *
 * - **Adapter / Device / Queue**: requestAdapter, adapterRequestDevice,
 *   adapterDrop, deviceDestroy, devicePoll
 * - **Buffer**: deviceCreateBuffer, bufferDestroy, bufferMapAsync,
 *   bufferGetMappedRange, bufferUnmap
 * - **Shader**: deviceCreateShaderModule (WGSL only)
 * - **Bindings**: deviceCreateBindGroupLayout / Pipeline /BindGroup
 *   (with buffer / sampler / textureView resources, dynamic offsets)
 * - **Pipelines**: compute + render, sync + async, getBindGroupLayout
 * - **Textures**: deviceCreateTexture / textureCreateView / destroy /
 *   deviceCreateSampler — full TextureFormat / TextureViewDimension /
 *   TextureAspect / FilterMode / AddressMode / CompareFunction surface
 * - **Render pass**: full method tower (set-pipeline / set-bind-group
 *   with dynamic offsets / set-vertex-buffer / set-index-buffer / draw
 *   / drawIndexed / set-viewport / set-scissor-rect / set-blend-constant
 *   / set-stencil-reference / occlusion-query begin/end / end)
 * - **Compute pass**: set-pipeline / set-bind-group with dynamic
 *   offsets / dispatch-workgroups / end
 * - **Command encoder**: begin-compute-pass / begin-render-pass /
 *   copy-buffer-to-buffer / copy-buffer-to-texture /
 *   copy-texture-to-buffer / copy-texture-to-texture /
 *   resolve-query-set / finish
 * - **Queue**: submit / writeBuffer / writeTexture /
 *   onSubmittedWorkDone
 * - **QuerySet**: occlusion + timestamp
 * - **Error scopes**: pushErrorScope / popErrorScope
 * - **Surface (on-screen)**: requestSurface / surfaceFromNativeView /
 *   surfaceGetViewPtr / surfaceGetPreferredFormat / surfaceConfigure /
 *   surfaceGetCurrentTexture / surfacePresent / surfaceUnconfigure /
 *   surfaceDrop — swapchain into a perry-ui window
 *
 * # Why a flat functional surface
 *
 * The browser's WebGPU API is class-based (`GPUDevice.createBuffer`,
 * `GPUBuffer.mapAsync`, …). Perry's native-library FFI manifest only
 * supports flat `extern "C"` calls (numbers / strings / handles), so
 * the binding exposes a functional shape: receiver-then-args, with
 * the receiver type encoded in the function name to keep them
 * collision-free in the flat namespace. Same convention as
 * `@perryts/iroh`'s `streamWrite` / `connClose` / `endpointConnections`.
 *
 * Concretely:
 *
 * ```ts
 * // Browser:
 * const buf = device.createBuffer({ size: 1024, usage });
 * await buf.mapAsync(GPUMapMode.READ);
 * // Perry:
 * const buf = deviceCreateBuffer(device, { size: 1024, usage });
 * await bufferMapAsync(buf, GPUMapMode.READ);
 * ```
 *
 * Class wrappers that restore literal spec parity will land once
 * Perry's compiler grows codegen support for class-method dispatch
 * on registered handles (the "property/method tower used for Web
 * Fetch handles" called out in the spec issue).
 *
 * # Why descriptors are typed objects (and become JSON on the seam)
 *
 * Most descriptor objects (`GPUBindGroupLayoutDescriptor`,
 * `GPURenderPipelineDescriptor`, `GPURenderPassDescriptor`, …) are
 * deeply nested. Rather than exploding each into a wide function-arg
 * list, the TS surface accepts the typed descriptor object and the
 * binding hands it across as `JSON.stringify(...)`. The Rust side
 * deserialises with `serde_json`. This keeps the FFI surface small
 * without sacrificing the spec's exact descriptor shape on the TS
 * side — IDE autocomplete still works.
 *
 * # V8 fallback
 *
 * Each function body throws under V8 / Node — these bodies are never
 * reached under Perry, since the compiler intercepts call sites and
 * routes them to the corresponding `js_webgpu_*` FFI symbol declared
 * in `package.json`'s `perry.nativeLibrary.functions` manifest.
 */

// ═══════════════════════════════════════════════════════════════════
// Branded handle types
// ═══════════════════════════════════════════════════════════════════

/**
 * Opaque handle types — these are NaN-boxed integers under the hood,
 * but the brand prevents you from passing a `GPUDevice` where a
 * `GPUBuffer` is expected. Treat them as black-box values; never
 * inspect or arithmetic on them.
 */
export type GPUAdapter = number & { readonly __webgpuAdapter: unique symbol };
export type GPUDevice = number & { readonly __webgpuDevice: unique symbol };
export type GPUQueue = number & { readonly __webgpuQueue: unique symbol };
export type GPUBuffer = number & { readonly __webgpuBuffer: unique symbol };
export type GPUShaderModule = number & { readonly __webgpuShaderModule: unique symbol };
export type GPUBindGroupLayout = number & { readonly __webgpuBindGroupLayout: unique symbol };
export type GPUPipelineLayout = number & { readonly __webgpuPipelineLayout: unique symbol };
export type GPUBindGroup = number & { readonly __webgpuBindGroup: unique symbol };
export type GPUComputePipeline = number & { readonly __webgpuComputePipeline: unique symbol };
export type GPURenderPipeline = number & { readonly __webgpuRenderPipeline: unique symbol };
export type GPUCommandEncoder = number & { readonly __webgpuCommandEncoder: unique symbol };
export type GPUComputePassEncoder = number & { readonly __webgpuComputePassEncoder: unique symbol };
export type GPURenderPassEncoder = number & { readonly __webgpuRenderPassEncoder: unique symbol };
export type GPUCommandBuffer = number & { readonly __webgpuCommandBuffer: unique symbol };
export type GPUTexture = number & { readonly __webgpuTexture: unique symbol };
export type GPUTextureView = number & { readonly __webgpuTextureView: unique symbol };
export type GPUSampler = number & { readonly __webgpuSampler: unique symbol };
export type GPUQuerySet = number & { readonly __webgpuQuerySet: unique symbol };
/**
 * An on-screen swapchain surface. Stands in for the spec's
 * `GPUCanvasContext` (there's no `<canvas>` element natively) — created
 * with {@link requestSurface}, mounted into a perry-ui window via
 * {@link surfaceGetViewPtr} + `embedNativeView`, then driven with
 * {@link surfaceConfigure} / {@link surfaceGetCurrentTexture} /
 * {@link surfacePresent}.
 */
export type GPUSurface = number & { readonly __webgpuSurface: unique symbol };

// ═══════════════════════════════════════════════════════════════════
// Spec enum constants — runtime values, mirror the W3C spec exactly
// ═══════════════════════════════════════════════════════════════════

/** `GPUBufferUsage` flag values — OR them into `GPUBufferDescriptor.usage`. */
export const GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
} as const;

/** `GPUTextureUsage` flag values — OR them into `GPUTextureDescriptor.usage`. */
export const GPUTextureUsage = {
  COPY_SRC: 0x01,
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  STORAGE_BINDING: 0x08,
  RENDER_ATTACHMENT: 0x10,
} as const;

/** `GPUShaderStage` flags for `GPUBindGroupLayoutEntry.visibility`. */
export const GPUShaderStage = {
  VERTEX: 0x1,
  FRAGMENT: 0x2,
  COMPUTE: 0x4,
} as const;

/** `GPUMapMode` flags for `bufferMapAsync`. */
export const GPUMapMode = {
  READ: 0x1,
  WRITE: 0x2,
} as const;

/** `GPUColorWrite` flags for `GPUColorTargetState.writeMask`. */
export const GPUColorWrite = {
  RED: 0x1,
  GREEN: 0x2,
  BLUE: 0x4,
  ALPHA: 0x8,
  ALL: 0xF,
} as const;

// ═══════════════════════════════════════════════════════════════════
// Spec string-enum types
// ═══════════════════════════════════════════════════════════════════

export type GPUBufferBindingType = "uniform" | "storage" | "read-only-storage";
export type GPUSamplerBindingType = "filtering" | "non-filtering" | "comparison";
export type GPUTextureSampleType =
  | "float"
  | "unfilterable-float"
  | "depth"
  | "sint"
  | "uint";
export type GPUStorageTextureAccess = "write-only" | "read-only" | "read-write";
export type GPUTextureViewDimension = "1d" | "2d" | "2d-array" | "cube" | "cube-array" | "3d";
export type GPUTextureDimension = "1d" | "2d" | "3d";
export type GPUTextureAspect = "all" | "stencil-only" | "depth-only";
export type GPUErrorFilter = "validation" | "out-of-memory" | "internal";
export type GPUAddressMode = "clamp-to-edge" | "repeat" | "mirror-repeat";
export type GPUFilterMode = "nearest" | "linear";
export type GPUCompareFunction =
  | "never"
  | "less"
  | "equal"
  | "less-equal"
  | "greater"
  | "not-equal"
  | "greater-equal"
  | "always";
export type GPUStencilOperation =
  | "keep"
  | "zero"
  | "replace"
  | "invert"
  | "increment-clamp"
  | "decrement-clamp"
  | "increment-wrap"
  | "decrement-wrap";
export type GPUPrimitiveTopology =
  | "point-list"
  | "line-list"
  | "line-strip"
  | "triangle-list"
  | "triangle-strip";
export type GPUFrontFace = "ccw" | "cw";
export type GPUCullMode = "none" | "front" | "back";
export type GPUIndexFormat = "uint16" | "uint32";
export type GPUVertexStepMode = "vertex" | "instance";
export type GPUVertexFormat =
  | "uint8x2" | "uint8x4" | "sint8x2" | "sint8x4"
  | "unorm8x2" | "unorm8x4" | "snorm8x2" | "snorm8x4"
  | "uint16x2" | "uint16x4" | "sint16x2" | "sint16x4"
  | "unorm16x2" | "unorm16x4" | "snorm16x2" | "snorm16x4"
  | "float16x2" | "float16x4"
  | "float32" | "float32x2" | "float32x3" | "float32x4"
  | "uint32" | "uint32x2" | "uint32x3" | "uint32x4"
  | "sint32" | "sint32x2" | "sint32x3" | "sint32x4";
export type GPULoadOp = "load" | "clear";
export type GPUStoreOp = "store" | "discard";
export type GPUBlendFactor =
  | "zero" | "one"
  | "src" | "one-minus-src" | "src-alpha" | "one-minus-src-alpha"
  | "dst" | "one-minus-dst" | "dst-alpha" | "one-minus-dst-alpha"
  | "src-alpha-saturated"
  | "constant" | "one-minus-constant";
export type GPUBlendOperation = "add" | "subtract" | "reverse-subtract" | "min" | "max";
export type GPUQueryType = "occlusion" | "timestamp";

/**
 * Subset of the spec's `GPUTextureFormat` that the binding maps
 * natively. Anything else falls back to `rgba8unorm` on the Rust
 * side (with a v0.3 plan to surface unknown formats as a hard error
 * via the device error scope).
 */
export type GPUTextureFormat =
  | "r8unorm" | "r8snorm" | "r8uint" | "r8sint"
  | "r16uint" | "r16sint" | "r16float"
  | "rg8unorm" | "rg8snorm" | "rg8uint" | "rg8sint"
  | "r32uint" | "r32sint" | "r32float"
  | "rg16uint" | "rg16sint" | "rg16float"
  | "rgba8unorm" | "rgba8unorm-srgb" | "rgba8snorm" | "rgba8uint" | "rgba8sint"
  | "bgra8unorm" | "bgra8unorm-srgb"
  | "rgb10a2unorm"
  | "rg32uint" | "rg32sint" | "rg32float"
  | "rgba16uint" | "rgba16sint" | "rgba16float"
  | "rgba32uint" | "rgba32sint" | "rgba32float"
  | "depth16unorm" | "depth24plus" | "depth24plus-stencil8" | "depth32float";

// ═══════════════════════════════════════════════════════════════════
// Spec descriptor types
// ═══════════════════════════════════════════════════════════════════

export interface GPUBufferDescriptor {
  label?: string;
  size: number;
  usage: number;
  mappedAtCreation?: boolean;
}

export interface GPUBufferBindingLayout {
  type?: GPUBufferBindingType;
  hasDynamicOffset?: boolean;
  minBindingSize?: number;
}

export interface GPUSamplerBindingLayout {
  type?: GPUSamplerBindingType;
}

export interface GPUTextureBindingLayout {
  sampleType?: GPUTextureSampleType;
  viewDimension?: GPUTextureViewDimension;
  multisampled?: boolean;
}

export interface GPUStorageTextureBindingLayout {
  access?: GPUStorageTextureAccess;
  format: GPUTextureFormat;
  viewDimension?: GPUTextureViewDimension;
}

export interface GPUBindGroupLayoutEntry {
  binding: number;
  visibility: number;
  buffer?: GPUBufferBindingLayout;
  sampler?: GPUSamplerBindingLayout;
  texture?: GPUTextureBindingLayout;
  storageTexture?: GPUStorageTextureBindingLayout;
}

export interface GPUBindGroupLayoutDescriptor {
  label?: string;
  entries: GPUBindGroupLayoutEntry[];
}

export interface GPUPipelineLayoutDescriptor {
  label?: string;
  bindGroupLayouts: GPUBindGroupLayout[];
}

/**
 * `GPUBindGroupEntry.resource` shapes — see the
 * [spec][gpu-binding-resource]. The wire format wraps the bare
 * sampler / textureView number in a single-key object so the parser
 * can disambiguate them from a buffer binding (which is also an
 * object). External textures are deferred until the canvas
 * integration crate exposes them.
 *
 * [gpu-binding-resource]: https://www.w3.org/TR/webgpu/#typedefdef-gpubindingresource
 */
export type GPUBindGroupResource =
  | GPUBufferBinding
  | { sampler: GPUSampler }
  | { textureView: GPUTextureView };

export interface GPUBufferBinding {
  buffer: GPUBuffer;
  offset?: number;
  /** `0` means "to the end of the buffer", matching the spec's `undefined` sentinel. */
  size?: number;
}

export interface GPUBindGroupEntry {
  binding: number;
  resource: GPUBindGroupResource;
}

export interface GPUBindGroupDescriptor {
  label?: string;
  layout: GPUBindGroupLayout;
  entries: GPUBindGroupEntry[];
}

export interface GPUProgrammableStage {
  module: GPUShaderModule;
  entryPoint?: string;
}

export interface GPUComputePipelineDescriptor {
  label?: string;
  /** `"auto"` per the spec, or an explicit `GPUPipelineLayout`. */
  layout: "auto" | GPUPipelineLayout;
  compute: GPUProgrammableStage;
}

// ─── Render pipeline ──────────────────────────────────────────────

export interface GPUVertexAttribute {
  format: GPUVertexFormat;
  offset: number;
  shaderLocation: number;
}

export interface GPUVertexBufferLayout {
  arrayStride: number;
  stepMode?: GPUVertexStepMode;
  attributes: GPUVertexAttribute[];
}

export interface GPUVertexState extends GPUProgrammableStage {
  buffers?: GPUVertexBufferLayout[];
}

export interface GPUPrimitiveState {
  topology?: GPUPrimitiveTopology;
  stripIndexFormat?: GPUIndexFormat;
  frontFace?: GPUFrontFace;
  cullMode?: GPUCullMode;
}

export interface GPUStencilFaceState {
  compare?: GPUCompareFunction;
  failOp?: GPUStencilOperation;
  depthFailOp?: GPUStencilOperation;
  passOp?: GPUStencilOperation;
}

export interface GPUDepthStencilState {
  format: GPUTextureFormat;
  depthWriteEnabled?: boolean;
  depthCompare?: GPUCompareFunction;
  stencilFront?: GPUStencilFaceState;
  stencilBack?: GPUStencilFaceState;
  stencilReadMask?: number;
  stencilWriteMask?: number;
  depthBias?: number;
  depthBiasSlopeScale?: number;
  depthBiasClamp?: number;
}

export interface GPUMultisampleState {
  count?: number;
  mask?: number;
  alphaToCoverageEnabled?: boolean;
}

export interface GPUBlendComponent {
  srcFactor?: GPUBlendFactor;
  dstFactor?: GPUBlendFactor;
  operation?: GPUBlendOperation;
}

export interface GPUBlendState {
  color: GPUBlendComponent;
  alpha: GPUBlendComponent;
}

export interface GPUColorTargetState {
  format: GPUTextureFormat;
  blend?: GPUBlendState;
  /** Bitmask of `GPUColorWrite.*`. Defaults to `ALL` (0xF) per spec. */
  writeMask?: number;
}

export interface GPUFragmentState extends GPUProgrammableStage {
  targets: (GPUColorTargetState | null)[];
}

export interface GPURenderPipelineDescriptor {
  label?: string;
  layout: "auto" | GPUPipelineLayout;
  vertex: GPUVertexState;
  primitive?: GPUPrimitiveState;
  depthStencil?: GPUDepthStencilState;
  multisample?: GPUMultisampleState;
  fragment?: GPUFragmentState;
}

// ─── Texture / sampler ────────────────────────────────────────────

export interface GPUExtent3D {
  width: number;
  height?: number;
  depthOrArrayLayers?: number;
}

export interface GPUOrigin3D {
  x?: number;
  y?: number;
  z?: number;
}

export interface GPUTextureDescriptor {
  label?: string;
  size: GPUExtent3D;
  mipLevelCount?: number;
  sampleCount?: number;
  dimension?: GPUTextureDimension;
  format: GPUTextureFormat;
  /** Bitmask of `GPUTextureUsage.*`. */
  usage: number;
  viewFormats?: GPUTextureFormat[];
}

export interface GPUTextureViewDescriptor {
  label?: string;
  format?: GPUTextureFormat;
  dimension?: GPUTextureViewDimension;
  aspect?: GPUTextureAspect;
  baseMipLevel?: number;
  /** `0` means "all remaining levels" (the spec's `undefined` sentinel). */
  mipLevelCount?: number;
  baseArrayLayer?: number;
  /** `0` means "all remaining layers". */
  arrayLayerCount?: number;
}

export interface GPUSamplerDescriptor {
  label?: string;
  addressModeU?: GPUAddressMode;
  addressModeV?: GPUAddressMode;
  addressModeW?: GPUAddressMode;
  magFilter?: GPUFilterMode;
  minFilter?: GPUFilterMode;
  mipmapFilter?: GPUFilterMode;
  lodMinClamp?: number;
  lodMaxClamp?: number;
  compare?: GPUCompareFunction;
  maxAnisotropy?: number;
}

// ─── Render pass ──────────────────────────────────────────────────

export interface GPUColor {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface GPURenderPassColorAttachment {
  view: GPUTextureView;
  resolveTarget?: GPUTextureView;
  loadOp?: GPULoadOp;
  storeOp?: GPUStoreOp;
  clearValue?: GPUColor;
}

export interface GPURenderPassDepthStencilAttachment {
  view: GPUTextureView;
  depthClearValue?: number;
  depthLoadOp?: GPULoadOp;
  depthStoreOp?: GPUStoreOp;
  depthReadOnly?: boolean;
  stencilClearValue?: number;
  stencilLoadOp?: GPULoadOp;
  stencilStoreOp?: GPUStoreOp;
  stencilReadOnly?: boolean;
}

export interface GPURenderPassDescriptor {
  label?: string;
  colorAttachments: (GPURenderPassColorAttachment | null)[];
  depthStencilAttachment?: GPURenderPassDepthStencilAttachment;
  occlusionQuerySet?: GPUQuerySet;
}

// ─── Query set ────────────────────────────────────────────────────

export interface GPUQuerySetDescriptor {
  label?: string;
  type: GPUQueryType;
  count: number;
}

// ─── Queue.writeTexture + texture copy ops ────────────────────────

export interface GPUImageCopyTexture {
  texture: GPUTexture;
  mipLevel?: number;
  origin?: GPUOrigin3D;
  aspect?: GPUTextureAspect;
}

export interface GPUImageDataLayout {
  offset?: number;
  bytesPerRow?: number;
  rowsPerImage?: number;
}

export interface GPUImageCopyBuffer extends GPUImageDataLayout {
  buffer: GPUBuffer;
}

// ═══════════════════════════════════════════════════════════════════
// Public functional API
//
// Function names are mechanical mappings of the underlying
// `js_webgpu_*` symbol with the prefix stripped:
// `js_webgpu_device_create_buffer` ↔ `deviceCreateBuffer`. Same
// pattern as `@perryts/iroh`'s `streamWrite` / `connClose`.
//
// Each body throws under the V8 / Node fallback path — Perry's
// compiler intercepts the call site and routes it to the FFI symbol
// declared in `package.json`'s `perry.nativeLibrary.functions`
// manifest, so these throws never execute under a native Perry build.
// ═══════════════════════════════════════════════════════════════════

const NOT_NATIVE = "Not implemented under V8 fallback — compile with `perry compile`.";

// ─── Adapter / Device / Queue ──────────────────────────────────────

/**
 * Equivalent to the spec's `navigator.gpu.requestAdapter()` with no
 * descriptor — high-perf, fallback allowed. Rejects rather than
 * resolving to `null` if no adapter is available.
 */
export function requestAdapter(): Promise<GPUAdapter> {
  throw new Error(NOT_NATIVE);
}

/**
 * Equivalent to the spec's `adapter.requestDevice()`. Resolves with
 * `{ device, queue }` so the queue is available without a second
 * round-trip — browsers expose `device.queue` synchronously after
 * `requestDevice` resolves.
 */
export function adapterRequestDevice(
  _adapter: GPUAdapter
): Promise<{ device: GPUDevice; queue: GPUQueue }> {
  throw new Error(NOT_NATIVE);
}

/** Free the adapter handle. Idempotent. */
export function adapterDrop(_adapter: GPUAdapter): void {
  throw new Error(NOT_NATIVE);
}

/** `device.destroy()` — release the device and its queue. Idempotent. */
export function deviceDestroy(_device: GPUDevice): void {
  throw new Error(NOT_NATIVE);
}

/**
 * Drive the device's poll loop forward — necessary between
 * `bufferMapAsync` and the ensuing `bufferGetMappedRange` /
 * `bufferUnmap` cycle. Spec-extra; the browser version of this code
 * wouldn't call `poll`.
 */
export function devicePoll(_device: GPUDevice): void {
  throw new Error(NOT_NATIVE);
}

// ─── Buffer ────────────────────────────────────────────────────────

/** `device.createBuffer(descriptor)` — synchronous. */
export function deviceCreateBuffer(
  _device: GPUDevice,
  _descriptor: GPUBufferDescriptor
): GPUBuffer {
  throw new Error(NOT_NATIVE);
}

/** `buffer.destroy()` — release the GPU memory. Idempotent. */
export function bufferDestroy(_buffer: GPUBuffer): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `buffer.mapAsync(mode, offset?, size?)` — make the buffer's
 * contents accessible to the host.
 *
 * Resolves once the GPU is done with the buffer. The caller must
 * call `devicePoll(device)` between issuing the `bufferMapAsync` and
 * awaiting it (browsers do this implicitly via the event loop).
 *
 * @param mode   `GPUMapMode.READ` (`1`) or `GPUMapMode.WRITE` (`2`).
 * @param offset Defaults to `0`.
 * @param size   Defaults to `0`, meaning "to the end of the buffer".
 */
export function bufferMapAsync(
  _buffer: GPUBuffer,
  _mode: number,
  _offset?: number,
  _size?: number
): Promise<void> {
  throw new Error(NOT_NATIVE);
}

/**
 * `buffer.getMappedRange(offset?, size?)` — copies the mapped bytes
 * into a fresh `Uint8Array`. The browser returns an `ArrayBuffer`
 * aliasing the GPU's staging memory; the binding copies because
 * Perry-runtime buffers are independently GC-managed.
 */
export function bufferGetMappedRange(
  _buffer: GPUBuffer,
  _offset?: number,
  _size?: number
): Uint8Array {
  throw new Error(NOT_NATIVE);
}

/** `buffer.unmap()` — release the host mapping. */
export function bufferUnmap(_buffer: GPUBuffer): void {
  throw new Error(NOT_NATIVE);
}

// ─── Shader module ─────────────────────────────────────────────────

/**
 * `device.createShaderModule({ code })` — WGSL only (matches the
 * browser spec). `code` is the source string; label / sourceMap /
 * compilationHints are spec-optional and the binding ignores them.
 */
export function deviceCreateShaderModule(
  _device: GPUDevice,
  _code: string
): GPUShaderModule {
  throw new Error(NOT_NATIVE);
}

// ─── Bind groups + pipeline layouts ────────────────────────────────

export function deviceCreateBindGroupLayout(
  _device: GPUDevice,
  _descriptor: GPUBindGroupLayoutDescriptor
): GPUBindGroupLayout {
  throw new Error(NOT_NATIVE);
}

export function deviceCreatePipelineLayout(
  _device: GPUDevice,
  _descriptor: GPUPipelineLayoutDescriptor
): GPUPipelineLayout {
  throw new Error(NOT_NATIVE);
}

export function deviceCreateBindGroup(
  _device: GPUDevice,
  _descriptor: GPUBindGroupDescriptor
): GPUBindGroup {
  throw new Error(NOT_NATIVE);
}

// ─── Compute pipeline ──────────────────────────────────────────────

export function deviceCreateComputePipeline(
  _device: GPUDevice,
  _descriptor: GPUComputePipelineDescriptor
): GPUComputePipeline {
  throw new Error(NOT_NATIVE);
}

/**
 * `device.createComputePipelineAsync(descriptor)` — same shape as
 * the sync call, but the pipeline build runs off the JS thread.
 */
export function deviceCreateComputePipelineAsync(
  _device: GPUDevice,
  _descriptor: GPUComputePipelineDescriptor
): Promise<GPUComputePipeline> {
  throw new Error(NOT_NATIVE);
}

/**
 * `pipeline.getBindGroupLayout(index)` for compute pipelines —
 * useful when the pipeline was created with `layout: "auto"`.
 */
export function computePipelineGetBindGroupLayout(
  _pipeline: GPUComputePipeline,
  _index: number
): GPUBindGroupLayout {
  throw new Error(NOT_NATIVE);
}

// ─── Render pipeline ───────────────────────────────────────────────

export function deviceCreateRenderPipeline(
  _device: GPUDevice,
  _descriptor: GPURenderPipelineDescriptor
): GPURenderPipeline {
  throw new Error(NOT_NATIVE);
}

/**
 * `device.createRenderPipelineAsync(descriptor)` — same shape as the
 * sync call, but the pipeline build runs off the JS thread.
 */
export function deviceCreateRenderPipelineAsync(
  _device: GPUDevice,
  _descriptor: GPURenderPipelineDescriptor
): Promise<GPURenderPipeline> {
  throw new Error(NOT_NATIVE);
}

/** `pipeline.getBindGroupLayout(index)` for render pipelines. */
export function renderPipelineGetBindGroupLayout(
  _pipeline: GPURenderPipeline,
  _index: number
): GPUBindGroupLayout {
  throw new Error(NOT_NATIVE);
}

// ─── Texture / view / sampler ──────────────────────────────────────

export function deviceCreateTexture(
  _device: GPUDevice,
  _descriptor: GPUTextureDescriptor
): GPUTexture {
  throw new Error(NOT_NATIVE);
}

/**
 * `texture.createView(descriptor?)` — descriptor is optional; pass
 * `{}` (or omit) for a default view of the whole texture.
 */
export function textureCreateView(
  _texture: GPUTexture,
  _descriptor?: GPUTextureViewDescriptor
): GPUTextureView {
  throw new Error(NOT_NATIVE);
}

/** `texture.destroy()` — idempotent. */
export function textureDestroy(_texture: GPUTexture): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `device.createSampler(descriptor?)` — all fields are optional;
 * defaults match the spec (clamp-to-edge, nearest, no compare,
 * anisotropy 1).
 */
export function deviceCreateSampler(
  _device: GPUDevice,
  _descriptor?: GPUSamplerDescriptor
): GPUSampler {
  throw new Error(NOT_NATIVE);
}

// ─── QuerySet ──────────────────────────────────────────────────────

export function deviceCreateQuerySet(
  _device: GPUDevice,
  _descriptor: GPUQuerySetDescriptor
): GPUQuerySet {
  throw new Error(NOT_NATIVE);
}

/** `querySet.destroy()` — idempotent. */
export function querySetDestroy(_querySet: GPUQuerySet): void {
  throw new Error(NOT_NATIVE);
}

// ─── Command encoder ───────────────────────────────────────────────

/** `device.createCommandEncoder()` — synchronous. */
export function deviceCreateCommandEncoder(_device: GPUDevice): GPUCommandEncoder {
  throw new Error(NOT_NATIVE);
}

/** `encoder.beginComputePass()` — synchronous. */
export function commandEncoderBeginComputePass(
  _encoder: GPUCommandEncoder
): GPUComputePassEncoder {
  throw new Error(NOT_NATIVE);
}

/** `encoder.beginRenderPass(descriptor)` — synchronous. */
export function commandEncoderBeginRenderPass(
  _encoder: GPUCommandEncoder,
  _descriptor: GPURenderPassDescriptor
): GPURenderPassEncoder {
  throw new Error(NOT_NATIVE);
}

/**
 * `encoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size)`.
 * `size` must be a multiple of 4 per the spec.
 */
export function commandEncoderCopyBufferToBuffer(
  _encoder: GPUCommandEncoder,
  _src: GPUBuffer,
  _srcOffset: number,
  _dst: GPUBuffer,
  _dstOffset: number,
  _size: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `encoder.copyBufferToTexture(source, destination, copySize)`. */
export function commandEncoderCopyBufferToTexture(
  _encoder: GPUCommandEncoder,
  _descriptor: { source: GPUImageCopyBuffer; destination: GPUImageCopyTexture; size: GPUExtent3D }
): void {
  throw new Error(NOT_NATIVE);
}

/** `encoder.copyTextureToBuffer(source, destination, copySize)`. */
export function commandEncoderCopyTextureToBuffer(
  _encoder: GPUCommandEncoder,
  _descriptor: { source: GPUImageCopyTexture; destination: GPUImageCopyBuffer; size: GPUExtent3D }
): void {
  throw new Error(NOT_NATIVE);
}

/** `encoder.copyTextureToTexture(source, destination, copySize)`. */
export function commandEncoderCopyTextureToTexture(
  _encoder: GPUCommandEncoder,
  _descriptor: { source: GPUImageCopyTexture; destination: GPUImageCopyTexture; size: GPUExtent3D }
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `encoder.resolveQuerySet(querySet, firstQuery, queryCount, destination, destinationOffset)`.
 */
export function commandEncoderResolveQuerySet(
  _encoder: GPUCommandEncoder,
  _querySet: GPUQuerySet,
  _firstQuery: number,
  _queryCount: number,
  _destination: GPUBuffer,
  _destinationOffset: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `encoder.finish()` — consumes the encoder. */
export function commandEncoderFinish(_encoder: GPUCommandEncoder): GPUCommandBuffer {
  throw new Error(NOT_NATIVE);
}

// ─── Compute pass ──────────────────────────────────────────────────

/** `pass.setPipeline(pipeline)`. */
export function computePassSetPipeline(
  _pass: GPUComputePassEncoder,
  _pipeline: GPUComputePipeline
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setBindGroup(index, bindGroup)`. */
export function computePassSetBindGroup(
  _pass: GPUComputePassEncoder,
  _index: number,
  _bindGroup: GPUBindGroup
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `pass.setBindGroup(index, bindGroup, dynamicOffsets)` — variant
 * with the spec's optional `dynamicOffsets` array. Pass it as a
 * `number[]`; the binding JSON-encodes it to keep the FFI surface
 * tractable.
 */
export function computePassSetBindGroupDyn(
  _pass: GPUComputePassEncoder,
  _index: number,
  _bindGroup: GPUBindGroup,
  _dynamicOffsetsJson: string
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.dispatchWorkgroups(x, y?, z?)` — `y` and `z` default to `1`. */
export function computePassDispatchWorkgroups(
  _pass: GPUComputePassEncoder,
  _x: number,
  _y?: number,
  _z?: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.end()` — finalises the pass. */
export function computePassEnd(_pass: GPUComputePassEncoder): void {
  throw new Error(NOT_NATIVE);
}

// ─── Render pass ───────────────────────────────────────────────────

/** `pass.setPipeline(pipeline)`. */
export function renderPassSetPipeline(
  _pass: GPURenderPassEncoder,
  _pipeline: GPURenderPipeline
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setBindGroup(index, bindGroup)`. */
export function renderPassSetBindGroup(
  _pass: GPURenderPassEncoder,
  _index: number,
  _bindGroup: GPUBindGroup
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setBindGroup(index, bindGroup, dynamicOffsets)`. */
export function renderPassSetBindGroupDyn(
  _pass: GPURenderPassEncoder,
  _index: number,
  _bindGroup: GPUBindGroup,
  _dynamicOffsetsJson: string
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `pass.setVertexBuffer(slot, buffer, offset?, size?)`. `size: 0`
 * means "to the end of the buffer" (the spec's `undefined` sentinel).
 */
export function renderPassSetVertexBuffer(
  _pass: GPURenderPassEncoder,
  _slot: number,
  _buffer: GPUBuffer,
  _offset?: number,
  _size?: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setIndexBuffer(buffer, indexFormat, offset?, size?)`. */
export function renderPassSetIndexBuffer(
  _pass: GPURenderPassEncoder,
  _buffer: GPUBuffer,
  _indexFormat: GPUIndexFormat,
  _offset?: number,
  _size?: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.draw(vertexCount, instanceCount?, firstVertex?, firstInstance?)`. */
export function renderPassDraw(
  _pass: GPURenderPassEncoder,
  _vertexCount: number,
  _instanceCount?: number,
  _firstVertex?: number,
  _firstInstance?: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.drawIndexed(indexCount, instanceCount?, firstIndex?, baseVertex?, firstInstance?)`. */
export function renderPassDrawIndexed(
  _pass: GPURenderPassEncoder,
  _indexCount: number,
  _instanceCount?: number,
  _firstIndex?: number,
  _baseVertex?: number,
  _firstInstance?: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setViewport(x, y, w, h, minDepth, maxDepth)`. */
export function renderPassSetViewport(
  _pass: GPURenderPassEncoder,
  _x: number,
  _y: number,
  _w: number,
  _h: number,
  _minDepth: number,
  _maxDepth: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setScissorRect(x, y, w, h)`. */
export function renderPassSetScissorRect(
  _pass: GPURenderPassEncoder,
  _x: number,
  _y: number,
  _w: number,
  _h: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setBlendConstant({r,g,b,a})` — components in 0..=1 for unorm formats. */
export function renderPassSetBlendConstant(
  _pass: GPURenderPassEncoder,
  _r: number,
  _g: number,
  _b: number,
  _a: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.setStencilReference(reference)`. */
export function renderPassSetStencilReference(
  _pass: GPURenderPassEncoder,
  _reference: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.beginOcclusionQuery(queryIndex)`. */
export function renderPassBeginOcclusionQuery(
  _pass: GPURenderPassEncoder,
  _queryIndex: number
): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.endOcclusionQuery()`. */
export function renderPassEndOcclusionQuery(_pass: GPURenderPassEncoder): void {
  throw new Error(NOT_NATIVE);
}

/** `pass.end()` — finalises the pass. */
export function renderPassEnd(_pass: GPURenderPassEncoder): void {
  throw new Error(NOT_NATIVE);
}

// ─── Queue ─────────────────────────────────────────────────────────

/** `queue.submit(commandBuffers)`. Each command buffer is consumed. */
export function queueSubmit(
  _queue: GPUQueue,
  _commandBuffersJson: string
): void {
  throw new Error(NOT_NATIVE);
}

/** `queue.writeBuffer(buffer, bufferOffset, data)`. */
export function queueWriteBuffer(
  _queue: GPUQueue,
  _buffer: GPUBuffer,
  _bufferOffset: number,
  _data: Uint8Array | Buffer
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `queue.writeTexture(destination, data, dataLayout, size)` — the
 * four spec args are packed into one descriptor object and a
 * separate `data` buffer.
 */
export function queueWriteTexture(
  _queue: GPUQueue,
  _descriptor: {
    destination: GPUImageCopyTexture;
    dataLayout: GPUImageDataLayout;
    size: GPUExtent3D;
  },
  _data: Uint8Array | Buffer
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `queue.onSubmittedWorkDone()` — resolves once every command buffer
 * submitted to the queue *before this call* has finished executing.
 * Implemented via wgpu's `on_submitted_work_done` callback + a poll
 * loop on the queue's parent device.
 */
export function queueOnSubmittedWorkDone(_queue: GPUQueue): Promise<void> {
  throw new Error(NOT_NATIVE);
}

// ─── Error scopes ──────────────────────────────────────────────────

/** `device.pushErrorScope(filter)`. */
export function devicePushErrorScope(
  _device: GPUDevice,
  _filter: GPUErrorFilter
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `device.popErrorScope()` — resolves with a JSON-encoded
 * `{type, message}` describing the captured error, or the empty
 * string when the scope captured nothing (which the call site can
 * map to `null`, matching the spec).
 */
export function devicePopErrorScope(_device: GPUDevice): Promise<string> {
  throw new Error(NOT_NATIVE);
}

// ─── On-screen surface (GPUSurface / GPUCanvasContext) ─────────────

/** `GPUCanvasAlphaMode` — how the swapchain composites with the window. */
export type GPUCanvasAlphaMode =
  | "opaque"
  | "premultiplied"
  | "postmultiplied"
  | "inherit"
  | "auto";

/**
 * Swapchain present mode. Spec-extra (the browser picks this for you);
 * `"fifo"` is vsync and the safe default supported everywhere.
 */
export type GPUPresentMode = "fifo" | "fifo-relaxed" | "immediate" | "mailbox";

/** Descriptor for {@link requestSurface}. */
export interface GPUSurfaceDescriptor {
  /** Initial backing-view size in physical pixels. */
  width: number;
  height: number;
  label?: string;
}

/**
 * `GPUCanvasConfiguration`, extended with the native swapchain knobs the
 * browser infers from the `<canvas>` element. `width`/`height` are
 * required here since there is no element to read them from.
 */
export interface GPUCanvasConfiguration {
  device: GPUDevice;
  format: GPUTextureFormat;
  /** Defaults to `GPUTextureUsage.RENDER_ATTACHMENT`. */
  usage?: number;
  alphaMode?: GPUCanvasAlphaMode;
  width: number;
  height: number;
  /** Spec-extra; defaults to `"fifo"`. */
  presentMode?: GPUPresentMode;
  viewFormats?: GPUTextureFormat[];
}

/**
 * Allocate an on-screen swapchain surface backed by a fresh native view.
 *
 * Currently the view is allocated natively on macOS; on other platforms
 * create the platform view through perry-ui and adopt it with
 * {@link surfaceFromNativeView} instead. Must be called on the main
 * thread.
 *
 * After creating the surface, mount it: `embedNativeView(surfaceGetViewPtr(s))`
 * (from `@perryts/ui`), then `surfaceConfigure(s, …)` and run the render
 * loop. Throws under the V8 fallback.
 */
export function requestSurface(_descriptor: GPUSurfaceDescriptor): GPUSurface {
  throw new Error(NOT_NATIVE);
}

/**
 * Wrap a native view the host already created (the inverse of
 * {@link requestSurface}). `viewPtr` is a platform handle: `NSView*`
 * (macOS), `UIView*` (iOS), `HWND` (Windows), or `ANativeWindow*`
 * (Android). The view must be GPU-capable and outlive the surface; we do
 * not take ownership of it.
 */
export function surfaceFromNativeView(_viewPtr: number): GPUSurface {
  throw new Error(NOT_NATIVE);
}

/**
 * The native view pointer backing the surface — pass it to
 * `@perryts/ui`'s `embedNativeView()` to mount it in the widget tree.
 */
export function surfaceGetViewPtr(_surface: GPUSurface): number {
  throw new Error(NOT_NATIVE);
}

/**
 * The surface's preferred swapchain format for the given adapter —
 * the native analogue of `navigator.gpu.getPreferredCanvasFormat()`.
 * Feed it straight into {@link surfaceConfigure}'s `format`.
 */
export function surfaceGetPreferredFormat(
  _surface: GPUSurface,
  _adapter: GPUAdapter
): GPUTextureFormat {
  throw new Error(NOT_NATIVE);
}

/** `context.configure(configuration)` — bind device + swapchain format. */
export function surfaceConfigure(
  _surface: GPUSurface,
  _configuration: GPUCanvasConfiguration
): void {
  throw new Error(NOT_NATIVE);
}

/**
 * `context.getCurrentTexture()` — acquire this frame's swapchain image.
 * The returned texture is usable with {@link textureCreateView} just
 * like a device texture (it is the render-pass color attachment). Call
 * {@link surfacePresent} once the frame's command buffers are submitted.
 */
export function surfaceGetCurrentTexture(_surface: GPUSurface): GPUTexture {
  throw new Error(NOT_NATIVE);
}

/**
 * Present the frame acquired by {@link surfaceGetCurrentTexture}.
 * Spec-extra: browsers present implicitly at task end; native wgpu
 * needs this explicit hand-back, so it's the one call a ported render
 * loop must add after `queueSubmit`.
 */
export function surfacePresent(_surface: GPUSurface): void {
  throw new Error(NOT_NATIVE);
}

/** `context.unconfigure()` — drop the swapchain (e.g. before a resize). */
export function surfaceUnconfigure(_surface: GPUSurface): void {
  throw new Error(NOT_NATIVE);
}

/** Release the surface and, if `requestSurface` created it, its view. Idempotent. */
export function surfaceDrop(_surface: GPUSurface): void {
  throw new Error(NOT_NATIVE);
}
