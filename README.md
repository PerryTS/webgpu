# @perryts/webgpu

Native [WebGPU](https://www.w3.org/TR/webgpu/) bindings — backed by [`wgpu`](https://github.com/gfx-rs/wgpu) — for the [Perry TypeScript-to-native compiler](https://github.com/PerryTS/perry).

Closes [PerryTS/perry#571](https://github.com/PerryTS/perry/issues/571).

## What this is

A Perry "native library" package: a Rust crate that exports `extern "C"` symbols the Perry compiler links into your TypeScript program. From your TS code you `import { ... } from "@perryts/webgpu"`; under the hood every call resolves to a direct call into the bundled staticlib — no Node addon, no IPC, no JSON round-trip on the hot path.

The point of WebGPU as a Perry binding is **portability**: shaders + pipelines authored against the browser's WebGPU spec run unmodified under Perry. Same WGSL, same descriptor objects, same async lifecycle.

This package contains:

- `src/lib.rs` — the Rust crate wrapping `wgpu` and exporting `js_webgpu_*` `extern "C"` symbols
- `src/index.ts` — the TypeScript surface (functions + spec descriptor types + runtime constants like `GPUBufferUsage`)
- `Cargo.toml` — staticlib build config consumed by the Perry linker
- `package.json` — the `perry.nativeLibrary` manifest (60 functions)

## Install

```sh
bun add @perryts/webgpu
# or
npm install @perryts/webgpu
```

The package's `package.json` declares a `perry.nativeLibrary` block (see the [manifest spec](https://github.com/PerryTS/perry/blob/main/docs/src/native-libraries/manifest-v1.md)) which Perry's compiler reads at link time to discover the staticlib + `extern "C"` symbols. No post-install build step — Perry compiles the Rust crate as part of your project's build.

## Quick start — compute shader hello world

The canonical compute-shader smoke test: dispatch a workgroup that doubles every element of an input buffer, then map the output buffer back and read the result.

```typescript
import {
  requestAdapter,
  adapterRequestDevice,
  deviceCreateBuffer,
  deviceCreateShaderModule,
  deviceCreateBindGroupLayout,
  deviceCreatePipelineLayout,
  deviceCreateBindGroup,
  deviceCreateComputePipeline,
  deviceCreateCommandEncoder,
  commandEncoderBeginComputePass,
  commandEncoderCopyBufferToBuffer,
  commandEncoderFinish,
  computePassSetPipeline,
  computePassSetBindGroup,
  computePassDispatchWorkgroups,
  computePassEnd,
  queueSubmit,
  queueWriteBuffer,
  bufferMapAsync,
  bufferGetMappedRange,
  bufferUnmap,
  devicePoll,
  GPUBufferUsage,
  GPUShaderStage,
  GPUMapMode,
} from "@perryts/webgpu";

const adapter = await requestAdapter();
const { device, queue } = await adapterRequestDevice(adapter);

// Input data: [1, 2, 3, 4] as u32 little-endian.
const input = new Uint8Array([1,0,0,0, 2,0,0,0, 3,0,0,0, 4,0,0,0]);

const inBuf = deviceCreateBuffer(device, { size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
const outBuf = deviceCreateBuffer(device, { size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const stagingBuf = deviceCreateBuffer(device, { size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

queueWriteBuffer(queue, inBuf, 0, input);

const shader = deviceCreateShaderModule(device, `
  @group(0) @binding(0) var<storage, read>       input  : array<u32>;
  @group(0) @binding(1) var<storage, read_write> output : array<u32>;
  @compute @workgroup_size(4)
  fn main(@builtin(global_invocation_id) id: vec3u) {
    output[id.x] = input[id.x] * 2u;
  }
`);

const bgl = deviceCreateBindGroupLayout(device, {
  entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ],
});
const bg = deviceCreateBindGroup(device, {
  layout: bgl,
  entries: [
    { binding: 0, resource: { buffer: inBuf } },
    { binding: 1, resource: { buffer: outBuf } },
  ],
});
const pipeline = deviceCreateComputePipeline(device, {
  layout: deviceCreatePipelineLayout(device, { bindGroupLayouts: [bgl] }),
  compute: { module: shader, entryPoint: "main" },
});

const enc = deviceCreateCommandEncoder(device);
const pass = commandEncoderBeginComputePass(enc);
computePassSetPipeline(pass, pipeline);
computePassSetBindGroup(pass, 0, bg);
computePassDispatchWorkgroups(pass, 1);
computePassEnd(pass);
commandEncoderCopyBufferToBuffer(enc, outBuf, 0, stagingBuf, 0, 16);
queueSubmit(queue, JSON.stringify([commandEncoderFinish(enc)]));

await bufferMapAsync(stagingBuf, GPUMapMode.READ);
devicePoll(device);  // ← spec-extra: native runtime needs the explicit poll
const bytes = bufferGetMappedRange(stagingBuf);
console.log(new Uint32Array(bytes.buffer, bytes.byteOffset, 4));
// → Uint32Array(4) [ 2, 4, 6, 8 ]
bufferUnmap(stagingBuf);
```

## Render-pipeline sketch

The render path mirrors the spec one-for-one — same descriptor shapes, same WGSL, same `setVertexBuffer` / `draw` rhythm. Sampler + textureView bind-group entries use a `{sampler:n}` / `{textureView:n}` wrapping so the FFI parser can disambiguate them from buffer bindings:

```typescript
const tex = deviceCreateTexture(device, {
  size: { width: 256, height: 256 },
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
});
const view = textureCreateView(tex);              // default view
const sampler = deviceCreateSampler(device, { magFilter: "linear", minFilter: "linear" });

const bg = deviceCreateBindGroup(device, {
  layout: bgl,
  entries: [
    { binding: 0, resource: { sampler } },        // ← sampler-flavoured resource
    { binding: 1, resource: { textureView: view } }, // ← textureView-flavoured
    { binding: 2, resource: { buffer: uniforms } },
  ],
});

const pipeline = deviceCreateRenderPipeline(device, {
  layout: pipelineLayout,
  vertex: {
    module: shader,
    entryPoint: "vs_main",
    buffers: [
      { arrayStride: 32, stepMode: "vertex", attributes: [
          { format: "float32x3", offset: 0,  shaderLocation: 0 },
          { format: "float32x2", offset: 12, shaderLocation: 1 },
        ] },
    ],
  },
  fragment: { module: shader, entryPoint: "fs_main", targets: [{ format: "bgra8unorm" }] },
  primitive: { topology: "triangle-list", cullMode: "back" },
  depthStencil: { format: "depth32float", depthWriteEnabled: true, depthCompare: "less" },
  multisample: { count: 1 },
});

const enc = deviceCreateCommandEncoder(device);
const pass = commandEncoderBeginRenderPass(enc, {
  colorAttachments: [{ view: targetView, loadOp: "clear", storeOp: "store", clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
  depthStencilAttachment: { view: depthView, depthLoadOp: "clear", depthStoreOp: "store", depthClearValue: 1.0 },
});
renderPassSetPipeline(pass, pipeline);
renderPassSetBindGroup(pass, 0, bg);
renderPassSetVertexBuffer(pass, 0, vbuf);
renderPassSetIndexBuffer(pass, ibuf, "uint16");
renderPassDrawIndexed(pass, indexCount);
renderPassEnd(pass);
queueSubmit(queue, JSON.stringify([commandEncoderFinish(enc)]));
```

## What's a "spec-faithful" binding

WebGPU is a W3C spec. The wgpu crate is the canonical native implementation of that spec. This package is a **direct binding** to wgpu — every `js_webgpu_*` FFI function is a thin wrapper around the corresponding wgpu method. Two consequences:

1. **Bring your shader from a browser**: WGSL is the same. Descriptor shapes (`GPUBufferDescriptor`, `GPURenderPipelineDescriptor`, …) are the same JSON-serialisable objects with identical field names. `GPUBufferUsage.STORAGE` is `0x80` here just like in the browser.
2. **Wgpu's gotchas are the spec's gotchas**. If wgpu rejects an unaligned offset, this binding rejects too — we don't paper over the difference.

The one un-spec-y thing is the **flat function shape**: `deviceCreateBuffer(device, descriptor)` instead of `device.createBuffer(descriptor)`. Same arguments, just receiver-as-first-arg. A class wrapper that restores literal spec parity will land once Perry's compiler grows codegen support for class-method dispatch on registered handles (the "property/method tower used for Web Fetch handles" called out in [#571](https://github.com/PerryTS/perry/issues/571)).

## Surface coverage

- **Adapter / Device / Queue**: `requestAdapter`, `adapterRequestDevice`, `adapterDrop`, `deviceDestroy`, `devicePoll`
- **Buffer**: `deviceCreateBuffer`, `bufferDestroy`, `bufferMapAsync`, `bufferGetMappedRange`, `bufferUnmap`
- **Shader**: `deviceCreateShaderModule` (WGSL only — same as browser)
- **Bindings**: `deviceCreateBindGroupLayout` / `deviceCreatePipelineLayout` / `deviceCreateBindGroup` (with buffer / sampler / textureView resources, plus dynamic offsets)
- **Pipelines**: compute + render, sync + async, plus `getBindGroupLayout` accessors
- **Textures**: `deviceCreateTexture` / `textureCreateView` / `textureDestroy` / `deviceCreateSampler` — full TextureFormat / TextureViewDimension / TextureAspect / FilterMode / AddressMode / CompareFunction surface
- **Render pass**: full method tower — `setPipeline` / `setBindGroup` (with dynamic offsets) / `setVertexBuffer` / `setIndexBuffer` / `draw` / `drawIndexed` / `setViewport` / `setScissorRect` / `setBlendConstant` / `setStencilReference` / `beginOcclusionQuery` / `endOcclusionQuery` / `end`
- **Compute pass**: `setPipeline` / `setBindGroup` (with dynamic offsets) / `dispatchWorkgroups` / `end`
- **Command encoder**: `beginComputePass` / `beginRenderPass` / `copyBufferToBuffer` / `copyBufferToTexture` / `copyTextureToBuffer` / `copyTextureToTexture` / `resolveQuerySet` / `finish`
- **Queue**: `submit` / `writeBuffer` / `writeTexture` / `onSubmittedWorkDone` (real device-poll, not a stub)
- **QuerySet**: `deviceCreateQuerySet` (occlusion + timestamp) / `querySetDestroy`
- **Error scopes**: `devicePushErrorScope` / `devicePopErrorScope` (real wgpu errorScope wired through, returning `{type, message}` JSON)
- **Surface (on-screen)**: `requestSurface` / `surfaceFromNativeView` / `surfaceGetViewPtr` / `surfaceGetPreferredFormat` / `surfaceConfigure` / `surfaceGetCurrentTexture` / `surfacePresent` / `surfaceUnconfigure` / `surfaceDrop` — render a swapchain into a perry-ui window

## On-screen surface

WebGPU presents to a window through a swapchain. The browser hides this behind `canvas.getContext("webgpu")`; natively, wgpu builds a `Surface` from a platform window/view handle. Perry programs are headless by default, so the on-screen path rides on **perry-ui**: its `BloomView` widget ("render-surface host for an external GPU renderer") reserves a GPU-capable native view in the window and hands back the platform handle. `@perryts/webgpu` *wraps* that handle into a swapchain — it never creates windows itself, so compute / render-to-texture programs still link without dragging in the UI toolkit.

The render loop is the spec's, plus one explicit `surfacePresent` (browsers present implicitly at task end; native wgpu needs the hand-back). A complete, runnable program is in [`examples/triangle-macos`](examples/triangle-macos/) — the shape is:

```typescript
import { App, BloomView, bloomViewGetNativeHandle, onFrame } from "perry/ui";
import {
  requestAdapter, adapterRequestDevice,
  surfaceFromNativeView, surfaceGetPreferredFormat, surfaceConfigure,
  surfaceGetCurrentTexture, surfacePresent, textureCreateView,
  deviceCreateCommandEncoder, commandEncoderBeginRenderPass,
  renderPassSetPipeline, renderPassDraw, renderPassEnd,
  commandEncoderFinish, queueSubmit,
} from "@perryts/webgpu";

const view = BloomView(800, 600);                  // perry-ui reserves the native view
const adapter = await requestAdapter();
const { device, queue } = await adapterRequestDevice(adapter);

const surface = surfaceFromNativeView(bloomViewGetNativeHandle(view)); // wrap it
const format = surfaceGetPreferredFormat(surface, adapter);
surfaceConfigure(surface, { device, format, width: 800, height: 600 });

// …build `pipeline` against `format` exactly as in the browser…

onFrame(function frame() {
  const tex = surfaceGetCurrentTexture(surface);   // this frame's swapchain image
  const target = textureCreateView(tex);           // → render-pass color attachment
  const enc = deviceCreateCommandEncoder(device);
  const pass = commandEncoderBeginRenderPass(enc, {
    colorAttachments: [{ view: target, loadOp: "clear", storeOp: "store",
                         clearValue: { r: 0, g: 0, b: 0, a: 1 } }],
  });
  renderPassSetPipeline(pass, pipeline);
  renderPassDraw(pass, 3);
  renderPassEnd(pass);
  queueSubmit(queue, JSON.stringify([commandEncoderFinish(enc)]));
  surfacePresent(surface);                         // hand the frame to the compositor
});

App({ title: "WebGPU Triangle", width: 800, height: 600, body: view });
```

> **Main-thread note**: `onFrame` fires on perry-ui's main-thread frame pump, so each `getCurrentTexture` + `surfacePresent` blocks it on GPU/vsync sync. Render only as often as the image actually changes — a static scene can render a short burst and stop re-registering; the `CAMetalLayer` keeps showing the last presented frame and the UI stays responsive. (See the example.)

`surfaceFromNativeView(ptr)` is the canonical seam, and the one a future "wrap an existing wgpu device" bloom entry point builds on (see below). `requestSurface({ width, height })` is the inverse — the binding allocates its *own* `NSView` and returns its pointer via `surfaceGetViewPtr` for the host to embed; it's macOS-only today, since other toolkits own view creation.

### Requirements

The ergonomic camelCase API (`requestAdapter`, not `js_webgpu_request_adapter`) routes to native through Perry's native-library support, which needs a Perry build with **ergonomic-export routing** ([PerryTS/perry#5621](https://github.com/PerryTS/perry/issues/5621)) and the **`json` descriptor param type** ([#5626](https://github.com/PerryTS/perry/issues/5626)). The package manifest declares each target's `crate` + `lib` and marks descriptor params `json` accordingly.

### Platform status

| Platform | View creation (`requestSurface`) | Adopt existing (`surfaceFromNativeView`) | Backend |
|----------|----------------------------------|------------------------------------------|---------|
| macOS    | ✅ implemented & verified (`NSView` → `CAMetalLayer`) | ✅ (`NSView*`) | Metal |
| iOS      | — (UIKit owns the view) | ✅ (`UIView*`, must be `CAMetalLayer`-backed) | Metal |
| Windows  | — (create the `HWND` via perry-ui) | ✅ (`HWND`) | DX12 / Vulkan |
| Android  | — (the framework owns the `Surface`) | ✅ (`ANativeWindow*`) | Vulkan |
| Linux    | — | ⏳ needs a display handle alongside the window (X11/Wayland) — follow-up | Vulkan |

**macOS is verified end-to-end** — [`examples/triangle-macos`](examples/triangle-macos/) renders a live triangle into a perry-ui window. The other arms compile against their platform window handles but await on-device validation. Mobile and Windows invert view ownership (the toolkit/OS creates the view), so they go through `surfaceFromNativeView` with the toolkit's pointer rather than `requestSurface`. Linux still needs a richer entry point because a single pointer can't carry the X11/Wayland display connection wgpu also requires.

## Out of scope

- **External textures**: depend on the canvas integration crate exposing `ImageBitmap` (#570 acceptance criterion); the descriptor parser treats the v0.2 wire format as a forwards-compatible no-op.

## Bloom interop

Out of scope for this version — every adapter/device is freshly allocated. The bloom-side hook is a future addition: bloom will expose its `wgpu::Device` + `wgpu::Queue` via a stable internal API, and `@perryts/webgpu` will gain a "wrap an existing device" entry point so a user can drop into raw WebGPU for a custom pass inside a bloom app. The surface side already has its half of that seam — `surfaceFromNativeView` wraps a view bloom hands over — so the remaining work is the device/queue adoption path.

## Versioning

Pre-1.0. The `perry.nativeLibrary.abiVersion` (currently `0.5`) is a hard pin against Perry's perry-ffi ABI — bump in lockstep with the Perry release that the bindings target.

## License

MIT — see [LICENSE](./LICENSE).
