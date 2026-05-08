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

## Out of scope

- **`GPUSurface` / on-screen canvas**: window-bound, follow-up ticket (the issue's "On-screen canvas widget (separate ticket)" carve-out). Off-screen render-to-texture is fully supported via the v0.2 surface above.
- **External textures**: depend on the canvas integration crate exposing `ImageBitmap` (#570 acceptance criterion); the descriptor parser treats the v0.2 wire format as a forwards-compatible no-op.

## Bloom interop

Out of scope for this version — every adapter/device is freshly allocated. The bloom-side hook is a future addition: bloom will expose its `wgpu::Device` + `wgpu::Queue` via a stable internal API, and `@perryts/webgpu` will gain a "wrap an existing device" entry point so a user can drop into raw WebGPU for a custom pass inside a bloom app.

## Versioning

Pre-1.0. The `perry.nativeLibrary.abiVersion` (currently `0.5`) is a hard pin against Perry's perry-ffi ABI — bump in lockstep with the Perry release that the bindings target.

## License

MIT — see [LICENSE](./LICENSE).
