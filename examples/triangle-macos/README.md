# On-screen WebGPU triangle (macOS)

A minimal on-screen WebGPU program: a single orange triangle rendered into a
perry-ui window via a swapchain. It exercises the full surface path —
`surfaceFromNativeView` → `surfaceConfigure` → `surfaceGetCurrentTexture` →
render → `surfacePresent` — driven by `onFrame`.

See [`main.ts`](./main.ts). The key integration is:

```ts
const view = BloomView(800, 600);                          // perry-ui owns the native view
const surface = surfaceFromNativeView(bloomViewGetNativeHandle(view)); // we wrap its NSView
surfaceConfigure(surface, { device, format, width: 800, height: 600 });
// …each frame: getCurrentTexture → createView → render pass → submit → present
```

`BloomView` is perry-ui's "render-surface host for an external GPU renderer" —
it reserves a GPU-capable native view and hands back its pointer
(`NSView*` on macOS). wgpu attaches the `CAMetalLayer` itself.

## Setup

```sh
# From the repo root, link the package into this example's node_modules
# (until @perryts/webgpu is npm-installed):
mkdir -p node_modules/@perryts
ln -sfn ../../../.. node_modules/@perryts/webgpu
```

The host `package.json` already allow-lists the native library
(`perry.allow.nativeLibrary`), required because it links native code.

## Run

```sh
perry compile main.ts -o triangle && ./triangle
```

You'll need `perry-ui-macos` built once in your Perry checkout:

```sh
cargo build --release -p perry-ui-macos
```

## Status

**Runs end-to-end on macOS** against a `perry` containing
[#5621](https://github.com/PerryTS/perry/issues/5621) (ergonomic camelCase →
`js_<pkg>_*` routing) and
[#5626](https://github.com/PerryTS/perry/issues/5626) (the `json` manifest param
type). Verified the full render loop executes:

```
configured surface, format = bgra8unorm-srgb   # surfaceConfigure (json descriptor)
creating pipeline...                            # deviceCreateRenderPipeline (json descriptor)
rendered frame 1
rendered frame 2
rendered frame 3                                # getCurrentTexture → createView → beginRenderPass → draw → submit → present, looping
```

Three pieces were needed; all are in place:

1. **Linking** — manifest `targets` declare `crate` + `lib` (`"crate": "."`,
   `"lib": "libperry_ext_webgpu.a"`); without them perry builds the staticlib but
   never links it.

2. **Routing classification** — an `exports.perry` entry → `src/index.ts` so the
   import is `ModuleKind::NativeCompiled`, which enables #5621's alias routing.

3. **Descriptor marshalling** — descriptor params are declared `"json"` in the
   manifest (per #5626), so perry `JSON.stringify`s the object at the FFI
   boundary; the Rust side `serde_json`-deserializes it unchanged. Genuine-string
   params (WGSL source, index format, error filter, the pre-stringified
   `*Json` args) stay `"string"`.

Requires `perry-ui-macos` built once (`cargo build --release -p perry-ui-macos`)
and, for the native-library compile, `PERRY_ALLOW_PERRY_FEATURES=1` (or the host
`perry.allow.nativeLibrary` allow-list, already in this example's `package.json`).
