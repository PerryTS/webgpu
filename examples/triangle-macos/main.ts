// Minimal on-screen WebGPU triangle, hosted in a perry-ui window.
//
// Run (macOS):
//   cd examples/triangle-macos
//   perry compile main.ts -o triangle && ./triangle
//
// The flow mirrors the browser's, with two perry-specific seams:
//   • perry-ui's `BloomView` reserves a GPU-capable native view in the
//     window; `bloomViewGetNativeHandle` hands us its `NSView*`, which
//     `surfaceFromNativeView` wraps into a wgpu swapchain (wgpu attaches
//     the CAMetalLayer itself).
//   • native wgpu presents explicitly — `surfacePresent` after submit.

import { App, BloomView, bloomViewGetNativeHandle, onFrame } from "perry/ui";
import {
  requestAdapter,
  adapterRequestDevice,
  deviceCreateShaderModule,
  deviceCreateRenderPipeline,
  surfaceFromNativeView,
  surfaceGetPreferredFormat,
  surfaceConfigure,
  surfaceGetCurrentTexture,
  surfacePresent,
  textureCreateView,
  deviceCreateCommandEncoder,
  commandEncoderBeginRenderPass,
  renderPassSetPipeline,
  renderPassDraw,
  renderPassEnd,
  commandEncoderFinish,
  queueSubmit,
} from "@perryts/webgpu";

const WIDTH = 800;
const HEIGHT = 600;

// Reserve the render-surface view first — perry-ui owns it, we draw into it.
const view = BloomView(WIDTH, HEIGHT);

const adapter = await requestAdapter();
const { device, queue } = await adapterRequestDevice(adapter);

// Wrap the view's native handle into a swapchain and configure it.
const surface = surfaceFromNativeView(bloomViewGetNativeHandle(view));
const format = surfaceGetPreferredFormat(surface, adapter);
surfaceConfigure(surface, { device, format, width: WIDTH, height: HEIGHT });

// A self-contained triangle — positions come from @builtin(vertex_index),
// so there are no vertex buffers and an "auto" pipeline layout suffices.
const shader = deviceCreateShaderModule(
  device,
  `
  @vertex
  fn vs_main(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
    var p = array<vec2f, 3>(
      vec2f( 0.0,  0.5),
      vec2f(-0.5, -0.5),
      vec2f( 0.5, -0.5),
    );
    return vec4f(p[i], 0.0, 1.0);
  }
  @fragment
  fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 0.45, 0.1, 1.0); // perry orange
  }
`
);

const pipeline = deviceCreateRenderPipeline(device, {
  layout: "auto",
  vertex: { module: shader, entryPoint: "vs_main" },
  fragment: { module: shader, entryPoint: "fs_main", targets: [{ format }] },
  primitive: { topology: "triangle-list" },
});

// The triangle is static, so we render a short bounded burst — enough frames to
// reliably latch the drawn frame onto the swapchain, then stop. perry-ui drives
// `onFrame` from a main-thread timer, so rendering on EVERY tick forever would
// block that thread on GPU present/vsync and beachball the UI. Once we stop, the
// CAMetalLayer keeps showing the last presented frame and the window is free.
let frames = 0;
let attempts = 0;
const TARGET_FRAMES = 15;

function drawFrame(): void {
  if (frames >= TARGET_FRAMES) return;
  attempts++;
  const tex = surfaceGetCurrentTexture(surface);
  // 0 = swapchain not presentable yet (view not yet on screen); wait + retry.
  if ((tex as number) === 0) {
    if (attempts < 600) onFrame(drawFrame);
    return;
  }

  const target = textureCreateView(tex);
  const enc = deviceCreateCommandEncoder(device);
  const pass = commandEncoderBeginRenderPass(enc, {
    colorAttachments: [
      {
        view: target,
        loadOp: "clear",
        storeOp: "store",
        clearValue: { r: 0.07, g: 0.07, b: 0.09, a: 1.0 },
      },
    ],
  });
  renderPassSetPipeline(pass, pipeline);
  renderPassDraw(pass, 3);
  renderPassEnd(pass);
  queueSubmit(queue, JSON.stringify([commandEncoderFinish(enc)]));
  surfacePresent(surface);
  frames++;
  if (frames < TARGET_FRAMES) onFrame(drawFrame); // brief burst, then idle
}

onFrame(drawFrame);

App({ title: "WebGPU Triangle", width: WIDTH, height: HEIGHT, body: view });
