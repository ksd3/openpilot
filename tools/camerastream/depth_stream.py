#!/usr/bin/env python3
"""
Live monocular depth estimation from comma v4 camera stream.

Usage: python tools/camerastream/depth_stream.py 192.168.63.120

Opens http://localhost:8098 — shows camera feed + depth map side by side.
Uses Depth Anything V2 (SOTA monocular depth estimation).

Prerequisites:
  - Run `cereal/messaging/bridge` on the comma device
  - pip install torch torchvision transformers
"""

import argparse
import io
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import av
import zmq
import numpy as np
from PIL import Image

from cereal import log

V4L2_BUF_FLAG_KEYFRAME = 8

STREAMS = {
  "road": "roadEncodeData",
  "wide": "wideRoadEncodeData",
  "driver": "driverEncodeData",
}

latest_combined_jpeg: bytes | None = None
latest_combined_lock = threading.Lock()

depth_fps_val = 0.0
stream_fps_val = 0.0


def get_port(endpoint: str) -> int:
  fnv_prime = 0x100000001b3
  hash_value = 0xcbf29ce484222325
  for c in endpoint.encode():
    hash_value ^= c
    hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
  return 8023 + (hash_value % (65535 - 8023))


def colorize_depth(depth: np.ndarray) -> np.ndarray:
  """Convert single-channel depth to a colorized RGB image using plasma colormap."""
  depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
  depth_u8 = (depth_norm * 255).astype(np.uint8)

  # plasma-style colormap (purple -> blue -> teal -> yellow)
  # build a simple 256-entry LUT
  lut = np.zeros((256, 3), dtype=np.uint8)
  for i in range(256):
    t = i / 255.0
    if t < 0.25:
      s = t / 0.25
      lut[i] = [int(13 + s * (86 - 13)), int(8 + s * (1 - 8)), int(135 + s * (164 - 135))]
    elif t < 0.5:
      s = (t - 0.25) / 0.25
      lut[i] = [int(86 + s * (187 - 86)), int(1 + s * (55 - 1)), int(164 + s * (120 - 164))]
    elif t < 0.75:
      s = (t - 0.5) / 0.25
      lut[i] = [int(187 + s * (249 - 187)), int(55 + s * (142 - 55)), int(120 + s * (9 - 120))]
    else:
      s = (t - 0.75) / 0.25
      lut[i] = [int(249 + s * (240 - 249)), int(142 + s * (249 - 142)), int(9 + s * (33 - 9))]

  return lut[depth_u8]


class DepthEstimator:
  def __init__(self, model_size: str = "small"):
    import torch
    from transformers import pipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
    print(f"Loading {model_name} on {device}...")

    self.pipe = pipeline("depth-estimation", model=model_name, device=device)
    self.device = device
    print(f"Model loaded on {device}")

  def estimate(self, rgb_image: Image.Image) -> np.ndarray:
    result = self.pipe(rgb_image)
    return np.array(result["depth"])


def decode_and_depth_loop(addr: str, camera: str, model_size: str):
  global latest_combined_jpeg, depth_fps_val, stream_fps_val

  sock_name = STREAMS[camera]
  port = get_port(sock_name)
  zmq_endpoint = f"tcp://{addr}:{port}"
  print(f"[{camera}] Connecting to {zmq_endpoint}...")

  ctx = zmq.Context()
  sock = ctx.socket(zmq.SUB)
  sock.setsockopt(zmq.SUBSCRIBE, b"")
  sock.setsockopt(zmq.RECONNECT_IVL_MAX, 500)
  sock.setsockopt(zmq.CONFLATE, 1)  # only keep latest frame, drop old ones
  sock.connect(zmq_endpoint)

  codec = av.CodecContext.create("hevc", "r")

  # load model
  estimator = DepthEstimator(model_size)

  seen_iframe = False
  frame_count = 0
  depth_count = 0
  fps_time = time.monotonic()

  while True:
    try:
      data = sock.recv()
    except zmq.ZMQError:
      time.sleep(0.01)
      continue

    with log.Event.from_bytes(data) as evt:
      evta = getattr(evt, evt.which())

      if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
        continue

      if not seen_iframe:
        codec.decode(av.packet.Packet(evta.header))
        seen_iframe = True

      frames = codec.decode(av.packet.Packet(evta.data))
      if len(frames) == 0:
        continue

      frame_bgr = frames[0].to_ndarray(format=av.video.format.VideoFormat('bgr24'))
      frame_rgb = frame_bgr[:, :, ::-1]
      frame_count += 1

      # run depth estimation
      h, w = frame_rgb.shape[:2]
      # downscale for speed
      scale = min(518 / h, 518 / w, 1.0)
      if scale < 1.0:
        small_h, small_w = int(h * scale), int(w * scale)
        pil_img = Image.fromarray(frame_rgb).resize((small_w, small_h), Image.BILINEAR)
      else:
        pil_img = Image.fromarray(frame_rgb)

      depth_map = estimator.estimate(pil_img)

      # resize depth back to original size
      depth_colored = colorize_depth(depth_map)
      depth_pil = Image.fromarray(depth_colored).resize((w, h), Image.BILINEAR)
      depth_count += 1

      # combine side by side
      combined = Image.new('RGB', (w * 2, h))
      combined.paste(Image.fromarray(frame_rgb), (0, 0))
      combined.paste(depth_pil, (w, 0))

      buf = io.BytesIO()
      combined.save(buf, format='JPEG', quality=80)

      with latest_combined_lock:
        latest_combined_jpeg = buf.getvalue()

      now = time.monotonic()
      if now - fps_time >= 2.0:
        stream_fps_val = frame_count / (now - fps_time)
        depth_fps_val = depth_count / (now - fps_time)
        print(f"  stream: {stream_fps_val:.1f} fps, depth: {depth_fps_val:.1f} fps, {w}x{h}")
        frame_count = 0
        depth_count = 0
        fps_time = now


HTML_PAGE = b"""\
<!DOCTYPE html>
<html><head><title>comma v4 depth estimation</title>
<style>
  body { margin: 0; background: #000; display: flex; flex-direction: column;
         align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; }
  img { max-width: 100vw; max-height: 90vh; }
  .info { color: #aaa; padding: 8px; font-size: 14px; }
  .label { position: absolute; top: 10px; color: #fff; font-size: 16px;
           background: rgba(0,0,0,0.6); padding: 4px 12px; border-radius: 4px; }
</style></head>
<body>
  <div class="info">camera feed (left) | depth estimation (right) | Depth Anything V2</div>
  <img id="stream" />
  <div class="info" id="fps"></div>
  <script>
    const img = document.getElementById('stream');
    const fpsEl = document.getElementById('fps');
    let lastTime = 0, frameCount = 0;
    function refresh() {
      img.src = '/frame.jpg?t=' + Date.now();
    }
    img.onload = () => {
      frameCount++;
      const now = performance.now();
      if (now - lastTime > 1000) {
        fpsEl.textContent = `browser: ${(frameCount / ((now - lastTime) / 1000)).toFixed(1)} fps`;
        frameCount = 0;
        lastTime = now;
      }
      setTimeout(refresh, 10);
    };
    img.onerror = () => setTimeout(refresh, 500);
    refresh();
  </script>
</body></html>
"""


class StreamHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == '/' or self.path == '/index.html':
      self.send_response(200)
      self.send_header('Content-Type', 'text/html')
      self.end_headers()
      self.wfile.write(HTML_PAGE)

    elif self.path.startswith('/frame.jpg'):
      with latest_combined_lock:
        jpeg = latest_combined_jpeg
      if jpeg is None:
        self.send_response(503)
        self.end_headers()
        return
      self.send_response(200)
      self.send_header('Content-Type', 'image/jpeg')
      self.send_header('Content-Length', str(len(jpeg)))
      self.send_header('Cache-Control', 'no-store')
      self.end_headers()
      self.wfile.write(jpeg)

    else:
      self.send_response(404)
      self.end_headers()

  def log_message(self, format, *args):
    pass


def main():
  parser = argparse.ArgumentParser(description="Depth estimation on comma v4 camera stream")
  parser.add_argument("addr", help="Device IP address (e.g. 192.168.63.120)")
  parser.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
  parser.add_argument("--port", type=int, default=8098)
  parser.add_argument("--model-size", default="small", choices=["small", "base", "large"],
                       help="Depth Anything V2 model size (small=fastest, large=best quality)")
  args = parser.parse_args()

  t = threading.Thread(target=decode_and_depth_loop,
                       args=(args.addr, args.camera, args.model_size), daemon=True)
  t.start()

  print(f"Open http://localhost:{args.port} in your browser")
  server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
  server.serve_forever()


if __name__ == "__main__":
  main()
