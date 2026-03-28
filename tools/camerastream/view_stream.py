#!/usr/bin/env python3
"""
View live camera stream from comma v4 in your browser.
Usage: python tools/camerastream/view_stream.py 192.168.63.120

Opens http://localhost:8099 — shows MJPEG stream of the road camera.
Switch cameras live: http://localhost:8099/?camera=wide
  Options: road, wide, driver

Prerequisites:
  Run `cereal/messaging/bridge` on the comma device (no args = msgq-to-zmq).
  This script connects to the ZMQ publisher the bridge exposes.
"""

import argparse
import io
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import av
import zmq
from PIL import Image

from cereal import log

V4L2_BUF_FLAG_KEYFRAME = 8

STREAMS = {
  "road": "roadEncodeData",
  "wide": "wideRoadEncodeData",
  "driver": "driverEncodeData",
}

# Per-camera state: each camera gets its own decoder thread + latest JPEG
camera_jpegs: dict[str, bytes | None] = {cam: None for cam in STREAMS}
camera_jpegs_lock = threading.Lock()

active_camera = "road"
active_camera_lock = threading.Lock()


def get_port(endpoint: str) -> int:
  """Replicate the FNV-1a port hash from cereal/messaging/bridge_zmq.cc"""
  fnv_prime = 0x100000001b3
  hash_value = 0xcbf29ce484222325
  for c in endpoint.encode():
    hash_value ^= c
    hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
  start_port = 8023
  max_port = 65535
  return start_port + (hash_value % (max_port - start_port))


def decode_loop(addr: str, camera: str):
  sock_name = STREAMS[camera]
  port = get_port(sock_name)
  zmq_endpoint = f"tcp://{addr}:{port}"
  print(f"[{camera}] Connecting to {zmq_endpoint} for {sock_name}...")

  ctx = zmq.Context()
  sock = ctx.socket(zmq.SUB)
  sock.setsockopt(zmq.SUBSCRIBE, b"")
  sock.setsockopt(zmq.RECONNECT_IVL_MAX, 500)
  sock.connect(zmq_endpoint)

  codec = av.CodecContext.create("hevc", "r")

  seen_iframe = False
  fps_count = 0
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

      frame = frames[0].to_ndarray(format=av.video.format.VideoFormat('bgr24'))

      img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
      buf = io.BytesIO()
      img.save(buf, format='JPEG', quality=80)

      with camera_jpegs_lock:
        camera_jpegs[camera] = buf.getvalue()

      fps_count += 1
      now = time.monotonic()
      if now - fps_time >= 2.0:
        print(f"  [{camera}] {fps_count / (now - fps_time):.1f} fps, {frame.shape[1]}x{frame.shape[0]}")
        fps_count = 0
        fps_time = now


def make_html_page(current_camera: str) -> bytes:
  buttons = ""
  for cam in STREAMS:
    style = "font-weight:bold;text-decoration:underline" if cam == current_camera else ""
    buttons += f'<a href="/?camera={cam}" style="color:#fff;margin:0 12px;font-size:18px;{style}">{cam}</a>'

  return f"""\
<!DOCTYPE html>
<html><head><title>comma v4 stream</title>
<style>
  body {{ margin: 0; background: #000; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; font-family: sans-serif; }}
  img {{ max-width: 100vw; max-height: 90vh; }}
  .controls {{ padding: 10px; }}
</style></head>
<body>
  <div class="controls">{buttons}</div>
  <img id="stream" />
  <script>
    const img = document.getElementById('stream');
    function refresh() {{
      img.src = '/frame.jpg?t=' + Date.now();
    }}
    img.onload = () => setTimeout(refresh, 50);
    img.onerror = () => setTimeout(refresh, 500);
    refresh();
  </script>
</body></html>
""".encode()


class StreamHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    global active_camera
    parsed = urlparse(self.path)
    params = parse_qs(parsed.query)

    if parsed.path == '/' or parsed.path == '/index.html':
      requested = params.get('camera', [None])[0]
      if requested and requested in STREAMS:
        with active_camera_lock:
          if active_camera != requested:
            print(f"Switching to {requested} camera")
            active_camera = requested

      with active_camera_lock:
        current = active_camera

      self.send_response(200)
      self.send_header('Content-Type', 'text/html')
      self.end_headers()
      self.wfile.write(make_html_page(current))

    elif parsed.path == '/frame.jpg':
      with active_camera_lock:
        current = active_camera
      with camera_jpegs_lock:
        jpeg = camera_jpegs[current]

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
  global active_camera

  parser = argparse.ArgumentParser(description="View comma v4 camera stream in browser")
  parser.add_argument("addr", help="Device IP address (e.g. 192.168.63.120)")
  parser.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
  parser.add_argument("--port", type=int, default=8099)
  args = parser.parse_args()

  active_camera = args.camera

  for cam in STREAMS:
    t = threading.Thread(target=decode_loop, args=(args.addr, cam), daemon=True)
    t.start()

  print(f"Open http://localhost:{args.port} in your browser")
  print(f"Switch cameras: http://localhost:{args.port}/?camera=wide")
  server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
  server.serve_forever()


if __name__ == "__main__":
  main()
