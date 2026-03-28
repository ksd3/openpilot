#!/usr/bin/env python3
"""
View live camera stream from comma v4 in your browser.
Usage: python tools/camerastream/view_stream.py 192.168.63.120

Opens http://localhost:8099 — shows MJPEG stream of the road camera.
"""

import argparse
import os

os.environ["ZMQ"] = "1"

import io
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

import av
import numpy as np
from PIL import Image

import cereal.messaging as messaging

V4L2_BUF_FLAG_KEYFRAME = 8

STREAMS = {
  "road": "roadEncodeData",
  "wide": "wideRoadEncodeData",
  "driver": "driverEncodeData",
}

latest_jpeg = None
latest_jpeg_lock = threading.Lock()


def decode_loop(addr: str, camera: str):
  global latest_jpeg

  sock_name = STREAMS[camera]
  print(f"Connecting to {addr} for {sock_name}...")

  codec = av.CodecContext.create("hevc", "r")
  messaging.reset_context()
  sock = messaging.sub_sock(sock_name, None, addr=addr, conflate=False)

  seen_iframe = False
  fps_count = 0
  fps_time = time.monotonic()

  while True:
    msgs = messaging.drain_sock(sock, wait_for_one=True)
    for evt in msgs:
      evta = getattr(evt, evt.which())

      if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
        continue

      if not seen_iframe:
        codec.decode(av.packet.Packet(evta.header))
        seen_iframe = True

      frames = codec.decode(av.packet.Packet(evta.data))
      if len(frames) == 0:
        continue

      # convert to RGB numpy array
      frame = frames[0].to_ndarray(format=av.video.format.VideoFormat('bgr24'))

      # encode as JPEG
      img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
      buf = io.BytesIO()
      img.save(buf, format='JPEG', quality=80)
      jpeg_bytes = buf.getvalue()

      with latest_jpeg_lock:
        latest_jpeg = jpeg_bytes

      fps_count += 1
      now = time.monotonic()
      if now - fps_time >= 2.0:
        print(f"  {fps_count / (now - fps_time):.1f} fps, frame {frame.shape[1]}x{frame.shape[0]}")
        fps_count = 0
        fps_time = now


HTML_PAGE = b"""\
<!DOCTYPE html>
<html><head><title>comma v4 stream</title>
<style>
  body { margin: 0; background: #000; display: flex; align-items: center; justify-content: center; height: 100vh; }
  img { max-width: 100vw; max-height: 100vh; }
</style></head>
<body>
  <img id="stream" />
  <script>
    const img = document.getElementById('stream');
    function refresh() {
      img.src = '/frame.jpg?t=' + Date.now();
    }
    img.onload = () => setTimeout(refresh, 50);
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
      with latest_jpeg_lock:
        jpeg = latest_jpeg
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
    pass  # silence request logs


def main():
  parser = argparse.ArgumentParser(description="View comma v4 camera stream in browser")
  parser.add_argument("addr", help="Device IP address (e.g. 192.168.63.120)")
  parser.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
  parser.add_argument("--port", type=int, default=8099)
  args = parser.parse_args()

  # start decoder thread
  t = threading.Thread(target=decode_loop, args=(args.addr, args.camera), daemon=True)
  t.start()

  print(f"Open http://localhost:{args.port} in your browser")
  server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
  server.serve_forever()


if __name__ == "__main__":
  main()
