#!/usr/bin/env python3
"""
Eye Hunter: Drives comma body v2 toward people with closed eyes.

Usage:
  # On the comma v4 device (SSH in):
  cd /data/openpilot
  cereal/messaging/bridge                                    # msgq->zmq (already running if body is on)
  cereal/messaging/bridge <your_pc_ip> testJoystick          # zmq->msgq for receiving commands

  # On your PC:
  python tools/camerastream/eye_hunter.py 192.168.63.120

  Open http://localhost:8097 to see the annotated feed.

How it works:
  1. Receives road camera H.265 stream from comma v4 via ZMQ
  2. Runs MediaPipe Face Mesh to detect faces and eye state (open/closed)
  3. When a closed-eyes target is found, computes steering + throttle
  4. Publishes testJoystick commands back via ZMQ for the body to drive toward them
"""

import argparse
import io
import os
import socket
import subprocess
import time
import threading

import av
import zmq
import numpy as np
from PIL import Image, ImageDraw

from cereal import log

V4L2_BUF_FLAG_KEYFRAME = 8

STREAMS = {
  "road": "roadEncodeData",
  "wide": "wideRoadEncodeData",
  "driver": "driverEncodeData",
}

# Blendshape blink threshold — above this = eyes closed
BLINK_THRESHOLD = 0.5

# Model path
FACE_LANDMARKER_MODEL = "/tmp/face_landmarker_v2_with_blendshapes.task"
FACE_LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

# Control params
MAX_STEER = 0.5      # max steering command [-1, 1]
MAX_THROTTLE = 0.3   # max forward throttle
MIN_THROTTLE = 0.1   # creep speed when target found
FACE_SIZE_CLOSE = 0.25  # face width as fraction of frame = "close enough", slow down
STEER_DEADZONE = 0.05   # don't steer if target is near center

latest_annotated_jpeg: bytes | None = None
latest_jpeg_lock = threading.Lock()

# Status info
status_info = {"state": "searching", "targets": 0, "fps": 0.0, "steer": 0.0, "throttle": 0.0}
status_lock = threading.Lock()

# Shared control commands (updated by detection loop, sent by command thread)
current_cmd = {"accel": 0.0, "steer": 0.0}
current_cmd_lock = threading.Lock()


def get_port(endpoint: str) -> int:
  fnv_prime = 0x100000001b3
  hash_value = 0xcbf29ce484222325
  for c in endpoint.encode():
    hash_value ^= c
    hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
  return 8023 + (hash_value % (65535 - 8023))


def download_model():
  """Download the face landmarker model if not present."""
  if not os.path.exists(FACE_LANDMARKER_MODEL):
    import urllib.request
    print("[model] Downloading face landmarker model...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    print("[model] Done.")


def build_joystick_msg(accel: float, steer: float) -> bytes:
  """Build a cereal testJoystick message."""
  import capnp
  msg = log.Event.new_message()
  msg.logMonoTime = int(time.monotonic() * 1e9)
  msg.valid = True
  msg.init('testJoystick')
  msg.testJoystick.axes = [accel, steer]
  msg.testJoystick.buttons = []
  return msg.to_bytes()


def command_sender_loop(addr: str):
  """Send testJoystick commands at 100Hz so joystickd doesn't time out."""
  joystick_port = get_port("testJoystick")
  ctx = zmq.Context()
  pub_sock = ctx.socket(zmq.PUB)
  pub_sock.bind(f"tcp://0.0.0.0:{joystick_port}")
  print(f"[control] Publishing testJoystick at 100Hz on port {joystick_port}")

  while True:
    with current_cmd_lock:
      accel = current_cmd["accel"]
      steer = current_cmd["steer"]
    msg_bytes = build_joystick_msg(accel, steer)
    pub_sock.send(msg_bytes)
    time.sleep(0.01)  # 100Hz


def detection_and_control_loop(addr: str, camera: str, control_enabled: bool):
  global latest_annotated_jpeg

  import mediapipe as mp
  from ultralytics import YOLO

  download_model()

  sock_name = STREAMS[camera]
  port = get_port(sock_name)
  zmq_endpoint = f"tcp://{addr}:{port}"
  print(f"[{camera}] Connecting to {zmq_endpoint}...")

  ctx = zmq.Context()

  # subscriber for video
  sub_sock = ctx.socket(zmq.SUB)
  sub_sock.setsockopt(zmq.SUBSCRIBE, b"")
  sub_sock.setsockopt(zmq.RECONNECT_IVL_MAX, 500)
  sub_sock.setsockopt(zmq.CONFLATE, 1)
  sub_sock.connect(zmq_endpoint)

  codec = av.CodecContext.create("hevc", "r")

  # YOLO for person detection (runs on GPU)
  yolo = YOLO("yolov8n.pt")
  print("[model] YOLOv8n loaded on GPU")

  # MediaPipe FaceLandmarker for blink detection (runs on CPU, only on face crops)
  options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
  )
  landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
  print("[model] FaceLandmarker loaded (for blink on crops)")

  seen_iframe = False
  fps_count = 0
  fps_time = time.monotonic()

  while True:
    try:
      data = sub_sock.recv()
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
      frame_rgb = frame_bgr[:, :, ::-1].copy()
      h, w = frame_rgb.shape[:2]

      # YOLO person detection on GPU (~5ms)
      results = yolo(frame_bgr, classes=[0], conf=0.5, verbose=False, imgsz=640)

      target_x = None
      closed_eyes_faces = []
      open_eyes_faces = []

      for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
        person_w = (x2 - x1) / w  # normalized width
        person_center_x = ((x1 + x2) / 2) / w  # normalized center

        # estimate head region: top 30% of person bbox
        head_y2 = y1 + int((y2 - y1) * 0.35)
        head_x1 = max(0, x1 - int((x2 - x1) * 0.1))
        head_x2 = min(w, x2 + int((x2 - x1) * 0.1))
        head_y1 = max(0, y1)
        head_y2 = min(h, head_y2)

        # crop head region for blink detection
        head_crop = frame_rgb[head_y1:head_y2, head_x1:head_x2]
        if head_crop.shape[0] < 20 or head_crop.shape[1] < 20:
          open_eyes_faces.append(((x1, y1, x2, y2), 0.0))
          continue

        # run MediaPipe on small head crop only
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(head_crop))
        face_results = landmarker.detect(mp_image)

        if not face_results.face_landmarks or not face_results.face_blendshapes:
          open_eyes_faces.append(((x1, y1, x2, y2), 0.0))
          continue

        blink_left = 0.0
        blink_right = 0.0
        for bs in face_results.face_blendshapes[0]:
          if bs.category_name == "eyeBlinkLeft":
            blink_left = bs.score
          elif bs.category_name == "eyeBlinkRight":
            blink_right = bs.score
        blink_avg = (blink_left + blink_right) / 2.0

        eyes_closed = blink_avg > BLINK_THRESHOLD
        box = (x1, y1, x2, y2)

        if eyes_closed:
          closed_eyes_faces.append((box, blink_avg, person_center_x, person_w))
        else:
          open_eyes_faces.append((box, blink_avg))

      # pick the largest closed-eyes person as target
      steer_cmd = 0.0
      throttle_cmd = 0.0
      state = "searching"

      if closed_eyes_faces:
        closed_eyes_faces.sort(key=lambda f: f[3], reverse=True)
        target = closed_eyes_faces[0]
        target_x = target[2]
        target_face_size = target[3]

        error = target_x - 0.5
        if abs(error) > STEER_DEADZONE:
          # negate: body v2 carcontroller negates torque, so we invert here
          steer_cmd = np.clip(-error * 2.0, -MAX_STEER, MAX_STEER)

        if target_face_size > FACE_SIZE_CLOSE:
          throttle_cmd = 0.0
          state = "arrived"
        else:
          throttle_cmd = -np.clip(MAX_THROTTLE * (1.0 - target_face_size / FACE_SIZE_CLOSE),
                                   MIN_THROTTLE, MAX_THROTTLE)
          state = "targeting"

      # update shared control command (sent at 100Hz by command_sender_loop)
      now = time.monotonic()
      if control_enabled:
        with current_cmd_lock:
          current_cmd["accel"] = throttle_cmd
          current_cmd["steer"] = steer_cmd

      # annotate frame
      pil_img = Image.fromarray(frame_rgb)
      draw = ImageDraw.Draw(pil_img)

      for box, blink in open_eyes_faces:
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        draw.text((box[0], box[1] - 15), f"blink:{blink:.2f} OPEN", fill=(0, 255, 0))

      for box, blink, cx, fw in closed_eyes_faces:
        is_target = (cx == target_x) if target_x is not None else False
        color = (255, 0, 0) if is_target else (255, 165, 0)
        lw = 4 if is_target else 2
        draw.rectangle(box, outline=color, width=lw)
        label = f"blink:{blink:.2f} CLOSED"
        if is_target:
          label += " [TARGET]"
        draw.text((box[0], box[1] - 15), label, fill=color)

      # HUD
      hud_y = 10
      draw.text((10, hud_y), f"State: {state.upper()}", fill=(255, 255, 255))
      hud_y += 20
      draw.text((10, hud_y), f"Steer: {steer_cmd:+.2f}  Throttle: {throttle_cmd:+.2f}", fill=(255, 255, 255))
      hud_y += 20
      draw.text((10, hud_y), f"Faces: {len(open_eyes_faces) + len(closed_eyes_faces)}  Targets: {len(closed_eyes_faces)}", fill=(255, 255, 255))
      if not control_enabled:
        draw.text((10, hud_y + 20), "CONTROL DISABLED (--no-control)", fill=(255, 255, 0))

      buf = io.BytesIO()
      pil_img.save(buf, format='JPEG', quality=80)
      with latest_jpeg_lock:
        latest_annotated_jpeg = buf.getvalue()

      with status_lock:
        status_info["state"] = state
        status_info["targets"] = len(closed_eyes_faces)
        status_info["steer"] = steer_cmd
        status_info["throttle"] = throttle_cmd

      fps_count += 1
      if now - fps_time >= 2.0:
        fps_val = fps_count / (now - fps_time)
        with status_lock:
          status_info["fps"] = fps_val
        print(f"  {fps_val:.1f} fps | {state} | steer:{steer_cmd:+.2f} throttle:{throttle_cmd:+.2f} | persons:{len(open_eyes_faces)+len(closed_eyes_faces)} targets:{len(closed_eyes_faces)}")
        fps_count = 0
        fps_time = now


HTML_PAGE = b"""\
<!DOCTYPE html>
<html><head><title>eye hunter</title>
<style>
  body { margin: 0; background: #111; display: flex; flex-direction: column;
         align-items: center; justify-content: center; height: 100vh; font-family: monospace; }
  img { max-width: 100vw; max-height: 85vh; }
  .hud { color: #0f0; padding: 8px 16px; font-size: 16px; }
  .title { color: #f00; font-size: 24px; padding: 8px; font-weight: bold; }
</style></head>
<body>
  <div class="title">EYE HUNTER</div>
  <img id="stream" />
  <div class="hud" id="hud"></div>
  <script>
    const img = document.getElementById('stream');
    const hud = document.getElementById('hud');
    let frames = 0, lastTime = performance.now();
    function refresh() {
      img.src = '/frame.jpg?t=' + Date.now();
    }
    img.onload = () => {
      frames++;
      const now = performance.now();
      if (now - lastTime > 1000) {
        const fps = (frames / ((now - lastTime) / 1000)).toFixed(1);
        hud.textContent = `browser: ${fps} fps`;
        frames = 0;
        lastTime = now;
      }
      setTimeout(refresh, 10);
    };
    img.onerror = () => setTimeout(refresh, 500);
    refresh();
  </script>
</body></html>
"""


from http.server import HTTPServer, BaseHTTPRequestHandler


class StreamHandler(BaseHTTPRequestHandler):
  def do_GET(self):
    if self.path == '/' or self.path == '/index.html':
      self.send_response(200)
      self.send_header('Content-Type', 'text/html')
      self.end_headers()
      self.wfile.write(HTML_PAGE)
    elif self.path.startswith('/frame.jpg'):
      with latest_jpeg_lock:
        jpeg = latest_annotated_jpeg
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


def get_local_ip(device_addr: str) -> str:
  """Get our local IP as seen from the device's network."""
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect((device_addr, 80))
      return s.getsockname()[0]
  except Exception:
    return "127.0.0.1"


def start_reverse_bridge(device_addr: str, local_ip: str) -> subprocess.Popen | None:
  """SSH into the comma v4 and start a zmq->msgq bridge for testJoystick."""
  cmd = [
    "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
    f"comma@{device_addr}",
    f"cd /data/openpilot && exec cereal/messaging/bridge {local_ip} testJoystick"
  ]
  print(f"[bridge] Starting reverse bridge via SSH: {' '.join(cmd[-1:])}")
  try:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(1)
    if proc.poll() is not None:
      stderr = proc.stderr.read().decode().strip()
      print(f"[bridge] Failed to start: {stderr}")
      return None
    print(f"[bridge] Reverse bridge running (pid {proc.pid})")
    return proc
  except Exception as e:
    print(f"[bridge] SSH failed: {e}")
    return None


def main():
  parser = argparse.ArgumentParser(description="Eye Hunter: drive toward people with closed eyes")
  parser.add_argument("addr", help="Comma v4 IP address")
  parser.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
  parser.add_argument("--port", type=int, default=8097, help="Web viewer port")
  parser.add_argument("--no-control", action="store_true", help="Vision only, don't send drive commands")
  args = parser.parse_args()

  control_enabled = not args.no_control
  bridge_proc = None

  print("=" * 50)
  print("  EYE HUNTER")
  print("=" * 50)

  if control_enabled:
    local_ip = get_local_ip(args.addr)
    print(f"\n  Local IP: {local_ip}")
    print(f"  Starting reverse bridge for joystick commands...")
    bridge_proc = start_reverse_bridge(args.addr, local_ip)
    if bridge_proc is None:
      print("  WARNING: Could not start reverse bridge. Control may not work.")
      print(f"  Manually run on device: cereal/messaging/bridge {local_ip} testJoystick")
  else:
    print("\n  Control DISABLED (vision only mode)")

  print(f"\n  Open http://localhost:{args.port} to see the feed")
  print("=" * 50)

  if control_enabled:
    cmd_thread = threading.Thread(target=command_sender_loop, args=(args.addr,), daemon=True)
    cmd_thread.start()

  t = threading.Thread(target=detection_and_control_loop,
                       args=(args.addr, args.camera, control_enabled), daemon=True)
  t.start()

  try:
    server = HTTPServer(('0.0.0.0', args.port), StreamHandler)
    server.serve_forever()
  finally:
    if bridge_proc:
      print("\n[bridge] Stopping reverse bridge...")
      bridge_proc.terminate()
      bridge_proc.wait(timeout=3)


if __name__ == "__main__":
  main()
