#!/usr/bin/env python3
"""
SCP-173 / Eye Hunter: comma body v2 approaches people only when unobserved.

Usage:
  # On the comma v4 device (SSH in):
  cd /data/openpilot
  cereal/messaging/bridge
  cereal/messaging/bridge <your_pc_ip> testJoystick

  # On your PC:
  python tools/camerastream/eye_hunter.py 192.168.63.120

  Open http://localhost:8097 to see the annotated feed.

Behavior:
  - FREEZE if any visible person has eyes open AND faces the camera (being watched).
  - ADVANCE only toward people whose eyes are classified CLOSED (MediaPipe blink/squint),
    unless --chase-largest-person (debug / ignores eyes).
  - Range: monocular depth or bbox width; optional depth strip steer bias.
"""

from __future__ import annotations

import argparse
import io
import os
import socket
import subprocess
import time
import threading
from dataclasses import dataclass
from typing import Any

import av
import numpy as np
import zmq
from PIL import Image, ImageDraw

from cereal import log

V4L2_BUF_FLAG_KEYFRAME = 8

STREAMS = {
  "road": "roadEncodeData",
  "wide": "wideRoadEncodeData",
  "driver": "driverEncodeData",
}

# MediaPipe landmark indices (face mesh) for yaw proxy in crop space
LM_NOSE_TIP = 1
LM_LEFT_EYE_OUTER = 33
LM_RIGHT_EYE_OUTER = 263

FACE_LANDMARKER_MODEL = "/tmp/face_landmarker_v2_with_blendshapes.task"
FACE_LANDMARKER_URL = (
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

latest_annotated_jpeg: bytes | None = None
latest_jpeg_lock = threading.Lock()

status_info: dict[str, Any] = {
  "state": "searching",
  "targets": 0,
  "fps": 0.0,
  "steer": 0.0,
  "throttle": 0.0,
  "observed": False,
  "depth_ok": False,
}
status_lock = threading.Lock()

current_cmd = {"accel": 0.0, "steer": 0.0}
current_cmd_lock = threading.Lock()


def get_port(endpoint: str) -> int:
  fnv_prime = 0x100000001b3
  hash_value = 0xcbf29ce484222325
  for c in endpoint.encode():
    hash_value ^= c
    hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
  return 8023 + (hash_value % (65535 - 8023))


def download_face_model() -> None:
  if not os.path.exists(FACE_LANDMARKER_MODEL):
    import urllib.request
    print("[model] Downloading face landmarker model...")
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
    print("[model] Done.")


def build_joystick_msg(accel: float, steer: float) -> bytes:
  msg = log.Event.new_message()
  msg.logMonoTime = int(time.monotonic() * 1e9)
  msg.valid = True
  msg.init('testJoystick')
  msg.testJoystick.axes = [accel, steer]
  msg.testJoystick.buttons = []
  return msg.to_bytes()


def command_sender_loop(addr: str) -> None:
  joystick_port = get_port("testJoystick")
  ctx = zmq.Context()
  pub_sock = ctx.socket(zmq.PUB)
  pub_sock.bind(f"tcp://0.0.0.0:{joystick_port}")
  print(f"[control] Publishing testJoystick at 100Hz on port {joystick_port}")

  while True:
    with current_cmd_lock:
      accel = current_cmd["accel"]
      steer = current_cmd["steer"]
    pub_sock.send(build_joystick_msg(accel, steer))
    time.sleep(0.01)


@dataclass
class PersonPerception:
  box: tuple[int, int, int, int]
  has_face: bool
  blink_avg: float
  eyes_closed: bool
  facing_camera: bool
  yaw_proxy: float
  person_center_x_norm: float
  person_w_norm: float


def yaw_proxy_from_landmarks(lm_list: list) -> float:
  """Horizontal nose offset / inter-ocular distance. ~0 when facing camera."""
  try:
    n = lm_list[LM_NOSE_TIP]
    le = lm_list[LM_LEFT_EYE_OUTER]
    re = lm_list[LM_RIGHT_EYE_OUTER]
  except (IndexError, TypeError):
    return 0.0
  eye_mid_x = (le.x + re.x) / 2.0
  iod = abs(re.x - le.x) + 1e-6
  return float((n.x - eye_mid_x) / iod)


def extract_eye_blendshapes(blendshapes: list) -> tuple[float, float, float, float]:
  """Returns blink_left, blink_right, squint_left, squint_right (0–1)."""
  blink_left = blink_right = squint_left = squint_right = 0.0
  for bs in blendshapes:
    n = bs.category_name
    if n == "eyeBlinkLeft":
      blink_left = bs.score
    elif n == "eyeBlinkRight":
      blink_right = bs.score
    elif n == "eyeSquintLeft":
      squint_left = bs.score
    elif n == "eyeSquintRight":
      squint_right = bs.score
  return blink_left, blink_right, squint_left, squint_right


def blink_combined_score(
  blink_l: float,
  blink_r: float,
  squint_l: float,
  squint_r: float,
  mode: str,
  use_squint: bool,
) -> float:
  if mode == "max":
    m = max(blink_l, blink_r)
  else:
    m = (blink_l + blink_r) / 2.0
  if use_squint:
    s = (squint_l + squint_r) / 2.0
    m = 0.62 * m + 0.38 * s
  return float(m)


def eye_look_off_camera_score(blendshapes: list) -> float:
  """Heuristic: how much eyes are rolled away from straight-ahead (0 = neutral)."""
  scores: dict[str, float] = {bs.category_name: bs.score for bs in blendshapes}
  keys = (
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookUpLeft", "eyeLookUpRight",
  )
  return float(sum(scores.get(k, 0.0) for k in keys) / max(len(keys), 1))


class DepthEstimator:
  """Monocular depth via HuggingFace Depth Anything V2 Small (relative, uint8-style)."""

  def __init__(self, device: str | None) -> None:
    import torch
    from transformers import pipeline

    if device == "cpu":
      dev = -1
    elif device == "cuda" or device is None:
      dev = 0 if torch.cuda.is_available() else -1
    elif device.lstrip("-").isdigit():
      dev = int(device)
    else:
      dev = 0 if torch.cuda.is_available() else -1

    print(f"[depth] Loading Depth-Anything-V2-Small (device={dev})...")
    self.pipe = pipeline(
      task="depth-estimation",
      model="depth-anything/Depth-Anything-V2-Small-hf",
      device=dev,
    )

  def infer(self, rgb: np.ndarray, out_wh: tuple[int, int], max_input_dim: int = 448) -> np.ndarray:
    """rgb uint8 HxWx3 -> depth float32 HxW resized to out_wh (width, height)."""
    ow, oh = out_wh
    h0, w0 = rgb.shape[:2]
    m = max(h0, w0)
    if m > max_input_dim and m > 0:
      s = max_input_dim / m
      nw, nh = max(1, int(w0 * s)), max(1, int(h0 * s))
      pil_in = Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR)
    else:
      pil_in = Image.fromarray(rgb)
    out = self.pipe(pil_in)
    d = np.asarray(out["depth"], dtype=np.float32)
    if d.max() > 1.5:
      d = d / 255.0
    pil_d = Image.fromarray(d)
    pil_d = pil_d.resize((ow, oh), Image.BILINEAR)
    return np.asarray(pil_d, dtype=np.float32)


def median_depth_roi(depth: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
  x1, y1 = max(0, x1), max(0, y1)
  x2, y2 = min(depth.shape[1] - 1, x2), min(depth.shape[0] - 1, y2)
  if x2 <= x1 or y2 <= y1:
    return float(np.median(depth))
  patch = depth[y1:y2, x1:x2]
  return float(np.median(patch))


def ema(prev: float, new: float, alpha: float) -> float:
  if alpha <= 0.0:
    return new
  return float(alpha * new + (1.0 - alpha) * prev)


def path_bias_from_strips(
  depth: np.ndarray,
  strip_y0_ratio: float,
  margin: float,
  max_bias: float,
) -> float:
  """
  Compare median depth in bottom left/center/right strips.
  Depth model: larger value ~= closer. If center is much closer than sides, bias steer
  toward the side that is farther (smaller median).
  Returns additive steer bias in [-max_bias, max_bias] (same sign convention as main steer).
  """
  h, w = depth.shape[:2]
  y0 = int(h * strip_y0_ratio)
  band = depth[y0:h, :]
  if band.size == 0:
    return 0.0
  third = w // 3
  mL = float(np.median(band[:, :third]))
  mC = float(np.median(band[:, third:2 * third]))
  mR = float(np.median(band[:, 2 * third:]))
  side_avg = (mL + mR) / 2.0
  if mC <= side_avg + margin:
    return 0.0
  # Center is more "close" than sides — nudge toward freer side (lower median = farther in inv-depth encoding; for DA uint8 higher=near, so freer = lower value)
  if mL < mR:
    return max_bias
  if mR < mL:
    return -max_bias
  return 0.0


def detection_and_control_loop(cfg: argparse.Namespace) -> None:
  global latest_annotated_jpeg

  import warnings

  # Torch probes CUDA even when we run on CPU; old drivers spam stderr every run.
  warnings.filterwarnings(
    "ignore",
    message=".*The NVIDIA driver on your system is too old.*",
    category=UserWarning,
  )

  import mediapipe as mp
  from ultralytics import YOLO

  download_face_model()

  addr = cfg.addr
  camera = cfg.camera
  control_enabled = not cfg.no_control

  sock_name = STREAMS[camera]
  port = get_port(sock_name)
  zmq_endpoint = f"tcp://{addr}:{port}"
  print(f"[{camera}] Connecting to {zmq_endpoint}...")

  ctx = zmq.Context()
  sub_sock = ctx.socket(zmq.SUB)
  sub_sock.setsockopt(zmq.SUBSCRIBE, b"")
  sub_sock.setsockopt(zmq.RECONNECT_IVL_MAX, 500)
  sub_sock.setsockopt(zmq.CONFLATE, 1)
  sub_sock.connect(zmq_endpoint)

  codec = av.CodecContext.create("hevc", "r")

  yolo = YOLO("yolov8n.pt")
  print("[model] YOLOv8n loaded")

  depth_model: DepthEstimator | None = None
  if not cfg.no_depth:
    try:
      depth_model = DepthEstimator(cfg.depth_device)
    except Exception as e:
      print(f"[depth] Disabled (failed to load): {e}")

  # Note: face_landmarks are always returned on detection; older mediapipe has no output_face_landmarks flag.
  options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    output_face_blendshapes=True,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4,
  )
  landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
  print("[model] FaceLandmarker loaded")

  poller = zmq.Poller()
  poller.register(sub_sock, zmq.POLLIN)
  print(
    f"[stream] Waiting for ZMQ packets ({sock_name} @ {zmq_endpoint}). "
    "If this hangs, confirm cereal/messaging/bridge is running on the comma and the IP is correct.",
    flush=True,
  )

  seen_iframe = False
  fps_count = 0
  fps_time = time.monotonic()
  frame_idx = 0
  last_depth: np.ndarray | None = None
  pre_keyframe_packets = 0
  keyframe_warn_t = time.monotonic()
  steer_filt = 0.0
  throttle_filt = 0.0

  while True:
    loop_t0 = time.monotonic()
    try:
      if not poller.poll(5000):
        print(
          f"[stream] No data for 5s — still listening on {zmq_endpoint}. "
          "On device: ps aux | grep bridge; encoderd/camerad must be producing *EncodeData.",
          flush=True,
        )
        continue
      data = sub_sock.recv()
    except zmq.ZMQError:
      time.sleep(0.01)
      continue

    with log.Event.from_bytes(data) as evt:
      evta = getattr(evt, evt.which())

      if not seen_iframe and not (evta.idx.flags & V4L2_BUF_FLAG_KEYFRAME):
        pre_keyframe_packets += 1
        nowk = time.monotonic()
        if nowk - keyframe_warn_t >= 3.0:
          print(
            f"[stream] Waiting for HEVC keyframe ({pre_keyframe_packets} non-key packets skipped so far)...",
            flush=True,
          )
          keyframe_warn_t = nowk
        continue
      if not seen_iframe:
        codec.decode(av.packet.Packet(evta.header))
        seen_iframe = True
        print("[stream] Got keyframe — decoding video.", flush=True)

      frames = codec.decode(av.packet.Packet(evta.data))
      if len(frames) == 0:
        continue

      frame_bgr = frames[0].to_ndarray(format=av.video.format.VideoFormat('bgr24'))
      frame_rgb = frame_bgr[:, :, ::-1].copy()
      h, w = frame_rgb.shape[:2]

      frame_idx += 1
      # Uncompressed camera to browser immediately so /frame.jpg is not 503 while depth/YOLO run (CPU depth can take many seconds).
      if not cfg.no_preview:
        pv = io.BytesIO()
        Image.fromarray(frame_rgb).save(pv, format='JPEG', quality=cfg.preview_jpeg_quality)
        with latest_jpeg_lock:
          latest_annotated_jpeg = pv.getvalue()
        if frame_idx == 1:
          print("[feed] Raw camera preview is live; annotated frames replace it after each full pipeline pass.", flush=True)

      results = yolo(frame_bgr, classes=[0], conf=cfg.yolo_conf, verbose=False, imgsz=cfg.yolo_imgsz)

      persons: list[PersonPerception] = []
      closed_eyes: list[PersonPerception] = []

      for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
        person_w = (x2 - x1) / w
        person_center_x = ((x1 + x2) / 2) / w

        head_y2 = y1 + int((y2 - y1) * cfg.head_crop_fraction)
        head_x1 = max(0, x1 - int((x2 - x1) * 0.1))
        head_x2 = min(w, x2 + int((x2 - x1) * 0.1))
        head_y1 = max(0, y1)
        head_y2 = min(h, head_y2)

        head_crop = frame_rgb[head_y1:head_y2, head_x1:head_x2]
        if head_crop.shape[0] < 20 or head_crop.shape[1] < 20:
          persons.append(
            PersonPerception(
              (x1, y1, x2, y2), False, 0.0, False, False, 0.0, person_center_x, person_w,
            )
          )
          continue

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(head_crop))
        face_results = landmarker.detect(mp_image)

        if not face_results.face_landmarks or not face_results.face_blendshapes:
          persons.append(
            PersonPerception(
              (x1, y1, x2, y2), False, 0.0, False, False, 0.0, person_center_x, person_w,
            )
          )
          continue

        lm0 = face_results.face_landmarks[0]
        yaw = yaw_proxy_from_landmarks(lm0)
        look_off = eye_look_off_camera_score(face_results.face_blendshapes[0])

        blink_l, blink_r, sq_l, sq_r = extract_eye_blendshapes(face_results.face_blendshapes[0])
        blink_metric = blink_combined_score(
          blink_l, blink_r, sq_l, sq_r, cfg.blink_mode, cfg.blink_use_squint,
        )
        eyes_closed = blink_metric > cfg.blink_threshold

        facing = abs(yaw) < cfg.yaw_threshold and look_off < cfg.eye_look_threshold
        p = PersonPerception(
          (x1, y1, x2, y2), True, blink_metric, eyes_closed, facing, yaw, person_center_x, person_w,
        )
        persons.append(p)
        if eyes_closed:
          closed_eyes.append(p)

      depth_ok = False
      if depth_model is not None and frame_idx % cfg.depth_every_n == 0:
        try:
          last_depth = depth_model.infer(frame_rgb, (w, h), max_input_dim=cfg.depth_max_input)
          depth_ok = True
        except Exception as e:
          print(f"[depth] inference error: {e}", flush=True)
      elif last_depth is not None:
        depth_ok = True

      observed = False
      for p in persons:
        if not p.has_face:
          continue
        if cfg.strict_open_eyes:
          if not p.eyes_closed:
            observed = True
            break
        else:
          if (not p.eyes_closed) and p.facing_camera:
            observed = True
            break

      steer_cmd = 0.0
      throttle_cmd = 0.0
      state = "searching"
      target_x: float | None = None
      target_person: PersonPerception | None = None

      chase_pool = list(persons) if cfg.chase_largest_person else list(closed_eyes)

      if observed:
        state = "frozen"
      elif chase_pool:
        chase_pool.sort(key=lambda p: p.person_w_norm, reverse=True)
        tgt = chase_pool[0]
        target_person = tgt
        target_x = tgt.person_center_x_norm
        tx1, ty1, tx2, ty2 = tgt.box

        error = target_x - 0.5
        if abs(error) > cfg.steer_deadzone:
          steer_cmd = float(np.clip(-error * cfg.steer_gain, -cfg.max_steer, cfg.max_steer))

        if last_depth is not None:
          ph = ty2 - ty1
          y_feet0 = ty1 + int(0.6 * ph)
          y_feet1 = ty2
          d_tgt = median_depth_roi(last_depth, tx1, y_feet0, tx2, y_feet1)
          d_img = float(np.percentile(last_depth, cfg.depth_close_percentile))
          close_enough = d_tgt >= d_img * cfg.depth_arrived_ratio

          if close_enough:
            throttle_cmd = 0.0
            state = "arrived"
          else:
            t = float(np.clip((d_tgt / (d_img + 1e-6)), 0.0, 1.2))
            throttle_cmd = -float(
              np.clip(cfg.max_throttle * (1.0 - t * 0.85), cfg.min_throttle, cfg.max_throttle)
            )
            state = "targeting"

          if cfg.path_max_bias > 0.0:
            pb = path_bias_from_strips(
              last_depth,
              cfg.path_strip_y0,
              cfg.path_center_margin,
              cfg.path_max_bias,
            )
            steer_cmd = float(np.clip(steer_cmd + pb, -cfg.max_steer, cfg.max_steer))
        else:
          if tgt.person_w_norm > cfg.face_size_close:
            throttle_cmd = 0.0
            state = "arrived"
          else:
            throttle_cmd = -float(
              np.clip(
                cfg.max_throttle * (1.0 - tgt.person_w_norm / cfg.face_size_close),
                cfg.min_throttle,
                cfg.max_throttle,
              )
            )
            state = "targeting"

      if cfg.steer_smooth_alpha <= 0.0:
        steer_cmd_out = steer_cmd
        throttle_cmd_out = throttle_cmd
      elif state in ("frozen", "searching", "arrived"):
        steer_filt = ema(steer_filt, 0.0, max(cfg.steer_smooth_alpha, 0.15))
        throttle_filt = ema(throttle_filt, 0.0, max(cfg.throttle_smooth_alpha, 0.2))
        steer_cmd_out = steer_filt
        throttle_cmd_out = throttle_filt
      else:
        steer_filt = ema(steer_filt, steer_cmd, cfg.steer_smooth_alpha)
        throttle_filt = ema(throttle_filt, throttle_cmd, cfg.throttle_smooth_alpha)
        steer_cmd_out = steer_filt
        throttle_cmd_out = throttle_filt

      now = time.monotonic()
      if control_enabled:
        with current_cmd_lock:
          current_cmd["accel"] = throttle_cmd_out
          current_cmd["steer"] = steer_cmd_out

      pil_img = Image.fromarray(frame_rgb)
      if last_depth is not None and cfg.show_depth_overlay:
        d_vis = last_depth.copy()
        d_vis -= d_vis.min()
        if d_vis.max() > 1e-6:
          d_vis /= d_vis.max()
        d_u8 = (d_vis * 255).astype(np.uint8)
        heat = Image.fromarray(d_u8).convert("L").resize((w // 4, h // 4), Image.BILINEAR)
        heat = heat.resize((w, h), Image.NEAREST)
        heat_rgb = Image.merge("RGB", (heat, heat, heat))
        pil_img = Image.blend(pil_img, heat_rgb, alpha=0.35)

      draw = ImageDraw.Draw(pil_img)
      hud_note_depth = last_depth is not None and cfg.show_depth_overlay

      for p in persons:
        x1, y1, x2, y2 = p.box
        if not p.has_face:
          draw.rectangle(p.box, outline=(128, 128, 128), width=2)
          draw.text((x1, y1 - 15), "no face", fill=(128, 128, 128))
          continue
        if cfg.strict_open_eyes:
          obs = not p.eyes_closed
        else:
          obs = (not p.eyes_closed) and p.facing_camera
        is_chase = target_person is not None and p.box == target_person.box
        if obs:
          color = (255, 0, 255)
          label = "OBSERVER" if not cfg.strict_open_eyes else "OPEN EYES"
        elif p.eyes_closed:
          color = (255, 0, 0) if is_chase else (255, 140, 0)
          label = f"CLOSED blink={p.blink_avg:.2f} ({cfg.blink_mode})"
        elif is_chase and cfg.chase_largest_person:
          color = (0, 200, 255)
          label = f"CHASE blink={p.blink_avg:.2f} (--chase-largest-person)"
        else:
          color = (0, 255, 0)
          label = f"OPEN blink={p.blink_avg:.2f} yaw={p.yaw_proxy:+.2f} ({cfg.blink_mode})"

        lw = 4 if is_chase else 2
        draw.rectangle(p.box, outline=color, width=lw)
        if is_chase and (p.eyes_closed or cfg.chase_largest_person):
          label += " [TARGET]"
        draw.text((x1, y1 - 15), label, fill=color)

      hud_y = 10
      draw.text((10, hud_y), f"State: {state.upper()}", fill=(255, 255, 255))
      hud_y += 20
      draw.text((10, hud_y), f"Observed: {observed}  (strict_open={cfg.strict_open_eyes})", fill=(255, 100, 255))
      hud_y += 20
      sm_lab = "raw" if cfg.steer_smooth_alpha <= 0.0 else "smoothed"
      draw.text((10, hud_y), f"Steer: {steer_cmd_out:+.2f}  Throttle: {throttle_cmd_out:+.2f} ({sm_lab})", fill=(255, 255, 255))
      hud_y += 20
      draw.text((10, hud_y), f"Persons: {len(persons)}  closed-eye: {len(closed_eyes)}", fill=(255, 255, 255))
      hud_y += 20
      if state == "searching" and persons and not closed_eyes and not cfg.chase_largest_person:
        draw.text(
          (10, hud_y),
          "SCP: need CLOSED eyes to advance (lower --blink-threshold or --blink-mode max). "
          "Or --chase-largest-person ignores eyes.",
          fill=(255, 200, 0),
        )
        hud_y += 20
      if hud_note_depth:
        draw.text((10, hud_y), "Depth overlay ON", fill=(0, 255, 255))
        hud_y += 20

      if not control_enabled:
        draw.text((10, hud_y), "CONTROL DISABLED (--no-control)", fill=(255, 255, 0))

      buf = io.BytesIO()
      pil_img.save(buf, format='JPEG', quality=cfg.annotated_jpeg_quality)
      with latest_jpeg_lock:
        latest_annotated_jpeg = buf.getvalue()

      with status_lock:
        status_info["state"] = state
        status_info["targets"] = len(closed_eyes)
        status_info["steer"] = steer_cmd_out
        status_info["throttle"] = throttle_cmd_out
        status_info["observed"] = observed
        status_info["depth_ok"] = depth_ok or (last_depth is not None)

      fps_count += 1
      if now - fps_time >= 2.0:
        fps_val = fps_count / (now - fps_time)
        with status_lock:
          status_info["fps"] = fps_val
        print(
          f"  {fps_val:.1f} fps | {state} | obs={observed} | "
          f"steer:{steer_cmd_out:+.2f} thr:{throttle_cmd_out:+.2f} | persons:{len(persons)} "
          f"closed_eyes:{len(closed_eyes)} chase_mode={'largest' if cfg.chase_largest_person else 'scp'}",
          flush=True,
        )
        fps_count = 0
        fps_time = now

      if cfg.vision_max_fps > 0:
        elapsed = time.monotonic() - loop_t0
        period = 1.0 / float(cfg.vision_max_fps)
        if elapsed < period:
          time.sleep(period - elapsed)


HTML_PAGE = b"""\
<!DOCTYPE html>
<html><head><title>SCP-173 eye hunter</title>
<style>
  body { margin: 0; background: #111; display: flex; flex-direction: column;
         align-items: center; justify-content: center; height: 100vh; font-family: monospace; }
  img { max-width: 100vw; max-height: 85vh; }
  .hud { color: #0f0; padding: 8px 16px; font-size: 16px; }
  .title { color: #f00; font-size: 24px; padding: 8px; font-weight: bold; }
</style></head>
<body>
  <div class="title">SCP-173</div>
  <img id="stream" />
  <div class="hud" id="hud"></div>
  <script>
    const img = document.getElementById('stream');
    const hud = document.getElementById('hud');
    let frames = 0, lastTime = performance.now();
    function refresh() { img.src = '/frame.jpg?t=' + Date.now(); }
    img.onload = () => {
      frames++;
      const now = performance.now();
      if (now - lastTime > 1000) {
        hud.textContent = 'browser: ' + (frames / ((now - lastTime) / 1000)).toFixed(1) + ' fps';
        frames = 0; lastTime = now;
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
  def do_GET(self) -> None:
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

  def log_message(self, format, *args) -> None:
    pass


class ReusableHTTPServer(HTTPServer):
  allow_reuse_address = True


def get_local_ip(device_addr: str) -> str:
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
      s.connect((device_addr, 80))
      return s.getsockname()[0]
  except Exception:
    return "127.0.0.1"


def start_reverse_bridge(device_addr: str, local_ip: str) -> subprocess.Popen | None:
  cmd = [
    "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=5",
    f"comma@{device_addr}",
    f"cd /data/openpilot && exec cereal/messaging/bridge {local_ip} testJoystick",
  ]
  print(f"[bridge] Starting reverse bridge via SSH: {cmd[-1]}")
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


def build_arg_parser() -> argparse.ArgumentParser:
  p = argparse.ArgumentParser(description="SCP-173: depth-guided approach, freeze when observed")
  p.add_argument("addr", help="Comma v4 IP address")
  p.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
  p.add_argument("--port", type=int, default=8097, help="Web viewer port")
  p.add_argument("--no-control", action="store_true", help="Vision only, no drive commands")
  p.add_argument("--no-depth", action="store_true", help="Skip depth model; use bbox width for range")
  p.add_argument(
    "--no-preview",
    action="store_true",
    help="Do not push raw camera JPEG before pipeline (browser may 503 until first annotated frame)",
  )
  p.add_argument("--depth-device", default=None, help="Transformers device: cuda | cpu | 0 | -1 (default: cuda if available)")
  p.add_argument("--depth-every-n", type=int, default=2, help="Run depth every N frames (reuse map in between)")
  p.add_argument(
    "--depth-max-input",
    type=int,
    default=448,
    help="Max longer side (px) fed to depth model; lower=faster on CPU",
  )
  p.add_argument("--depth-close-percentile", type=float, default=88.0, help="Percentile of frame depth for 'close' scale")
  p.add_argument("--depth-arrived-ratio", type=float, default=0.82, help="Target ROI median / percentile >= this -> arrived")
  p.add_argument("--path-strip-y0", type=float, default=0.55, help="Bottom strip band starts at this fraction of height")
  p.add_argument("--path-center-margin", type=float, default=0.06, help="Min excess (normalized) for center obstacle")
  p.add_argument(
    "--path-max-bias",
    type=float,
    default=0.0,
    help="Extra steer from depth strips (0=off). Was a common source of twitchy turning.",
  )
  p.add_argument("--yolo-imgsz", type=int, default=480, help="YOLO letterbox size (lower=faster, 320–640)")
  p.add_argument("--steer-gain", type=float, default=1.25, help="Bearing P gain (lower=calmer)")
  p.add_argument(
    "--steer-smooth-alpha",
    type=float,
    default=0.28,
    help="EMA blend toward new steer (0=off, 0.15–0.35 typical). Larger=snappier.",
  )
  p.add_argument("--throttle-smooth-alpha", type=float, default=0.38, help="EMA for throttle axis")
  p.add_argument(
    "--vision-max-fps",
    type=float,
    default=0.0,
    help="Cap vision/control updates (0=unlimited). Lowers PC CPU; does not reduce comma encode bitrate.",
  )
  p.add_argument("--preview-jpeg-quality", type=int, default=68, help="MJPEG preview quality 1–95")
  p.add_argument("--annotated-jpeg-quality", type=int, default=74, help="Annotated frame JPEG quality")
  p.add_argument(
    "--low-end",
    action="store_true",
    help="Preset: small YOLO, slower depth cadence, cap ~8 vision FPS, heavier smoothing, no path bias, smaller JPEGs",
  )
  p.add_argument(
    "--blink-mode",
    choices=("avg", "max"),
    default="max",
    help="max=either eye closing counts (better for SCP); avg=both eyes averaged",
  )
  p.add_argument(
    "--blink-use-squint",
    action="store_true",
    help="Mix in eyeSquint L/R blendshapes (helps when lids close without high blink score)",
  )
  p.add_argument(
    "--blink-threshold",
    type=float,
    default=0.38,
    help="Combined blink score above this => eyes closed (try 0.32–0.45)",
  )
  p.add_argument(
    "--head-crop-fraction",
    type=float,
    default=0.42,
    help="Person box height fraction used as head region for MediaPipe (larger=easier blink)",
  )
  p.add_argument("--yaw-threshold", type=float, default=0.28, help="|yaw proxy| below = facing camera")
  p.add_argument("--eye-look-threshold", type=float, default=0.45, help="Sum eye-look blendshapes; below = looking at cam")
  p.add_argument("--strict-open-eyes", action="store_true", help="Freeze if any person has open eyes (ignore head pose)")
  p.add_argument(
    "--chase-largest-person",
    action="store_true",
    help="When not frozen, chase largest YOLO person even if eyes are not classified closed (hackathon / tuning)",
  )
  p.add_argument("--yolo-conf", type=float, default=0.5)
  p.add_argument("--max-steer", type=float, default=0.5)
  p.add_argument("--max-throttle", type=float, default=0.3)
  p.add_argument("--min-throttle", type=float, default=0.1)
  p.add_argument("--face-size-close", type=float, default=0.25, help="BBox width frac fallback when no depth")
  p.add_argument("--steer-deadzone", type=float, default=0.08, help="Ignore bearing error smaller than this (fraction of frame)")
  p.add_argument("--show-depth-overlay", action="store_true", help="Blend depth heatmap on viewer")
  return p


def apply_low_end_preset(args: argparse.Namespace) -> None:
  args.yolo_imgsz = min(args.yolo_imgsz, 320)
  args.depth_every_n = max(args.depth_every_n, 5)
  args.depth_max_input = min(args.depth_max_input, 320)
  args.path_max_bias = 0.0
  args.steer_smooth_alpha = max(args.steer_smooth_alpha, 0.26)
  args.throttle_smooth_alpha = max(args.throttle_smooth_alpha, 0.42)
  args.steer_deadzone = max(args.steer_deadzone, 0.09)
  args.steer_gain = min(args.steer_gain, 1.05)
  args.preview_jpeg_quality = min(args.preview_jpeg_quality, 52)
  args.annotated_jpeg_quality = min(args.annotated_jpeg_quality, 62)
  if args.vision_max_fps <= 0.0:
    args.vision_max_fps = 8.0
  else:
    args.vision_max_fps = min(args.vision_max_fps, 12.0)
  print(
    "[low-end] yolo=320 depth_every≥5 vision_fps≤8 path_bias=0 smoothing↑ jpeg↓ "
    "(comma still streams full H.265; use device settings to cut Wi‑Fi bitrate if needed)",
    flush=True,
  )
  args.blink_mode = "max"
  args.blink_use_squint = True
  if args.blink_threshold > 0.42:
    args.blink_threshold = 0.38


def main() -> None:
  parser = build_arg_parser()
  args = parser.parse_args()
  if args.low_end:
    apply_low_end_preset(args)

  control_enabled = not args.no_control
  bridge_proc = None

  print("=" * 50)
  print("  SCP-173 / EYE HUNTER")
  print("=" * 50)
  if not args.chase_largest_person:
    print(
      "  [SCP] Only advances when someone’s eyes are CLOSED (blink score > threshold). "
      "Observers (open eyes + facing cam) freeze you.",
      flush=True,
    )
  else:
    print(
      "  [warn] --chase-largest-person: chases largest person even with open eyes (SCP off).",
      flush=True,
    )

  if control_enabled:
    local_ip = get_local_ip(args.addr)
    print(f"\n  Local IP: {local_ip}")
    print(
      "  [hint] Wheels only turn when: openpilot is ENGAGED on the comma (body/joystick ready), "
      "AND testJoystick reaches the device (reverse bridge above or manual SSH command), "
      "AND HUD state is TARGETING (not FROZEN / SEARCHING / ARRIVED)."
    )
    bridge_proc = start_reverse_bridge(args.addr, local_ip)
    if bridge_proc is None:
      print("  WARNING: Could not start reverse bridge. Control may not work.")
      print(f"  Manually run on device: cereal/messaging/bridge {local_ip} testJoystick")
  else:
    print("\n  Control DISABLED (vision only mode)")

  print(f"\n  Open http://127.0.0.1:{args.port}/ or http://localhost:{args.port}/ (same machine as this script)")
  print("=" * 50)

  if control_enabled:
    threading.Thread(target=command_sender_loop, args=(args.addr,), daemon=True).start()

  threading.Thread(target=detection_and_control_loop, args=(args,), daemon=True).start()

  try:
    server = ReusableHTTPServer(('0.0.0.0', args.port), StreamHandler)
    server.serve_forever()
  finally:
    if bridge_proc:
      print("\n[bridge] Stopping reverse bridge...")
      bridge_proc.terminate()
      bridge_proc.wait(timeout=3)


if __name__ == "__main__":
  main()
