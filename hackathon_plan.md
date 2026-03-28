# Objective

Make SCP-173 from SCP Containment Breach

i.e. if it sees a person that has their eyes closed, it makes a beeline towards that person

unanswered questions:
- what if two people have their eyes closed?
- how does it navigate?
- how do we send get it to move?

tentative plan:

1. ~~figure out how camera data is sent over to us from the computer.~~ **DONE**
   if we manage to get the raw camera stream, we can run a model offline that allows us to
   a. do monocular depth estimation
   b. figure out whose eyes are closed

2. once that's done, we want to set a path towards that person. this means streaming motion commands
   to the model. so figure out how to do that

3. then go from deploying it on a PC to deploying it on the comma v4 (stretch)

---

# Step 1 Breakdown: Camera Streaming from Comma v4 to PC

## Architecture

The comma body has no camera of its own. A comma v4 device is mounted on the body
and provides all cameras + compute. Three cameras exist:

- **road** (forward-facing, narrow)
- **wide** (forward-facing, wide FOV — best for person detection)
- **driver** (inward-facing)

The data flow on the comma v4 looks like this:

```
camerad (C++, ISP) -> NV12 frames -> VisionIPC (shared memory)
                                          |
                                     encoderd --stream
                                          |
                                     H.265 encoded frames
                                          |
                                   cereal msgq (shared memory)
                                          |
                                   cereal/messaging/bridge
                                          |
                                     ZMQ TCP publisher
                                          |
                                    ~~~ network ~~~
                                          |
                              view_stream.py on PC (ZMQ sub)
                                          |
                                   PyAV HEVC decode -> RGB frames -> JPEG
                                          |
                                   HTTP server on localhost:8099
```

## The Problem

`tools/camerastream/view_stream.py` was using `cereal.messaging.sub_sock()` with
the remote device IP. This crashes immediately:

```
Assertion `address == "127.0.0.1"' failed.
Aborted (core dumped)
```

**Why:** The cereal messaging library (`msgq`) only supports shared-memory IPC
(localhost). ZMQ support was removed from msgq in recent openpilot. The `os.environ["ZMQ"] = "1"`
flag in the original script had no effect — there is no ZMQ backend to switch to.

## The Fix

Rewrote `view_stream.py` to bypass `cereal.messaging` entirely and use `pyzmq` directly:

1. The comma v4 already runs `cereal/messaging/bridge` which subscribes to all
   local msgq topics and re-publishes them over ZMQ on TCP ports.

2. Each service gets a deterministic port via FNV-1a hash (defined in
   `cereal/messaging/bridge_zmq.cc`). We replicated that hash in Python:
   - `roadEncodeData` -> port 51336
   - `wideRoadEncodeData` -> port 42305
   - `driverEncodeData` -> port 57332

3. The script connects a `zmq.SUB` socket to `tcp://<device_ip>:<port>`,
   receives raw capnp bytes, deserializes with `log.Event.from_bytes()`,
   decodes H.265 with PyAV, and serves JPEG frames over HTTP.

4. All three cameras decode in parallel. Switch cameras live in the browser
   via `http://localhost:8099/?camera=wide` (or `road` or `driver`).

## Key Files

- `tools/camerastream/view_stream.py` — the fixed streaming viewer (our code)
- `cereal/messaging/bridge.cc` — the bridge binary (runs on device, msgq <-> ZMQ)
- `cereal/messaging/bridge_zmq.cc` — ZMQ socket impl + FNV-1a port hash
- `system/camerad/cameras/hw.h` — camera hardware config (3 cameras)
- `system/loggerd/encoderd.cc` — H.265 encoder that produces `*EncodeData` streams

## How to Run

1. SSH into comma v4: `ssh comma@192.168.63.120`
2. Confirm bridge is running: `ps aux | grep bridge`
   (it should already be running; if not: `cd /data/openpilot && ./cereal/messaging/bridge`)
3. On PC: `.venv/bin/python tools/camerastream/view_stream.py 192.168.63.120`
4. Open http://localhost:8099 — click road/wide/driver to switch cameras

---

# FAQ

1. How to SSH into the thing
a. only one person can SSH in at once time. Be on `unifi` WIFI and ssh comma@192.168.63.120. 
   you should have a github account that you logged into `connect.comma.ai` with
   and also set the ssh username on the comma v4 to your github username
