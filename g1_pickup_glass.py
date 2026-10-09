#!/usr/bin/env python3
"""
G1 Pickup Glass - line up on the cup with the RealSense, then run the grasp
===========================================================================

The grasp SEQUENCE below is an open-loop recording that works for ONE
cup-to-robot pose. Everything before it exists to put the robot back into that
pose, wherever the cup is in front of it, as repeatably as possible.

Why the old centring did nothing
--------------------------------
  * Moves were tiny (a few cm/s for half a second) - below the G1's walking
    deadband, so the robot barely stepped and "centred" five times in place.
  * Sideways error was "fixed" by turning, which swings the cup out of the
    arm's plane instead of correcting it.
  * TARGET_DEPTH_M was an uncalibrated guess, so even a perfect servo would
    have aimed at the wrong spot.
  * After failing it grasped anyway.

How it works now (RealSense colour + depth; Opus finds the cup ONCE)
--------------------------------------------------------------------
  1. LLM boxes the cup once (re-asked only if tracking loses it).
  2. From then on the cup is TRACKED IN DEPTH every pulse - segment the blob
     at the cup's depth, take its centroid. Fast, no LLM, repeatable.
  3. Error is in METRES: sideways = (px offset / focal length) * depth,
     forward = depth - calibrated depth.
  4. Walking pulses sized to the error, never below a velocity floor; the
     floor rises automatically when a pulse fails to reduce the error (the
     deadband), up to a cap. Sideways is STRAFED, not turned.
  5. Must be inside tolerance on 2 readings in a row. If it cannot get there
     it REPORTS FAILURE and does NOT grasp.
  6. Runs the fixed SEQUENCE.

ONE-TIME CALIBRATION (required - there is no default)
-----------------------------------------------------
Stand the robot exactly where SEQUENCE works, cup in its original spot, then:

    python3 g1_pickup_glass.py --calibrate

It measures the cup 5 times and saves ~/pickup_calib.json (cup depth and
sideways offset, in metres). Without that file the program refuses to run.
Redo it if the sequence or the camera mount ever changes.

RUN
---
    python3 g1_pickup_glass.py --calibrate      # no motion; saves calibration
    python3 g1_pickup_glass.py --watch          # no motion; live tracking -
                                                # slide the cup by hand and
                                                # check the numbers move sanely
    python3 g1_pickup_glass.py wlan0 --dry-run  # full loop, no base/arm output
    python3 g1_pickup_glass.py wlan0            # for real
    python3 g1_pickup_glass.py wlan0 --no-center
"""

import argparse
import base64
import json
import math
import os
import signal
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: pip3 install opencv-python")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not importable")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("ERROR: pip3 install anthropic")
    sys.exit(1)

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDef
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from g1_brainco_hand import BrainCoHand


# ============================================================
# WHICH HAND / THE CAPTURED GRASP
# ============================================================
HAND = "left"

# Confirmed working by the user on real hardware (see g1_hand_sequence.py
# history) - pose_9 -> pose_8 -> pose_7, then a two-step fingertip grasp that
# holds until released.
SEQUENCE = [
    {"arm": {15: -0.223, 16: +0.763, 17: +0.043, 18: +0.965, 19: -0.054},
     "duration": 1.2, "note": "pose_9 - approach"},
    {"arm": {15: -0.519, 16: +0.811, 17: -0.033, 18: +0.437, 19: -0.054},
     "duration": 1.2, "note": "pose_8"},
    {"arm": {15: -0.519, 16: -0.028, 17: -0.144, 18: +0.437, 19: -0.054},
     "duration": 1.2, "note": "pose_7 - final position before grasp"},
    {"fingers": [0, 1, 0, 0, 0, 0],
     "duration": 0.6, "note": "thumb_aux only"},
    {"fingers": [0, 1, 0.4, 0.4, 0.4, 0.4],
     "duration": 0.8, "note": "fingertip grasp - holds here until released"},
]
DEFAULT_DURATION = 1.0

# ============================================================
# RE-CENTRING
# ============================================================
CALIB_PATH = os.path.expanduser("~/pickup_calib.json")

# Acceptable error at the grasp pose, in metres. Tighten only if the grasp
# proves sensitive; the walking itself is good to a couple of cm at best.
TOL_LAT_M = 0.020
TOL_DEPTH_M = 0.025
ACCEPT_LAT_M = 0.035        # if the time runs out but we're inside THIS,
ACCEPT_DEPTH_M = 0.045      # still grasp; outside it, report failure

# Velocity floors (m/s). The G1 ignores or barely answers very small
# commands, so every pulse is at least this fast. The floor climbs by
# FLOOR_STEP when a pulse does not reduce the error, up to FLOOR_CAP.
V_FLOOR_LAT = 0.12
V_FLOOR_FWD = 0.12
FLOOR_STEP = 0.04
FLOOR_CAP = 0.30
V_MAX = 0.25
GAIN = 1.2                  # m/s per metre of error, before floor/cap
PULSE_MIN_S, PULSE_MAX_S = 0.30, 1.2
PULSE_FRACTION = 0.8        # aim to cover 80% of the error per pulse
SETTLE_S = 0.5
CENTER_TIMEOUT_S = 45.0
MAX_PULSES = 30
NEED_STABLE = 2             # consecutive in-tolerance readings

# Depth-blob tracking
TRACK_BAND_M = 0.06         # pixels within this of the cup depth are "cup"
TRACK_PAD = 1.5             # search window = box grown by this fraction
MIN_BLOB_PX = 250
MAX_DEPTH_JUMP_M = 0.15     # bigger than this between pulses = lost it
DEPTH_RANGE_M = (0.20, 1.50)

WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP = 30
VISION_MODEL = "claude-opus-5-5"

LOCATE_PROMPT = """You are looking through the camera of a humanoid robot, \
which is angled slightly downward at a table. The image is {w} by {h} pixels. \
The colour may look slightly cyan-tinted.

Find the drinking glass or cup the robot is about to pick up.

Reply with ONLY JSON:
{{"found": <true|false>, "x": <left>, "y": <top>, "w": <width>, \
"h": <height>, "confidence": <0-1>}}

Box ONLY the cup itself, tightly - not its shadow, reflection or the table. \
If there is no glass or cup clearly visible, set "found" to false. Do not \
guess a position."""


def load_calibration():
    try:
        with open(CALIB_PATH) as f:
            c = json.load(f)
        return float(c["depth_m"]), float(c["lat_m"])
    except Exception:
        return None


# ============================================================
ARM_JOINTS = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
HAND_JOINTS = {"left": [15, 16, 17, 18, 19], "right": [22, 23, 24, 25, 26]}

ARM_LIMITS = {
    15: (-2.9, 2.7), 16: (-0.3, 2.9), 17: (-2.6, 2.6),
    18: (-0.2, 3.4), 19: (-2.6, 2.6),
    22: (-2.9, 2.7), 23: (-2.9, 0.3), 24: (-2.6, 2.6),
    25: (-0.2, 3.4), 26: (-2.6, 2.6),
}

ARM_SDK_TOPIC = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lf/lowstate"
NOT_USED_JOINT = 29
CONTROL_HZ = 100.0
ARM_KP, ARM_KD = 60.0, 1.5
TEARDOWN_SECONDS = 1.2


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def filter_to_hand(arm_dict, hand):
    allowed = set(HAND_JOINTS[hand])
    return {j: v for j, v in arm_dict.items() if j in allowed}


# A clean SIGTERM shutdown: g1_robot.py releases a grasp with SIGTERM, which
# Python does not turn into `finally:` blocks unless re-raised like this.
def _sigterm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)


# ============================================================
# CAMERA + LOCATE + TRACK
# ============================================================
class Camera:
    def __init__(self):
        self.pipe = None
        self.align = None
        self.scale = 1.0
        self.fx = 600.0
        self.ppx = WIDTH / 2.0

    def open(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        try:
            prof = self.pipe.start(cfg)
        except RuntimeError as e:
            if "busy" in str(e).lower():
                raise RuntimeError(
                    "RealSense is busy - videohub_pc4 holds the colour "
                    "node. Disable videohub in the tablet Service Manager.")
            raise
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        try:
            intr = prof.get_stream(rs.stream.color) \
                .as_video_stream_profile().get_intrinsics()
            self.fx, self.ppx = float(intr.fx), float(intr.ppx)
        except Exception:
            pass
        for _ in range(WARMUP):
            self.pipe.wait_for_frames()

    def close(self):
        if self.pipe:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def frame(self, fresh=True):
        """Newest aligned (color, depth_m). `fresh` drops a few queued
        frames first so a reading taken right after a move is not stale."""
        n = 4 if fresh else 1
        f = None
        for _ in range(n):
            f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return None, None
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        return color, depth


def locate_glass(client, color):
    """Opus boxes the cup. Returns ((x,y,w,h) in image px, seconds) or
    (None, seconds)."""
    h, w = color.shape[:2]
    ok, enc = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None, 0.0
    t0 = time.time()
    resp = client.messages.create(
        model=VISION_MODEL, max_tokens=1024,
        extra_body={"output_config": {"effort": "low"}},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.b64encode(enc.tobytes()).decode()}},
            {"type": "text", "text": LOCATE_PROMPT.format(w=w, h=h)}]}])
    dt = time.time() - t0
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip().replace("```json", "").replace("```", "")
    try:
        data = json.loads(text)
    except Exception:
        return None, dt
    if not data.get("found"):
        return None, dt
    return (int(data["x"]), int(data["y"]), int(data["w"]), int(data["h"])), dt


def _core_depth(depth, box):
    """Median depth of the middle of the box (the cup body, not background
    or the table edge)."""
    x, y, w, h = box
    x0, x1 = int(x + 0.3 * w), int(x + 0.7 * w)
    y0, y1 = int(y + 0.3 * h), int(y + 0.7 * h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(WIDTH, max(x1, x0 + 1)), min(HEIGHT, max(y1, y0 + 1))
    patch = depth[y0:y1, x0:x1]
    vals = patch[(patch > DEPTH_RANGE_M[0]) & (patch < DEPTH_RANGE_M[1])]
    if vals.size < 10:
        return 0.0
    return float(np.median(vals))


def track_cup(depth, box, ref_depth=None):
    """Segment the cup as the blob at its own depth inside a window around
    `box`. Returns dict {cx, cy, depth, box, area} or None if lost.

    Pure depth - no colour, no LLM - so it is fast, repeatable and not
    thrown by the cyan tint."""
    x, y, w, h = box
    pad_x, pad_y = int(w * TRACK_PAD), int(h * TRACK_PAD)
    wx0, wy0 = max(0, x - pad_x), max(0, y - pad_y)
    wx1, wy1 = min(WIDTH, x + w + pad_x), min(HEIGHT, y + h + pad_y)
    if wx1 - wx0 < 8 or wy1 - wy0 < 8:
        return None
    d0 = _core_depth(depth, box)
    if d0 <= 0:
        return None
    if ref_depth is not None and abs(d0 - ref_depth) > MAX_DEPTH_JUMP_M:
        return None
    win = depth[wy0:wy1, wx0:wx1]
    mask = ((np.abs(win - d0) < TRACK_BAND_M) & (win > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return None
    # the component containing the box centre, else the biggest
    cx0, cy0 = (x + w // 2) - wx0, (y + h // 2) - wy0
    pick = None
    if 0 <= cy0 < lab.shape[0] and 0 <= cx0 < lab.shape[1] and lab[cy0, cx0]:
        pick = int(lab[cy0, cx0])
    if pick is None:
        pick = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[pick, cv2.CC_STAT_AREA])
    if area < MIN_BLOB_PX:
        return None
    bx = int(stats[pick, cv2.CC_STAT_LEFT]) + wx0
    by = int(stats[pick, cv2.CC_STAT_TOP]) + wy0
    bw = int(stats[pick, cv2.CC_STAT_WIDTH])
    bh = int(stats[pick, cv2.CC_STAT_HEIGHT])
    vals = win[lab == pick]
    return {"cx": bx + bw / 2.0, "cy": by + bh / 2.0,
            "depth": float(np.median(vals)),
            "box": (bx, by, bw, bh), "area": area}


def measure(cam, client, box=None, prev_depth=None, tries=3):
    """One cup reading: (lat_m, depth_m, box, info) or (None,...,reason).
    Tracks from `box` if given; asks the LLM when there is no box or the
    track is lost."""
    for attempt in range(tries):
        color, depth = cam.frame()
        if color is None:
            continue
        t = track_cup(depth, box, prev_depth) if box else None
        src = "track"
        if t is None:
            src = "llm"
            nb, dt = locate_glass(client, color)
            if nb is None:
                box, prev_depth = None, None
                continue
            t = track_cup(depth, nb)
            if t is None:
                box, prev_depth = None, None
                continue
        lat = (t["cx"] - cam.ppx) / cam.fx * t["depth"]
        return lat, t["depth"], t["box"], src
    return None, None, None, "cup not found"


def center_on_glass(loco_client, cam, client, calib, dry_run=False):
    """Pulse-walk until the cup is at the calibrated depth/sideways offset.
    Returns (ok, message)."""
    tgt_depth, tgt_lat = calib
    floor_lat, floor_fwd = V_FLOOR_LAT, V_FLOOR_FWD
    t_start = time.time()
    box = prev_depth = None
    stable = 0
    last_err = None
    stall = {"lat": 0, "fwd": 0}
    pulses = 0
    el = ed = None

    while pulses <= MAX_PULSES and time.time() - t_start < CENTER_TIMEOUT_S:
        lat, dep, box, src = measure(cam, client, box, prev_depth)
        if lat is None:
            return False, "I can't see the cup to line up on it."
        prev_depth = dep
        el, ed = lat - tgt_lat, dep - tgt_depth     # +el: cup right of target
        print(f"  [{pulses:02d}] ({src}) sideways {el * 100:+5.1f}cm  "
              f"forward {ed * 100:+5.1f}cm   floors "
              f"{floor_lat:.2f}/{floor_fwd:.2f}")

        if abs(el) <= TOL_LAT_M and abs(ed) <= TOL_DEPTH_M:
            stable += 1
            if stable >= NEED_STABLE:
                return True, f"Lined up after {pulses} pulse(s)."
            time.sleep(SETTLE_S)
            continue
        stable = 0

        # Did the last pulse help? If not, the deadband is eating it.
        if last_err is not None:
            axis, before = last_err
            now = abs(el) if axis == "lat" else abs(ed)
            if now > before - 0.005:
                stall[axis] += 1
                if stall[axis] >= 2:
                    if axis == "lat":
                        floor_lat = min(FLOOR_CAP, floor_lat + FLOOR_STEP)
                    else:
                        floor_fwd = min(FLOOR_CAP, floor_fwd + FLOOR_STEP)
                    stall[axis] = 0
                    print(f"      no progress on {axis} - raising the "
                          f"velocity floor")
            else:
                stall[axis] = 0

        # One axis at a time, sideways first (the arm reaches along a line).
        if abs(el) > TOL_LAT_M:
            axis, err, floor = "lat", el, floor_lat
        else:
            axis, err, floor = "fwd", ed, floor_fwd
        v = clamp(GAIN * abs(err), floor, max(V_MAX, floor))
        dur = clamp(PULSE_FRACTION * abs(err) / v, PULSE_MIN_S, PULSE_MAX_S)
        if axis == "lat":
            vx, vy = 0.0, (-v if err > 0 else v)   # cup right -> go right
        else:
            vx, vy = (v if err > 0 else -v), 0.0   # cup far  -> go forward
        last_err = (axis, abs(err))

        if dry_run:
            print(f"      (dry-run: Move({vx:+.2f}, {vy:+.2f}, 0) for "
                  f"{dur:.2f}s)")
        else:
            loco_client.Move(vx, vy, 0.0)
            time.sleep(dur)
            loco_client.Move(0.0, 0.0, 0.0)
        time.sleep(SETTLE_S)
        pulses += 1

    if el is not None and abs(el) <= ACCEPT_LAT_M and abs(ed) <= ACCEPT_DEPTH_M:
        return True, (f"Close enough (off by {el * 100:+.1f}cm sideways, "
                      f"{ed * 100:+.1f}cm forward).")
    off = "" if el is None else (f" (still off by {el * 100:+.1f}cm sideways, "
                                 f"{ed * 100:+.1f}cm forward)")
    return False, "I couldn't line up on the cup well enough to pick it up" + off


def run_calibrate(cam, client):
    """5 readings of the cup from where the grasp works -> ~/pickup_calib.json"""
    box = None
    rows = []
    for i in range(5):
        lat, dep, box, src = measure(cam, client, box, None if not rows else rows[-1][1])
        if lat is None:
            print("Could not find the cup in view - aborting, nothing saved.")
            return 1
        rows.append((lat, dep))
        print(f"  reading {i + 1}: sideways {lat * 100:+.1f}cm, "
              f"depth {dep * 100:.1f}cm ({src})")
        time.sleep(0.4)
    lats = [r[0] for r in rows]
    deps = [r[1] for r in rows]
    spread = max(max(lats) - min(lats), max(deps) - min(deps))
    if spread > 0.02:
        print(f"\nReadings vary by {spread * 100:.1f}cm - the cup or robot "
              f"moved, or tracking is noisy. Nothing saved; try again.")
        return 1
    calib = {"depth_m": float(np.median(deps)), "lat_m": float(np.median(lats)),
             "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved {CALIB_PATH}: depth {calib['depth_m'] * 100:.1f}cm, "
          f"sideways {calib['lat_m'] * 100:+.1f}cm")
    return 0


def run_watch(cam, client):
    """Live tracking, no motion. Slide the cup by hand; sideways should go
    + when it moves to the robot's right, depth up when it moves away."""
    calib = load_calibration()
    box = prev = None
    print("Watching (Ctrl-C to stop)...")
    try:
        while True:
            lat, dep, box, src = measure(cam, client, box, prev)
            if lat is None:
                print("  lost the cup"); box = prev = None
                continue
            prev = dep
            extra = ""
            if calib:
                extra = (f"   vs calibration: sideways "
                         f"{(lat - calib[1]) * 100:+5.1f}cm  forward "
                         f"{(dep - calib[0]) * 100:+5.1f}cm")
            print(f"  ({src}) sideways {lat * 100:+6.1f}cm  depth "
                  f"{dep * 100:5.1f}cm{extra}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    return 0


# ============================================================
# ARM + FINGER PLAYBACK (unchanged from g1_hand_sequence.py)
# ============================================================
class ArmMotion:
    def __init__(self, iface, rate=CONTROL_HZ):
        self.rate = rate
        self.dt = 1.0 / rate
        self.weight = 0.0

        self._st = {"msg": None}
        sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        sub.Init(lambda m: self._st.__setitem__("msg", m), 1)

        print("Waiting for lowstate...")
        deadline = time.time() + 10.0
        while self._st["msg"] is None:
            if time.time() > deadline:
                raise RuntimeError("no lowstate - check the interface name")
            time.sleep(0.05)

        self.crc = CRC()
        self.msg = LowCmdDef()
        self.msg.mode_machine = self._st["msg"].mode_machine
        self.pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self.pub.Init()

        self.pos = {j: self._st["msg"].motor_state[j].q for j in ARM_JOINTS}
        self.home = dict(self.pos)

    def _publish(self):
        for j in ARM_JOINTS:
            lo, hi = ARM_LIMITS[j]
            mc = self.msg.motor_cmd[j]
            mc.q = clamp(self.pos[j], lo, hi)
            mc.dq = 0.0
            mc.kp = ARM_KP
            mc.kd = ARM_KD
            mc.tau = 0.0
        self.msg.motor_cmd[NOT_USED_JOINT].q = self.weight
        self.msg.crc = self.crc.Crc(self.msg)
        self.pub.Write(self.msg)

    def start(self, seconds=1.5):
        print("Ramping arm_sdk weight up...")
        steps = int(seconds * self.rate)
        for i in range(steps):
            self.weight = (i + 1) / steps
            self._publish()
            time.sleep(self.dt)

    def stop(self, seconds=1.0):
        print("Ramping weight down...")
        steps = int(seconds * self.rate)
        w0 = self.weight
        for i in range(steps):
            self.weight = w0 * (1 - (i + 1) / steps)
            self._publish()
            time.sleep(self.dt)

    def go(self, target, seconds):
        if not target:
            time.sleep(seconds)
            return
        joints = list(target.keys())
        start = {j: self.pos[j] for j in joints}
        steps = max(1, int(seconds * self.rate))
        for i in range(1, steps + 1):
            t = i / steps
            for j in joints:
                self.pos[j] = clamp(start[j] + (target[j] - start[j]) * t,
                                    *ARM_LIMITS[j])
            self._publish()
            time.sleep(self.dt)


def play(arms, hand_ctrl, sequence, hand, tempo=1.0):
    for i, step in enumerate(sequence):
        duration = step.get("duration", DEFAULT_DURATION) * tempo
        arm = filter_to_hand(step.get("arm", {}), hand)
        label = f" ({step['note']})" if step.get("note") else ""
        print(f"[{i + 1}/{len(sequence)}] {duration:.2f}s{label}")
        if arm:
            print("    arm -> " +
                  ", ".join(f"{j}:{v:+.3f}" for j, v in sorted(arm.items())))
        if "fingers" in step:
            print(f"    fingers -> {step['fingers']}")
            hand_ctrl.set_all(step["fingers"])
        arms.go(arm, duration)


def main():
    ap = argparse.ArgumentParser(description="G1 pick up a glass, lining up first")
    ap.add_argument("iface", nargs="?")
    ap.add_argument("--hand", choices=["left", "right"], default=HAND)
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full flow but send no base/arm commands")
    ap.add_argument("--no-center", action="store_true",
                    help="skip lining up - grasp from where it stands")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the cup from the pose where SEQUENCE works "
                         "and save ~/pickup_calib.json. No motion.")
    ap.add_argument("--watch", action="store_true",
                    help="live cup tracking, no motion - for checking")
    ap.add_argument("--tempo", type=float, default=1.0)
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in this process")
        return 1
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    if args.calibrate or args.watch:
        cam = Camera()
        print("Opening RealSense...")
        cam.open()
        try:
            return run_calibrate(cam, client) if args.calibrate \
                else run_watch(cam, client)
        finally:
            cam.close()

    if not args.iface:
        print("ERROR: iface required (or use --calibrate / --watch)")
        return 1

    calib = None
    if not args.no_center:
        calib = load_calibration()
        if calib is None:
            msg = (f"no calibration yet - run 'python3 g1_pickup_glass.py "
                   f"--calibrate' once with the robot at the grasp pose "
                   f"({CALIB_PATH} missing)")
            print("ERROR: " + msg)
            print("PICKUP_STATUS " + json.dumps(
                {"ok": False, "stage": "calibration", "message":
                 "the pickup hasn't been calibrated yet"}))
            return 1
        print(f"calibration: depth {calib[0] * 100:.1f}cm, "
              f"sideways {calib[1] * 100:+.1f}cm")

    ChannelFactoryInitialize(0, args.iface)

    loco_client = LocoClient()
    loco_client.SetTimeout(10.0)
    loco_client.Init()

    cam = Camera()
    print("Opening RealSense...")
    cam.open()

    if not args.no_center:
        print("\n=== lining up on the cup ===")
        try:
            ok, msg = center_on_glass(loco_client, cam, client, calib,
                                      dry_run=args.dry_run)
        except KeyboardInterrupt:
            loco_client.Move(0.0, 0.0, 0.0)
            cam.close()
            return 1
        finally:
            if not args.dry_run:
                try:
                    loco_client.Move(0.0, 0.0, 0.0)
                except Exception:
                    pass
        print(f"  {msg}")
        if not ok:
            print("PICKUP_STATUS " + json.dumps({"ok": False, "stage": "center",
                                                  "message": msg}))
            cam.close()
            return 1

    cam.close()   # free the RealSense before the grasp

    arms = ArmMotion(args.iface)
    hand_ctrl = BrainCoHand(args.hand)
    time.sleep(0.3)

    arms.start()
    try:
        print(f"\n=== grasping ({args.hand} hand) ===")
        play(arms, hand_ctrl, SEQUENCE, args.hand, tempo=args.tempo)
        print("PICKUP_STATUS " + json.dumps({"ok": True, "stage": "holding",
                                             "hand": args.hand}))
        print("\n=== holding - Ctrl-C or SIGTERM releases and resets ===")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nReleasing.")
    finally:
        print("=== resetting ===")
        hand_ctrl.open()
        arms.go({j: arms.home[j] for j in HAND_JOINTS[args.hand]},
               TEARDOWN_SECONDS)
        arms.stop()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
