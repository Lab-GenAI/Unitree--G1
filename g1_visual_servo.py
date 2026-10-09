#!/usr/bin/env python3
"""
G1 Visual Servo Grasp - close the loop instead of computing it
===============================================================

Puts the hands around a box by watching both in the same camera frame and
driving the difference to zero.

WHY THIS AND NOT KINEMATICS
---------------------------
The open-loop route is: measure where the box is, compute joint angles, move
there. That needs hand-eye calibration, link lengths and joint accuracy all
to be right, and the errors compound. Every one of those is a number nobody
has measured on this robot.

Closed loop needs none of them. The camera sees the box at one pixel and the
hands at another; we move the hands toward the box, look again, correct,
repeat. Because the error is MEASURED rather than PREDICTED, whatever the
calibration is wrong by cancels out. The loop converges regardless.

This only works because the hands are visible from the head camera when the
arms are extended - confirmed on the robot.

WHY THE LLM DOES LOCALISATION
-----------------------------
RANSAC plane segmentation worked but was slow, and colour thresholding is
hopeless here - the RealSense sits under a cyan face LED that tints
everything, and anyone wearing red breaks it.

The LLM only has to be ROUGHLY right, which is what vision models are
actually good at. Given a rough box, depth INSIDE that box gives a precise
centroid - no scene-wide segmentation, no plane fitting, much faster.

THE LOOP
--------
    1. LLM: where is the box, and where are the hands?      (~1.3 s)
    2. Compute the pixel offset between them
    3. Move the arms a small step to reduce it
    4. Repeat until the offset is within tolerance
    5. LLM: "are the hands positioned to grasp this?"       - a judgement
    6. Close the hands

Slow and deliberate - a handful of seconds per grasp. That is the trade for
not needing anything calibrated.

SAFETY
------
  - --dry-run does everything except publish arm commands
  - every joint clamped to the crane-mode limits
  - a step limit, so a bad reading cannot drive the arms far
  - loses sight of the box, it stops rather than continuing blind

RUN
---
    python3 g1_visual_servo.py --look-only          # what does it see
    python3 g1_visual_servo.py wlan0 --dry-run      # full loop, no motion
    python3 g1_visual_servo.py wlan0                # for real
"""

import argparse
import base64
import json
import math
import os
import sys
import threading
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


# ============================================================
# CONFIG
# ============================================================
WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP = 30

VISION_MODEL = "claude-haiku-4-5-20251001"
VISION_MAX_TOKENS = 400

ARM_SDK_TOPIC  = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lf/lowstate"
NOT_USED_JOINT = 29
CONTROL_HZ = 50.0
ARM_KP, ARM_KD = 60.0, 1.5
MAX_VEL, ACCEL = 0.45, 1.40

L_PITCH, L_ROLL, L_YAW, L_ELBOW, L_WRIST = 15, 16, 17, 18, 19
R_PITCH, R_ROLL, R_YAW, R_ELBOW, R_WRIST = 22, 23, 24, 25, 26
ARM_JOINTS = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]

ARM_LIMITS = {
    15: (-2.9, 2.7), 16: (-0.3, 2.9), 17: (-2.6, 2.6),
    18: (-0.2, 3.4), 19: (-2.6, 2.6),
    22: (-2.9, 2.7), 23: (-2.9, 0.3), 24: (-2.6, 2.6),
    25: (-0.2, 3.4), 26: (-2.6, 2.6),
}

# Recorded with crane mode. The loop starts from "mid" and corrects from
# there, so these only need to be in the right neighbourhood.
ARM_POSES = {
    "low":  {15: -0.274, 16: +0.266, 17: -0.082, 18: +0.832, 19: -0.007,
             22: -0.267, 23: -0.263, 24: +0.086, 25: +0.815, 26: -0.001},
    "mid":  {15: -0.841, 16: +0.266, 17: -0.084, 18: +0.840, 19: -0.007,
             22: -0.823, 23: -0.263, 24: +0.091, 25: +0.818, 26: -0.001},
    "high": {15: -1.250, 16: +0.494, 17: -0.086, 18: +0.848, 19: -0.007,
             22: -1.239, 23: -0.490, 24: +0.092, 25: +0.818, 26: -0.001},
}

# SERVO GAINS
# How much joint movement per pixel of error. Deliberately small - the loop
# runs slowly, so it can afford several gentle corrections, and a large gain
# on a bad reading is how arms hit furniture.
PITCH_PER_PX_Y = 0.0016     # box above the hands -> raise (pitch negative)
ROLL_PER_PX_X  = 0.0010     # box left of the hands -> open arms that way
ELBOW_PER_M    = 0.55       # per metre of range error

MAX_STEP_RAD = 0.18         # hard cap on any single correction
TOL_PX       = 28           # within this, call it aligned
TOL_DEPTH_M  = 0.06
MAX_ITERS    = 8

# TRACKING
# An LLM call takes ~1.3 s. Asking it where the box is on every iteration
# caps the loop at under 1 Hz, which cannot follow anything that moves and
# burns an API call per correction.
#
# So: ask ONCE, then track. A CSRT tracker follows the box between frames in
# a few milliseconds with no network involved, and the LLM is re-consulted
# only when the tracker loses confidence or after RELOCATE_EVERY_S as a
# sanity check.
SERVO_HZ = 10.0
RELOCATE_EVERY_S = 8.0
# Measured on this machine at 640x480:
#   KCF    15 ms  -> 67 Hz
#   CSRT  114 ms  ->  9 Hz
#   MIL   178 ms  ->  6 Hz
# CSRT is steadier on hard targets but cannot hold 10 Hz once camera and
# depth work are added. KCF has the headroom; if it drifts on your box,
# --tracker CSRT and drop --hz to 5.
TRACKER_TYPE = "KCF"

# HAND PLAUSIBILITY
# The model will invent hands when told to expect them and none are in frame -
# it put a "hand" box on empty floor. Asking it more politely does not fix
# that, so every detection is checked against things we can measure
# independently.
#
# A real gripper is at arm's length from a head-mounted camera. The floor is
# much further away. A detection outside this range is provably not a hand.
HAND_DEPTH_MIN_M = 0.15
HAND_DEPTH_MAX_M = 0.95

# Grippers enter from below. Anything in the top of the frame is not a hand.
HAND_MIN_Y_FRAC = 0.35

# Arms must actually be raised for the hands to be in view at all. Shoulder
# pitch is NEGATIVE when raised forward, so a value above this means the arms
# are down and any "hand" is imaginary.
ARMS_RAISED_PITCH_MAX = -0.35


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# CAMERA
# ============================================================
class Camera:
    """Held open for the life of the run - reopening costs ~0.5 s of warmup
    and the loop needs a fresh frame every couple of seconds."""

    def __init__(self):
        self.pipe = None
        self.align = None
        self.scale = 1.0
        self.intr = None

    def open(self):
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        for _ in range(WARMUP):
            self.pipe.wait_for_frames()

    def close(self):
        if self.pipe:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def frame(self):
        f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return None, None
        if self.intr is None:
            self.intr = c.profile.as_video_stream_profile().intrinsics
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale
        return color, depth

    @staticmethod
    def depth_in_box(depth, box, percentile=35):
        """Range to the object inside a bounding box.

        The LLM's box includes background around the edges, so a plain median
        would be pulled toward whatever is behind. A low percentile favours
        the NEAR surface, which is the face we want to grasp.
        """
        x, y, w, h = box
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        patch = depth[y0:y1, x0:x1]
        vals = patch[patch > 0]
        if vals.size < 20:
            return 0.0
        return float(np.percentile(vals, percentile))


# ============================================================
# LLM
# ============================================================
LOCATE_PROMPT = """You are looking through a camera mounted in a humanoid \
robot's head. The image is {w} by {h} pixels.

Find three things:
1. The box or package the robot should pick up.
2. The robot's LEFT gripper - a metallic mechanical hand on a robotic arm. \
It appears on the LEFT side of the image.
3. The robot's RIGHT gripper - the same, on the RIGHT side.

Box each GRIPPER tightly - just the hand itself, NOT the arm behind it, and \
NOT the space between the two hands. Do not confuse the robot's metallic \
grippers with any human hands that may also be in the picture.

Reply with ONLY a JSON object, no other text:

{{"box": {{"x": <left>, "y": <top>, "w": <width>, "h": <height>, \
"confidence": <0-1>, "what": "<short description>"}},
 "left_hand": {{"x": <left>, "y": <top>, "w": <width>, "h": <height>}},
 "right_hand": {{"x": <left>, "y": <top>, "w": <width>, "h": <height>}}}}

IMPORTANT: the robot's arms are often DOWN and out of shot. That is the \
normal case, not an error. If you cannot clearly see a metallic robotic \
gripper, set that field to null. An empty floor is not a hand. Never place a \
box where a hand "should" be - only where one actually is.

Set any of the three to null if it is genuinely not visible. Pixel \
coordinates, origin at top-left."""

CHECK_PROMPT = """You are looking through a humanoid robot's head camera. \
The robot is trying to position its hands around a box in order to lift it.

Answer ONLY with JSON:

{"ready": <true|false>,
 "reason": "<one short sentence>",
 "adjust": "<none|left|right|up|down|closer|further>"}

"ready" is true only if the hands are positioned so that closing them would \
grip the box. If not, "adjust" says which way the hands should move to fix \
it, from the robot's point of view."""


class Eyes:
    def __init__(self, model=VISION_MODEL):
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in this process")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    @staticmethod
    def _encode(color, width=640):
        h, w = color.shape[:2]
        img = color
        if w > width:
            img = cv2.resize(color, (width, int(h * width / w)),
                             interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(enc.tobytes()).decode(), img.shape[1] / w

    def _ask(self, color, prompt):
        b64, scale = self._encode(color)
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model, max_tokens=VISION_MAX_TOKENS,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": b64}},
                {"type": "text", "text": prompt}]}])
        text = resp.content[0].text.strip()
        dt = time.time() - t0
        # Models sometimes wrap JSON in fences despite instructions.
        text = text.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(text), scale, dt
        except Exception:
            print(f"  [llm] unparseable: {text[:200]}")
            return None, scale, dt

    def locate(self, color, depth=None, cam=None):
        """Rough box and each gripper separately.

        Asking for ONE box around both hands put its centroid on the floor
        between them, which inflated the vertical error and made the depth
        reading a floor measurement rather than a hand measurement. Two
        boxes give the real grasp midpoint and two usable depth samples.
        """
        data, scale, dt = self._ask(
            color, LOCATE_PROMPT.format(w=WIDTH, h=HEIGHT))
        if data is None:
            return None, None, dt

        def rescale(b):
            if not b:
                return None
            return (int(b["x"] / scale), int(b["y"] / scale),
                    int(b["w"] / scale), int(b["h"] / scale))

        box = data.get("box")
        lh = rescale(data.get("left_hand"))
        rh = rescale(data.get("right_hand"))

        # Validate each gripper against depth before believing it.
        rejected = []
        if depth is not None and cam is not None:
            for name, rect in (("left", lh), ("right", rh)):
                if rect is None:
                    continue
                d = cam.depth_in_box(depth, rect)
                ok, why = hand_is_plausible(rect, d, name)
                if not ok:
                    rejected.append(f"{name}: {why}")
                    if name == "left":
                        lh = None
                    else:
                        rh = None

        hands = None
        if lh or rh:
            hands = {"left": lh, "right": rh,
                     "grasp_point": grasp_point(lh, rh)}

        if rejected:
            for r in rejected:
                print(f"  [rejected] {r}")

        return (
            {"rect": rescale(box),
             "confidence": (box or {}).get("confidence", 0.0),
             "what": (box or {}).get("what", "")} if box else None,
            hands, dt)

    def ready_to_grasp(self, color):
        data, _, dt = self._ask(color, CHECK_PROMPT)
        return data, dt


# ============================================================
# ARMS
# ============================================================
class Arms:
    def __init__(self, iface, dry_run=False, rate=CONTROL_HZ):
        self.dry_run = dry_run
        self.rate = rate
        self.dt = 1.0 / rate
        self.weight = 0.0
        self.running = True

        ChannelFactoryInitialize(0, iface)
        self._st = {"msg": None}
        sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        sub.Init(lambda m: self._st.__setitem__("msg", m), 1)

        print("Waiting for lowstate...")
        deadline = time.time() + 10.0
        while self._st["msg"] is None:
            if time.time() > deadline:
                raise RuntimeError("no lowstate - check the interface")
            time.sleep(0.05)

        self.crc = CRC()
        self.msg = LowCmdDef()
        self.msg.mode_machine = self._st["msg"].mode_machine
        self.pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self.pub.Init()

        self.pos = {j: self._st["msg"].motor_state[j].q for j in ARM_JOINTS}
        self.goal = dict(self.pos)
        self.vel = {j: 0.0 for j in ARM_JOINTS}
        self.home = dict(self.pos)

        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _publish(self):
        if self.dry_run:
            return
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

    def _loop(self):
        max_dv = ACCEL * self.dt
        while self.running:
            with self._lock:
                for j in ARM_JOINTS:
                    err = self.goal[j] - self.pos[j]
                    want = clamp(err / max(self.dt * 8.0, 1e-6),
                                 -MAX_VEL, MAX_VEL)
                    self.vel[j] += clamp(want - self.vel[j], -max_dv, max_dv)
                    lo, hi = ARM_LIMITS[j]
                    self.pos[j] = clamp(self.pos[j] + self.vel[j] * self.dt,
                                        lo, hi)
                self._publish()
            time.sleep(self.dt)

    def start(self):
        if self.dry_run:
            print("(dry run - arm_sdk not engaged)")
            self.weight = 1.0
        else:
            print("Ramping arm_sdk weight up...")
            for i in range(int(2.0 * self.rate)):
                self.weight = (i + 1) / (2.0 * self.rate)
                self._publish()
                time.sleep(self.dt)
        self._thread.start()

    def move_to(self, targets, settle=1.2):
        with self._lock:
            for j, v in targets.items():
                lo, hi = ARM_LIMITS[j]
                self.goal[j] = clamp(v, lo, hi)
        time.sleep(settle)

    def arms_are_raised(self):
        """Can the grippers be in the camera's view at all?

        Shoulder pitch is NEGATIVE when the arm is raised forward. If the
        arms are down, any hand the model reports is imaginary - so do not
        even consider the detection.
        """
        with self._lock:
            pitch = min(self.pos[L_PITCH], self.pos[R_PITCH])
        return pitch <= ARMS_RAISED_PITCH_MAX, pitch

    def nudge(self, deltas, settle=0.0):
        """Apply a bounded correction relative to the current goal.

        Non-blocking by default: the background thread moves the arms toward
        the new goal while perception keeps running. Sleeping here would peg
        the loop at 1 Hz again.
        """
        with self._lock:
            for j, d in deltas.items():
                d = clamp(d, -MAX_STEP_RAD, MAX_STEP_RAD)
                lo, hi = ARM_LIMITS[j]
                self.goal[j] = clamp(self.goal[j] + d, lo, hi)
        if settle:
            time.sleep(settle)

    def stop(self):
        print("\nReturning to start pose...")
        self.move_to(self.home, settle=2.0)
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self.dry_run:
            print("Ramping weight down...")
            w0 = self.weight
            for i in range(int(1.0 * self.rate)):
                self.weight = w0 * (1 - (i + 1) / (1.0 * self.rate))
                self._publish()
                time.sleep(self.dt)


# ============================================================
# SERVO
# ============================================================
def centre(rect):
    x, y, w, h = rect
    return x + w // 2, y + h // 2


def make_tracker(kind=None):
    kind = kind or TRACKER_TYPE
    """OpenCV moved trackers between namespaces across versions."""
    for factory in (getattr(cv2, f"Tracker{kind}_create", None),
                    getattr(getattr(cv2, "legacy", None),
                            f"Tracker{kind}_create", None)):
        if factory is not None:
            return factory()
    raise RuntimeError(f"no {kind} tracker in this OpenCV build")


class BoxTracker:
    """Follows a bounding box between LLM calls.

    Holds the last known rect and re-seeds whenever the LLM speaks. Reports
    when it has lost the target so the caller can ask the LLM again rather
    than servoing toward a stale position.
    """

    def __init__(self, relocate_every=RELOCATE_EVERY_S, kind=None):
        self.kind = kind
        self.tracker = None
        self.rect = None
        self.label = ""
        self.last_llm = 0.0
        self.relocate_every = relocate_every
        self.lost = True

    def seed(self, frame, rect, label=""):
        self.tracker = make_tracker(self.kind)
        self.tracker.init(frame, tuple(int(v) for v in rect))
        self.rect = tuple(int(v) for v in rect)
        self.label = label
        self.last_llm = time.time()
        self.lost = False

    def update(self, frame):
        if self.tracker is None:
            return None
        ok, rect = self.tracker.update(frame)
        if not ok:
            self.lost = True
            return None
        self.rect = tuple(int(v) for v in rect)
        return self.rect

    def needs_llm(self):
        """Re-ask when lost, never seeded, or the check interval has passed."""
        if self.lost or self.tracker is None:
            return True
        return time.time() - self.last_llm > self.relocate_every


def hand_is_plausible(rect, depth_m, label=""):
    """Reject hand detections that cannot physically be hands.

    Returns (ok, reason). Checked against measurements, not the model's own
    confidence - which was 0.95 on a leg.
    """
    if rect is None:
        return False, "not detected"
    x, y, w, h = rect
    cy = y + h // 2
    if cy < HEIGHT * HAND_MIN_Y_FRAC:
        return False, (f"too high in frame (y={cy}, grippers enter from "
                       f"below)")
    if depth_m <= 0:
        return False, "no depth reading there"
    if depth_m > HAND_DEPTH_MAX_M:
        return False, (f"{depth_m:.2f} m away - too far to be a gripper, "
                       f"probably floor or background")
    if depth_m < HAND_DEPTH_MIN_M:
        return False, f"{depth_m:.2f} m - too close to be real"
    return True, f"{depth_m:.2f} m"


def grasp_point(left, right):
    """Where the hands would close, in pixels.

    Midpoint of the two grippers when both are visible. With one visible, use
    it alone - better a single real hand than a midpoint guessed from one.
    """
    if left and right:
        lx, ly = centre(left)
        rx, ry = centre(right)
        return (lx + rx) // 2, (ly + ry) // 2
    if left:
        return centre(left)
    if right:
        return centre(right)
    return None


def hand_depth(cam, depth, hands):
    """Range to the grippers - the mean of whichever are visible.

    Sampling a single box spanning both hands measured the FLOOR between
    them, so the elbow correction was driven by a floor distance.
    """
    vals = []
    for key in ("left", "right"):
        rect = hands.get(key)
        if rect:
            d = cam.depth_in_box(depth, rect)
            if d > 0:
                vals.append(d)
    return float(np.mean(vals)) if vals else 0.0


def corrections(box_rect, grasp_px, box_depth, hands_depth):
    """Joint deltas that reduce the box-to-hands offset.

    Everything here is a DIFFERENCE between two things seen in the SAME
    frame. That is what makes calibration irrelevant - any error in where
    the camera thinks things are applies equally to both, and cancels.
    """
    bx, by = centre(box_rect)
    hx, hy = grasp_px
    dx, dy = bx - hx, by - hy          # pixels the grasp point must travel

    deltas = {}

    # Box above the hands (dy negative) -> raise the arms -> pitch negative.
    pitch = dy * PITCH_PER_PX_Y
    deltas[L_PITCH] = pitch
    deltas[R_PITCH] = pitch

    # Box left of the hands (dx negative) -> swing both arms left. Roll is
    # mirrored, so the same physical direction is opposite signs.
    roll = -dx * ROLL_PER_PX_X
    deltas[L_ROLL] = roll
    deltas[R_ROLL] = roll

    # Range: box further than the hands -> straighten the elbow.
    if box_depth > 0 and hands_depth > 0:
        dz = box_depth - hands_depth
        if abs(dz) > TOL_DEPTH_M:
            elbow = -dz * ELBOW_PER_M
            deltas[L_ELBOW] = elbow
            deltas[R_ELBOW] = elbow
    else:
        dz = 0.0

    return deltas, (dx, dy, dz)


def annotate(color, box, hands, note=""):
    img = color.copy()
    if box:
        x, y, w, h = box["rect"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img, centre(box["rect"]), 5, (0, 255, 0), -1)
        cv2.putText(img, f"box {box.get('confidence', 0):.2f}", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if hands:
        for key, colour in (("left", (255, 128, 0)),
                            ("right", (255, 200, 0))):
            rect = hands.get(key)
            if rect:
                x, y, w, h = rect
                cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(img, key, (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
        gp = hands.get("grasp_point")
        if gp:
            cv2.circle(img, gp, 6, (255, 255, 255), -1)
            cv2.circle(img, gp, 6, (0, 0, 0), 1)
            if hands.get("left") and hands.get("right"):
                cv2.line(img, centre(hands["left"]), centre(hands["right"]),
                         (255, 255, 255), 1)
    if box and hands and hands.get("grasp_point"):
        cv2.arrowedLine(img, hands["grasp_point"], centre(box["rect"]),
                        (0, 0, 255), 2, tipLength=0.15)
    if note:
        cv2.putText(img, note, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
    return img


def servo(cam, eyes, arms, save=None, timeout_s=60.0, hz=SERVO_HZ,
          verbose_every=5, tracker_kind=None):
    """Drive the grippers onto the box at ~hz, not at LLM speed.

    The LLM locates the box once and is re-consulted only when the tracker
    loses it or the check interval elapses. Everything in between is tracking
    plus depth, which costs milliseconds.

    Hands are re-detected on each LLM call rather than tracked: they only
    move when we command them, and a stale hand position is more dangerous
    than a stale box position.
    """
    if arms:
        print("\nMoving to the starting reach pose...")
        arms.move_to(ARM_POSES["mid"], settle=2.0)

    tracker = BoxTracker(kind=tracker_kind)
    hands = None
    period = 1.0 / hz
    deadline = time.time() + timeout_s
    n = 0
    llm_calls = 0
    last_correction = 0.0

    while time.time() < deadline:
        loop_start = time.time()
        n += 1

        color, depth = cam.frame()
        if color is None:
            continue

        # ---- relocate with the LLM only when needed ----
        if tracker.needs_llm():
            why = "lost" if tracker.lost else (
                "first look" if tracker.tracker is None else "periodic check")
            box_llm, hands_llm, dt = eyes.locate(color, depth, cam)
            llm_calls += 1
            print(f"\n  [llm {why}] {dt * 1000:.0f}ms  "
                  f"(call {llm_calls})")

            if box_llm is None:
                print("  no box in view - stopping rather than moving blind")
                return False
            tracker.seed(color, box_llm["rect"], box_llm.get("what", ""))
            print(f"  box {tracker.rect}  {tracker.label}")

            if arms:
                raised, pitch = arms.arms_are_raised()
                if not raised and hands_llm:
                    print(f"  [rejected] arms are down (pitch {pitch:+.2f})")
                    hands_llm = None
            hands = hands_llm
            if hands:
                seen = [k for k in ("left", "right") if hands.get(k)]
                print(f"  hands {', '.join(seen)}")

        box_rect = tracker.update(color)
        if box_rect is None:
            continue          # lost - next iteration asks the LLM

        if hands is None or not hands.get("grasp_point"):
            print("  grippers not visible - raise the arms into view")
            return False

        box_d = cam.depth_in_box(depth, box_rect)
        hand_d = hand_depth(cam, depth, hands)
        deltas, (dx, dy, dz) = corrections(box_rect, hands["grasp_point"],
                                           box_d, hand_d)

        aligned = (abs(dx) < TOL_PX and abs(dy) < TOL_PX
                   and abs(dz) < TOL_DEPTH_M)

        if n % verbose_every == 0 or aligned:
            rate = n / max(time.time() - (deadline - timeout_s), 1e-6)
            print(f"  [{n:>4}] dx {dx:+5d} dy {dy:+5d} dz {dz:+.3f}  "
                  f"box {box_d:.2f} hands {hand_d:.2f}  "
                  f"{rate:.1f} Hz  llm x{llm_calls}")

        if save and n % verbose_every == 0:
            cv2.imwrite(save, annotate(
                color, {"rect": box_rect, "confidence": 1.0,
                        "what": tracker.label},
                hands, f"dx{dx:+d} dy{dy:+d} dz{dz:+.2f}"))

        if aligned:
            print("\n  ALIGNED")
            if save:
                cv2.imwrite(save, annotate(
                    color, {"rect": box_rect, "confidence": 1.0,
                            "what": tracker.label}, hands, "ALIGNED"))
            return True

        # Apply corrections at a lower rate than perception. The arms take
        # time to move, and re-correcting before they have arrived just
        # makes the loop oscillate.
        if arms and time.time() - last_correction > 0.5:
            arms.nudge(deltas, settle=0.0)
            last_correction = time.time()

        elapsed = time.time() - loop_start
        if elapsed < period:
            time.sleep(period - elapsed)

    print(f"\nTimed out after {timeout_s:.0f}s ({llm_calls} llm calls).")
    return False


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="G1 visual servo grasp")
    ap.add_argument("iface", nargs="?", default="wlan0")
    ap.add_argument("--dry-run", action="store_true",
                    help="run the full loop but publish nothing")
    ap.add_argument("--look-only", action="store_true",
                    help="one LLM look, no DDS, no motion")
    ap.add_argument("--save", default="/tmp/servo.jpg")
    ap.add_argument("--timeout", type=float, default=60.0,
                    help="give up after this long")
    ap.add_argument("--hz", type=float, default=SERVO_HZ,
                    help="perception/correction rate")
    ap.add_argument("--relocate-every", type=float, default=RELOCATE_EVERY_S,
                    help="seconds between LLM sanity checks")
    ap.add_argument("--tracker", choices=["KCF", "CSRT", "MIL"],
                    default=TRACKER_TYPE,
                    help="KCF ~15ms, CSRT ~114ms but steadier")
    ap.add_argument("--model", default=VISION_MODEL)
    ap.add_argument("--check", action="store_true",
                    help="ask the LLM to confirm the grasp when aligned")
    args = ap.parse_args()

    cam = Camera()
    print("Opening RealSense...")
    cam.open()
    print("ok")

    try:
        eyes = Eyes(args.model)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        cam.close()
        return 1

    if args.look_only:
        color, depth = cam.frame()
        box, hands, dt = eyes.locate(color, depth, cam)
        print(f"llm {dt * 1000:.0f}ms")
        print(f"box   {box}")
        print(f"hands {hands}")
        if box:
            print(f"box depth   {cam.depth_in_box(depth, box['rect']):.3f} m")
        if hands:
            for key in ("left", "right"):
                if hands.get(key):
                    print(f"{key:<6} depth "
                          f"{cam.depth_in_box(depth, hands[key]):.3f} m")
            print(f"grasp point {hands['grasp_point']}  "
                  f"depth {hand_depth(cam, depth, hands):.3f} m")
        cv2.imwrite(args.save, annotate(color, box, hands, "look-only"))
        print(f"wrote {args.save}")
        cam.close()
        return 0

    arms = Arms(args.iface, dry_run=args.dry_run)
    arms.start()

    try:
        ok = servo(cam, eyes, arms, save=args.save,
                   timeout_s=args.timeout, hz=args.hz,
                   tracker_kind=args.tracker)

        if ok and args.check:
            print("\nAsking whether this is grippable...")
            color, _ = cam.frame()
            verdict, dt = eyes.ready_to_grasp(color)
            print(f"  llm {dt * 1000:.0f}ms  {verdict}")
            if verdict and verdict.get("ready"):
                print("\n  Hands are in position.")
                print("  Closing them needs the BrainCo message type -")
                print("  grep -rn 'brainco' ~/brainco_hand_service/")
            else:
                print("  Not ready. Suggested adjustment: "
                      f"{(verdict or {}).get('adjust', '?')}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        arms.stop()
        cam.close()
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
