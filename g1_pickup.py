#!/usr/bin/env python3
"""
G1 Pickup - scripted grasp, confirmed by looking
=================================================

Position the robot in front of the object first (by hand, by crane mode, or
by the visual servo), then run this.

PICKUP
    1. open    arms swing wide, clear of the object
    2. close   arms close to the recorded grasp pose
    3. lift    shoulders forward, taking the weight
    4. check   photograph the arms and confirm something is held
    5. hold    stay alive keeping the arms up

Step 5 matters. Ramping the arm_sdk weight down hands arm control back to
the robot, which drops whatever is being held no matter what pose the joints
were left in. So a successful pickup blocks, keeping the weight up, until
Ctrl-C or a --drop from another terminal. --release skips that and lets go.

The verification photo comes from the REALSENSE, not the Brio. The voice
daemon holds the Brio open for its whole run, and two processes fighting over
one camera is a problem worth avoiding. The RealSense image is tinted cyan by
the face LED beneath it, which makes it poor for describing a scene but
perfectly adequate for "is there a box between the arms".

DROP
    1. lower   shoulders back down
    2. release arms open
    3. back    step away
    4. home    return to rest

WHY THERE IS NO FORCE SENSING
-----------------------------
There was, and it did not work. A cardboard box shifts the shoulder-roll
torque by roughly 0.05 N.m; motor noise is about 0.035 N.m. The signal
barely clears the floor.

Four approaches were tried against that ratio:

  fixed torque threshold   fires instantly - the roll joints pull 5 N.m
                           holding the arms wide and 1 N.m closed, so
                           gravity load dwarfs contact load
  position lag             a soft box compresses, so lag builds too slowly
                           to distinguish from normal tracking error
  velocity deficit         better, but noisy at the low closing speed
  recorded torque profile  removes the gravity dependence properly, but
                           0.05 against 0.035 is still a poor ratio and
                           detection came out intermittent

None of them beat the signal-to-noise ratio, and no amount of filtering
creates information that is not in the measurement.

So the motion is simply scripted. That works because POSE_GRASP was recorded
WITH the box already in the arms - closing to exactly that pose is the right
width for that box by construction. A different box means recording a
different grasp pose, which takes a minute with g1_arm_control.py.

Whether anything was actually caught is then answered by a photograph, which
is a far better sensor for that question than a strain reading.

The force sensing is still in here behind --use-contact, for experimenting.

POSES
-----
Captured on the robot with g1_arm_control.py, not invented. Re-record them
for a different object and paste the numbers into POSE_OPEN / POSE_GRASP.

RUN
---
    python3 g1_pickup.py wlan0 --goto grasp    # check the arms reach it
    python3 g1_pickup.py wlan0 --verify-only   # test the camera check
    python3 g1_pickup.py wlan0 --pickup
    python3 g1_pickup.py wlan0 --drop
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import threading
import time

try:
    import cv2
    import numpy as np
    import anthropic
    VERIFY_AVAILABLE = True
except ImportError:
    VERIFY_AVAILABLE = False

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDef
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

try:
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    LOCO_AVAILABLE = True
except ImportError:
    LOCO_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
ARM_SDK_TOPIC  = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lf/lowstate"
NOT_USED_JOINT = 29

CONTROL_HZ = 100.0
ARM_KP, ARM_KD = 60.0, 1.5

L_PITCH, L_ROLL, L_YAW, L_ELBOW, L_WRIST = 15, 16, 17, 18, 19
R_PITCH, R_ROLL, R_YAW, R_ELBOW, R_WRIST = 22, 23, 24, 25, 26
ARM_JOINTS = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
ROLL_JOINTS = [L_ROLL, R_ROLL]

NAMES = {
    15: "L pitch", 16: "L roll", 17: "L yaw", 18: "L elbow", 19: "L wrist",
    22: "R pitch", 23: "R roll", 24: "R yaw", 25: "R elbow", 26: "R wrist",
}

ARM_LIMITS = {
    15: (-2.9, 2.7), 16: (-0.3, 2.9), 17: (-2.6, 2.6),
    18: (-0.2, 3.4), 19: (-2.6, 2.6),
    22: (-2.9, 2.7), 23: (-2.9, 0.3), 24: (-2.6, 2.6),
    25: (-0.2, 3.4), 26: (-2.6, 2.6),
}

# POSES - captured on the robot with g1_arm_control.py, not invented.
#
#   OPEN  : arms spread wide, clear of the object
#   GRASP : arms closed around a box, confirmed by placing one in them
#
# The pickup interpolates OPEN -> GRASP and stops early on contact. The
# object decides where it stops; GRASP is only the far end of the travel.
POSE_OPEN = {
    15: -0.748, 22: -0.731,     # pitch
    16: +0.983, 23: -0.983,     # roll  - wide
    17: -0.374, 24: +0.383,     # yaw
    18: +1.084, 25: +1.084,     # elbow
    19: -0.713, 26: +0.713,     # wrist
}
POSE_GRASP = {
    15: -0.100, 22: -0.084,
    16: +0.107, 23: -0.107,     # roll  - closed
    17: -0.205, 24: +0.213,
    18: +0.115, 25: +0.115,
    19: -0.413, 26: +0.413,
}

# LIFT - once gripped, raise the shoulders slightly to take the object's
# weight. Shoulder pitch only: bending the elbow would drag the box in
# against the chest, and holding it out in front is what we want.
#
# Relative to wherever contact happened, so a wide box lifts from a wider
# grip. Small on purpose - this is "pick it up", not "raise it overhead".
LIFT_PITCH_DELTA = -0.20     # negative raises the shoulders forward

# CONTACT DETECTION - position error, not torque.
#
# Torque was the obvious choice and it was wrong. Measured on the robot, the
# roll joints pull about 5 N.m simply holding the arms wide, and near 1 N.m
# when closed - so any fixed torque threshold either fires immediately at the
# open pose or never fires at all. Gravity load dominates contact load.
#
# Position error does not care. A joint that cannot reach where it was told
# to go falls behind, and that lag is what a blocked arm looks like.
# Measured normal lag was 0.043 rad holding and 0.091 rad at full extension,
# so the threshold sits above both.
# Cardboard is the hard case. It compresses, so the arm creeps into it and
# lag builds slowly rather than jumping - a threshold set high enough to
# ignore normal lag never fires on a soft box. Three detectors run together;
# any one firing counts as contact.
CONTACT_LAG = 0.09           # rad between commanded and measured

# VELOCITY DEFICIT
# The sweep moves the goal at a known rate. A free joint tracks it. A joint
# pressing on something falls below. This catches soft contact the moment
# resistance starts, rather than after lag has accumulated.
CONTACT_DQ_FRACTION = 0.35   # fraction of expected speed

# TORQUE JUMP
# Absolute torque is useless here - the roll joints read 5 N.m holding the
# arms wide and 1 N.m closed, so gravity dominates. But torque changes
# SMOOTHLY as the pose changes, and jumps when something is hit. Comparing
# against a lagged moving average of itself removes the pose dependence.
# Measured on the robot: pushing an arm outward with about the force a
# cardboard box applies moved torque by roughly 0.05 N.m. So the threshold
# has to be small - which means it will sometimes fire on noise.
#
# That is acceptable BECAUSE the grip is verified visually afterwards. A
# false stop costs one retry; a missed contact crushes the box. Bias toward
# stopping early and checking.
# TORQUE PROFILE
# Gravity load is a deterministic function of arm POSITION - the same pose
# pulls the same torque every time. So rather than guessing at trends, record
# what a free close actually looks like once, then compare against it.
#
# Trend extrapolation was tried first and is not sensitive enough: a
# cardboard box shifts torque by only ~0.05 N.m, and a threshold that small
# also fires on the ordinary variation of unobstructed motion. Comparing
# against a recorded profile removes the variation instead of trying to
# out-guess it.
TAU_PROFILE_FILE = os.path.expanduser("~/g1_torque_profile.json")
TAU_PROFILE_BINS = 40        # roll-angle buckets across the close
TAU_DEVIATION = 0.06         # N.m above the recorded profile
TAU_PROFILE_RUNS = 3         # averaged, to see through noise

CONTACT_SAMPLES = 3          # consecutive reads before believing it

# Torque is kept only as a secondary guard against something much heavier
# than a cardboard box. Set well above the ~5 N.m of normal free motion.
CONTACT_TAU_ABORT = 12.0

# VISUAL CONFIRMATION
# The contact threshold is deliberately sensitive, so a stop does not prove
# a grip. A photo does. If the check says nothing is held, the arms release
# and the pickup reports failure instead of walking off empty-handed.
VERIFY_MODEL = "claude-haiku-4-5-20251001"

# The RealSense, NOT the Brio. The voice daemon holds the Brio open for the
# whole of its run, so opening it here would be a fight over the device.
#
# The RealSense colour image is tinted cyan by the face LED sitting under it,
# which makes it poor for describing a scene - but "is there a box between
# the arms" survives a colour cast perfectly well.
VERIFY_WIDTH, VERIFY_HEIGHT, VERIFY_FPS = 640, 480, 30
VERIFY_WARMUP = 30           # a cold RealSense returns black for a moment

CLOSE_SECONDS = 6.0          # how long the full OPEN -> GRASP squeeze takes
SETTLE_AFTER_CONTACT = 0.06  # extra squeeze once contact is confirmed, rad

STEP_BACK_SPEED = -0.25
STEP_BACK_TIME  = 1.6


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ============================================================
# VISUAL CONFIRMATION
# ============================================================
VERIFY_PROMPT = """You are looking through a camera on a humanoid robot that \
has just tried to pick up a cardboard box by closing both arms around it.

Is the robot actually holding something between its arms right now?

Answer ONLY with JSON:
{"holding": <true|false>, "what": "<short description>", \
"confidence": <0-1>, "reason": "<one short sentence>"}

Set "holding" true ONLY if an object is clearly gripped between the robot's \
two arms. Empty arms closed on nothing, or an object sitting nearby but not \
held, are both false."""


def grab_frame(warmup=VERIFY_WARMUP):
    """One colour frame from the RealSense.

    Opened and closed per call. Holding the pipeline open would block the
    visual servo or anything else that wants depth, and an uncleanly closed
    pipeline leaves the USB device claimed.
    """
    if not REALSENSE_AVAILABLE:
        raise RuntimeError("pyrealsense2 not importable")

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, VERIFY_WIDTH, VERIFY_HEIGHT,
                      rs.format.bgr8, VERIFY_FPS)
    started = False
    try:
        pipeline.start(cfg)
        started = True
        frame = None
        for _ in range(warmup):
            frame = pipeline.wait_for_frames().get_color_frame()
        if not frame:
            raise RuntimeError("no colour frames from the RealSense")
        return np.asanyarray(frame.get_data()).copy()
    except RuntimeError as e:
        if "busy" in str(e).lower() or "resource" in str(e).lower():
            raise RuntimeError(
                "RealSense colour node is busy - videohub_pc4 holds it. "
                "Disable videohub in the tablet Service Manager.")
        raise
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass


def verify_grip(save=None):
    """Photograph the arms and ask whether anything is actually held.

    Returns (holding, detail). `holding` is None when the check could not
    run at all - treated as "assume held", since refusing to trust a grip we
    cannot inspect would make the whole sequence useless without a camera.
    """
    if not VERIFY_AVAILABLE:
        return None, "opencv, numpy or anthropic not installed"
    if not REALSENSE_AVAILABLE:
        return None, "pyrealsense2 not importable"
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None, "ANTHROPIC_API_KEY not set in this process"

    try:
        frame = grab_frame()
    except Exception as e:
        return None, f"camera: {e}"

    if save:
        cv2.imwrite(save, frame)

    h, w = frame.shape[:2]
    if w > 640:
        frame = cv2.resize(frame, (640, int(h * 640 / w)),
                           interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return None, "jpeg encode failed"

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        t0 = time.time()
        resp = client.messages.create(
            model=VERIFY_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": base64.b64encode(enc.tobytes()).decode()}},
                {"type": "text", "text": VERIFY_PROMPT}]}])
        text = resp.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        import json as _json
        data = _json.loads(text)
        dt = time.time() - t0
        return bool(data.get("holding")), (
            f"{data.get('what', '?')} "
            f"(confidence {data.get('confidence', 0):.2f}, "
            f"{data.get('reason', '')}) [{dt * 1000:.0f}ms]")
    except Exception as e:
        return None, f"llm: {e}"


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

        self.goal = {j: self._st["msg"].motor_state[j].q for j in ARM_JOINTS}
        self.home = dict(self.goal)
        self._profile = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---------- sensing ----------
    def q(self, j):
        return self._st["msg"].motor_state[j].q

    def dq(self, j):
        return self._st["msg"].motor_state[j].dq

    def tau(self, j):
        return self._st["msg"].motor_state[j].tau_est

    def lag(self, j):
        """How far behind its command a joint is."""
        return abs(self.goal[j] - self.q(j))

    def profile_expect(self, j, roll_angle):
        """Torque this joint pulled at this angle when nothing was in the way.

        Returns (mean, spread) or None if no profile has been recorded.
        """
        prof = self._profile.get(str(j)) if self._profile else None
        if not prof:
            return None
        lo, hi = prof["lo"], prof["hi"]
        if hi <= lo:
            return None
        frac = (roll_angle - lo) / (hi - lo)
        idx = int(clamp(frac, 0.0, 0.999) * len(prof["mean"]))
        return prof["mean"][idx], prof["std"][idx]

    def load_profile(self, path=TAU_PROFILE_FILE):
        try:
            with open(path) as f:
                self._profile = json.load(f)
            return True
        except Exception:
            self._profile = None
            return False

    def blocked(self, joints, expected_dq=None):
        """Any of three independent signs that a joint is being resisted.

        Returns (contact, joint, reason, detail).

        Three detectors rather than one because a cardboard box compresses -
        it resists gently and progressively, so a single threshold set high
        enough to ignore normal motion never fires on it.
        """
        for j in joints:
            l = self.lag(j)
            if l > CONTACT_LAG:
                return True, j, "lag", f"{l:.3f} rad behind"

        if expected_dq:
            for j in joints:
                want = abs(expected_dq)
                if want > 1e-4 and abs(self.dq(j)) < want * CONTACT_DQ_FRACTION:
                    return True, j, "slowed", (
                        f"{abs(self.dq(j)):.3f} of {want:.3f} rad/s")

        for j in joints:
            expect = self.profile_expect(j, self.q(j))
            if expect is None:
                continue
            mean, spread = expect
            # Allow for the noise seen during calibration, so a quiet joint
            # gets a tight threshold and a noisy one does not cry wolf.
            margin = max(TAU_DEVIATION, spread * 3.0)
            excess = abs(self.tau(j)) - mean
            if excess > margin:
                return True, j, "torque", (
                    f"{abs(self.tau(j)):.2f} vs {mean:.2f} expected "
                    f"(+{excess:.3f}, margin {margin:.3f})")

        return False, None, "", ""

    def overloaded(self, joints, tau_thr=CONTACT_TAU_ABORT):
        """Secondary guard - something far heavier than a cardboard box."""
        for j in joints:
            if abs(self.tau(j)) > tau_thr:
                return True, j
        return False, None

    # ---------- motion ----------
    def _publish(self):
        if self.dry_run:
            return
        for j in ARM_JOINTS:
            lo, hi = ARM_LIMITS[j]
            mc = self.msg.motor_cmd[j]
            mc.q = clamp(self.goal[j], lo, hi)
            mc.dq = 0.0
            mc.kp = ARM_KP
            mc.kd = ARM_KD
            mc.tau = 0.0
        self.msg.motor_cmd[NOT_USED_JOINT].q = self.weight
        self.msg.crc = self.crc.Crc(self.msg)
        self.pub.Write(self.msg)

    def _loop(self):
        while self.running:
            with self._lock:
                self._publish()
            time.sleep(self.dt)

    def start(self):
        if self.dry_run:
            print("(dry run - arm_sdk not engaged)")
            self.weight = 1.0
        else:
            print("Ramping arm_sdk weight up...")
            steps = int(2.0 * self.rate)
            for i in range(steps):
                self.weight = (i + 1) / steps
                self._publish()
                time.sleep(self.dt)
        self._thread.start()

    def sweep(self, target, seconds, watch=None, label="", debug=False):
        """Interpolate every joint from here to `target` over `seconds`,
        stopping early if a watched joint falls behind its command.

        Returns (reason, fraction_completed).
        """
        start = {j: self.goal[j] for j in ARM_JOINTS}
        steps = max(1, int(seconds / self.dt))
        hits = 0

        # How fast the watched joints are being asked to move. A free joint
        # tracks this; a resisted one falls below it.
        expected_dq = None
        if watch:
            travel = max(abs(target.get(j, start[j]) - start[j])
                         for j in watch)
            expected_dq = travel / max(seconds, 1e-6)
        self._profile = None

        # Ignore the first moments - the arms are still accelerating, so lag
        # and velocity are both misleading.
        settle_steps = int(0.4 / self.dt)

        for i in range(1, steps + 1):
            t = i / steps
            with self._lock:
                for j in ARM_JOINTS:
                    g = target.get(j, start[j])
                    self.goal[j] = clamp(start[j] + (g - start[j]) * t,
                                         *ARM_LIMITS[j])
            time.sleep(self.dt)

            if not watch:
                continue

            if i < settle_steps:
                continue

            over, j = self.overloaded(watch)
            if over:
                print(f"    ABORT: joint {j} at {self.tau(j):+.1f} N.m "
                      f"- far heavier than expected")
                return "overload", t

            stuck, j, why, detail = self.blocked(watch, expected_dq)

            if debug and i % 25 == 0:
                jl = L_ROLL
                exp = self.profile_expect(jl, self.q(jl))
                tail = ""
                if exp:
                    tail = (f"  tau {abs(self.tau(jl)):.2f} vs "
                            f"{exp[0]:.2f} expected "
                            f"({abs(self.tau(jl)) - exp[0]:+.3f})")
                print(f"      {t * 100:3.0f}%  lag {self.lag(jl):.3f}  "
                      f"dq {abs(self.dq(jl)):.3f}/{expected_dq:.3f}"
                      f"{tail}  {why if stuck else ''}")

            hits = hits + 1 if stuck else 0
            if hits >= CONTACT_SAMPLES:
                print(f"    contact [{why}]: joint {j}, {detail} "
                      f"({t * 100:.0f}% of travel)")
                return "contact", t

        return "complete", 1.0

    def nudge_pairs(self, deltas, seconds=1.5):
        """Shift joints by a relative amount, from wherever they are."""
        target = {j: self.goal[j] + d for j, d in deltas.items()}
        return self.sweep(target, seconds)

    def hold_forever(self):
        """Keep publishing the current pose until interrupted.

        Necessary after a successful pickup. Ramping the arm_sdk weight down
        hands arm control back to the robot, which drops whatever is being
        held regardless of what pose the joints were left in - so the process
        has to stay alive and keep the weight up.
        """
        print("\n  Holding the object. The arms will stay here.")
        print("  Ctrl-C to release, or run --drop from another terminal.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n  Released by Ctrl-C.")

    def stop(self, return_home=False):
        if return_home:
            print("Returning to rest...")
            self.sweep(self.home, 3.0)
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if not self.dry_run:
            print("Ramping weight down...")
            steps = int(1.0 * self.rate)
            w0 = self.weight
            for i in range(steps):
                self.weight = w0 * (1 - (i + 1) / steps)
                self._publish()
                time.sleep(self.dt)


# ============================================================
# SEQUENCES
# ============================================================
def calibrate(arms, close_seconds=CLOSE_SECONDS, runs=TAU_PROFILE_RUNS):
    """Record what a free close looks like, with NOTHING in the way.

    Sweeps open -> grasp several times, bucketing torque by roll angle.
    Contact is then any excursion above that recorded curve, which sidesteps
    the whole problem of separating gravity load from contact load.
    """
    print("\n=== CALIBRATE ===")
    print("  NOTHING must be between the arms. Clear the space.\n")

    lo = min(POSE_GRASP[j] for j in ROLL_JOINTS)
    hi = max(POSE_OPEN[j] for j in ROLL_JOINTS)
    samples = {j: [[] for _ in range(TAU_PROFILE_BINS)]
               for j in ROLL_JOINTS}

    for run in range(1, runs + 1):
        print(f"  run {run}/{runs}: opening")
        arms.sweep(POSE_OPEN, 3.0)
        print(f"           closing over {close_seconds:.0f}s")

        start = {j: arms.goal[j] for j in ARM_JOINTS}
        steps = int(close_seconds / arms.dt)
        for i in range(1, steps + 1):
            t = i / steps
            with arms._lock:
                for j in ARM_JOINTS:
                    g = POSE_GRASP.get(j, start[j])
                    arms.goal[j] = clamp(start[j] + (g - start[j]) * t,
                                         *ARM_LIMITS[j])
            time.sleep(arms.dt)

            if i < int(0.4 / arms.dt):
                continue          # ignore the initial acceleration
            for j in ROLL_JOINTS:
                frac = (arms.q(j) - lo) / (hi - lo) if hi > lo else 0.0
                idx = int(clamp(frac, 0.0, 0.999) * TAU_PROFILE_BINS)
                samples[j][idx].append(abs(arms.tau(j)))

    profile = {"lo": lo, "hi": hi}
    for j in ROLL_JOINTS:
        means, stds = [], []
        last_mean = 0.0
        for bucket in samples[j]:
            if bucket:
                m = sum(bucket) / len(bucket)
                var = sum((v - m) ** 2 for v in bucket) / len(bucket)
                last_mean = m
                means.append(m)
                stds.append(var ** 0.5)
            else:
                # Empty bucket - carry the neighbour forward rather than
                # leaving a hole the detector would read as zero expected.
                means.append(last_mean)
                stds.append(0.05)
        profile[str(j)] = {"lo": lo, "hi": hi, "mean": means, "std": stds}

    with open(TAU_PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n  wrote {TAU_PROFILE_FILE}")
    for j in ROLL_JOINTS:
        mm = profile[str(j)]["mean"]
        ss = profile[str(j)]["std"]
        print(f"  joint {j}: torque {min(mm):.2f} to {max(mm):.2f} N.m, "
              f"typical noise {sum(ss) / len(ss):.3f} N.m")
    print(f"\n  contact fires above the recorded curve by "
          f"{TAU_DEVIATION} N.m or 3x the noise, whichever is larger.")
    arms.sweep(POSE_OPEN, 3.0)


def report_errors(arms, tol=0.08):
    """Which joints did not get where they were told."""
    bad = [(j, arms.goal[j] - arms.q(j)) for j in ARM_JOINTS
           if abs(arms.goal[j] - arms.q(j)) > tol]
    if not bad:
        print("        all joints reached their commanded angles")
        return
    print("        joints short of their command:")
    for j, err in bad:
        print(f"          {NAMES[j]:<10} cmd {arms.goal[j]:+.3f}  "
              f"measured {arms.q(j):+.3f}  off by {err:+.3f}  "
              f"tau {arms.tau(j):+.2f}")


def pickup(arms, close_seconds=CLOSE_SECONDS, verify=True,
           save=None, debug=False, use_contact=False):
    """Scripted grasp, confirmed by looking.

    NO TORQUE SENSING by default, and deliberately so. A cardboard box shifts
    the roll torque by about 0.05 N.m while motor noise is around 0.035 N.m -
    the signal barely clears the floor, so any threshold either fires during
    free motion or misses the box. Lag, velocity deficit, trend
    extrapolation and a recorded torque profile were all tried; none of them
    beat that ratio reliably.

    The scripted close works because POSE_GRASP was recorded WITH the box in
    the arms. Closing to exactly that pose is already the right width for
    that box - the arms stop where they should by construction rather than
    by sensing. A different box means recording a different grasp pose,
    which is a minute of work with g1_arm_control.py.

    Whether anything was actually caught is then answered by a photograph,
    which is a far better sensor for that question than a strain reading.
    """
    print("\n=== PICKUP ===")

    print("  [1/4] opening to the wide pose")
    arms.sweep(POSE_OPEN, 3.0)
    print(f"        {fmt_pairs(arms)}")

    print(f"  [2/4] closing to the grasp pose over {close_seconds:.0f}s")
    watch = ROLL_JOINTS if use_contact else None
    why, t = arms.sweep(POSE_GRASP, close_seconds, watch=watch, debug=debug)
    if use_contact and why == "contact":
        print(f"        stopped early at {t * 100:.0f}% of travel")
    elif use_contact and why == "overload":
        print("        stopped - something is much heavier than a box")
        arms.sweep(POSE_OPEN, 2.5)
        return False
    print(f"        {fmt_pairs(arms)}")
    report_errors(arms)

    print("  [3/4] lifting - shoulders forward")
    arms.nudge_pairs({15: LIFT_PITCH_DELTA, 22: LIFT_PITCH_DELTA},
                     seconds=2.0)

    if not verify:
        print(f"\n  HOLDING (unverified) - {fmt_pairs(arms)}")
        return True

    print("  [4/4] checking whether anything is actually held")
    holding, detail = verify_grip(save)

    if holding is None:
        # Could not check at all. Do not discard a probably-good grip.
        print(f"        check unavailable - {detail}")
        print(f"\n  HOLDING (unverified) - {fmt_pairs(arms)}")
        return True

    if holding:
        print(f"        confirmed: {detail}")
        print(f"\n  HOLDING - {fmt_pairs(arms)}")
        return True

    print(f"        NOT HOLDING: {detail}")
    print("        Releasing and returning to the open pose.")
    arms.nudge_pairs({15: -LIFT_PITCH_DELTA, 22: -LIFT_PITCH_DELTA},
                     seconds=1.5)
    arms.sweep(POSE_OPEN, 2.5)
    return False


def drop(arms, loco=None, step_back=True):
    print("\n=== DROP ===")

    print("  [1/4] lowering the shoulders back down")
    arms.nudge_pairs({15: -LIFT_PITCH_DELTA, 22: -LIFT_PITCH_DELTA},
                     seconds=2.0)

    print("  [2/4] releasing")
    # Open the roll joints only - lifting the whole way back to POSE_OPEN
    # would also raise and extend, flinging the object.
    arms.nudge_pairs({16: POSE_OPEN[16] - arms.goal[16],
                      23: POSE_OPEN[23] - arms.goal[23]}, seconds=2.0)

    if step_back:
        if loco is not None:
            print("  [3/4] stepping back")
            loco.Move(STEP_BACK_SPEED, 0.0, 0.0)
            time.sleep(STEP_BACK_TIME)
            loco.Move(0.0, 0.0, 0.0)
        else:
            print("  [3/4] stepping back - SKIPPED (no loco client)")

    print("  [4/4] returning to rest")
    arms.sweep(arms.home, 3.0)
    print("\n  DONE")
    return True


def fmt_pairs(arms):
    return ("pitch {:+.3f}  roll {:+.3f}  yaw {:+.3f}  "
            "elbow {:+.3f}  wrist {:+.3f}").format(
        arms.goal[15], arms.goal[16], arms.goal[17],
        arms.goal[18], arms.goal[19])


def measure(arms, seconds=15.0):
    """Show all three contact signals live while the arms close.

    Runs an actual closing sweep so the numbers are the ones the detector
    sees. Put the box in the way partway through and watch which signal
    fires first.
    """
    print("\n=== MEASURE ===")
    print(f"  lag      > {CONTACT_LAG} rad")
    print(f"  slowed   < {CONTACT_DQ_FRACTION:.0%} of commanded speed")
    print(f"  torque   > {TAU_DEVIATION} N.m above the recorded "
          f"profile\n")

    print("Opening...")
    arms.sweep(POSE_OPEN, 3.0)

    travel = max(abs(POSE_GRASP[j] - POSE_OPEN[j]) for j in ROLL_JOINTS)
    expected = travel / seconds
    print(f"Closing over {seconds:.0f}s at {expected:.3f} rad/s.")
    print("Put the box in the way now.\n")
    print(f"{'t':>5} {'lag L':>7} {'lag R':>7} {'dq L':>7} {'dq R':>7} "
          f"{'tauL':>7} {'trend':>7} {'jump':>6}  fires")
    print("-" * 74)

    start = {j: arms.goal[j] for j in ARM_JOINTS}
    steps = int(seconds / arms.dt)
    last_print = 0.0
    t0 = time.time()

    try:
        for i in range(1, steps + 1):
            frac = i / steps
            with arms._lock:
                for j in ARM_JOINTS:
                    g = POSE_GRASP.get(j, start[j])
                    arms.goal[j] = clamp(start[j] + (g - start[j]) * frac,
                                         *ARM_LIMITS[j])
            time.sleep(arms.dt)

            now = time.time() - t0
            if now - last_print < 0.3:
                continue
            last_print = now

            lL, lR = arms.lag(L_ROLL), arms.lag(R_ROLL)
            dL, dR = abs(arms.dq(L_ROLL)), abs(arms.dq(R_ROLL))
            tL = abs(arms.tau(L_ROLL))
            exp = arms.profile_expect(L_ROLL, arms.q(L_ROLL))
            trend = exp[0] if exp else float("nan")
            stuck, j, why, detail = arms.blocked(ROLL_JOINTS, expected)
            print(f"{now:5.1f} {lL:>7.3f} {lR:>7.3f} {dL:>7.3f} {dR:>7.3f} "
                  f"{tL:>7.2f} {trend:>7.2f} {tL - trend:>+6.2f}  "
                  f"{why if stuck else ''}")
    except KeyboardInterrupt:
        pass

    print("\nWhichever column moved first when you blocked the arms is the")
    print("signal to trust. Tune it with --lag, --dq-frac or --tau-jump.")


# ============================================================
def main():
    global CONTACT_LAG, LIFT_PITCH_DELTA
    global CONTACT_DQ_FRACTION, TAU_DEVIATION
    ap = argparse.ArgumentParser(description="G1 scripted pickup")
    ap.add_argument("iface", nargs="?", default="wlan0")
    ap.add_argument("--pickup", action="store_true")
    ap.add_argument("--drop", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-step-back", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the visual grip check")
    ap.add_argument("--retries", type=int, default=1,
                    help="attempts before reporting failure")
    ap.add_argument("--save", default="/tmp/grip_check.jpg",
                    help="where to write the verification frame")
    ap.add_argument("--verify-only", action="store_true",
                    help="just run the grip check and exit")
    ap.add_argument("--debug", action="store_true",
                    help="print all three contact signals during the close")
    ap.add_argument("--calibrate", action="store_true",
                    help="record the free-motion torque profile. Run this "
                         "ONCE with nothing between the arms, before the "
                         "first pickup.")
    ap.add_argument("--runs", type=int, default=TAU_PROFILE_RUNS,
                    help="calibration sweeps to average")
    ap.add_argument("--deviation", type=float, default=TAU_DEVIATION,
                    help=f"N.m above the recorded profile that counts as "
                         f"contact (default {TAU_DEVIATION})")
    ap.add_argument("--release", action="store_true",
                    help="let go at the end of a pickup instead of holding. "
                         "Without this the process stays alive keeping the "
                         "arms up, because releasing arm_sdk drops the box.")
    ap.add_argument("--use-contact", action="store_true",
                    help="EXPERIMENTAL: stop early on torque/lag sensing. "
                         "Off by default - the signal from a cardboard box "
                         "is roughly the size of the motor noise.")
    ap.add_argument("--goto", choices=["open", "grasp", "home"],
                    help="move straight to a pose and report the error")
    ap.add_argument("--lag", type=float, default=CONTACT_LAG,
                    help=f"position-error threshold, rad "
                         f"(default {CONTACT_LAG})")
    ap.add_argument("--dq-frac", type=float, default=CONTACT_DQ_FRACTION,
                    help=f"contact if speed drops below this fraction of "
                         f"commanded (default {CONTACT_DQ_FRACTION})")

    ap.add_argument("--close-seconds", type=float, default=CLOSE_SECONDS,
                    help="how long the squeeze takes; slower is safer")
    ap.add_argument("--lift", type=float, default=LIFT_PITCH_DELTA,
                    help=f"shoulder pitch change when lifting, rad. More "
                         f"negative lifts higher (default "
                         f"{LIFT_PITCH_DELTA})")
    args = ap.parse_args()

    if args.verify_only:
        holding, detail = verify_grip(args.save)
        print(f"holding: {holding}")
        print(f"detail : {detail}")
        print(f"frame  : {args.save}")
        return 0

    if not (args.pickup or args.drop or args.measure or args.goto
            or args.calibrate):
        ap.print_help()
        return 1

    CONTACT_LAG = args.lag
    CONTACT_DQ_FRACTION = args.dq_frac
    TAU_DEVIATION = args.deviation
    LIFT_PITCH_DELTA = args.lift

    arms = Arms(args.iface, dry_run=args.dry_run)

    loco = None
    if args.drop and not args.no_step_back and LOCO_AVAILABLE \
            and not args.dry_run:
        try:
            loco = LocoClient()
            loco.SetTimeout(10.0)
            loco.Init()
        except Exception as e:
            print(f"loco client failed: {e}")

    if not arms.load_profile():
        if not args.calibrate:
            print(f"\nNo torque profile at {TAU_PROFILE_FILE}.")
            print("The torque detector is disabled without one - run:")
            print("  python3 g1_pickup.py wlan0 --calibrate")
            print("with nothing between the arms. Lag and velocity still "
                  "work.\n")
    else:
        print(f"Loaded torque profile from {TAU_PROFILE_FILE}")

    held = False
    arms.start()
    try:
        if args.calibrate:
            calibrate(arms, close_seconds=args.close_seconds,
                      runs=args.runs)
        elif args.goto:
            target = {"open": POSE_OPEN, "grasp": POSE_GRASP}.get(
                args.goto, arms.home)
            print(f"\nMoving to {args.goto}...")
            arms.sweep(target, 4.0)
            time.sleep(1.5)          # let the motors settle
            print(f"\n{'joint':<10} {'commanded':>10} {'measured':>10} "
                  f"{'error':>8} {'tau':>8}")
            print("-" * 50)
            for j in ARM_JOINTS:
                err = arms.goal[j] - arms.q(j)
                flag = "  <-- not reached" if abs(err) > 0.08 else ""
                print(f"{NAMES[j]:<10} {arms.goal[j]:>10.3f} "
                      f"{arms.q(j):>10.3f} {err:>+8.3f} "
                      f"{arms.tau(j):>8.2f}{flag}")
            print("\nErrors above ~0.08 rad mean the motor cannot hold that")
            print("angle - gravity load beyond what kp resists, or something")
            print("physically in the way. Around 0.02-0.05 is normal.")
        elif args.measure:
            measure(arms)
        elif args.pickup:
            ok = False
            for attempt in range(1, args.retries + 1):
                if attempt > 1:
                    print(f"\n--- retry {attempt} of {args.retries} ---")
                ok = pickup(arms, close_seconds=args.close_seconds,
                            verify=not args.no_verify, save=args.save,
                            debug=args.debug,
                            use_contact=args.use_contact)
                if ok:
                    break
            if not ok:
                print("\nFAILED: nothing picked up after "
                      f"{args.retries} attempt(s).")
                return 2

            # Something IS held, so do not let go. Ramping the weight down
            # here would drop it.
            if not args.release:
                held = True
                arms.hold_forever()
        elif args.drop:
            drop(arms, loco, step_back=not args.no_step_back)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Never return home while holding something - and note that the
        # weight ramp inside stop() releases the arms either way, which is
        # why a successful pickup blocks in hold_forever() first.
        arms.stop(return_home=(args.drop or args.measure) and not held)
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
