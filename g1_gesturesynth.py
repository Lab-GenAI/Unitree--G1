#!/usr/bin/env python3
"""
G1 GestureSynth - play a tune as a list of finger/arm steps
==============================================================

Both arms go up beside the face once at the start and stay there. From then
on, everything is finger movement plus a small left-elbow tilt to switch
between major and minor. Edit TUNE at the top of this file to build a song -
nothing else needs to change.

CAPTURED POSES (g1_arm_control_v2.py)
--------------------------------------
pose_5 and pose_6 are IDENTICAL except for the left elbow and a small
compensating left wrist shift. Yaw did not change between them (0.548 in
both) - if you actually meant a yaw tilt rather than elbow, tell me and I'll
swap which joint MODE switches.

    RIGHT ARM - fixed, never moves once in position:
        pitch -1.189  roll -1.431  yaw -0.309  elbow +0.015  wrist -0.260

    LEFT ARM - pitch/roll/yaw fixed; elbow+wrist change with mode:
        pitch -1.168  roll +1.303  yaw +0.548
        major:  elbow -0.081  wrist +0.485   (pose_5)
        minor:  elbow +0.230  wrist +0.402   (pose_6)

THE TUNE FORMAT
----------------
TUNE is a plain list of steps. Each step is a dict. Only include what
CHANGES for that step - anything you leave out carries over from the
previous step, so a long tune does not need every hand repeated on every
line. The very first step should usually set both hands, since there is no
"previous" to carry over from yet.

    {"left": [t, ta, i, m, r, p],   # BrainCo finger order, each 0-1
     "right": [...],               # either, both, or neither may appear
     "mode": "major" | "minor",    # switches the left elbow/wrist
     "duration": 0.4,              # seconds to hold this step
     "note": "C4"}                 # your own label, ignored by the code -
                                    # purely so you can read your own tune

Example:

    TUNE = [
        {"left": [0,0,1,0,0,0], "right": [0,0,0,0,0,0],
         "mode": "major", "duration": 0.4, "note": "open"},
        {"right": [0,0,1,1,0,0], "duration": 0.4, "note": "two fingers"},
        {"mode": "minor", "duration": 0.3, "note": "switch to minor"},
        {"left": [1,1,1,1,1,1], "duration": 0.6, "note": "fist chord"},
    ]

FINGER ORDER (from g1_brainco_hand.py, confirmed on hardware)
-----------------------------------------------------------------
    [thumb, thumb_aux, index, middle, ring, pinky]
    0.0 = open/extended, 1.0 = closed/curled

RUN
---
    python3 g1_gesturesynth.py wlan0 --dry-run   # print the resolved
                                                  # sequence, no hardware
    python3 g1_gesturesynth.py wlan0
    python3 g1_gesturesynth.py wlan0 --tempo 1.5 # 50% slower - scales every
                                                  # step's duration
"""

import argparse
import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDef
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_brainco_hand import BrainCoHand


# ============================================================
# EDIT YOUR TUNE HERE
# ============================================================
# Only specify what CHANGES. Everything else carries over from the step
# before it. "left"/"right" default to fully open [0,0,0,0,0,0] before the
# first step that sets them. "mode" defaults to "major". "duration" defaults
# to DEFAULT_DURATION below if never set.
TUNE = [
    {"left": [1, 1, 0, 1, 1, 1], "right": [0, 0, 0, 1, 1, 1],
     "mode": "major", "duration": 2.0, "note": "example - replace me"},
    {"left": [1, 1, 0, 1, 1, 1], "right": [0, 0, 0, 0, 0, 1],
     "mode": "major", "duration": 2.0, "note": "example - replace me"},
    {"left": [1, 1, 0, 0, 1, 1], "right": [0, 0, 0, 1, 1, 1],
     "mode": "major", "duration": 2.0, "note": "example - replace me"},
     {"left": [1, 1, 0, 0, 0, 0], "right": [1, 1, 0, 1, 1, 1],
     "mode": "major", "duration": 1.0, "note": "example - replace me"},
    {"left": [1, 1, 0, 0, 0, 0], "right": [1, 1, 0, 1, 1, 1],
     "mode": "minor", "duration": 1.0, "note": "example - replace me"},
    {"left": [1, 1, 0, 1, 1, 1], "right": [0, 0, 0, 1, 1, 1],
     "mode": "major", "duration": 2.0, "note": "example - replace me"},
]

DEFAULT_DURATION = 0.4


# ============================================================
# CAPTURED POSES - from g1_arm_control_v2.py pose_5 / pose_6
# ============================================================
LEFT_FIXED = {15: -1.168, 16: +1.303, 17: +0.548}      # pitch, roll, yaw
RIGHT_ARM = {22: -1.189, 23: -1.431, 24: -0.309,        # never changes
             25: +0.015, 26: -0.260}

MODE_LEFT = {
    "major": {18: -0.081, 19: +0.485},   # pose_5 - elbow, wrist
    "minor": {18: +0.230, 19: +0.402},   # pose_6 - elbow, wrist
}

ARM_JOINTS = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
LEFT_JOINTS = [15, 16, 17, 18, 19]
LEFT_MODE_JOINTS = [18, 19]   # only these move on a mode switch

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

READY_SECONDS = 1.5      # moving both arms up to start
MODE_SWITCH_SECONDS = 0.25
TEARDOWN_SECONDS = 1.2

OPEN_FINGERS = [0, 0, 0, 0, 0, 0]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def full_pose(mode):
    """The complete ten-joint pose for a given mode."""
    pose = dict(LEFT_FIXED)
    pose.update(MODE_LEFT[mode])
    pose.update(RIGHT_ARM)
    return pose


class ArmMotion:
    """Minimal arm_sdk driver - ramp weight, interpolate to a target."""

    def __init__(self, iface, rate=CONTROL_HZ):
        self.rate = rate
        self.dt = 1.0 / rate
        self.weight = 0.0

        ChannelFactoryInitialize(0, iface)
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

    def start(self, seconds=1.0):
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

    def go(self, target, seconds, joints=None):
        joints = joints or list(target.keys())
        start = {j: self.pos[j] for j in joints}
        span = max(abs(target[j] - start[j]) for j in joints) or 1e-6
        steps = max(1, int(seconds * self.rate))
        for i in range(1, steps + 1):
            t = i / steps
            for j in joints:
                self.pos[j] = clamp(start[j] + (target[j] - start[j]) * t,
                                    *ARM_LIMITS[j])
            self._publish()
            time.sleep(self.dt)


def resolve_tune(tune, tempo=1.0):
    """Fill in the carry-over state so every step is fully explicit. Returns
    a list of resolved steps - used by both playback and --dry-run so they
    can never disagree about what a step actually means."""
    resolved = []
    left = list(OPEN_FINGERS)
    right = list(OPEN_FINGERS)
    mode = "major"

    for i, step in enumerate(tune):
        if "left" in step:
            if len(step["left"]) != 6:
                raise ValueError(f"step {i}: 'left' needs 6 values, "
                                 f"got {len(step['left'])}")
            left = list(step["left"])
        if "right" in step:
            if len(step["right"]) != 6:
                raise ValueError(f"step {i}: 'right' needs 6 values, "
                                 f"got {len(step['right'])}")
            right = list(step["right"])
        if "mode" in step:
            if step["mode"] not in MODE_LEFT:
                raise ValueError(f"step {i}: mode must be 'major' or "
                                 f"'minor', got {step['mode']!r}")
            mode = step["mode"]
        duration = step.get("duration", DEFAULT_DURATION) * tempo

        resolved.append({
            "left": list(left), "right": list(right), "mode": mode,
            "duration": duration, "note": step.get("note", ""),
        })
    return resolved


def play(arms, left_hand, right_hand, resolved):
    current_mode = None
    for i, step in enumerate(resolved):
        label = f" ({step['note']})" if step["note"] else ""
        print(f"[{i + 1}/{len(resolved)}] L={step['left']} "
              f"R={step['right']} mode={step['mode']} "
              f"{step['duration']:.2f}s{label}")

        if step["mode"] != current_mode:
            arms.go(full_pose(step["mode"]), MODE_SWITCH_SECONDS,
                    joints=LEFT_MODE_JOINTS)
            current_mode = step["mode"]

        left_hand.set_all(step["left"])
        right_hand.set_all(step["right"])
        time.sleep(step["duration"])


def dry_run(resolved):
    print(f"{len(resolved)} steps, resolved (nothing carries over silently "
          f"- this is exactly what will run):\n")
    total = 0.0
    for i, step in enumerate(resolved):
        label = f"  ({step['note']})" if step["note"] else ""
        print(f"  [{i + 1:>3}] L={step['left']}  R={step['right']}  "
              f"mode={step['mode']:<6}  {step['duration']:.2f}s{label}")
        total += step["duration"]
    print(f"\nTotal runtime: {total:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="G1 GestureSynth")
    ap.add_argument("iface", nargs="?")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved tune, touch no hardware")
    ap.add_argument("--tempo", type=float, default=1.0,
                    help="multiply every step's duration - 2.0 is half "
                         "speed, 0.5 is double speed")
    args = ap.parse_args()

    try:
        resolved = resolve_tune(TUNE, tempo=args.tempo)
    except ValueError as e:
        print(f"TUNE error: {e}")
        return 1

    if args.dry_run:
        dry_run(resolved)
        return 0

    if not args.iface:
        print("ERROR: iface required unless --dry-run")
        return 1

    arms = ArmMotion(args.iface)
    left_hand = BrainCoHand("left")
    right_hand = BrainCoHand("right")
    time.sleep(0.3)

    arms.start(READY_SECONDS)
    try:
        print("\n=== raising both hands ===")
        arms.go(full_pose("major"), READY_SECONDS)
        left_hand.set_all(OPEN_FINGERS)
        right_hand.set_all(OPEN_FINGERS)

        print("\n=== playing ===")
        play(arms, left_hand, right_hand, resolved)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        print("\n=== resetting ===")
        left_hand.open()
        right_hand.open()
        arms.go(arms.home, TEARDOWN_SECONDS)
        arms.stop()
        print("Done.")


if __name__ == "__main__":
    main()
