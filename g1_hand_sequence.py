#!/usr/bin/env python3
"""
G1 Hand Sequence - one arm, an ordered list of pose + grasp steps
====================================================================

Record poses with g1_arm_control_v2.py, paste them into SEQUENCE below, run.
Each step moves toward an arm pose and/or a finger shape, then holds for
`duration` before the next step - built for something like a one-handed cup
grasp, but the format is generic: any ordered arm+finger sequence on a
single hand.

WHY THIS IS SIMPLER THAN g1_gesturesynth.py OR g1_rps_v2.py
--------------------------------------------------------------
Those needed explicit carry-over bookkeeping (fingers) or two-arm collision
ordering. Neither applies here: one arm, one hand, moving through a list top
to bottom. Anything a step does not mention simply is not touched - arm_sdk
and the BrainCo hand both hold their last commanded position on their own,
so there is nothing to track between steps. Leave a field out and it stays
exactly where it was.

SEQUENCE FORMAT
----------------
A list of steps. Each step is a dict with either or both of:

    "arm":      {joint_id: value, ...}   - only the joints that should move
    "fingers":  [t, ta, i, m, r, p]       - full 6-value BrainCo pose
    "duration": seconds to move+hold before the next step
    "note":     your own label, ignored by the code

"arm" only needs the joints for the ACTIVE hand (see HAND below) - the other
five are ignored if present, so you can paste a full two-arm dict straight
out of g1_arm_control_v2.py's --export without editing it down by hand.

Example:

    SEQUENCE = [
        {"arm": {22: -0.90, 23: 0.10, 24: 0.00, 25: 0.60, 26: 0.00},
         "fingers": [0, 0, 0, 0, 0, 0], "duration": 1.2, "note": "approach"},
        {"fingers": [0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
         "duration": 1.0, "note": "fingertip grasp"},
        {"arm": {22: -1.30}, "duration": 1.0, "note": "lift"},
    ]

JOINTS
------
    left  arm: 15 pitch  16 roll  17 yaw  18 elbow  19 wrist
    right arm: 22 pitch  23 roll  24 yaw  25 elbow  26 wrist

FINGER ORDER (BrainCo, confirmed on hardware)
------------------------------------------------
    [thumb, thumb_aux, index, middle, ring, pinky] - 0.0 open, 1.0 closed

RUN
---
    python3 g1_hand_sequence.py --dry-run          # print the list, no hw
    python3 g1_hand_sequence.py wlan0
    python3 g1_hand_sequence.py wlan0 --hand left
    python3 g1_hand_sequence.py wlan0 --tempo 1.5  # 50% slower
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
# WHICH HAND
# ============================================================
HAND = "left"  # "left" or "right" - override with --hand

# ============================================================
# EDIT YOUR SEQUENCE HERE
# ============================================================

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
     "duration": 0.8, "note": "fingertip grasp - holds here until Ctrl-C"},
]
DEFAULT_DURATION = 1.0


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
    """Keep only the joints belonging to the active hand.

    Lets you paste a full two-arm dict straight from
    g1_arm_control_v2.py --export without trimming it by hand first -
    anything for the other arm is silently ignored.
    """
    allowed = set(HAND_JOINTS[hand])
    return {j: v for j, v in arm_dict.items() if j in allowed}


class ArmMotion:
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

        # Start from wherever the arms actually are. The INACTIVE arm's
        # joints are set here once and never touched again - they simply
        # hold at whatever this startup position was.
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
        """Move only the joints present in `target`. Anything not listed is
        never touched - it holds at its current commanded value, which is
        exactly the "everything else stays put" behaviour a sequence like
        this needs, with no extra bookkeeping required."""
        if not target:
            time.sleep(seconds)
            return
        joints = list(target.keys())
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


def validate(sequence, hand):
    """Catch mistakes before anything moves, not partway through."""
    for i, step in enumerate(sequence):
        if "fingers" in step and len(step["fingers"]) != 6:
            raise ValueError(f"step {i}: 'fingers' needs 6 values, got "
                             f"{len(step['fingers'])}")
        if "arm" in step:
            bad = set(step["arm"]) - set(ARM_JOINTS)
            if bad:
                raise ValueError(f"step {i}: unknown joint id(s) {bad}")
        if "duration" in step and step["duration"] < 0:
            raise ValueError(f"step {i}: duration cannot be negative")


def dry_run(sequence, hand):
    print(f"Hand: {hand}\n")
    total = 0.0
    for i, step in enumerate(sequence):
        duration = step.get("duration", DEFAULT_DURATION)
        total += duration
        arm = filter_to_hand(step.get("arm", {}), hand)
        parts = []
        if arm:
            parts.append("arm " + ", ".join(f"{j}:{v:+.3f}"
                                            for j, v in sorted(arm.items())))
        if "fingers" in step:
            parts.append(f"fingers {step['fingers']}")
        label = f"  ({step['note']})" if step.get("note") else ""
        print(f"  [{i + 1:>3}] {duration:.2f}s  " +
              ("; ".join(parts) if parts else "(hold only)") + label)
    print(f"\nTotal runtime: {total:.1f}s")


def play(arms, hand_ctrl, sequence, hand, tempo=1.0):
    for i, step in enumerate(sequence):
        duration = step.get("duration", DEFAULT_DURATION) * tempo
        arm = filter_to_hand(step.get("arm", {}), hand)
        label = f" ({step['note']})" if step.get("note") else ""
        print(f"[{i + 1}/{len(sequence)}] {duration:.2f}s{label}")

        if arm:
            print(f"    arm -> " +
                  ", ".join(f"{j}:{v:+.3f}" for j, v in sorted(arm.items())))
        if "fingers" in step:
            print(f"    fingers -> {step['fingers']}")
            hand_ctrl.set_all(step["fingers"])

        arms.go(arm, duration)


def main():
    ap = argparse.ArgumentParser(description="G1 single-arm hand sequence")
    ap.add_argument("iface", nargs="?")
    ap.add_argument("--hand", choices=["left", "right"], default=HAND)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tempo", type=float, default=1.0)
    args = ap.parse_args()

    try:
        validate(SEQUENCE, args.hand)
    except ValueError as e:
        print(f"SEQUENCE error: {e}")
        return 1

    if args.dry_run:
        dry_run(SEQUENCE, args.hand)
        return 0

    if not args.iface:
        print("ERROR: iface required unless --dry-run")
        return 1

    arms = ArmMotion(args.iface)
    hand_ctrl = BrainCoHand(args.hand)
    time.sleep(0.3)

    arms.start()
    try:
        print(f"\n=== playing ({args.hand} hand) ===")
        play(arms, hand_ctrl, SEQUENCE, args.hand, tempo=args.tempo)
        print("\n=== holding final pose - Ctrl-C to release and reset ===")
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


if __name__ == "__main__":
    main()
