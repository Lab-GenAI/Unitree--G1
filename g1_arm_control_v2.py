#!/usr/bin/env python3
"""
G1 Arm Control v2 - INDEPENDENT left/right, not mirrored
==========================================================

The original g1_arm_control.py always moved both arms together as a
mirrored pair - fine for symmetric gestures, useless for something like RPS's
countdown, where one arm holds flat and still while the other pumps up and
down on top of it. This version selects ONE arm at a time and moves only
that arm's joints; the other stays exactly where it was.

CONTROLLER
----------
    L1              select LEFT arm as active   (tap alone, not with Select)
    R1              select RIGHT arm as active
    Left / Right    cycle which joint of the ACTIVE arm is selected
    Up / Down       move the selected joint
    L2 / R2 (hold)  fine adjustment
    A               record the CURRENT FULL POSE - both arms, as they
                    actually are right now, not just the active one
    B               show every joint, both arms, commanded vs measured
    Y               return BOTH arms to the pose they started in
    L1 + Select     exit

KEYBOARD
--------
    left                    make the left arm active
    right                   make the right arm active
    pitch -0.85             set that joint on the ACTIVE arm only
    roll / yaw / elbow / wrist    likewise
    15 -0.85                set one joint by number directly, either arm
    all | compact | home | poses | export
    save <name> | go <name>        FULL pose - both arms together
    q

JOINTS
------
    pitch   15 (L) / 22 (R)   negative reaches/raises FORWARD
    roll    16 (L) / 23 (R)   positive(L)/negative(R) = arm OUTWARD
    yaw     17 (L) / 24 (R)
    elbow   18 (L) / 25 (R)
    wrist   19 (L) / 26 (R)

Recorded poses are the FULL ten-joint state, so a "ready" pose captures both
arms in their correct asymmetric positions at once - one flat and low, one
raised above it - not a mirrored pair.

WORKFLOW FOR THE RPS CHOREOGRAPHY
----------------------------------
    1. L1 to select the left arm. Move it flat, palm up, held low. Do not
       record yet - the right arm is still wherever it started, which would
       be wrong in the snapshot.
    2. R1 to select the right arm. Move it above the left hand.
    3. A - this records BOTH arms as they now stand. Name it "ready".
    4. Adjust the right arm slightly lower (the "down" part of the pump),
       leaving the left arm untouched. A again, name it "pump_low".
    5. Raise the right arm back. A again, name it "pump_high".
    6. Move the right arm into whichever throw shape, A, name it
       "throw_rock" / "throw_paper" / "throw_scissors".

That gives g1_rps.py six saved poses (ready, pump_low, pump_high, and three
throws) to sequence through, all captured on the real robot rather than
invented.

RUN
---
    python3 g1_arm_control_v2.py wlan0
    python3 g1_arm_control_v2.py wlan0 --keyboard
"""

import argparse
import json
import os
import struct
import sys
import threading
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDef
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


ARM_SDK_TOPIC  = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lf/lowstate"
NOT_USED_JOINT = 29
POSE_FILE = os.path.expanduser("~/g1_arm_poses_v2.json")

CONTROL_HZ = 100.0
ARM_KP, ARM_KD = 60.0, 1.5

MAX_VEL  = 0.60
FINE_VEL = 0.12
ACCEL    = 2.00

ARM_JOINTS = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
LEFT_JOINTS  = [15, 16, 17, 18, 19]
RIGHT_JOINTS = [22, 23, 24, 25, 26]

ARM_LIMITS = {
    15: (-2.9, 2.7), 16: (-0.3, 2.9), 17: (-2.6, 2.6),
    18: (-0.2, 3.4), 19: (-2.6, 2.6),
    22: (-2.9, 2.7), 23: (-2.9, 0.3), 24: (-2.6, 2.6),
    25: (-0.2, 3.4), 26: (-2.6, 2.6),
}

NAMES = {
    15: "L pitch", 16: "L roll", 17: "L yaw", 18: "L elbow", 19: "L wrist",
    22: "R pitch", 23: "R roll", 24: "R yaw", 25: "R elbow", 26: "R wrist",
}

# (name, left_joint, right_joint, hint) - no mirroring flag needed any more,
# each side is addressed independently.
JOINT_DEFS = [
    ("pitch", 15, 22, "negative = reach/raise FORWARD"),
    ("roll",  16, 23, "positive(L) / negative(R) = OUTWARD"),
    ("yaw",   17, 24, "rotates the upper arm"),
    ("elbow", 18, 25, "bends the forearm"),
    ("wrist", 19, 26, "rotates the hand"),
]
JOINT_BY_NAME = {d[0]: d for d in JOINT_DEFS}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Remote:
    def __init__(self):
        self.Lx = self.Rx = self.Ry = self.Ly = 0.0
        self.L1 = self.L2 = self.R1 = self.R2 = 0
        self.A = self.B = self.X = self.Y = 0
        self.Up = self.Down = self.Left = self.Right = 0
        self.Select = self.F1 = self.F3 = self.Start = 0

    def parse(self, data):
        self.Lx = struct.unpack('<f', data[4:8])[0]
        self.Rx = struct.unpack('<f', data[8:12])[0]
        self.Ry = struct.unpack('<f', data[12:16])[0]
        self.Ly = struct.unpack('<f', data[20:24])[0]
        d1, d2 = data[2], data[3]
        self.R1 = (d1 >> 0) & 1
        self.L1 = (d1 >> 1) & 1
        self.Start = (d1 >> 2) & 1
        self.Select = (d1 >> 3) & 1
        self.R2 = (d1 >> 4) & 1
        self.L2 = (d1 >> 5) & 1
        self.F1 = (d1 >> 6) & 1
        self.F3 = (d1 >> 7) & 1
        self.A = (d2 >> 0) & 1
        self.B = (d2 >> 1) & 1
        self.X = (d2 >> 2) & 1
        self.Y = (d2 >> 3) & 1
        self.Up = (d2 >> 4) & 1
        self.Right = (d2 >> 5) & 1
        self.Down = (d2 >> 6) & 1
        self.Left = (d2 >> 7) & 1


class ArmControl:
    def __init__(self, iface, rate=CONTROL_HZ):
        self.rate = rate
        self.dt = 1.0 / rate
        self.weight = 0.0
        self.running = True
        self.active_side = "left"      # which arm Up/Down currently moves
        self.selected_joint = 0        # index into JOINT_DEFS

        ChannelFactoryInitialize(0, iface)
        self._st = {"msg": None}
        self.remote = Remote()

        def on_state(m):
            self._st["msg"] = m
            try:
                self.remote.parse(m.wireless_remote)
            except Exception:
                pass

        sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
        sub.Init(on_state, 1)

        print("Waiting for lowstate...")
        deadline = time.time() + 10.0
        while self._st["msg"] is None:
            if time.time() > deadline:
                raise RuntimeError("no lowstate - check the interface name")
            time.sleep(0.05)
        print("ok")

        self.crc = CRC()
        self.msg = LowCmdDef()
        self.msg.mode_machine = self._st["msg"].mode_machine
        self.pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
        self.pub.Init()

        # Start from where the arms actually are - nothing jumps on entry.
        self.pos = {j: self._st["msg"].motor_state[j].q for j in ARM_JOINTS}
        self.vel = {j: 0.0 for j in ARM_JOINTS}
        self.home = dict(self.pos)
        self.poses = self._load()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    # ---------- state ----------
    def measured(self, j):
        m = self._st["msg"]
        return m.motor_state[j].q if m else float("nan")

    def active_joint(self):
        """(name, joint_id) for the currently selected joint on the active
        arm - e.g. active_side='right', selected='elbow' -> ('elbow', 25)."""
        name, lj, rj, _ = JOINT_DEFS[self.selected_joint]
        return name, (lj if self.active_side == "left" else rj)

    # ---------- publishing ----------
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

    def _loop(self):
        max_dv = ACCEL * self.dt
        prev = {"sel": 0, "A": 0, "B": 0, "Y": 0, "L1": 0, "R1": 0}
        while self.running:
            r = self.remote

            if r.L1 and r.Select:
                self.running = False
                break

            # L1 / R1 ALONE (not combined with Select) pick the active arm.
            if r.L1 and not r.Select and not prev["L1"]:
                self.active_side = "left"
                print(f"\n  >> ACTIVE ARM: LEFT")
                self.show_selected()
            prev["L1"] = r.L1 and not r.Select

            if r.R1 and not prev["R1"]:
                self.active_side = "right"
                print(f"\n  >> ACTIVE ARM: RIGHT")
                self.show_selected()
            prev["R1"] = r.R1

            sel = (1 if r.Right else 0) - (1 if r.Left else 0)
            if sel and not prev["sel"]:
                self.selected_joint = (self.selected_joint + sel) % len(
                    JOINT_DEFS)
                self.show_selected()
            prev["sel"] = sel

            if r.A and not prev["A"]:
                self.record()
            prev["A"] = r.A
            if r.B and not prev["B"]:
                self.show_all()
            prev["B"] = r.B
            if r.Y and not prev["Y"]:
                print("\n  returning BOTH arms to the starting pose")
                self.go(self.home)
            prev["Y"] = r.Y

            # ---- move the active arm's selected joint ----
            direction = (1 if r.Up else 0) - (1 if r.Down else 0)
            speed = FINE_VEL if (r.L2 or r.R2) else MAX_VEL
            _, j = self.active_joint()

            want = {jj: 0.0 for jj in ARM_JOINTS}
            if direction:
                want[j] = direction * speed

            with self._lock:
                for jj in ARM_JOINTS:
                    dv = clamp(want[jj] - self.vel[jj], -max_dv, max_dv)
                    self.vel[jj] += dv
                    if abs(self.vel[jj]) > 1e-6:
                        lo, hi = ARM_LIMITS[jj]
                        new = clamp(self.pos[jj] + self.vel[jj] * self.dt,
                                    lo, hi)
                        if new in (lo, hi) and new != self.pos[jj]:
                            self.vel[jj] = 0.0
                        self.pos[jj] = new
                self._publish()
            time.sleep(self.dt)

    def start(self):
        print("Ramping arm_sdk weight up...")
        steps = int(2.0 * self.rate)
        for i in range(steps):
            self.weight = (i + 1) / steps
            self._publish()
            time.sleep(self.dt)
        self._thread.start()

    def stop(self):
        self.running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("Ramping weight down...")
        steps = int(1.0 * self.rate)
        w0 = self.weight
        for i in range(steps):
            self.weight = w0 * (1 - (i + 1) / steps)
            self._publish()
            time.sleep(self.dt)

    # ---------- commands (used by both controller and keyboard) ----------
    def set_active_joint(self, value):
        """Set the currently selected joint on the currently active arm."""
        _, j = self.active_joint()
        with self._lock:
            self.pos[j] = clamp(value, *ARM_LIMITS[j])
        return self.pos[j]

    def set_named(self, name, value):
        """Set a joint pair's name (pitch/roll/...) on the ACTIVE arm only -
        NOT mirrored, unlike the original tool."""
        d = JOINT_BY_NAME.get(name)
        if not d:
            return None
        _, lj, rj, _ = d
        j = lj if self.active_side == "left" else rj
        with self._lock:
            self.pos[j] = clamp(value, *ARM_LIMITS[j])
        return j, self.pos[j]

    def set_joint(self, j, value):
        with self._lock:
            self.pos[j] = clamp(value, *ARM_LIMITS[j])
        return self.pos[j]

    def go(self, target, speed=0.5):
        start = {j: self.pos[j] for j in ARM_JOINTS}
        span = max(abs(target.get(j, start[j]) - start[j])
                   for j in ARM_JOINTS) or 1e-6
        steps = max(1, int(span / (speed * self.dt)))
        for i in range(1, steps + 1):
            t = i / steps
            with self._lock:
                for j in ARM_JOINTS:
                    g = target.get(j, start[j])
                    self.pos[j] = clamp(start[j] + (g - start[j]) * t,
                                        *ARM_LIMITS[j])
            time.sleep(self.dt)

    def show_selected(self):
        name, j = self.active_joint()
        print(f"     active: {self.active_side.upper()} arm, {name} "
              f"(joint {j}) = {self.pos[j]:+.3f}")

    def show_all(self):
        print(f"\n  {'joint':<10} {'cmd':>8} {'measured':>9} "
              f"{'limits':>16}")
        print("  " + "-" * 46)
        active_j = self.active_joint()[1]
        for j in ARM_JOINTS:
            lo, hi = ARM_LIMITS[j]
            marker = " <-- selected" if j == active_j else ""
            print(f"  {NAMES[j]:<10} {self.pos[j]:>8.3f} "
                  f"{self.measured(j):>9.3f} "
                  f"{f'[{lo:+.1f},{hi:+.1f}]':>16}{marker}")
        print()

    def compact(self):
        left = "  ".join(f"{d[0]} {self.pos[d[1]]:+.3f}" for d in JOINT_DEFS)
        right = "  ".join(f"{d[0]} {self.pos[d[2]]:+.3f}" for d in JOINT_DEFS)
        return f"L: {left}\n  R: {right}"

    # ---------- poses: FULL state, both arms, not mirrored ----------
    def _load(self):
        try:
            with open(POSE_FILE) as f:
                return {k: {int(a): b for a, b in v.items()}
                        for k, v in json.load(f).items()}
        except Exception:
            return {}

    def record(self, name=None):
        name = name or f"pose_{len(self.poses) + 1}"
        self.poses[name] = {j: round(self.pos[j], 4) for j in ARM_JOINTS}
        with open(POSE_FILE, "w") as f:
            json.dump({k: {str(a): b for a, b in v.items()}
                       for k, v in self.poses.items()}, f, indent=2)
        print(f"\n  [RECORDED] {name}  (both arms, full pose)")
        print(f"  {self.compact()}")
        print(f"  -> {POSE_FILE}")
        return name

    def export(self):
        if not self.poses:
            print("no poses saved")
            return
        print("\nARM_POSES = {")
        for n, p in self.poses.items():
            inner = ", ".join(f"{j}: {v:+.3f}" for j, v in sorted(p.items()))
            print(f'    "{n}": {{{inner}}},')
        print("}\n")


def keyboard_loop(ac):
    print("\nType commands. 'help' for the list, 'q' to exit.\n")
    while ac.running:
        try:
            line = input(f"[{ac.active_side}] arm> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            ac.running = False
            break
        if cmd == "help":
            print("  left | right                  make that arm active")
            print("  pitch/roll/yaw/elbow/wrist <v>  set it on the ACTIVE "
                  "arm only")
            print("  15 <v>                        set one joint by number")
            print("  all | compact | home | poses | export")
            print("  save <name> | go <name>       FULL pose, both arms")
            continue
        if cmd in ("left", "right"):
            ac.active_side = cmd
            print(f"  active arm: {cmd.upper()}")
            continue
        if cmd == "all":
            ac.show_all()
            continue
        if cmd == "compact":
            print("  " + ac.compact())
            continue
        if cmd == "home":
            ac.go(ac.home)
            print("  both arms at starting pose")
            continue
        if cmd == "poses":
            print("  " + (" ".join(ac.poses) or "(none)"))
            continue
        if cmd == "export":
            ac.export()
            continue
        if cmd == "save":
            ac.record(parts[1] if len(parts) > 1 else None)
            continue
        if cmd == "go" and len(parts) > 1:
            p = ac.poses.get(parts[1])
            if p is None:
                print(f"  no pose '{parts[1]}'")
            else:
                ac.go(p)
                print(f"  at '{parts[1]}'")
            continue

        if len(parts) < 2:
            print("  need a value, e.g. 'pitch -0.85'")
            continue
        try:
            val = float(parts[1])
        except ValueError:
            print(f"  '{parts[1]}' is not a number")
            continue

        if cmd in JOINT_BY_NAME:
            res = ac.set_named(cmd, val)
            if res:
                j, v = res
                print(f"  {ac.active_side} {cmd} (joint {j}) -> {v:+.3f}")
        elif cmd.isdigit() and int(cmd) in ARM_LIMITS:
            j = int(cmd)
            print(f"  {NAMES[j]} -> {ac.set_joint(j, val):+.3f}")
        else:
            print(f"  unknown '{cmd}'")


def main():
    ap = argparse.ArgumentParser(description="G1 independent arm control")
    ap.add_argument("iface")
    ap.add_argument("--keyboard", action="store_true")
    args = ap.parse_args()

    ac = ArmControl(args.iface)

    print("=" * 66)
    print("  G1 ARM CONTROL v2 - LEFT and RIGHT move INDEPENDENTLY")
    print("-" * 66)
    for name, lj, rj, hint in JOINT_DEFS:
        print(f"     {name:<6} L={lj} R={rj}  {hint}")
    print("-" * 66)
    print("  L1 = active arm LEFT    R1 = active arm RIGHT")
    print("  Left/Right cycle joint  Up/Down move   L2/R2 fine")
    print("  A record FULL pose (both arms)   B show all   Y home")
    print("  L1+Select exit")
    print("  Keyboard also works - type 'help'")
    print("=" * 66)
    print("\nStart with the arm that must stay STILL, position it, THEN")
    print("switch to the other arm - A captures both as they stand.\n")

    ac.start()
    ac.show_selected()

    try:
        if args.keyboard:
            keyboard_loop(ac)
        else:
            t = threading.Thread(target=keyboard_loop, args=(ac,),
                                 daemon=True)
            t.start()
            while ac.running:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        ac.running = False
        ac.stop()
        print("\nFinal pose:")
        print("  " + ac.compact())
        ac.export()
        print("Done.")


if __name__ == "__main__":
    main()
