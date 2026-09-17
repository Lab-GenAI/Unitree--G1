#!/usr/bin/env python3
"""
G1 Crane Mode - smooth motion
==============================

Same controls as before, but the arms accelerate and decelerate instead of
snapping.

WHAT CHANGED AND WHY
--------------------
The original applied a fixed STEP of 0.03 rad to the target on every control
cycle a button was held. At 50 Hz that is a 1.5 rad/s step command that
starts and stops instantly - no ramp in, no ramp out - so every press jerks
and every release stops dead.

Now each joint carries a VELOCITY. Holding a button sets a target velocity;
the actual velocity moves toward it under an acceleration limit, and the
joint position integrates from that. Press and release are both smooth, and
the motion has a natural feel.

    hold   -> velocity ramps 0 -> MAX_VEL over (MAX_VEL / ACCEL) seconds
    release-> velocity ramps back to 0 over the same time

ANALOG STICKS
-------------
The d-pad is on/off, so it can only ever give one speed. The sticks are
proportional - push a little for fine positioning, push hard for fast moves.
Both work at once.

    Left stick  Y  -> both arms pitch  (up/down)
    Left stick  X  -> both arms roll   (out/in)
    Right stick Y  -> both elbows
    Right stick X  -> both wrists

CONTROLS
--------
  Up / Down     both arms raise / lower      (shoulder pitch, joints 15/22)
  Left / Right  both arms outward / inward   (shoulder roll,  joints 16/23)
  Sticks        proportional control (see above)
  R2 (hold)     close hands
  L2 (hold)     open hands
  A             lock arms where they are
  B             unlock
  L1 + Select   exit

JOINT DIRECTIONS (confirmed from the original)
  15/22 pitch : NEGATIVE raises the arm forward
  16    roll  : POSITIVE  moves the LEFT arm outward
  23    roll  : NEGATIVE  moves the RIGHT arm outward   (mirrored)

HANDS
-----
Defaults to the Dex3 topics. THIS ROBOT HAS BRAINCO REVO2 HANDS, which use
different topics and a different message - pass --hands brainco and make sure
brainco_hand_service is running. --hands none skips hand control entirely.

RUN
---
    python3 g1_crane_mode.py wlan0
    python3 g1_crane_mode.py wlan0 --hands none
    python3 g1_crane_mode.py wlan0 --max-vel 0.5 --accel 1.5
"""

import argparse
import struct
import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
)
from unitree_sdk2py.idl.default import (
    unitree_hg_msg_dds__LowCmd_ as LowCmdDefault,
    unitree_hg_msg_dds__HandCmd_ as HandCmdDefault,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, HandCmd_
from unitree_sdk2py.utils.crc import CRC


# ============================================================
# CONFIG
# ============================================================
ARM_SDK_TOPIC    = "rt/arm_sdk"
LOWSTATE_TOPIC   = "rt/lf/lowstate"
LEFT_HAND_TOPIC  = "rt/dex3/left/cmd"
RIGHT_HAND_TOPIC = "rt/dex3/right/cmd"

CONTROL_HZ = 100.0     # higher than the original 50 - smoother command stream
ARM_KP     = 60.0
ARM_KD     = 1.5
HAND_KP    = 1.5
HAND_KD    = 0.1

# Motion feel. MAX_VEL is the top speed of a joint with a button fully held;
# ACCEL is how quickly it gets there. Time to full speed = MAX_VEL / ACCEL.
MAX_VEL   = 0.70       # rad/s
ACCEL     = 2.00       # rad/s^2  -> ~0.35 s to reach full speed
STICK_DEADZONE = 0.12  # sticks rest slightly off-centre

NOT_USED_JOINT = 29

ARM_JOINTS = {
    "L_pitch": 15, "L_roll": 16, "L_yaw": 17, "L_elbow": 18, "L_wrist": 19,
    "R_pitch": 22, "R_roll": 23, "R_yaw": 24, "R_elbow": 25, "R_wrist": 26,
}

ARM_LIMITS = {
    15: (-2.9,  2.7), 16: (-0.3,  2.9), 17: (-2.6,  2.6),
    18: (-0.2,  3.4), 19: (-2.6,  2.6),
    22: (-2.9,  2.7), 23: (-2.9,  0.3), 24: (-2.6,  2.6),
    25: (-0.2,  3.4), 26: (-2.6,  2.6),
}

LEFT_MAX  = [ 1.05,  1.05,  1.75,  0.00,  0.00,  0.00,  0.00]
LEFT_MIN  = [-1.05, -0.724, 0.00, -1.57, -1.75, -1.57, -1.75]
RIGHT_MAX = [ 1.05,  0.742, 0.00,  1.57,  1.75,  1.57,  1.75]
RIGHT_MIN = [-1.05, -1.05, -1.75,  0.00,  0.00,  0.00,  0.00]
MOTOR_MAX = 7


class unitreeRemoteController:
    def __init__(self):
        self.Lx = self.Rx = self.Ry = self.Ly = 0.0
        self.L1 = self.L2 = self.R1 = self.R2 = 0
        self.A = self.B = self.X = self.Y = 0
        self.Up = self.Down = self.Left = self.Right = 0
        self.Select = self.F1 = self.F3 = self.Start = 0

    def parse_botton(self, d1, d2):
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

    def parse_key(self, data):
        self.Lx = struct.unpack('<f', data[4:8])[0]
        self.Rx = struct.unpack('<f', data[8:12])[0]
        self.Ry = struct.unpack('<f', data[12:16])[0]
        self.Ly = struct.unpack('<f', data[20:24])[0]

    def parse(self, remoteData):
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2], remoteData[3])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def deadzone(v, dz=STICK_DEADZONE):
    """Rescale so motion starts smoothly at the edge of the deadzone rather
    than jumping to `dz` worth of speed the moment the stick moves."""
    if abs(v) < dz:
        return 0.0
    return (abs(v) - dz) / (1.0 - dz) * (1.0 if v > 0 else -1.0)


def make_hand_cmd(positions, kp, kd):
    msg = HandCmdDefault()
    for i in range(MOTOR_MAX):
        msg.motor_cmd[i].mode = (i & 0x0F) | (0x01 << 4)
        msg.motor_cmd[i].q = float(positions[i])
        msg.motor_cmd[i].dq = 0.0
        msg.motor_cmd[i].tau = 0.0
        msg.motor_cmd[i].kp = float(kp)
        msg.motor_cmd[i].kd = float(kd)
        msg.motor_cmd[i].reserve = 0
    return msg


def hand_positions(is_left, grip):
    """grip 0.0 = fully open, 1.0 = closed. Interpolating rather than sending
    open/closed outright means the fingers move smoothly too."""
    if is_left:
        open_p, closed_p = LEFT_MAX, [(LEFT_MAX[i] + LEFT_MIN[i]) / 2.0
                                      for i in range(MOTOR_MAX)]
    else:
        open_p, closed_p = RIGHT_MIN, [(RIGHT_MAX[i] + RIGHT_MIN[i]) / 2.0
                                       for i in range(MOTOR_MAX)]
    return [open_p[i] + (closed_p[i] - open_p[i]) * grip
            for i in range(MOTOR_MAX)]


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="G1 crane mode (smooth)")
    ap.add_argument("iface")
    ap.add_argument("--hands", choices=["dex3", "brainco", "none"],
                    default="dex3",
                    help="THIS ROBOT HAS BRAINCO HANDS - 'dex3' will publish "
                         "to topics nothing is listening on")
    ap.add_argument("--max-vel", type=float, default=MAX_VEL,
                    help="top joint speed in rad/s")
    ap.add_argument("--accel", type=float, default=ACCEL,
                    help="acceleration in rad/s^2; lower is gentler")
    ap.add_argument("--rate", type=float, default=CONTROL_HZ)
    args = ap.parse_args()

    if args.hands == "brainco":
        print("NOTE: BrainCo hands use rt/brainco/{left,right}/cmd with a")
        print("      different message type. Hand control is DISABLED here -")
        print("      drive them from your brainco_hand_service client.")

    ChannelFactoryInitialize(0, args.iface)

    latest = {"msg": None}
    remote = unitreeRemoteController()

    def on_lowstate(msg):
        latest["msg"] = msg
        try:
            remote.parse(msg.wireless_remote)
        except Exception:
            pass

    sub = ChannelSubscriber(LOWSTATE_TOPIC, LowState_)
    sub.Init(on_lowstate, 10)

    print("Waiting for lowstate...")
    deadline = time.time() + 10.0
    while latest["msg"] is None:
        if time.time() > deadline:
            raise RuntimeError("No lowstate - check the interface name.")
        time.sleep(0.05)
    print(f"Got lowstate. mode_machine = {latest['msg'].mode_machine}")
    time.sleep(0.5)

    arm_pub = ChannelPublisher(ARM_SDK_TOPIC, LowCmd_)
    arm_pub.Init()
    crc = CRC()
    arm_msg = LowCmdDefault()
    arm_msg.mode_machine = latest["msg"].mode_machine

    # Seed from where the arms actually are, so nothing jumps on entry.
    targets = {j: latest["msg"].motor_state[j].q for j in ARM_JOINTS.values()}
    vels = {j: 0.0 for j in ARM_JOINTS.values()}
    locked = False

    hands_enabled = args.hands == "dex3"
    if hands_enabled:
        lh = ChannelPublisher(LEFT_HAND_TOPIC, HandCmd_)
        lh.Init()
        rh = ChannelPublisher(RIGHT_HAND_TOPIC, HandCmd_)
        rh.Init()
    grip = 0.0

    weight = 0.0
    dt = 1.0 / args.rate

    def publish_arms(w):
        for j in ARM_JOINTS.values():
            lo, hi = ARM_LIMITS[j]
            mc = arm_msg.motor_cmd[j]
            mc.q = clamp(targets[j], lo, hi)
            mc.dq = 0.0
            mc.kp = ARM_KP
            mc.kd = ARM_KD
            mc.tau = 0.0
        arm_msg.motor_cmd[NOT_USED_JOINT].q = w
        arm_msg.crc = crc.Crc(arm_msg)
        arm_pub.Write(arm_msg)

    print("Ramping arm_sdk weight up...")
    steps = int(2.0 * args.rate)
    for i in range(steps):
        weight = (i + 1) / steps
        publish_arms(weight)
        time.sleep(dt)

    print(f"Crane mode active  (max {args.max_vel:.2f} rad/s, "
          f"accel {args.accel:.2f} rad/s^2, {args.rate:.0f} Hz)")
    print("  D-pad / sticks: move arms   R2: grip   L2: open")
    print("  A: lock   B: unlock   L1+Select: exit")

    try:
        while True:
            r = remote

            if r.L1 and r.Select:
                print("Exiting crane mode...")
                break
            if r.A and not locked:
                locked = True
                print("Arms LOCKED.")
            if r.B and locked:
                locked = False
                print("Arms UNLOCKED.")

            # ---- desired velocity per joint, in [-1, 1] ----
            want = {j: 0.0 for j in ARM_JOINTS.values()}

            if not locked:
                # D-pad: full speed. Negative pitch raises the arm forward.
                pitch = (-1.0 if r.Up else 0.0) + (1.0 if r.Down else 0.0)
                # Left arm rolls out on POSITIVE, right arm on NEGATIVE.
                roll = (1.0 if r.Left else 0.0) + (-1.0 if r.Right else 0.0)

                # Sticks: proportional, added on top of the d-pad.
                pitch += -deadzone(r.Ly)
                roll += deadzone(r.Lx)
                elbow = -deadzone(r.Ry)
                wrist = deadzone(r.Rx)

                pitch = clamp(pitch, -1.0, 1.0)
                roll = clamp(roll, -1.0, 1.0)

                want[15] = pitch
                want[22] = pitch
                want[16] = roll
                want[23] = -roll        # mirrored
                want[18] = elbow
                want[25] = elbow
                want[19] = wrist
                want[26] = -wrist       # mirrored

            # ---- ramp actual velocity toward desired, then integrate ----
            max_delta = args.accel * dt
            for j in ARM_JOINTS.values():
                target_v = want[j] * args.max_vel
                dv = clamp(target_v - vels[j], -max_delta, max_delta)
                vels[j] += dv

                if abs(vels[j]) > 1e-6:
                    lo, hi = ARM_LIMITS[j]
                    new = clamp(targets[j] + vels[j] * dt, lo, hi)
                    # Hitting a limit should stop the joint, not keep winding
                    # velocity up against it.
                    if new in (lo, hi) and new != targets[j]:
                        vels[j] = 0.0
                    targets[j] = new

            publish_arms(weight)

            # ---- hands: interpolate rather than snap ----
            if hands_enabled:
                if r.R2:
                    grip = clamp(grip + 2.0 * dt, 0.0, 1.0)
                elif r.L2:
                    grip = clamp(grip - 2.0 * dt, 0.0, 1.0)
                lh.Write(make_hand_cmd(hand_positions(True, grip),
                                       HAND_KP, HAND_KD))
                rh.Write(make_hand_cmd(hand_positions(False, grip),
                                       HAND_KP, HAND_KD))

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # Decelerate before releasing, so the arms do not drop from speed.
        print("Stopping arms...")
        for _ in range(int(0.5 * args.rate)):
            for j in ARM_JOINTS.values():
                dv = clamp(-vels[j], -args.accel * dt, args.accel * dt)
                vels[j] += dv
                targets[j] = clamp(targets[j] + vels[j] * dt,
                                   *ARM_LIMITS[j])
            publish_arms(weight)
            time.sleep(dt)

        print("Ramping weight down...")
        steps = int(1.0 * args.rate)
        for i in range(steps):
            publish_arms(weight * (1 - (i + 1) / steps))
            time.sleep(dt)
        print("Done.")


if __name__ == "__main__":
    main()
