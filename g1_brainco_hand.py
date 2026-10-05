#!/usr/bin/env python3
"""
G1 BrainCo Hand - per-finger control
=====================================

Direct finger-level control of the BrainCo Revo2 hands, over the SAME message
type the working test client uses:

    unitree_go::msg::dds_::MotorCmds_   ->  rt/brainco/{left,right}/cmd
    unitree_go::msg::dds_::MotorStates_ ->  rt/brainco/{left,right}/state

This is a standard Go2 IDL type reused for the hand, not a BrainCo-specific
message - which is why it needs no CRC and no header ceremony, unlike
arm_sdk's LowCmd_. Six commands in the array, one per finger; set speed once,
then just update position per publish.

FINGER ORDER
------------
Index into the 6-element array. Confirmed from the working demo's fist
sequence (fingers curl first, thumb wraps last) matching the convention
established when this hand's DDS bridge was first reverse-engineered:

    0  thumb
    1  thumb_aux   (thumb rotation/opposition - switches pinch vs power grip)
    2  index
    3  middle
    4  ring
    5  pinky

Position is 0.0 = open/extended, 1.0 = closed/curled. Speed (dq) is set once
at startup to 1.0 (max) and not touched per command, matching the test
client.

PREREQUISITE
------------
brainco_hand_server must be running (the serial<->DDS bridge), same as for
the crane-mode/test-client work. This module only talks DDS; it does not
touch the serial ports itself.

USAGE
-----
    from g1_brainco_hand import BrainCoHand

    left = BrainCoHand("left")
    left.set_finger("thumb", 0.0)
    left.set_all([0, 1, 1, 1, 1, 1])   # curl everything but the thumb
    left.open()
    left.fist()

CLI
---
    python3 g1_brainco_hand.py wlan0 left --open
    python3 g1_brainco_hand.py wlan0 left --fist
    python3 g1_brainco_hand.py wlan0 left --finger index 0.8
    python3 g1_brainco_hand.py wlan0 left --pose 0 1 1 0 0 1
    python3 g1_brainco_hand.py wlan0 left --watch
"""

import argparse
import sys
import time

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)

try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import (
        MotorCmds_, MotorCmd_, MotorStates_)
    IDL_OK = True
except ImportError as e:
    IDL_OK = False
    _IMPORT_ERROR = str(e)


FINGERS = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]
FINGER_INDEX = {name: i for i, name in enumerate(FINGERS)}

OPEN_POSE = [0.0] * 6
FIST_POSE = [1.0] * 6

DEFAULT_SPEED = 1.0   # dq, set once at startup - matches the test client


def make_motor_cmd(q=0.0, dq=0.0):
    """Build a MotorCmd_.

    Like Request_ elsewhere in this project, this is a frozen dataclass -
    every field is required at construction, no default-then-assign. The C++
    test client only ever touches q and dq; mode/tau/kp/kd/reserve are left
    at whatever C++ default-construction gives them, which is zero.

    `reserve` is NOT a scalar - confirmed via dataclasses.fields() as
    Sequence[uint32], fixed length 3. Passing a bare 0 made the serializer
    try len(0) and fail. It needs a 3-element list.
    """
    return MotorCmd_(mode=0, q=q, dq=dq, tau=0.0, kp=0.0, kd=0.0,
                     reserve=[0, 0, 0])


def clamp01(v):
    return max(0.0, min(1.0, v))


class BrainCoHand:
    """One hand. Two of these for left + right."""

    def __init__(self, side, speed=DEFAULT_SPEED):
        if not IDL_OK:
            raise RuntimeError(
                f"unitree_go MotorCmds_/MotorStates_ not importable: "
                f"{_IMPORT_ERROR}")
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side

        self.pub = ChannelPublisher(f"rt/brainco/{side}/cmd", MotorCmds_)
        self.pub.Init()

        # cmds is a plain Python list, starts EMPTY - confirmed by
        # inspecting the class directly rather than trusting the C++ .cmds()
        # syntax, which does not carry over (it raised "list object is not
        # callable"). Build six real MotorCmd_ elements and assign the list.
        self.msg = MotorCmds_()
        self.msg.cmds = [make_motor_cmd(q=0.0, dq=speed) for _ in range(6)]

        self.positions = list(OPEN_POSE)

        self._state = {"msg": None}
        self.sub = ChannelSubscriber(f"rt/brainco/{side}/state", MotorStates_)
        self.sub.Init(lambda m: self._state.__setitem__("msg", m), 1)

    def _publish(self):
        # Also a frozen dataclass - cannot do cmds[i].q = pos, must replace
        # the whole element. dq (speed) carries over from what was set at
        # construction.
        speed = self.msg.cmds[0].dq if self.msg.cmds else DEFAULT_SPEED
        self.msg.cmds = [make_motor_cmd(q=pos, dq=speed)
                         for pos in self.positions]
        self.pub.Write(self.msg)

    def set_finger(self, name_or_index, position):
        idx = (FINGER_INDEX[name_or_index]
               if isinstance(name_or_index, str) else int(name_or_index))
        self.positions[idx] = clamp01(position)
        self._publish()

    def set_all(self, positions):
        if len(positions) != 6:
            raise ValueError("need exactly 6 positions")
        self.positions = [clamp01(p) for p in positions]
        self._publish()

    def open(self):
        self.set_all(OPEN_POSE)

    def fist(self):
        self.set_all(FIST_POSE)

    def measured(self, name_or_index, timeout=1.0):
        """Read back the actual position from rt/brainco/{side}/state.

        Useful for confirming a command actually took effect, same as
        checking LowState_ elsewhere in this project - and for verifying the
        finger index order empirically rather than trusting the demo pattern.
        """
        idx = (FINGER_INDEX[name_or_index]
               if isinstance(name_or_index, str) else int(name_or_index))
        deadline = time.time() + timeout
        while self._state["msg"] is None and time.time() < deadline:
            time.sleep(0.02)
        m = self._state["msg"]
        if m is None:
            return None
        return m.states[idx].q

    def measured_all(self, timeout=1.0):
        deadline = time.time() + timeout
        while self._state["msg"] is None and time.time() < deadline:
            time.sleep(0.02)
        m = self._state["msg"]
        if m is None:
            return None
        return [m.states[i].q for i in range(6)]


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="BrainCo hand finger control")
    ap.add_argument("iface")
    ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--fist", action="store_true")
    ap.add_argument("--finger", nargs=2, metavar=("NAME", "POS"),
                    help="e.g. --finger index 0.8")
    ap.add_argument("--pose", nargs=6, type=float,
                    metavar=("THUMB", "THUMB_AUX", "INDEX", "MIDDLE",
                            "RING", "PINKY"))
    ap.add_argument("--watch", action="store_true",
                    help="cycle each finger alone, print measured position - "
                         "use this to VERIFY the index order on real hardware")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    args = ap.parse_args()

    ChannelFactoryInitialize(0, args.iface)
    hand = BrainCoHand(args.side, speed=args.speed)
    time.sleep(0.3)   # let the subscriber connect before reading state

    if args.open:
        hand.open()
        print("opened")
    elif args.fist:
        hand.fist()
        print("fist")
    elif args.finger:
        name, pos = args.finger[0], float(args.finger[1])
        hand.set_finger(name, pos)
        print(f"{name} -> {pos}")
    elif args.pose:
        hand.set_all(args.pose)
        print(f"pose -> {args.pose}")
    elif args.watch:
        print("Opening fully first...")
        hand.open()
        time.sleep(1.0)
        for name in FINGERS:
            print(f"\ncurling ONLY {name}...")
            positions = list(OPEN_POSE)
            positions[FINGER_INDEX[name]] = 1.0
            hand.set_all(positions)
            time.sleep(1.2)
            m = hand.measured_all()
            if m:
                print(f"  measured: " +
                      "  ".join(f"{f}={v:.2f}" for f, v in zip(FINGERS, m)))
            else:
                print("  no state received - is brainco_hand_server running?")
            input("  press ENTER to check the next finger... ")
        hand.open()
        print("\ndone - open again")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
