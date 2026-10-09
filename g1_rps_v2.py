#!/usr/bin/env python3
"""
G1 Rock Paper Scissors v2 - full two-arm choreography
=======================================================

Right arm holds flat and still below, palm up, as the base. Left arm pumps
above it three times, then drops into whichever throw was chosen - the
sign itself comes from the LEFT HAND's fingers (BrainCo), not from the arm,
since the arm pose for "throw" is identical for rock, paper and scissors.

POSES - captured on the robot with g1_arm_control_v2.py, not invented
------------------------------------------------------------------------
    pump_low   (pose_2)  left arm at the bottom of the pump
    pump_high  (pose_3)  left arm at the top of the pump
    throw      (pose_4)  left arm extended to present the sign

Right-arm joints (22-26) are IDENTICAL across all three captures, confirming
the base hand never moves once it is in place - only the left arm and its
fingers do anything after setup.

READY POSE - ASSUMED, NOT CAPTURED
------------------------------------
No separate "ready" pose was recorded. This assumes ready == pump_high,
since hovering at the top of the pump is the natural place to start from. If
that is wrong, record a real pose_1 with g1_arm_control_v2.py and paste its
values into READY_LEFT below - one dict, nothing else changes.

COLLISION AVOIDANCE ON SETUP
-----------------------------
The right arm sits BELOW the left. Moving both arms into place at once risks
them crossing paths on the way there. So setup happens in two strict steps:

    1. LEFT arm ONLY moves up into position. Right arm stays at rest,
       out of the way underneath.
    2. Right arm ONLY THEN moves into its base position, sliding in
       underneath the left hand which is already waiting above it.

Teardown reverses this: the LOWER hand (right) retreats first, then the
upper hand (left) comes down - the same reasoning run backwards.

RUN
---
    python3 g1_rps_v2.py wlan0
    python3 g1_rps_v2.py wlan0 --rounds 5
    python3 g1_rps_v2.py wlan0 --force rock
"""

import argparse
import base64
import json
import os
import random
import signal
import sys
import time

# g1_robot.py launches this as a subprocess (same pattern as crane mode) and
# can SIGTERM it to cut a round short. SIGTERM does not run `finally:`
# blocks by default - only SIGINT does, via KeyboardInterrupt - so re-raise
# it as one and the existing try/except/finally teardown handles both.
def _sigterm_to_keyboard_interrupt(signum, frame):
    raise KeyboardInterrupt()


signal.signal(signal.SIGTERM, _sigterm_to_keyboard_interrupt)

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize)
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmdDef
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from g1_brainco_hand import BrainCoHand

from g1_rps_judge import (JUDGE_FRAME_DELAYS, VISION_AVAILABLE,
                          capture_brio_frames, judge_frames)


ARM_SDK_TOPIC  = "rt/arm_sdk"
LOWSTATE_TOPIC = "rt/lf/lowstate"
NOT_USED_JOINT = 29
CONTROL_HZ = 100.0
ARM_KP, ARM_KD = 60.0, 1.5

ARM_JOINTS   = [15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
LEFT_JOINTS  = [15, 16, 17, 18, 19]
RIGHT_JOINTS = [22, 23, 24, 25, 26]

ARM_LIMITS = {
    15: (-2.9, 2.7), 16: (-0.3, 2.9), 17: (-2.6, 2.6),
    18: (-0.2, 3.4), 19: (-2.6, 2.6),
    22: (-2.9, 2.7), 23: (-2.9, 0.3), 24: (-2.6, 2.6),
    25: (-0.2, 3.4), 26: (-2.6, 2.6),
}

# ---- Captured poses, pasted directly from g1_arm_control_v2.py --export ----
PUMP_LOW = {
    15: -0.7706, 16: 0.1433, 17: -0.4791, 18: 0.3552, 19: -0.5706,
    22: -0.4791, 23: -0.0828, 24: 0.3338, 25: 0.2704, 26: 1.5296,
}
PUMP_HIGH = {
    15: -0.7706, 16: 0.1433, 17: -0.4791, 18: 0.1982, 19: -0.5706,
    22: -0.4791, 23: -0.0828, 24: 0.3338, 25: 0.2704, 26: 1.5296,
}
THROW_ARM = {
    15: -0.7706, 16: 0.1433, 17: -0.2631, 18: 0.464, 19: -0.529,
    22: -0.4791, 23: -0.0828, 24: 0.3338, 25: 0.2704, 26: 1.5296,
}

# ASSUMED equal to PUMP_HIGH - see the module docstring. Replace with a real
# captured pose_1 if the actual ready position differs.
READY = dict(PUMP_HIGH)

# Right-arm joints, held fixed once the base is in place - same in every
# captured pose above, extracted once here so setup/teardown can move it
# without depending on which pose happens to be "current" for the left arm.
RIGHT_BASE = {j: PUMP_HIGH[j] for j in RIGHT_JOINTS}

# ---- Finger shapes on the LEFT (playing) hand ----
THROWS = {
    # 1,1,1,1,1,1 made the thumb collide with the index finger on real
    # hardware. Backing the thumb off to 0.4 clears it while still reading
    # as a closed fist.
    "rock":     [0.4, 1, 1, 1, 1, 1],
    "paper":    [0, 0, 0, 0, 0, 0],
    # Was 1,1,0,0,1,1 - after a rock throw the retracted thumb trapped the
    # index/middle fingers behind it on the way to scissors. Releasing the
    # thumb (0) alongside index/middle clears that; ring+pinky (1) still
    # curl to keep the "blades" reading clearly.
    "scissors": [0, 0, 0, 0, 1, 1],
}
BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}
ALIASES = {"r": "rock", "rock": "rock", "p": "paper", "paper": "paper",
          "s": "scissors", "scissors": "scissors"}

RIGHT_HAND_OPEN = [0, 0, 0, 0, 0, 0]   # flat, palm up, set once and left alone

# ---- Vision judge ----
NO_OPPONENT_LINES = [
    "Guess I'm playing myself again.",
    "I don't see anyone brave enough to play me.",
    "Bring your hand closer - I can't play an empty room.",
    "Is anybody actually going to play, or am I just practising?",
]


def judge_human_throw():
    """STANDALONE: grab Brio frames ourselves, then judge them."""
    if not VISION_AVAILABLE:
        return judge_frames([])
    try:
        frames = capture_brio_frames()
    except Exception as e:
        return {"visible": False, "sign": None, "confidence": 0.0,
                "note": "", "error": f"camera: {e}"}
    return judge_frames(frames)


# Cut way down from the first pass, which ramped everything at 2s/move -
# visibly slow. These are also exposed as CLI flags so they can be tuned
# further without editing the file.
SETUP_SECONDS = 0.6
PUMP_SECONDS = 0.22
TEARDOWN_SECONDS = 0.6
WEIGHT_RAMP_UP_SECONDS = 0.6
WEIGHT_RAMP_DOWN_SECONDS = 0.5


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class TwoArmMotion:
    """Smooth interpolated moves, one arm at a time or both together.

    Not interactive - this is the playback half of what
    g1_arm_control_v2.py records poses with.
    """

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

    def start(self, seconds=WEIGHT_RAMP_UP_SECONDS):
        print("Ramping arm_sdk weight up...")
        steps = int(seconds * self.rate)
        for i in range(steps):
            self.weight = (i + 1) / steps
            self._publish()
            time.sleep(self.dt)

    def stop(self, seconds=WEIGHT_RAMP_DOWN_SECONDS):
        print("Ramping weight down...")
        steps = int(seconds * self.rate)
        w0 = self.weight
        for i in range(steps):
            self.weight = w0 * (1 - (i + 1) / steps)
            self._publish()
            time.sleep(self.dt)

    def go(self, target, seconds, joints=None):
        """Interpolate toward `target`. If `joints` is given, ONLY those
        move - everything else holds at its current commanded position,
        which is what makes the collision-avoiding one-arm-at-a-time setup
        possible."""
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


def resolve(human, robot):
    if human == robot:
        return "tie"
    return "robot" if BEATS[robot] == human else "human"


def wait_for_go():
    """Block until the operator presses ENTER, or 'q' to stop the session."""
    raw = input("Press ENTER to throw (q to quit): ").strip().lower()
    return raw not in ("q", "quit", "exit")


# Fixed pause used in --auto mode in place of the "press ENTER to retract"
# prompt - voice triggering has no terminal to type into. Long enough that a
# human sees the robot's throw before it retracts.
AUTO_REVEAL_HOLD_SECONDS = 2.0
# --no-vision: hold the sign this long so the agent can take both Brio frames.
NO_VISION_HOLD_SECONDS = JUDGE_FRAME_DELAYS[-1] + 1.2


def setup(arms, left_hand, right_hand):
    """Bring both arms into the ready position, left first, to avoid the
    two hands crossing paths. Left hand starts in ROCK, which is also the
    shape it pumps in and returns to between throws - not open palm."""
    print("\n=== SETUP ===")
    right_hand.set_all(RIGHT_HAND_OPEN)

    print("  [1/2] left arm moving into position (right stays clear)")
    arms.go(READY, SETUP_SECONDS, joints=LEFT_JOINTS)

    print("  [2/2] right arm sliding into base position underneath")
    arms.go(RIGHT_BASE, SETUP_SECONDS, joints=RIGHT_JOINTS)

    left_hand.set_all(THROWS["rock"])
    print("  ready.")


def teardown(arms, left_hand, right_hand):
    """Reverse of setup: the LOWER hand clears out first, then the upper
    hand comes down - same collision logic run backwards.

    Fingers end FULLY OPEN [0,0,0,0,0,0] on both hands, not rock - a round
    could end holding a fist mid-retract() and that used to be left as the
    final state. Opening explicitly here means the session always ends the
    same way regardless of what the last throw was.
    """
    print("\n=== RESETTING ===")
    right_target = {j: arms.home[j] for j in RIGHT_JOINTS}
    left_target = {j: arms.home[j] for j in LEFT_JOINTS}

    print("  [1/3] opening both hands")
    left_hand.open()
    right_hand.open()

    print("  [2/3] right arm retreating first")
    arms.go(right_target, TEARDOWN_SECONDS, joints=RIGHT_JOINTS)

    print("  [3/3] left arm coming down")
    arms.go(left_target, TEARDOWN_SECONDS, joints=LEFT_JOINTS)


def pump_and_reveal(arms, left_hand, robot_throw, beats=3):
    """Rock... paper... scissors... shoot!

    Fingers stay in ROCK for the whole pump - that is the normal RPS
    convention, a closed fist bouncing, not an open hand. Only the arm's
    elbow moves during the pump; the fingers do not change until the reveal.

    Ends on the reveal and STAYS THERE - it used to auto-retract after a
    fixed pause, which meant the throw was barely visible before it reset.
    Retracting is now a separate step the operator triggers explicitly with
    ENTER, so the sign stays up as long as needed.
    """
    left_hand.set_all(THROWS["rock"])   # ensure rock even if called stale

    words = ["Rock", "Paper", "Scissors"][:beats] if beats <= 3 else \
        (["Rock", "Paper", "Scissors"] * (beats // 3 + 1))[:beats]

    for i in range(beats):
        print(f"  {words[i]}...", end=" ", flush=True)
        arms.go(PUMP_LOW, PUMP_SECONDS, joints=LEFT_JOINTS)
        arms.go(PUMP_HIGH, PUMP_SECONDS, joints=LEFT_JOINTS)
    print()

    print(f"  Shoot! -> {robot_throw.upper()}")
    # Arm shape and finger shape land together - the arm is the same for
    # every sign, the fingers are what actually shows rock/paper/scissors.
    arms.go(THROW_ARM, PUMP_SECONDS, joints=LEFT_JOINTS)
    left_hand.set_all(THROWS[robot_throw])


def retract_to_ready(arms, left_hand, robot_throw):
    """Bring the arm and fingers back to the pumping position, so the next
    round starts from a known state. Only actually moves anything if the
    throw was not already rock."""
    if robot_throw != "rock":
        left_hand.set_all(THROWS["rock"])
        arms.go(PUMP_HIGH, PUMP_SECONDS, joints=LEFT_JOINTS)


def play_round(arms, left_hand, forced=None, beats=3, auto=False,
               no_vision=False):
    """One throw, then judge it against a human hand via camera + LLM.

    `forced` is a manual override for testing a specific sign - normal play
    always throws randomly, regardless of anything typed.

    Interactively, holds the reveal until the operator presses ENTER, then
    retracts - so the sign stays visible as long as wanted instead of
    resetting on a timer. In `auto` mode (voice-triggered, no terminal to
    type into) both ENTER prompts become fixed pauses instead.

    Returns a result dict (never None, in auto mode), or None if the
    operator quit interactively.
    """
    if not auto and not wait_for_go():
        return None

    robot_throw = forced or random.choice(list(THROWS))
    print()
    pump_and_reveal(arms, left_hand, robot_throw, beats=beats)
    print(f"\n  Robot threw: {robot_throw}")

    if no_vision:
        # The agent process owns the Brio and judges the throw itself. Tell
        # it the moment the sign is up, then hold the pose long enough for
        # its frames (JUDGE_FRAME_DELAYS) + the API call to land.
        print("RPS_REVEAL " + json.dumps(
            {"robot_throw": robot_throw,
             "delays": list(JUDGE_FRAME_DELAYS)}), flush=True)
        time.sleep(NO_VISION_HOLD_SECONDS)
        retract_to_ready(arms, left_hand, robot_throw)
        return {"robot_throw": robot_throw, "human_throw": None,
                "winner": None, "visible": False, "error": None,
                "note": "judged by the agent"}

    print("  Looking for your hand...")
    result = judge_human_throw()

    outcome = {"robot_throw": robot_throw, "human_throw": None,
              "winner": None, "visible": result["visible"],
              "error": result["error"], "note": result["note"]}

    if result["error"]:
        print(f"  (couldn't judge: {result['error']})")
    elif not result["visible"]:
        print(f"  {random.choice(NO_OPPONENT_LINES)}")
        if result["note"]:
            print(f"  ({result['note']})")
    else:
        human_throw = result["sign"]
        winner = resolve(human_throw, robot_throw)
        outcome["human_throw"] = human_throw
        outcome["winner"] = winner
        print(f"  You threw: {human_throw}  "
              f"(confidence {result['confidence']:.2f})")
        print("  Tie!" if winner == "tie" else
              ("  Robot wins!" if winner == "robot" else "  You win!"))

    if auto:
        time.sleep(AUTO_REVEAL_HOLD_SECONDS)
    else:
        input("\n  Press ENTER to retract...")
    retract_to_ready(arms, left_hand, robot_throw)
    return outcome


def main():
    global SETUP_SECONDS, PUMP_SECONDS, TEARDOWN_SECONDS

    ap = argparse.ArgumentParser(
        description="G1 RPS - throws at random on ENTER. No winner "
                    "judging yet - that needs a camera + LLM call on the "
                    "human's hand, not built here.")
    ap.add_argument("iface")
    ap.add_argument("--rounds", type=int, default=1,
                    help="throws to run before resetting; ENTER between "
                         "each")
    ap.add_argument("--force", choices=list(THROWS),
                    help="DEBUG ONLY - always throw this sign instead of "
                         "random. Not for normal play.")
    ap.add_argument("--auto", action="store_true",
                    help="no ENTER prompts - throw immediately, hold the "
                         "reveal for a fixed pause, retract, and print one "
                         "JSON result line to stdout. For voice/agent "
                         "triggering, where nothing is there to press "
                         "ENTER. Combine with --rounds for more than one "
                         "throw back to back (default 1).")
    ap.add_argument("--no-vision", action="store_true",
                    help="do NOT touch any camera or the API. Prints an "
                         "RPS_REVEAL line when the sign is up and leaves "
                         "judging to the calling process (the agent, which "
                         "owns the Brio). Implies --auto.")
    ap.add_argument("--beats", type=int, default=3)
    ap.add_argument("--setup-time", type=float, default=SETUP_SECONDS)
    ap.add_argument("--pump-time", type=float, default=PUMP_SECONDS)
    ap.add_argument("--teardown-time", type=float,
                    default=TEARDOWN_SECONDS)
    args = ap.parse_args()

    SETUP_SECONDS = args.setup_time
    PUMP_SECONDS = args.pump_time
    TEARDOWN_SECONDS = args.teardown_time

    rounds = args.rounds
    if args.no_vision:
        args.auto = True

    arms = TwoArmMotion(args.iface)
    left_hand = BrainCoHand("left")
    right_hand = BrainCoHand("right")
    time.sleep(0.3)

    last_outcome = None
    arms.start()
    try:
        setup(arms, left_hand, right_hand)

        for i in range(1, rounds + 1):
            if rounds > 1:
                print(f"\n=== Throw {i} of {rounds} ===")
            outcome = play_round(arms, left_hand, forced=args.force,
                                 beats=args.beats, auto=args.auto,
                                 no_vision=args.no_vision)
            if outcome is None:
                break
            last_outcome = outcome

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        teardown(arms, left_hand, right_hand)
        arms.stop()
        print("Done.")

    if args.auto and not args.no_vision:
        # One machine-readable line on stdout, printed last so a caller can
        # just take the final line of output. g1_robot.py reads this to
        # build what the agent says.
        print("RPS_RESULT " + json.dumps(last_outcome or {
            "robot_throw": None, "human_throw": None, "winner": None,
            "visible": False, "error": "interrupted before a throw completed",
            "note": ""}))


if __name__ == "__main__":
    main()
