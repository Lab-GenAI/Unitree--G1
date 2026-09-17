#!/usr/bin/env python3
"""
G1 Robot - action dispatch for the single-tool ElevenLabs design
=================================================================

The agent calls ONE client tool, humanoid_output, with:

    {"reply": "...", "action": "...", "destination": "..."}

The agent SPEAKS the reply itself; this module executes the action. That
keeps the v9 JSON contract while still using ElevenLabs for speech.

USAGE
-----
    import g1_robot

    # AFTER the ElevenLabs session is live - see the ordering note below
    g1_robot.init(iface="wlan0")

    result = g1_robot.dispatch(action, destination)   # returns a string

ORDERING MATTERS
----------------
Call init() only AFTER conversation.start_session() has returned. Bringing
DDS up first floods the interpreter and the websocket handshake times out.

ACTIONS
-------
  arm         shake_hand, high_five, hug, high_wave, clap, face_wave,
              left_kiss, right_kiss, two_hand_kiss, heart, right_heart,
              hands_up, x_ray, right_hand_up, reject
  locomotion  move_forward, move_backward, move_left, move_right,
              turn_left, turn_right, stop, stand_up, sit_down
  navigation  navigate, run_route, where_is, nav_status, stop_navigation,
              list_places        (navigate/where_is use `destination`)
  camera      look
  crane       crane_on, crane_off
  none
"""

import base64
import os
import subprocess
import threading
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import (
    G1ArmActionClient, action_map)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

try:
    from g1_nav_bridge import NavBridge
    NAV_AVAILABLE = True
except Exception as _e:
    NAV_AVAILABLE = False
    _NAV_ERROR = str(_e)

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
CAMERA_NAME  = "Brio"
CAMERA_INDEX = None          # None = resolve by name
VISION_MODEL = "claude-sonnet-4-5"
CRANE_SCRIPT = os.path.expanduser("~/g1_crane_mode.py")

HOLD_AND_RELEASE = {
    "shake_hand", "high_five", "hug", "heart",
    "right_heart", "hands_up", "x_ray", "right_hand_up", "reject",
}
ACTION_TO_SDK = {
    "shake_hand": "shake hand", "high_five": "high five",
    "hug": "hug", "high_wave": "high wave", "clap": "clap",
    "face_wave": "face wave", "left_kiss": "left kiss",
    "right_kiss": "right kiss", "two_hand_kiss": "two-hand kiss",
    "heart": "heart", "right_heart": "right heart",
    "hands_up": "hands up", "x_ray": "x-ray",
    "right_hand_up": "right hand up", "reject": "reject",
}
LOCO_ACTIONS = {
    "move_forward":  (0.3,  0.0,  0.0, 2.0),
    "move_backward": (-0.3, 0.0,  0.0, 2.0),
    "move_left":     (0.0,  0.3,  0.0, 2.0),
    "move_right":    (0.0, -0.3,  0.0, 2.0),
    "turn_left":     (0.0,  0.0,  0.5, 2.0),
    "turn_right":    (0.0,  0.0, -0.5, 2.0),
}
LOCO_ONLY = set(LOCO_ACTIONS) | {"stop", "stand_up", "sit_down"}
NAV_ACTIONS = {"navigate", "run_route", "where_is", "nav_status",
               "stop_navigation", "list_places"}

arm_client = None
loco_client = None
nav = None
crane_state = {"proc": None}
IFACE = "wlan0"


def all_actions():
    return sorted(set(ACTION_TO_SDK) | LOCO_ONLY | NAV_ACTIONS
                  | {"look", "crane_on", "crane_off", "none"})


# ============================================================
# INIT
# ============================================================
def init(iface="wlan0", use_nav=True, init_dds=True):
    """Bring up DDS and the robot clients. Call AFTER the agent session."""
    global arm_client, loco_client, nav, IFACE
    IFACE = iface

    if init_dds:
        ChannelFactoryInitialize(0, iface)

    try:
        arm_client = G1ArmActionClient()
        arm_client.SetTimeout(10.0)
        arm_client.Init()
        print("[ROBOT] arm client ready")
    except Exception as e:
        print(f"[ROBOT] arm client failed: {e}")
        arm_client = None

    try:
        loco_client = LocoClient()
        loco_client.SetTimeout(10.0)
        loco_client.Init()
        print("[ROBOT] loco client ready")
    except Exception as e:
        print(f"[ROBOT] loco client failed: {e}")
        loco_client = None

    if use_nav and NAV_AVAILABLE:
        try:
            nav = NavBridge()
            print(f"[ROBOT] nav ready - {len(nav.places())} place(s): "
                  f"{nav.places()}")
            print(f"[ROBOT] routes: {nav.route_names()}")
            ready = nav.is_ready()
            if ready is True:
                print("[ROBOT] relocalised - navigation available")
            elif ready is False:
                print("[ROBOT] NOT relocalised. Run first:")
                print("        python3 g1_slam_client_v6.py --prepare "
                      "<map>.pcd --from-waypoint home")
        except Exception as e:
            print(f"[ROBOT] nav failed: {e}")
            nav = None
    elif not use_nav:
        print("[ROBOT] navigation disabled")
    elif not NAV_AVAILABLE:
        print(f"[ROBOT] nav unavailable: {_NAV_ERROR}")


def shutdown():
    if crane_is_active():
        crane_state["proc"].terminate()


# ============================================================
# CAMERA
# ============================================================
def find_camera_device(name_substr=None):
    """Resolve the Brio's /dev/videoN by NAME.

    The RealSense claims six video nodes, so a hardcoded index lands on a raw
    IR/depth stream - which is why vision used to report "a black screen".
    """
    name_substr = name_substr or CAMERA_NAME
    try:
        out = subprocess.run(["v4l2-ctl", "--list-devices"],
                             capture_output=True, text=True,
                             timeout=5).stdout
    except Exception:
        return None
    current = None
    for line in out.splitlines():
        if line and not line.startswith(("\t", " ")):
            current = line
        elif current and name_substr.lower() in current.lower():
            dev = line.strip()
            if dev.startswith("/dev/video"):
                return dev
    return None


def capture_camera_frame(warmup_frames=25):
    """Capture from the Brio via OpenCV AFTER auto-exposure settles.

    The first frame off a cold webcam is dark - exposure and white balance
    have not converged - and the model then honestly describes "a very dark
    room". Discarding ~25 frames fixes it.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("opencv-python not installed")
    device = (f"/dev/video{CAMERA_INDEX}" if CAMERA_INDEX is not None
              else find_camera_device())
    if device is None:
        raise RuntimeError(f"no video device matching '{CAMERA_NAME}'")

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {device}")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for _ in range(warmup_frames):
            ok, f = cap.read()
            if ok:
                frame = f
            time.sleep(0.02)
        if frame is None:
            raise RuntimeError(f"no frames read from {device}")
        mean = float(frame.mean())
        if mean < 12:
            raise RuntimeError(f"frame almost black (mean {mean:.1f})")
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        return base64.b64encode(enc.tobytes()).decode("utf-8")
    finally:
        cap.release()


def describe_view(question=""):
    """ElevenLabs agents cannot accept images, so vision runs on Claude and
    the description is returned as the tool result."""
    if not ANTHROPIC_AVAILABLE or not os.getenv("ANTHROPIC_API_KEY"):
        return "My vision isn't set up right now."
    try:
        img = capture_camera_frame()
    except Exception as e:
        print(f"[VISION] capture failed: {e}")
        return "I'm not getting a clear picture from my camera right now."

    prompt = (question or "What do you see?") + (
        "\n\nYou are looking through a Logitech Brio 100 webcam mounted on a "
        "humanoid robot, facing forward. Describe what is actually in the "
        "photo in one to three short sentences, as if speaking aloud. If the "
        "image is unreadable, say so plainly rather than inventing a "
        "description. Never describe it as thermal or infrared - this camera "
        "does not produce those.")
    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=VISION_MODEL, max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": img}},
                {"type": "text", "text": prompt}]}])
        text = resp.content[0].text.strip()
        print(f"[VISION] {text}")
        return text
    except Exception as e:
        print(f"[VISION] Claude failed: {e}")
        return "I couldn't process what I'm seeing just now."


# ============================================================
# ACTIONS
# ============================================================
def handle_loco_action(action):
    if loco_client is None:
        return "My legs aren't connected right now."

    def _run():
        try:
            if action == "stop":
                loco_client.Move(0.0, 0.0, 0.0)
            elif action == "stand_up":
                loco_client.StandUp()
            elif action == "sit_down":
                loco_client.StandUp2Squat()
            elif action in LOCO_ACTIONS:
                vx, vy, vyaw, dur = LOCO_ACTIONS[action]
                loco_client.Move(vx, vy, vyaw)
                time.sleep(dur)
                loco_client.Move(0.0, 0.0, 0.0)
        except Exception as e:
            print(f"[LOCO] {action} failed: {e}")

    print(f"[LOCO] {action}")
    threading.Thread(target=_run, daemon=True).start()
    return f"Doing {action.replace('_', ' ')}."


def handle_arm_action(action):
    if arm_client is None:
        return "My arms aren't connected right now."
    sdk_key = ACTION_TO_SDK.get(action)
    if not sdk_key:
        return f"I don't know the action {action}."
    action_id = action_map.get(sdk_key)
    if action_id is None:
        return f"I don't have {action} available."

    def _run():
        try:
            arm_client.ExecuteAction(action_id)
            if action in HOLD_AND_RELEASE:
                time.sleep(2)
                arm_client.ExecuteAction(action_map.get("release arm"))
        except Exception as e:
            print(f"[ACTION] {action} failed: {e}")

    print(f"[ACTION] {sdk_key} (id={action_id})")
    threading.Thread(target=_run, daemon=True).start()
    return f"Doing {action.replace('_', ' ')}."


def crane_is_active():
    p = crane_state["proc"]
    return p is not None and p.poll() is None


def enter_crane():
    if crane_is_active():
        return "I'm already in crane mode."
    print("[MODE] entering crane mode")
    crane_state["proc"] = subprocess.Popen(["python3", CRANE_SCRIPT, IFACE])
    return "Switching to crane mode."


def exit_crane():
    if not crane_is_active():
        return "I'm already in normal mode."
    print("[MODE] exiting crane mode")
    crane_state["proc"].terminate()
    crane_state["proc"] = None
    return "Switching back to normal mode."


# ============================================================
# DISPATCH
# ============================================================
def dispatch(action, destination=None, reply=""):
    """Execute one action. Returns a short status string.

    `destination` is used by navigate and where_is. The agent may send the
    string "none" rather than omitting it, so that is treated as empty.
    """
    action = (action or "none").strip().lower()
    if destination in (None, "none", ""):
        destination = None

    if action in ("none", ""):
        return "ok"

    if action in LOCO_ONLY:
        return handle_loco_action(action)

    if action in ACTION_TO_SDK:
        return handle_arm_action(action)

    if action == "look":
        return describe_view(reply or "")

    if action == "crane_on":
        return enter_crane()
    if action == "crane_off":
        return exit_crane()

    if action in NAV_ACTIONS:
        if nav is None:
            return "I can't move on my own right now."

        if action == "list_places":
            places = ", ".join(p.replace("_", " ") for p in nav.places())
            routes = ", ".join(nav.route_names())
            return f"Places: {places or 'none'}. Routes: {routes or 'none'}."

        if action == "nav_status":
            return nav.status_text()

        if action == "stop_navigation":
            _, msg = nav.stop()
            return msg

        if action == "where_is":
            if not destination:
                return "Which place do you mean?"
            return nav.where_is(destination)

        if action == "navigate":
            if not destination:
                return "Where would you like me to go?"
            if nav.is_ready() is False:
                return "I'm not localised on the map yet."
            _, msg = nav.start(destination)
            return msg

        if action == "run_route":
            if nav.is_ready() is False:
                return "I'm not localised on the map yet."
            _, msg = nav.start_route(destination or "tour")
            return msg

    return f"I don't know how to {action}."
