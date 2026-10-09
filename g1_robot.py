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
  demos       play_rps, play_gesturesynth   (fire-and-forget performances -
              see PROCESSES below)
  grasp       pickup_glass, release_grasp   (pickup_glass walks the robot
              into position and holds the grasp until release_grasp)
  measure     measure_object      (`destination` carries what to measure,
              e.g. "the red box" - blank means "whatever's in front")
  none

ASYNC NARRATION
----------------
Several of the above take longer than an ElevenLabs tool call can wait for
(a multi-waypoint walk, a full RPS round with vision judging). Rather than
block the tool call, those report back through announce(), which pushes a
line into the LIVE conversation so the agent speaks about it unprompted -
"I've arrived", "you win", "that's about 30 by 20 centimetres" - the same
way arrival was always meant to work. This needs set_conversation() called
once, right after conversation.start_session() returns (same ordering
constraint as init() - see ORDERING MATTERS above).
"""

import base64
import json
import os
import random
import re
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
    from g1_obstacle_guard import ObstacleGuard
    OBSTACLE_GUARD_AVAILABLE = True
except Exception as _e:
    OBSTACLE_GUARD_AVAILABLE = False
    _OBSTACLE_GUARD_ERROR = str(_e)

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
# Vision runs inside an ElevenLabs client tool, which has a timeout. Blow
# past it and the agent answers without waiting - which showed up as the
# robot apologising that it "couldn't get a clear view" while the real
# description arrived seconds later.
#
# Opus: Haiku hallucinated scene details. Slower (expect a few seconds), so
# keep the answer short (VISION_MAX_TOKENS) and the image small
# (VISION_WIDTH) to stay inside the tool deadline.
VISION_MODEL = "claude-opus-5-5"
VISION_MAX_TOKENS = 1024   # headroom: Opus may spend some on thinking
VISION_WIDTH = 640          # downscale before sending; smaller = faster
CRANE_SCRIPT      = os.path.expanduser("~/g1_crane_mode.py")
RPS_SCRIPT         = os.path.expanduser("~/g1_rps_v2.py")
GESTURESYNTH_SCRIPT = os.path.expanduser("~/g1_gesturesynth.py")
# Waypoint the robot walks to BEFORE performing GestureSynth. Must match a key
# in ~/g1_waypoints.json (matched loosely, e.g. "demo floor" -> demo_floor).
# The agent can override it per request by passing a `destination`. Leave ""
# to perform wherever it is standing.
GESTURESYNTH_STAGE = "labcenterlatest"
PICKUP_GLASS_SCRIPT = os.path.expanduser("~/g1_pickup_glass.py")
MEASURE_SCRIPT      = os.path.expanduser("~/g1_measure.py")

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
IFACE = "wlan0"

# Every subprocess-backed "performance" (crane mode, RPS, GestureSynth, a
# held grasp) lives here under a name, the same way crane_state used to be
# its own special case. One process per slot; starting a new one in an
# occupied slot stops the old one first.
PROCESSES = {}   # name -> subprocess.Popen

DEMO_ACTIONS = {"play_rps", "play_gesturesynth"}
GRASP_ACTIONS = {"pickup_glass", "release_grasp"}
MEASURE_ACTIONS = {"measure_object"}

_conversation = {"obj": None}


def set_conversation(conversation):
    """Call this once, right after conversation.start_session() returns -
    gives background work (nav arrival, a finished RPS round, a grasp
    confirmation) a way to make the agent speak proactively instead of only
    replying inside a tool call. See announce()."""
    _conversation["obj"] = conversation


def announce(text):
    """Push an out-of-band note into the live session so the agent speaks
    about something that just finished in the background.

    UNVERIFIED: which method the installed elevenlabs SDK actually exposes
    for this was not confirmed against a running session - this tries a
    short list of plausible names (ElevenLabs' own docs describe a
    `contextual_update` event for background info and a `user_message`
    event that triggers an immediate spoken response; the Python SDK's
    method names for sending either were not available to check at the
    time this was written). Check once with:

        python3 -c "from elevenlabs.conversational_ai.conversation import \\
Conversation; print([m for m in dir(Conversation) if not m.startswith('_')])"

    and if the real name isn't in the list below, add it - everything that
    calls announce() elsewhere does not need to change.
    """
    print(f"[ANNOUNCE] {text}")
    conv = _conversation["obj"]
    if conv is None:
        print("[ANNOUNCE] no live session registered (set_conversation not "
              "called yet) - not spoken")
        return False
    for method_name in ("send_user_message", "send_contextual_update",
                        "send_text", "send_message"):
        method = getattr(conv, method_name, None)
        if callable(method):
            try:
                method(text)
                return True
            except Exception as e:
                print(f"[ANNOUNCE] {method_name} failed: {e}")
    print("[ANNOUNCE] no working send method found on this SDK version - "
          "see the docstring above")
    return False


def update_context(text):
    """Silently add a fact to the agent's context WITHOUT making it speak
    (ElevenLabs' contextual_update event). Used for robot state, so a later
    'where are you?' or 'how's it going?' is answered from the truth rather
    than from whatever the agent last said. Returns False if not delivered."""
    conv = _conversation["obj"]
    if conv is None:
        return False
    method = getattr(conv, "send_contextual_update", None)
    if not callable(method):
        print("[STATE] this SDK has no send_contextual_update")
        return False
    try:
        method(text)
        return True
    except Exception as e:
        print(f"[STATE] contextual update failed: {e}")
        return False


_last_state_text = {"v": None}


def robot_state_text():
    """One line of ground truth about where the robot is and what it's doing."""
    parts = []
    if nav is not None:
        dest = nav.current_destination
        if dest:
            parts.append(f"WALKING to {dest.replace('_', ' ')} right now "
                         f"(has NOT arrived yet)")
        elif nav.last_arrived_key:
            parts.append(f"standing at {nav.last_arrived_key.replace('_', ' ')}"
                         f" - you have ARRIVED there and are not walking")
        else:
            near = None
            try:
                near = nav.nearest_waypoint()
            except Exception:
                pass
            parts.append(f"not walking; nearest saved place is "
                         f"{near.replace('_', ' ')}" if near
                         else "not walking; exact position unknown")
    doing = [label for key, label in (
        ("rps", "playing rock-paper-scissors"),
        ("gesturesynth", "performing the finger show"),
        ("grasp", "holding / picking up a glass"),
        ("measure", "taking a measurement")) if proc_active(key)]
    parts.append("currently " + ", ".join(doing) if doing
                 else "not doing any demo")
    return "[ROBOT STATE] " + "; ".join(parts) + "."


def push_state(force=False):
    """Send the current state to the agent as silent context, if it changed."""
    try:
        text = robot_state_text()
    except Exception as e:
        print(f"[STATE] couldn't build state: {e}")
        return
    if not force and text == _last_state_text["v"]:
        return
    _last_state_text["v"] = text
    if update_context(text):
        print(f"[STATE] {text}")


def _nav_event(msg):
    """NavBridge's on_event: silent state first (so the agent's next reply is
    built on the new truth), then the spoken note."""
    push_state()
    announce(f"[NAV] {msg}")


# Appended to the tool result of anything that only STARTS something slow.
# The agent has been seen reading a bare "Heading to X." as "done" and
# announcing arrival at once - so every started-not-finished result says so
# in capitals, and names the note that will report the real outcome.
_NOT_DONE = {
    "walk": " STARTED - you have NOT arrived. Say nothing more about it; "
            "a [NAV] note will tell you when you really get there.",
    "show": " The show has NOT started yet - you are still walking to your "
            "stage. A [NAV] note comes when you get there.",
    "grasp": " STARTED - you are NOT holding it yet. Wait for the "
             "confirmation note before saying you have it.",
    "measure": " STARTED - there is NO result yet. Wait for the "
               "[MEASUREMENT] note; do not guess a number.",
    "rps": " The throw is happening now - the winner is NOT known yet. "
           "Wait for the [RPS RESULT] note.",
}


_guard_ref = {"g": None}


def preopen_camera_guard():
    """Persistent mode: open the RealSense BEFORE the audio streams exist.

    Measured with g1_audio_diag.py: the USB speaker dies the moment the
    RealSense starts, and does not come back. So the camera has to start
    while there is no speaker stream to kill - i.e. before the voice session
    opens audio. Whatever the camera's start does to the USB bus then
    happens once, at startup, with nothing playing. init() reuses this
    guard. Returns True if the camera is open."""
    if not OBSTACLE_GUARD_AVAILABLE:
        return False
    g = ObstacleGuard()
    g.persistent = True
    try:
        g.open()
    except Exception as e:
        print(f"[ROBOT] couldn't pre-open the guard camera: {e}")
        return False
    _guard_ref["g"] = g
    time.sleep(2.0)     # let the USB bus settle before audio comes up
    print("[ROBOT] guard camera pre-opened (before audio)")
    return True


def _camera_release():
    """Persistent-guard mode only: hand the RealSense to a subprocess
    (pickup_glass / measure) that needs it. No-op otherwise."""
    g = _guard_ref["g"]
    if g is not None and getattr(g, "persistent", False):
        g.close(force=True)


def _camera_restore():
    """Persistent-guard mode only: take the RealSense back once nothing else
    is using it."""
    g = _guard_ref["g"]
    if g is None or not getattr(g, "persistent", False):
        return
    if proc_active("grasp") or proc_active("measure"):
        return
    try:
        g.open()
    except Exception as e:
        print(f"[ROBOT] couldn't reopen the guard camera: {e}")


def proc_active(name):
    p = PROCESSES.get(name)
    return p is not None and p.poll() is None


_STOPPED_PIDS = set()    # pids WE terminated - so on_exit can tell a natural
                         # finish (applaud!) from an interrupted one (silence)


def stop_proc(name, timeout=3.0):
    """SIGTERM a tracked subprocess and wait briefly for it to exit. Every
    script this launches re-raises SIGTERM as KeyboardInterrupt so its own
    finally: block (open hand, retract, ramp down) still runs - see each
    script's own signal.signal(SIGTERM, ...) near the top."""
    p = PROCESSES.get(name)
    if p is None or p.poll() is not None:
        PROCESSES.pop(name, None)
        return False
    _STOPPED_PIDS.add(p.pid)
    p.terminate()
    try:
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
    PROCESSES.pop(name, None)
    return True


def start_proc(name, cmd, watch_prefix=None, on_result=None, on_exit=None,
               extra_watchers=None):
    """Launch a tracked subprocess. If watch_prefix is given, a background
    thread reads its stdout line by line and, on a line starting with
    watch_prefix, parses the JSON after it and calls on_result(data) -
    this is how play_rps and measure_object get their answer back to speak,
    and how pickup_glass reports whether centring succeeded.

    extra_watchers - optional {prefix: callback(data)} for further JSON
    status lines from the same process (e.g. RPS_REVEAL). Callbacks run on
    the reader thread, so anything slow must start its own thread.

    on_exit - optional callable(returncode), called once when the process
    ends ON ITS OWN. Not called if stop_proc() ended it (a stop, a new
    request replacing it, shutdown) - that is an interruption, not a finish."""
    if proc_active(name):
        stop_proc(name)
    kwargs = {}
    if watch_prefix or extra_watchers:
        kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                     text=True, bufsize=1)
    proc = subprocess.Popen(cmd, **kwargs)
    PROCESSES[name] = proc

    if watch_prefix or extra_watchers:
        handlers = dict(extra_watchers or {})
        if watch_prefix and on_result:
            handlers[watch_prefix] = on_result

        def _watch():
            for line in proc.stdout:
                print(f"[{name}] {line.rstrip()}")
                for prefix, cb in handlers.items():
                    if line.startswith(prefix):
                        try:
                            data = json.loads(line[len(prefix):].strip())
                        except Exception as e:
                            print(f"[{name}] couldn't parse {prefix.strip()}"
                                  f" line: {e}")
                            break
                        try:
                            cb(data)
                        except Exception as e:
                            print(f"[{name}] {prefix.strip()} handler "
                                  f"failed: {e}")
                        break
        threading.Thread(target=_watch, daemon=True).start()

    if on_exit:
        def _wait():
            rc = proc.wait()
            if proc.pid in _STOPPED_PIDS:
                _STOPPED_PIDS.discard(proc.pid)
                return
            try:
                on_exit(rc)
            except Exception as e:
                print(f"[{name}] on_exit failed: {e}")
        threading.Thread(target=_wait, daemon=True).start()
    return proc


def all_actions():
    return sorted(set(ACTION_TO_SDK) | LOCO_ONLY | NAV_ACTIONS
                  | {"look", "crane_on", "crane_off", "none"}
                  | DEMO_ACTIONS | GRASP_ACTIONS | MEASURE_ACTIONS)


# ============================================================
# INIT
# ============================================================
def init(iface="wlan0", use_nav=True, init_dds=True, use_obstacle_guard=True,
         guard_persistent=False):
    """Bring up DDS and the robot clients. Call AFTER the agent session."""
    global arm_client, loco_client, nav, IFACE
    IFACE = iface

    # Open the camera immediately and hold it. Opening per request cost
    # ~0.5 s of warmup, which was enough to miss the tool deadline.
    if CV2_AVAILABLE:
        _grabber.start()

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

    guard = None
    if use_nav and use_obstacle_guard:
        if OBSTACLE_GUARD_AVAILABLE:
            guard = (_guard_ref["g"]
                     if guard_persistent and _guard_ref["g"] is not None
                     else ObstacleGuard())
            _guard_ref["g"] = guard
            if guard_persistent and not hasattr(guard, "persistent"):
                print("[ROBOT] --guard-mode persistent needs the newer "
                      "g1_obstacle_guard.py - using per-walk")
                guard_persistent = False
            if guard_persistent:
                # Open the camera ONCE now and leave it streaming, instead of
                # starting/stopping it at every walk. For the case where the
                # start/stop transition is what upsets the USB audio dongle.
                guard.persistent = True
                try:
                    guard.open()
                    print("[ROBOT] obstacle guard wired in - PERSISTENT: "
                          "RealSense stays open (released only while "
                          "pickup_glass / measure use it)")
                except Exception as e:
                    guard.persistent = False
                    print(f"[ROBOT] persistent guard couldn't open the "
                          f"camera ({e}) - falling back to per-walk")
            else:
                print("[ROBOT] obstacle guard wired in (opens the RealSense "
                      "only while actually walking)")
        else:
            print(f"[ROBOT] obstacle guard unavailable: "
                  f"{_OBSTACLE_GUARD_ERROR} - navigating without avoidance")

    if use_nav and NAV_AVAILABLE:
        try:
            nav = NavBridge(on_event=_nav_event, loco_client=loco_client,
                            obstacle_guard=guard)
            print(f"[ROBOT] nav ready - {len(nav.places())} place(s): "
                  f"{nav.places()}")
            print(f"[ROBOT] routes: {nav.route_names()}")
            push_state(force=True)
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
    _grabber.stop()
    for name in list(PROCESSES):
        stop_proc(name)
    g = _guard_ref["g"]
    if g is not None:
        try:
            g.close(force=True)
        except TypeError:          # older guard file: close() has no force=
            try:
                g.close()
            except Exception:
                pass
        except Exception:
            pass


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


class _FrameGrabber:
    """Keeps the newest Brio frame permanently warm in a background thread.

    Opening the camera per request costs ~0.5 s of auto-exposure warmup
    before anything usable comes out, and that alone was enough to miss the
    tool deadline. Holding it open means look() pays only for the Claude
    call.
    """

    def __init__(self, warmup=15, fps=5.0):
        self.warmup = warmup
        self.period = 1.0 / fps
        self._frame = None
        self._lock = threading.Lock()
        self._run = False
        self._thread = None
        self._err = None

    def start(self):
        if self._run:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        cap = None
        try:
            device = (f"/dev/video{CAMERA_INDEX}" if CAMERA_INDEX is not None
                      else find_camera_device())
            if device is None:
                self._err = f"no device matching '{CAMERA_NAME}'"
                return
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self._err = f"could not open {device}"
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            # Keep the driver buffer shallow so we read the NEWEST frame
            # rather than a queued stale one.
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            for _ in range(self.warmup):
                cap.read()

            while self._run:
                ok, f = cap.read()
                if ok:
                    with self._lock:
                        self._frame = f
                time.sleep(self.period)
        except Exception as e:
            self._err = str(e)
        finally:
            if cap is not None:
                cap.release()

    def latest(self, wait=2.0):
        deadline = time.time() + wait
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            if self._err:
                raise RuntimeError(self._err)
            time.sleep(0.05)
        raise RuntimeError(self._err or "no frame available")


_grabber = _FrameGrabber()


def capture_camera_frame(warmup_frames=None):
    """Grab the newest frame from the always-open camera and JPEG-encode it.

    No open, no warmup, no release - the grabber holds the device and keeps a
    current frame ready, so this is essentially instant.
    """
    frame = _grabber.latest()

    mean = float(frame.mean())
    if mean < 12:
        raise RuntimeError(f"frame almost black (mean {mean:.1f})")

    # Downscale before sending. A 1280x720 JPEG takes noticeably longer to
    # upload and tokenize than 640x360, and the description is no better.
    h, w = frame.shape[:2]
    if w > VISION_WIDTH:
        scale = VISION_WIDTH / float(w)
        frame = cv2.resize(frame, (VISION_WIDTH, int(h * scale)),
                           interpolation=cv2.INTER_AREA)

    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("utf-8")


def vision_status():
    """Why vision would fail, checked in order. Returns None if healthy."""
    if not CV2_AVAILABLE:
        return "opencv-python is not installed"
    if not ANTHROPIC_AVAILABLE:
        return "the anthropic package is not installed"
    if not os.getenv("ANTHROPIC_API_KEY"):
        return ("ANTHROPIC_API_KEY is not set IN THIS PROCESS - exporting it "
                "in another terminal does not carry over")
    if find_camera_device() is None:
        return f"no video device whose name contains '{CAMERA_NAME}'"
    return None


def describe_view(question=""):
    """ElevenLabs agents cannot accept images, so vision runs on Claude and
    the description is returned as the tool result.

    Uses the BRIO, not the RealSense. The RealSense sits directly above a
    cyan face LED that cannot be switched off or filtered, so everything it
    sees is tinted - white paper reads cyan, red reads dark magenta. Fine for
    depth, useless for describing a scene.
    """
    problem = vision_status()
    if problem:
        print(f"[VISION] unavailable: {problem}")
        return "I can't see anything at the moment - my camera isn't working."
    t0 = time.time()
    try:
        img = capture_camera_frame()
    except Exception as e:
        print(f"[VISION] capture failed: {e}")
        return "I'm not getting a clear picture from my camera right now."
    t_cap = time.time() - t0

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
            model=VISION_MODEL, max_tokens=VISION_MAX_TOKENS,
            extra_body={"output_config": {"effort": "low"}},
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": img}},
                {"type": "text", "text": prompt}]}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        t_total = time.time() - t0
        # Print the split so it is obvious whether a slow answer is the
        # camera or the model. The ElevenLabs tool timeout is what matters.
        print(f"[VISION] {t_cap * 1000:.0f}ms capture + "
              f"{(t_total - t_cap) * 1000:.0f}ms claude = "
              f"{t_total * 1000:.0f}ms")
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
    return proc_active("crane")


def enter_crane():
    if crane_is_active():
        return "I'm already in crane mode."
    print("[MODE] entering crane mode")
    start_proc("crane", ["python3", CRANE_SCRIPT, IFACE])
    return "Switching to crane mode."


def exit_crane():
    if not crane_is_active():
        return "I'm already in normal mode."
    print("[MODE] exiting crane mode")
    stop_proc("crane")
    return "Switching back to normal mode."


# ============================================================
# DEMOS - RPS, GestureSynth
# ============================================================
# Style cues rotate so a run of games doesn't repeat the same joke. They are
# for the agent's eyes (it turns the note into speech - see the prompt's
# OUT-OF-BAND NOTES section), not read out.
_RPS_WIN_CUES = [      # the ROBOT won
    "Gloat shamelessly, in good fun - a robot's victory lap.",
    "Be mock-humble about how effortless that was.",
    "Trash-talk gently - you're clearly the superior intellect.",
    "Claim it was pure strategy, not luck.",
]
_RPS_LOSE_CUES = [     # the HUMAN won
    "Be wildly, theatrically offended, then congratulate them.",
    "Blame your processor, the lighting, anything but yourself.",
    "Demand a rematch, with great dignity.",
    "Accuse them of cheating, jokingly - hands move fast.",
]
_RPS_TIE_CUES = [
    "Declare it suspicious - great minds think alike.",
    "Call it a diplomatic draw and demand a rematch.",
    "Act like you read their mind and it was mutual.",
]


def _rps_note(data):
    """Build the out-of-band note for a finished RPS round. The facts are
    stated plainly and one rotating style cue tells the agent what flavour of
    reaction to give - the funny line itself is the agent's to write."""
    robot = data.get("robot_throw")
    human = data.get("human_throw")
    winner = data.get("winner")
    if data.get("error"):
        return (f"[RPS RESULT] You threw {robot}, but your camera couldn't "
                f"read their hand. Joke that they were too fast, and offer "
                f"another round. One short line.")
    if not data.get("visible"):
        return (f"[RPS RESULT] You threw {robot} but nobody's hand was out "
                f"there. Playfully call them a coward or say you won by "
                f"forfeit. One short line.")
    if winner == "robot":
        outcome, cue = "YOU WON", random.choice(_RPS_WIN_CUES)
    elif winner == "human":
        outcome, cue = "THEY WON", random.choice(_RPS_LOSE_CUES)
    else:
        outcome, cue = "IT'S A TIE", random.choice(_RPS_TIE_CUES)
    return (f"[RPS RESULT] You threw {robot}, they threw {human}. {outcome}. "
            f"{cue} One or two short spoken lines, name both throws.")


_RPS_BEATS = {"rock": "scissors", "scissors": "paper", "paper": "rock"}


def _rps_resolve(human, robot):
    if human == robot:
        return "tie"
    return "robot" if _RPS_BEATS[robot] == human else "human"


def _rps_judge_from_brio(reveal):
    """Runs on its own thread when the RPS program says the sign is up.
    Takes Brio frames from the always-open grabber (the RealSense is never
    used for this), has Opus read the human's hand, and announces the result.
    """
    robot = reveal.get("robot_throw")
    delays = reveal.get("delays") or [0.5, 1.3]
    result = {"robot_throw": robot, "human_throw": None, "winner": None,
              "visible": False, "error": None, "note": ""}
    try:
        from g1_rps_judge import judge_frames
        frames, t0 = [], time.time()
        for d in delays:
            wait = d - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
            frames.append(_grabber.latest(wait=1.0))
        j = judge_frames(frames)
        result["error"] = j["error"]
        result["note"] = j["note"]
        result["visible"] = j["visible"]
        print(f"[RPS] judge: {j}")
        if j["visible"] and not j["error"]:
            result["human_throw"] = j["sign"]
            result["winner"] = _rps_resolve(j["sign"], robot)
    except Exception as e:
        result["error"] = f"judge: {e}"
        print(f"[RPS] judging failed: {e}")
    announce(_rps_note(result))


def play_rps():
    if proc_active("rps"):
        return "I'm already mid-game."

    print("[DEMO] starting rock paper scissors (Brio judging)")
    # --no-vision: the RPS program must not open any camera (this process
    # holds the Brio, and the RealSense is never used). It prints RPS_REVEAL
    # when the sign is up and we judge from our own frames.
    start_proc("rps", ["python3", "-u", RPS_SCRIPT, IFACE, "--no-vision"],
              extra_watchers={
                  "RPS_REVEAL ": lambda d: threading.Thread(
                      target=_rps_judge_from_brio, args=(d,),
                      daemon=True).start()},
              on_exit=lambda rc: push_state())
    push_state()
    return "Let's play - rock, paper, scissors!" + _NOT_DONE["rps"]


_FINALE_CUES = [
    "Take a bow, thank everyone, and ask for a round of applause.",
    "Fish for applause shamelessly - you were magnificent and you know it.",
    "Thank the audience like it was a sold-out world tour, and invite a clap.",
    "Say you'll be here all week and ask them to make some noise.",
]


def _gesturesynth_finished(returncode):
    """Runs when the performance ends on its own (not when interrupted)."""
    push_state()
    if returncode != 0:
        announce("[PERFORMANCE] The finger performance glitched out partway. "
                 "Laugh it off in one short line - artistic differences.")
        return
    announce("[PERFORMANCE] You just finished your finger performance. "
             + random.choice(_FINALE_CUES) + " One or two short lines.")


def _launch_gesturesynth():
    if proc_active("gesturesynth"):
        return
    print("[DEMO] starting GestureSynth")
    start_proc("gesturesynth", ["python3", "-u", GESTURESYNTH_SCRIPT, IFACE],
              on_exit=_gesturesynth_finished)
    push_state()


def play_gesturesynth(destination=None):
    """Walk to the stage waypoint, THEN perform GestureSynth.

    Stage = the `destination` the agent passed if it names a real place, else
    GESTURESYNTH_STAGE. With neither (or no nav), it performs where it stands.
    """
    if proc_active("gesturesynth"):
        return "Already playing."

    key = None
    if nav is not None:
        for cand in (destination, GESTURESYNTH_STAGE):
            if cand:
                key = nav.resolve(cand)
                if key:
                    break

    if key is None:
        if destination or GESTURESYNTH_STAGE:
            print(f"[DEMO] !! GestureSynth stage {destination or GESTURESYNTH_STAGE!r} "
                  f"is not a saved place (saved: "
                  f"{nav.places() if nav is not None else 'none - nav is off'}) "
                  f"- performing IN PLACE")
        else:
            print("[DEMO] !! GESTURESYNTH_STAGE is empty - performing IN "
                  "PLACE. Set it near the top of g1_robot.py.")
        _launch_gesturesynth()
        return "Watch this."

    if nav.is_ready() is False:
        return "I'm not localised on the map yet, so I can't walk to my stage."

    spoken = key.replace("_", " ")

    def _arrived():
        push_state()
        announce(f"[NAV] I'm at {spoken} - showtime.")
        _launch_gesturesynth()

    ok, msg = nav.start(key, then=_arrived)
    if not ok:
        return msg
    return (f"Heading to {spoken} first, then I'll put on a show."
            + _NOT_DONE["show"])


# ============================================================
# GRASP - pickup_glass / release_grasp
# ============================================================
def pickup_glass_action():
    if proc_active("grasp"):
        return "I'm already holding something."

    def _on_result(data):
        if data.get("ok") and data.get("stage") == "holding":
            announce("[GRASP] You are now holding the glass. Say so in one short line.")
        elif not data.get("ok"):
            announce(f"[GRASP] You could NOT get the glass - "
                     f"{data.get('message', 'something went wrong')}. "
                     f"Say so briefly and offer to try again.")

    print("[GRASP] starting pickup_glass")
    _camera_release()
    start_proc("grasp", ["python3", "-u", PICKUP_GLASS_SCRIPT, IFACE],
              watch_prefix="PICKUP_STATUS ", on_result=_on_result,
              on_exit=lambda rc: _camera_restore())
    return "Let me get that." + _NOT_DONE["grasp"]


def release_grasp():
    if not proc_active("grasp"):
        return "I'm not holding anything."
    print("[GRASP] releasing")
    stop_proc("grasp")
    _camera_restore()
    return "There you go."


# ============================================================
# MEASURE
# ============================================================
def measure_object(what=None):
    if proc_active("measure"):
        return "Already measuring something."

    def _on_result(data):
        w, h = data.get("width_cm"), data.get("height_cm")
        what_text = data.get("what", "it")
        msg = f"{what_text} is about {w} by {h} centimetres."
        if data.get("thickness_is_lower_bound"):
            msg += " I'm only seeing it face-on, so depth is a rough lower bound."
        if "grams" in data:
            msg += f" I'd guess around {data['grams']} grams."
        announce(f"[MEASUREMENT] {msg} State it plainly, in one or two lines.")

    cmd = ["python3", "-u", MEASURE_SCRIPT, "--json", "--weight"]
    if what:
        cmd += ["--what", what]
    print(f"[MEASURE] measuring {what or '(whatever is in front)'}")
    _camera_release()
    start_proc("measure", cmd, watch_prefix="MEASURE_RESULT ",
              on_result=_on_result, on_exit=lambda rc: _camera_restore())
    return ("Let me take a measurement - one second."
            + _NOT_DONE["measure"])


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

    # Safety net for a mis-picked action. If the agent attaches a destination
    # that is a real saved place but chose a small-step movement (move_forward
    # etc.) - which happens when the dashboard's action enum has no
    # "navigate" to choose, so the model grabs the closest thing - go to the
    # place instead of shuffling 0.6 m forward. A real small step carries no
    # destination, so this never changes a genuine "step forward".
    if (destination and nav is not None
            and action in LOCO_ACTIONS and action != "stop"):
        try:
            matched = nav.resolve(destination)
        except Exception:
            matched = None
        if matched:
            print(f"[ROBOT] action {action!r} came with destination "
                  f"{destination!r} (-> {matched}); treating as navigate")
            action = "navigate"

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

    if action == "play_rps":
        return play_rps()
    if action == "play_gesturesynth":
        return play_gesturesynth(destination)

    if action == "pickup_glass":
        return pickup_glass_action()
    if action == "release_grasp":
        return release_grasp()

    if action == "measure_object":
        return measure_object(destination)

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
            push_state()
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
            ok, msg = nav.start(destination)
            push_state()
            return msg + (_NOT_DONE["walk"] if ok else "")

        if action == "run_route":
            if nav.is_ready() is False:
                return "I'm not localised on the map yet."
            ok, msg = nav.start_route(destination or "tour")
            push_state()
            return msg + (_NOT_DONE["walk"] if ok else "")

    return f"I don't know how to {action}."


# ============================================================
# INTENT FALLBACK - when the agent SAYS it will walk but never calls the tool
# ============================================================
_NUMWORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
             "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def _norm_place(text):
    t = (text or "").lower().replace("centre", "center")
    for w, d in _NUMWORDS.items():
        t = re.sub(rf"\b{w}\b", d, t)
    t = re.sub(r"[^a-z0-9]", "", t)
    return re.sub(r"latest", "", t)


def _match_spoken_place(text):
    """Map a SPOKEN place ("screen one", "go home") to a waypoint key, or None.
    Longest saved-place name contained in the sentence wins, so "homefront"
    beats "home" when both appear."""
    if nav is None:
        return None
    phrase = _norm_place(text)
    best, best_len = None, 0
    for key in nav.places():
        kn = _norm_place(key)
        if kn and kn in phrase and len(kn) > best_len:
            best, best_len = key, len(kn)
    return best


_GO_RE = re.compile(r"\b(go|walk|head|take me|bring me|navigate|come|move|"
                    r"drive|run)\b")
_NOT_GO_RE = re.compile(r"\b(where|how far|what|which|why|who|when)\b")
_STOP_RE = re.compile(r"^\W*(?:hey\W+)?(?:tony\W+)?(?:please\W+)?"
                      r"(stop|halt|freeze|wait)\b")


def intent_fallback(user_text):
    """Deterministic backstop for the agent talking without acting.

    The model sometimes answers "on my way to screen one" and never calls
    humanoid_output - so nothing happens. This reads the USER's words and
    returns (action, destination) for the clear-cut cases only:
      - "go / take me / walk / head / come ... <saved place>"  -> navigate
      - "stop / halt / freeze / wait" while a walk is in progress -> stop
    Returns None for anything ambiguous, and never fires on questions
    ("where is screen one?")."""
    if nav is None or not user_text:
        return None
    text = user_text.lower()
    negated = re.search(r"\b(don'?t|do not|never|not)\b", text) is not None
    wants_go = (not negated and _GO_RE.search(text) is not None
                and _NOT_GO_RE.search(text) is None)
    key = _match_spoken_place(user_text) if wants_go else None
    if key:
        return ("navigate", key)
    if _STOP_RE.match(text) and nav.is_busy():
        return ("stop_navigation", None)
    return None
