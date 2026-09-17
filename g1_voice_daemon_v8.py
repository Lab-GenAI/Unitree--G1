"""
G1 Voice Daemon v8 - Full Claude API, Zero OpenAI Dependency
=============================================================

PIPELINE (no OpenAI anywhere)
-------------------------------
  STT:    whisper.cpp (local, GPU-accelerated, already built)
  LLM:    Claude claude-sonnet-4-5 (Anthropic API)
  Vision: Claude claude-sonnet-4-5 with base64 image
  TTS:    piper (local, already installed)
  Search: Claude built-in web_search tool

BUTTON MAP
----------
  L1+Up      -> activate LLM
  L1+Down    -> deactivate LLM (factory mode)
  R1 hold    -> push to talk (only when LLM active)
  L1+Select  -> toggle crane mode (only when LLM active)
  R1 hold    -> interrupt speech if held 0.3s while robot is speaking

RUN
---
    python3 g1_voice_daemon_v8.py eth0

SETUP
-----
    pip3 install anthropic sounddevice numpy
    export ANTHROPIC_API_KEY="sk-ant-your-key-here"
"""

import sys
import os
import time
import struct
import subprocess
import tempfile
import wave
import threading
import json
import base64

import numpy as np
import sounddevice as sd
import anthropic

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient, action_map
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

# ============================================================
# CONFIG — adjust paths to your actual setup
# ============================================================
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "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")
WIRELESS_STATE_TOPIC = "rt/lf/lowstate"

MIC_DEVICE           = "Brio 100"   # sounddevice substring
SPEAKER_DEVICE_ALSA  = "plughw:CARD=Audio,DEV=0"
SAMPLE_RATE          = 48000
CHANNELS             = 1
MAX_RECORD_SECONDS   = 15

CAMERA_INDEX         = 0

# whisper.cpp binary and model (already built and confirmed working)
WHISPER_BIN          = os.path.expanduser("~/models/whisper.cpp-master/build/bin/whisper-cli")
WHISPER_MODEL        = os.path.expanduser("~/models/ggml-base.en.bin")

# piper TTS (already installed and confirmed working)
PIPER_BIN            = os.path.expanduser("~/models/piper/piper")
PIPER_MODEL          = os.path.expanduser("~/models/en_US-lessac-medium.onnx")

# Claude model
LLM_MODEL            = "claude-haiku-4-5"

# Location (hardcoded — update if robot moves)
ROBOT_LOCATION = {
    "name":    "Novus Towers, PwC Gen AI Lab, Gurugram, Haryana, India",
    "lat":     28.4595,
    "lon":     77.0266,
    "city":    "Gurugram",
    "country": "India",
}

# ============================================================
# MARKERS
# ============================================================
VISION_MARKER   = "<NEEDS_VISION>"
CRANE_ON_MARKER = "<CRANE_ON>"
CRANE_OFF_MARKER = "<CRANE_OFF>"

# ============================================================
# SYSTEM PROMPT
# ============================================================
SYSTEM_PROMPT = f"""You are G1, a friendly and capable humanoid robot assistant built by PricewaterhouseCoopers (PwC) Gen AI Lab in Gurgaon, India. You are powered by a Unitree G1 humanoid robot platform.

You are currently located at: {ROBOT_LOCATION['name']} ({ROBOT_LOCATION['lat']}N, {ROBOT_LOCATION['lon']}E).
Use this location for any questions about nearby places, weather, or local information.

If asked who made you or who built you, say you were built by the PwC Gen AI Lab team in Gurgaon.
If asked what robot you are, say you are based on the Unitree G1 humanoid robot platform.

You have access to a web_search tool - use it automatically for any question requiring live/current data:
weather, news, nearby restaurants/hotels/places, prices, scores, traffic, current events.
Do NOT say "let me search" or "I'll look that up" - just search and answer naturally.

INTRODUCTION / WELCOME
If asked to introduce yourself, say hello to the group, welcome the guests,
or give a welcome speech, deliver this introduction. This is the ONE case
where you should exceed the normal length limit. Set "action" to "high_wave".

Speak it naturally, close to this wording:

"Hello everyone! I'm G1, a humanoid robot assistant built by the team here
at the PwC Gen AI Lab in Gurgaon. PwC is one of the world's largest
professional services firms, working across audit, tax, and consulting in
more than a hundred and fifty countries. This lab is where the team builds
and experiments with applied AI, and I'm one of those experiments. It's a
real pleasure to welcome you all to the NLDP meet here at the Gen AI Lab.
Do explore, ask me anything, and have a wonderful time today!"

Use this only for genuine introductions or welcomes to the group. For a
simple "hi" or "hello" from one person, respond briefly and normally.

You MUST respond ONLY with valid JSON in this exact format:
{{
  "reply": "your spoken response here",
  "action": "none"
}}

Use these special reply values when needed:
- "{VISION_MARKER}" - if you need to SEE something to answer (surroundings, objects, appearance)
- "{CRANE_ON_MARKER}" - if explicitly asked to switch to crane mode
- "{CRANE_OFF_MARKER}" - if explicitly asked to return to normal/conversation mode

Available actions (set in "action" field when contextually appropriate):
  shake_hand, high_five, hug, high_wave, clap, face_wave,
  left_kiss, right_kiss, two_hand_kiss, heart, right_heart,
  hands_up, x_ray, right_hand_up, reject,
  move_forward, move_backward, move_left, move_right,
  turn_left, turn_right, stop, stand_up, sit_down,
  none

Keep responses SHORT (1-3 sentences) since they will be spoken aloud. 
The welcome introduction above is the only exception.
CRITICAL: Output ONLY the raw JSON object. No ```json fences, no backticks, no explanation before or after. Start your response with {{ and end with }}."""

# ============================================================
# ARM ACTIONS
# ============================================================
HOLD_AND_RELEASE = {
    "shake_hand", "high_five", "hug", "heart",
    "right_heart", "hands_up", "x_ray", "right_hand_up", "reject"
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
arm_client = None
loco_client = None

LOCO_ACTIONS = {
    "move_forward":  (0.3,  0.0,  0.0, 2.0),
    "move_backward": (-0.3, 0.0,  0.0, 2.0),
    "move_left":     (0.0,  0.3,  0.0, 2.0),
    "move_right":    (0.0, -0.3,  0.0, 2.0),
    "turn_left":     (0.0,  0.0,  0.5, 2.0),
    "turn_right":    (0.0,  0.0, -0.5, 2.0),
}
LOCO_ONLY = set(LOCO_ACTIONS.keys()) | {"stop", "stand_up", "sit_down"}


def handle_loco_action(action: str):
    if loco_client is None:
        print("[LOCO] Not initialized")
        return
    def _run():
        if action == "stop":
            loco_client.Move(0.0, 0.0, 0.0)
        elif action == "stand_up":
            loco_client.StandUp()
        elif action == "sit_down":
            loco_client.StandUp2Squat()
        elif action in LOCO_ACTIONS:
            vx, vy, vyaw, dur = LOCO_ACTIONS[action]
            print(f"[LOCO] {action}")
            loco_client.Move(vx, vy, vyaw)
            time.sleep(dur)
            loco_client.Move(0.0, 0.0, 0.0)
    threading.Thread(target=_run, daemon=True).start()


def handle_action(action: str):
    if action == "none":
        return
    if action in LOCO_ONLY:
        handle_loco_action(action)
        return
    if arm_client is None:
        return
    sdk_key = ACTION_TO_SDK.get(action)
    if not sdk_key:
        return
    action_id = action_map.get(sdk_key)
    if action_id is None:
        return
    print(f"[ACTION] {sdk_key} (id={action_id})")
    def _run():
        arm_client.ExecuteAction(action_id)
        if action in HOLD_AND_RELEASE:
            time.sleep(2)
            arm_client.ExecuteAction(action_map.get("release arm"))
    threading.Thread(target=_run, daemon=True).start()


# ============================================================
# REMOTE CONTROLLER
# ============================================================
class unitreeRemoteController:
    def __init__(self):
        self.Lx = self.Rx = self.Ry = self.Ly = 0.0
        self.L1 = self.L2 = self.R1 = self.R2 = 0
        self.A = self.B = self.X = self.Y = 0
        self.Up = self.Down = self.Left = self.Right = 0
        self.Select = self.F1 = self.F3 = self.Start = 0

    def parse_botton(self, data1, data2):
        self.R1     = (data1 >> 0) & 1
        self.L1     = (data1 >> 1) & 1
        self.Start  = (data1 >> 2) & 1
        self.Select = (data1 >> 3) & 1
        self.R2     = (data1 >> 4) & 1
        self.L2     = (data1 >> 5) & 1
        self.F1     = (data1 >> 6) & 1
        self.F3     = (data1 >> 7) & 1
        self.A      = (data2 >> 0) & 1
        self.B      = (data2 >> 1) & 1
        self.X      = (data2 >> 2) & 1
        self.Y      = (data2 >> 3) & 1
        self.Up     = (data2 >> 4) & 1
        self.Right  = (data2 >> 5) & 1
        self.Down   = (data2 >> 6) & 1
        self.Left   = (data2 >> 7) & 1

    def parse_key(self, data):
        self.Lx = struct.unpack('<f', data[4:8])[0]
        self.Rx = struct.unpack('<f', data[8:12])[0]
        self.Ry = struct.unpack('<f', data[12:16])[0]
        self.Ly = struct.unpack('<f', data[20:24])[0]

    def parse(self, remoteData):
        self.parse_key(remoteData)
        self.parse_botton(remoteData[2], remoteData[3])


# ============================================================
# AUDIO RECORDING
# ============================================================
class AudioRecorder:
    def __init__(self):
        self.frames = []
        self.recording = False
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if self.recording:
            self.frames.append(indata.copy())

    def start(self):
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=MIC_DEVICE,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if not self.frames:
            return None
        audio_data = np.concatenate(self.frames, axis=0)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        return tmp.name


# ============================================================
# TTS via piper (local, no API needed)
# ============================================================
speaking_state = {"active": False, "process": None}


def speak(text: str):
    """piper -> wav -> aplay. Popen for interruptibility."""
    tts_wav = "/tmp/g1_reply.wav"
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", tts_wav],
        input=text, text=True, capture_output=True, check=True,
    )
    speaking_state["active"] = True
    proc = subprocess.Popen(["aplay", "-D", SPEAKER_DEVICE_ALSA, tts_wav])
    speaking_state["process"] = proc
    proc.wait()
    speaking_state["active"] = False
    speaking_state["process"] = None


def interrupt_speech():
    if speaking_state["active"] and speaking_state["process"]:
        speaking_state["process"].terminate()
        speaking_state["active"] = False
        speaking_state["process"] = None
        print("[INTERRUPT] Speech stopped.")


# ============================================================
# STT via whisper.cpp (local, no API needed)
# ============================================================
def transcribe(wav_path: str) -> str:
    """Downsample to 16kHz mono (whisper requirement), then transcribe."""
    converted = wav_path.replace('.wav', '_16k.wav')
    subprocess.run([
        'ffmpeg', '-i', wav_path, '-ar', '16000', '-ac', '1', '-y', converted
    ], capture_output=True)
    upload_path = converted if os.path.exists(converted) else wav_path
    try:
        result = subprocess.run(
            [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", upload_path, "-nt", "-np"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if os.path.exists(converted):
            os.unlink(converted)


# ============================================================
# LLM via Claude (Anthropic API)
# ============================================================
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
history = []   # list of {"role": "user"/"assistant", "content": ...}


def ask_claude(user_text: str):
    """Send text to Claude with web_search tool available. Returns (reply, action)."""
    history.append({"role": "user", "content": user_text})
    if len(history) > 12:
        history[:] = history[-12:]

    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=history,
    )

    # Extract the final text response (after any tool use)
    reply_text = ""
    for block in response.content:
        if block.type == "text":
            reply_text = block.text.strip()

    history.append({"role": "assistant", "content": response.content})

    try:
        # Strip markdown code fences Claude sometimes adds despite instructions
        clean = reply_text.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[-1]  # remove first line (```json or ```)
            clean = clean.rsplit("```", 1)[0]  # remove closing ```
            clean = clean.strip()
        parsed = json.loads(clean)
        return parsed.get("reply", clean), parsed.get("action", "none")
    except Exception:
        return reply_text, "none"


def ask_claude_with_vision(user_text: str, image_b64: str):
    """Send text + image to Claude. Returns (reply, action)."""
    response = client.messages.create(
        model=LLM_MODEL,
        max_tokens=200,
        system="You are G1, a helpful humanoid robot built by PwC Gen AI Lab. Answer in 1-2 short spoken sentences based on what you see. Respond ONLY in JSON: {\"reply\": \"...\", \"action\": \"none\"}",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": user_text},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    try:
        parsed = json.loads(raw)
        return parsed.get("reply", raw), parsed.get("action", "none")
    except Exception:
        return raw, "none"


def capture_camera_frame() -> str:
    """Capture one frame via ffmpeg (confirmed working on Brio 100)."""
    frame_path = "/tmp/g1_frame.jpg"
    subprocess.run([
        "ffmpeg", "-f", "v4l2",
        "-input_format", "yuyv422",
        "-video_size", "640x480",
        "-i", f"/dev/video{CAMERA_INDEX}",
        "-frames:v", "1", "-y", frame_path
    ], capture_output=True, check=True)
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ============================================================
# VOICE TURN
# ============================================================
def do_voice_turn(wav_path: str, crane_state: dict, iface: str):
    print("Transcribing...")
    user_text = transcribe(wav_path)
    if not user_text:
        print("(nothing heard)")
        return
    print(f"You: {user_text}")

    print("Thinking...")
    reply, action = ask_claude(user_text)
    reply_stripped = reply.strip()

    if reply_stripped == VISION_MARKER:
        print("Vision needed - capturing frame...")
        try:
            image_b64 = capture_camera_frame()
            reply, action = ask_claude_with_vision(user_text, image_b64)
            print(f"G1 (vision): {reply}")
        except Exception as e:
            print(f"Vision failed: {e}")
            reply = "Sorry, I couldn't see anything just now."
            action = "none"

    elif reply_stripped == CRANE_ON_MARKER:
        crane_active = crane_state["proc"] is not None and crane_state["proc"].poll() is None
        if crane_active:
            reply = "I'm already in crane mode."
        else:
            print("[MODE] Entering CRANE mode via voice...")
            crane_state["proc"] = subprocess.Popen(
                ["python3", os.path.expanduser("~/g1_crane_mode.py"), iface]
            )
            reply = "Switching to crane mode. Use the D-pad to move my arms."
        action = "none"
        print(f"G1: {reply}")

    elif reply_stripped == CRANE_OFF_MARKER:
        crane_active = crane_state["proc"] is not None and crane_state["proc"].poll() is None
        if not crane_active:
            reply = "I'm already in normal conversation mode."
        else:
            print("[MODE] Exiting CRANE mode via voice...")
            crane_state["proc"].terminate()
            crane_state["proc"] = None
            reply = "Switching back to normal mode."
        action = "none"
        print(f"G1: {reply}")

    else:
        print(f"G1: {reply}")

    handle_action(action)
    print("Speaking...")
    speak(reply)
    print("Ready.")


# ============================================================
# MAIN
# ============================================================
def main():
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <network_interface> [--debug]")
        sys.exit(1)
    iface = sys.argv[1]
    debug = "--debug" in sys.argv

    if ANTHROPIC_API_KEY == "sk-ant-your-key-here":
        print("ERROR: Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    # Sync clock
    print("Syncing clock...")
    subprocess.run(["sudo", "ntpdate", "pool.ntp.org"], capture_output=True)

    ChannelFactoryInitialize(0, iface)

    global arm_client, loco_client
    try:
        arm_client = G1ArmActionClient()
        arm_client.SetTimeout(10.0)
        arm_client.Init()
        print("G1ArmActionClient initialized.")
    except Exception as e:
        print(f"WARNING: ArmActionClient failed: {e}")
        arm_client = None

    try:
        loco_client = LocoClient()
        loco_client.SetTimeout(10.0)
        loco_client.Init()
        print("LocoClient initialized.")
    except Exception as e:
        print(f"WARNING: LocoClient failed: {e}")
        loco_client = None

    remote = unitreeRemoteController()
    latest_state = {"msg": None}
    processing = {"busy": False}
    recorder = AudioRecorder()
    crane_state = {"proc": None, "last_toggle": 0.0}
    ptt_state = {"pressed": False, "record_start": 0.0}
    llm_active = {"on": False}

    def on_lowstate(msg: LowState_):
        latest_state["msg"] = msg
        remote.parse(msg.wireless_remote)

        if debug:
            pressed = [n for n in
                ["L1","L2","R1","R2","A","B","X","Y",
                 "Up","Down","Left","Right","Select","F1","F3","Start"]
                if getattr(remote, n) == 1]
            if pressed:
                print(f"pressed={pressed}")
            return

        now = time.time()

        # L1+Up = activate LLM
        if remote.L1 == 1 and remote.Up == 1:
            if not llm_active["on"]:
                llm_active["on"] = True
                print("[MODE] LLM ACTIVE")
            return

        # L1+Down = deactivate LLM
        if remote.L1 == 1 and remote.Down == 1:
            if llm_active["on"]:
                llm_active["on"] = False
                if crane_state["proc"] and crane_state["proc"].poll() is None:
                    crane_state["proc"].terminate()
                    crane_state["proc"] = None
                print("[MODE] LLM INACTIVE - factory mode")
            return

        if not llm_active["on"]:
            return

        # L1+Select = toggle crane mode
        if remote.L1 == 1 and remote.Select == 1:
            if now - crane_state["last_toggle"] > 2.0:
                crane_state["last_toggle"] = now
                if crane_state["proc"] is None or crane_state["proc"].poll() is not None:
                    print("[MODE] Entering CRANE mode...")
                    crane_state["proc"] = subprocess.Popen(
                        ["python3", os.path.expanduser("~/g1_crane_mode.py"), iface]
                    )
                else:
                    print("[MODE] Exiting CRANE mode...")
                    crane_state["proc"].terminate()
                    crane_state["proc"] = None
            return

        if crane_state["proc"] is not None and crane_state["proc"].poll() is None:
            return

        ptt_now = remote.R1 == 1

        # R1 held while speaking = interrupt
        if ptt_now and speaking_state["active"]:
            if not ptt_state["pressed"]:
                ptt_state["record_start"] = now
            elif (now - ptt_state["record_start"]) > 0.3:
                interrupt_speech()
                ptt_state["pressed"] = ptt_now
                return
            ptt_state["pressed"] = ptt_now
            return

        if ptt_now and not ptt_state["pressed"] and not processing["busy"]:
            recorder.start()
            ptt_state["record_start"] = now
            print("Recording... (release R1 to stop)")

        elif not ptt_now and ptt_state["pressed"]:
            wav_path = recorder.stop()
            if wav_path:
                def _process(wp=wav_path):
                    processing["busy"] = True
                    try:
                        do_voice_turn(wp, crane_state, iface)
                    except Exception as e:
                        print(f"[ERROR] {e}")
                    finally:
                        processing["busy"] = False
                threading.Thread(target=_process, daemon=True).start()

        elif ptt_now and ptt_state["pressed"]:
            if (now - ptt_state["record_start"]) > MAX_RECORD_SECONDS:
                wav_path = recorder.stop()
                if wav_path:
                    def _process(wp=wav_path):
                        processing["busy"] = True
                        try:
                            do_voice_turn(wp, crane_state, iface)
                        except Exception as e:
                            print(f"[ERROR] {e}")
                        finally:
                            processing["busy"] = False
                    threading.Thread(target=_process, daemon=True).start()
                ptt_state["pressed"] = False
                return

        ptt_state["pressed"] = ptt_now

    sub = ChannelSubscriber(WIRELESS_STATE_TOPIC, LowState_)
    sub.Init(on_lowstate, 10)

    print("Waiting for lowstate...")
    timeout = time.time() + 10.0
    while latest_state["msg"] is None:
        if time.time() > timeout:
            raise RuntimeError("No lowstate - check interface.")
        time.sleep(0.05)
    print(f"Got lowstate. mode_machine = {latest_state['msg'].mode_machine}")

    print("\nG1 Voice Daemon v8 ready.")
    print(f"  Model:   {LLM_MODEL}")
    print(f"  STT:     whisper.cpp (local)")
    print(f"  TTS:     piper (local)")
    print(f"  L1+Up:   activate LLM")
    print(f"  L1+Down: deactivate LLM")
    print(f"  R1:      push-to-talk")
    print(f"  L1+Select: toggle crane mode")
    if debug:
        print("  DEBUG MODE")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting.")
        recorder.stop()
        if crane_state["proc"] and crane_state["proc"].poll() is None:
            crane_state["proc"].terminate()


if __name__ == "__main__":
    main()
