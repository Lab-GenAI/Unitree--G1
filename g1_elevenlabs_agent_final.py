#!/usr/bin/env python3
"""
G1 ElevenLabs Agent - standalone pipeline test
===============================================

Minimal test of an ElevenLabs Conversational AI agent using specific audio
hardware rather than the system defaults:

    INPUT  : Brio 100 webcam's built-in microphone
    OUTPUT : AB13X USB Audio dongle

Nothing here touches the robot. No DDS, no unitree_sdk2py, no motion. It is
purely: mic -> ElevenLabs agent -> speaker.

WHY A CUSTOM AudioInterface
---------------------------
The SDK's DefaultAudioInterface uses pyaudio and the system DEFAULT devices.
Here input and output are two DIFFERENT USB devices, neither necessarily the
default, and their ALSA card indices have already been observed to move
between reboots. This implementation uses sounddevice (already installed) and
matches each device by NAME, so it survives renumbering.

SETUP
-----
    pip3 install elevenlabs sounddevice numpy

    # Rotate the key that was pasted in plaintext, then:
    export ELEVENLABS_API_KEY="sk-your-new-key"
    export ELEVENLABS_AGENT_ID="agent_2401m209dgv9eze99h2yx74q24dp"

USAGE
-----
    python3 g1_elevenlabs_agent.py --list-devices     # see what's available
    python3 g1_elevenlabs_agent.py                    # push-to-talk (default)
    python3 g1_elevenlabs_agent.py --always-on        # open mic
    python3 g1_elevenlabs_agent.py --input-device 3   # force a device index

PUSH TO TALK
------------
Default mode. Press ENTER to open the mic, speak, press ENTER again to close
it. Repeat as often as you like. 'q' then ENTER quits.

A terminal cannot detect key RELEASE, so this is a toggle rather than a held
button. While the gate is closed the agent is fed silence rather than nothing
at all - that keeps the websocket and the agent's turn detection stable.
Opening the mic also cancels any in-progress agent speech, so you can talk
over it.

    Ctrl-C to end the session.

AUDIO FORMAT
------------
ElevenLabs conversational AI uses 16-bit PCM mono. This script keeps the USB
hardware at 48 kHz but explicitly resamples microphone audio to 16 kHz before
sending it to ElevenLabs, then resamples ElevenLabs 16-kHz output back to
48 kHz for the USB speaker.
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time

import numpy as np
import sounddevice as sd

try:
    from elevenlabs.client import ElevenLabs
    import json
    from elevenlabs.conversational_ai.conversation import (
        Conversation, AudioInterface,ClientTools,)
except ImportError:
    print("ERROR: pip3 install elevenlabs")
    sys.exit(1)

try:
    import g1_robot
    ROBOT_AVAILABLE = True
except Exception as _e:
    ROBOT_AVAILABLE = False
    _ROBOT_ERROR = str(_e)


# ============================================================
# CONFIG
# ============================================================
MIC_NAME_MATCH     = "Brio"    # Brio 100 webcam's built-in microphone
SPEAKER_NAME_MATCH = "AB13X"   # AB13X USB Audio dongle
INPUT_RATE          = 48000  # Brio hardware capture rate
OUTPUT_RATE         = 48000  # USB speaker hardware playback rate
ELEVENLABS_RATE     = 16000  # PCM rate sent to / received from ElevenLabs
CHUNK_FRAMES        = 960    # 20 ms at 48 kHz; becomes 320 samples at 16 kHz


def find_device(name_substr, kind="input", retries=10, delay=1.0):
    """Locate a device by name substring, retrying while USB enumerates.

    The daemon crashed on startup once because the dongle had not enumerated
    yet when it looked. Retrying rides through that.
    """
    want_in = kind == "input"
    for attempt in range(retries):
        for idx, dev in enumerate(sd.query_devices()):
            chans = dev["max_input_channels"] if want_in \
                else dev["max_output_channels"]
            if name_substr.lower() in dev["name"].lower() and chans > 0:
                return idx, dev["name"]
        if attempt == 0:
            print(f"  waiting for '{name_substr}' {kind} device...")
        time.sleep(delay)
    return None, None


def list_devices():
    print(f"{'idx':<5} {'in':<4} {'out':<4} name")
    print("-" * 70)
    for i, d in enumerate(sd.query_devices()):
        print(f"{i:<5} {d['max_input_channels']:<4} "
              f"{d['max_output_channels']:<4} {d['name']}")
    print("\nDefaults:", sd.default.device)


# ============================================================
# AUDIO INTERFACE
# ============================================================
class SoundDeviceAudioInterface(AudioInterface):
    """AudioInterface backed by sounddevice, pinned to specific devices.

    Contract expected by the SDK:
      start(input_callback)  begin capture; call input_callback(bytes) with
                             16-bit PCM mono chunks
      stop()                 tear everything down
      output(audio)          play a chunk of 16-bit PCM mono
      interrupt()            drop queued audio immediately (barge-in)
    """

    def __init__(self, input_device=None, output_device=None,
                 input_rate=INPUT_RATE, output_rate=OUTPUT_RATE,
                 agent_rate=ELEVENLABS_RATE,
                 verbose=False, push_to_talk=True, echo_guard=False,
                 duck_factor=0.08, input_name=None, output_name=None,
                 watchdog=True):
        self.input_device = input_device
        self.output_device = output_device
        # Device NAME substrings, used by the watchdog to find the devices
        # again if a USB audio device drops off the bus and comes back with a
        # different index. None = device was forced by index, don't re-resolve.
        self.input_name = input_name
        self.output_name = output_name
        self._last_in_cb = time.time()
        self._last_out_cb = time.time()
        self._wd_run = False
        self.watchdog = watchdog
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.agent_rate = agent_rate
        self.verbose = verbose

        # Push-to-talk gate. When closed we still feed the agent audio, but
        # SILENCE rather than mic input - that keeps the websocket stream and
        # the agent's voice-activity detection in a stable state. Simply not
        # calling the callback tends to confuse turn detection.
        self.push_to_talk = push_to_talk
        self.mic_open = threading.Event()
        if not push_to_talk:
            self.mic_open.set()

        # ECHO GUARD - OFF by default, and deliberately so.
        #
        # Muting the mic while the robot speaks stops it hearing itself, but
        # it is half duplex: you cannot interrupt by voice either, because
        # the mic is deaf for as long as the robot is talking. Barge-in
        # matters more here. The wake word does the filtering instead - the
        # agent only acts on speech that starts with it, so hearing its own
        # voice is mostly harmless.
        #
        # Turn it on with --echo-guard if the speaker is loud enough that the
        # agent triggers on itself.
        self.echo_guard = echo_guard
        self._speaking_until = 0.0

        # DUCKING
        # PulseAudio's module-echo-cancel did not work here - two USB devices
        # on independent clocks defeated it (measured rms 725 on a silent
        # capture, i.e. no cancellation at all). So instead of subtracting
        # the speaker signal, attenuate the mic hard while the robot talks.
        #
        # The robot's own voice arriving at the mic drops below the agent's
        # VAD threshold and never triggers. A person speaking is far louder
        # at the mic than the speaker bleed, so they still get through.
        # Interruption survives - unlike a hard gate.
        #
        # This is a level trick, not cancellation. If the speaker is loud and
        # the speaker is far from the mic, bleed and speech converge and no
        # setting separates them. A directional mic is the real fix then.
        self.duck_factor = duck_factor
        self._meter_enabled = False
        self._meter_last = 0.0

        self._in_stream = None
        self._out_stream = None
        self._out_q = queue.Queue()
        self._residual = b""
        self._lock = threading.Lock()
        self._input_callback = None

    # ---------- low-latency resampling ----------
    def _mic_to_agent_rate(self, pcm_bytes):
        """Convert mono int16 mic PCM from hardware rate to ElevenLabs rate.

        The default path is exactly 48 kHz -> 16 kHz (3:1).  We use a
        lightweight 3-sample box low-pass before decimation.  This is fast,
        state-free, and well suited to speech/VAD.
        """
        if self.input_rate == self.agent_rate:
            return pcm_bytes
        if self.input_rate == 48000 and self.agent_rate == 16000:
            x = np.frombuffer(pcm_bytes, dtype=np.int16)
            usable = (len(x) // 3) * 3
            if usable == 0:
                return b""
            x = x[:usable].astype(np.int32).reshape(-1, 3)
            y = np.rint(x.mean(axis=1)).clip(-32768, 32767).astype(np.int16)
            return y.tobytes()
        raise ValueError(
            f"Unsupported input resample {self.input_rate} -> {self.agent_rate}. "
            "Use 48000->16000 or matching rates."
        )

    def _agent_to_speaker_rate(self, pcm_bytes):
        """Convert mono int16 ElevenLabs PCM to the speaker hardware rate.

        For 16 kHz -> 48 kHz, linear interpolation avoids the chipmunk-speed
        playback that would occur if 16-kHz samples were sent directly to a
        48-kHz device.
        """
        if self.agent_rate == self.output_rate:
            return pcm_bytes
        if self.agent_rate == 16000 and self.output_rate == 48000:
            x = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(x) == 0:
                return b""
            if len(x) == 1:
                return np.repeat(x, 3).astype(np.int16).tobytes()
            src = np.arange(len(x), dtype=np.float32)
            dst = np.arange(len(x) * 3, dtype=np.float32) / 3.0
            y = np.interp(dst, src, x.astype(np.float32))
            return np.rint(y).clip(-32768, 32767).astype(np.int16).tobytes()
        raise ValueError(
            f"Unsupported output resample {self.agent_rate} -> {self.output_rate}. "
            "Use 16000->48000 or matching rates."
        )

    # ---------- push to talk ----------
    def _duck(self, chunk):
        """Attenuate mic audio while the robot is speaking."""
        if self.push_to_talk or self.duck_factor >= 1.0:
            return chunk
        if not self.is_speaking():
            return chunk
        a = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        a *= self.duck_factor
        return a.astype(np.int16).tobytes()

    def _meter(self, chunk):
        """Print mic level so ducking can be tuned against real numbers."""
        if not self._meter_enabled:
            return
        now = time.time()
        if now - self._meter_last < 0.25:
            return
        self._meter_last = now
        a = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt((a * a).mean())) if a.size else 0.0
        bar = "#" * min(40, int(rms / 100))
        tag = "SPEAKING" if self.is_speaking() else "        "
        print(f"  [mic {tag}] rms={rms:7.0f} {bar}", flush=True)

    def is_speaking(self):
        """True while the robot still has audio to play, plus a short tail so
        the speaker's decay does not get transcribed."""
        return (not self._out_q.empty()) or time.time() < self._speaking_until

    def open_mic(self):
        self.mic_open.set()

    def close_mic(self):
        self.mic_open.clear()

    def toggle_mic(self):
        if self.mic_open.is_set():
            self.close_mic()
            return False
        self.open_mic()
        return True

    # ---------- capture ----------
    def _on_audio(self, indata, frames, time_info, status):
        self._last_in_cb = time.time()
        if status and self.verbose:
            print(f"[input status] {status}", file=sys.stderr)
        if not self._input_callback:
            return
        try:
            if self.mic_open.is_set():
                # Hardware mic is 48 kHz; ElevenLabs receives true 16-kHz PCM.
                chunk = self._mic_to_agent_rate(bytes(indata))
                if chunk:
                    chunk = self._duck(chunk)
                    self._meter(chunk)
                    self._input_callback(chunk)
            else:
                # Feed silence at the ELEVENLABS rate, not the hardware rate.
                # 960 frames @ 48 kHz = 20 ms = 320 samples @ 16 kHz.
                silence_samples = round(frames * self.agent_rate
                                        / self.input_rate)
                self._input_callback(b"\x00" * (silence_samples * 2))
        except Exception as e:
            print(f"[AUDIO] input callback error: {e}", file=sys.stderr)

    def _on_output(self, outdata, frames, time_info, status):
        self._last_out_cb = time.time()
        if status and self.verbose:
            print(f"[output status] {status}", file=sys.stderr)
        needed = frames * 2          # int16 mono
        try:
            buf = self._residual
            while len(buf) < needed:
                try:
                    buf += self._out_q.get_nowait()
                except queue.Empty:
                    break
            if len(buf) >= needed:
                outdata[:] = buf[:needed]
                self._residual = buf[needed:]
            else:
                outdata[:len(buf)] = buf
                outdata[len(buf):] = b"\x00" * (needed - len(buf))
                self._residual = b""
        except Exception as e:
            # An exception escaping a PortAudio callback aborts the stream
            # for good - the speaker would go permanently silent with the
            # rest of the program still running. Log it, play silence, live.
            print(f"[AUDIO] output callback error: {e}", file=sys.stderr)
            try:
                outdata[:] = b"\x00" * needed
            except Exception:
                pass

    def _open_in(self):
        self._in_stream = sd.RawInputStream(
            samplerate=self.input_rate,
            blocksize=CHUNK_FRAMES,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=self._on_audio,
        )
        self._last_in_cb = time.time()
        self._in_stream.start()

    def _open_out(self):
        self._out_stream = sd.RawOutputStream(
            samplerate=self.output_rate,
            blocksize=CHUNK_FRAMES,
            device=self.output_device,
            channels=1,
            dtype="int16",
            callback=self._on_output,
        )
        self._last_out_cb = time.time()
        self._out_stream.start()

    def start(self, input_callback):
        self._input_callback = input_callback
        self._open_in()
        self._open_out()
        if self.watchdog:
            self._wd_run = True
            threading.Thread(target=self._watchdog, daemon=True).start()
        print(f"[AUDIO] streams open: in={self.input_device} "
              f"out={self.output_device} "
              f"(watchdog {'on' if self.watchdog else 'OFF'})", flush=True)

    # ---------- self-healing ----------
    def _close_streams(self):
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass
                try:
                    s.close()
                except Exception:
                    pass
        self._in_stream = self._out_stream = None

    def _reopen_all(self, why):
        """Tear both streams down, make PortAudio re-scan the USB bus, find
        the devices again by NAME and reopen. Both are rebuilt together
        because re-initialising PortAudio invalidates every open stream."""
        print(f"[AUDIO] {why} - restarting audio streams", flush=True)
        self._close_streams()
        try:
            sd._terminate()
            sd._initialize()
        except Exception as e:
            print(f"[AUDIO] couldn't re-scan devices: {e}")
        if self.input_name:
            idx, name = find_device(self.input_name, "input",
                                    retries=3, delay=0.5)
            if idx is not None:
                self.input_device = idx
        if self.output_name:
            idx, name = find_device(self.output_name, "output",
                                    retries=3, delay=0.5)
            if idx is not None:
                self.output_device = idx
        try:
            self._open_in()
            self._open_out()
            print(f"[AUDIO] recovered (in={self.input_device} "
                  f"out={self.output_device})", flush=True)
            return True
        except Exception as e:
            print(f"[AUDIO] reopen failed: {e}", flush=True)
            self._close_streams()
            return False

    def _watchdog(self):
        """Notice a dead stream and rebuild it. Added after the speaker went
        silent for good partway through a walk while everything else (agent,
        transcripts, tools) carried on. Both streams run callbacks
        continuously - even when idle they play/feed silence - so a stream
        that is inactive, or hasn't called back for 2 s, is dead.
        If this fires, `dmesg -T | tail -30` right after will usually show
        the USB device that dropped."""
        last_try = 0.0
        while self._wd_run:
            time.sleep(1.0)
            if not self._wd_run:
                break
            now = time.time()
            why = None
            for kind, stream, last in (
                    ("output", self._out_stream, self._last_out_cb),
                    ("input", self._in_stream, self._last_in_cb)):
                if stream is None:
                    why = f"{kind} stream missing"
                    break
                try:
                    active = stream.active
                except Exception:
                    active = False
                if not active:
                    why = f"{kind} stream stopped"
                    break
                if now - last > 2.0:
                    why = f"{kind} stream silent for {now - last:.1f}s"
                    break
            if why and now - last_try > 3.0:
                last_try = now
                self._reopen_all(why)

    def stop(self):
        self._wd_run = False
        self._input_callback = None
        self._close_streams()
        self.interrupt()

    def output(self, audio):
        # ElevenLabs returns 16-kHz PCM. Convert it to the 48-kHz USB speaker
        # rate before queueing so playback speed/pitch remain correct.
        converted = self._agent_to_speaker_rate(audio)
        if converted:
            self._out_q.put(converted)
            # Keep the mic muted for a moment past the last chunk so the
            # speaker's tail is not picked up and transcribed.
            play_secs = len(converted) / 2.0 / max(self.output_rate, 1)
            self._speaking_until = max(
                self._speaking_until, time.time() + play_secs) + 0.25

    def interrupt(self):
        """Barge-in: dump anything not yet played."""
        self._speaking_until = 0.0
        with self._lock:
            while True:
                try:
                    self._out_q.get_nowait()
                except queue.Empty:
                    break
            self._residual = b""


# ============================================================
def push_to_talk_loop(audio, stopping):
    """Enter toggles the mic. 'q' quits.

    A terminal cannot detect key RELEASE, so this is a toggle rather than
    hold-to-talk: Enter opens the mic, Enter again closes it. Repeat as often
    as you like.
    """
    print("=" * 58)
    print("  PUSH TO TALK")
    print("    ENTER  - toggle mic open / closed")
    print("    q+ENTER- quit")
    print("=" * 58)
    print("\n  [mic CLOSED]  press ENTER to talk\n")

    while not stopping.is_set():
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        if line.strip().lower() == "q":
            stopping.set()
            break

        if audio.toggle_mic():
            # Opening the mic cancels whatever the agent is still saying,
            # so you can talk over it.
            audio.interrupt()
            print("  [mic OPEN]    speak now, ENTER when done")
        else:
            print("  [mic CLOSED]  press ENTER to talk\n")

# --- tool-skip backstop -------------------------------------------------
# The model sometimes SAYS "on my way to screen one" and never calls
# humanoid_output, so the robot stands still. We watch each user turn: if no
# tool call arrives within FALLBACK_SECONDS and the user's words were a clear
# go-to-a-saved-place (or stop) request, we do it ourselves.
FALLBACK_SECONDS = 3.5
_turn = {"text": None, "tool": False, "fallback_at": 0.0}


def on_user_transcript(text):
    print(f"You:   {text}")
    _turn["text"] = text
    _turn["tool"] = False
    t = threading.Timer(FALLBACK_SECONDS, _fallback_check, args=(text,))
    t.daemon = True
    t.start()


def _fallback_check(text):
    if _turn["text"] != text or _turn["tool"] or not ROBOT_AVAILABLE:
        return
    try:
        hit = g1_robot.intent_fallback(text)
    except Exception as e:
        print(f"[FALLBACK] intent check failed: {e}")
        return
    if not hit:
        return
    action, dest = hit
    print(f"[FALLBACK] agent spoke but never called the tool for {text!r} "
          f"- doing {action} {dest or ''}")
    _turn["fallback_at"] = time.time()
    try:
        result = g1_robot.dispatch(action, dest, "")
        print(f"[ROBOT] (fallback) {action} -> {result}")
    except Exception as e:
        print(f"[FALLBACK] dispatch failed: {e}")


def humanoid_output(parameters):
    """The agent's single tool. It SPEAKS `reply` itself; we execute the
    action here and return a short status string back to the agent.
    """
    _turn["tool"] = True
    data = {
        "reply": parameters.get("reply", ""),
        "action": parameters.get("action", "none"),
        "destination": parameters.get("destination", "none"),
    }
    print(json.dumps(data))

    if not ROBOT_AVAILABLE:
        return "Robot control unavailable."

    if (data["action"] == "navigate"
            and time.time() - _turn["fallback_at"] < 20.0):
        print("[ROBOT] navigate already started by the fallback - skipping "
              "the duplicate")
        return ("Already heading there. STARTED - you have NOT arrived. Say "
                "nothing more about it; a [NAV] note will tell you when you "
                "really get there.")

    try:
        result = g1_robot.dispatch(
            data["action"], data["destination"], data["reply"])
    except Exception as e:
        print(f"[ROBOT] dispatch failed: {e}")
        return "Something went wrong doing that."

    print(f"[ROBOT] {data['action']} -> {result}")
    return result

def look(parameters):
    """Dedicated vision tool.

    WHY THIS IS SEPARATE FROM humanoid_output
    -----------------------------------------
    humanoid_output carries a `reply` that the agent speaks immediately. The
    tool result comes back afterwards, so a description returned that way
    arrives too late to be spoken in the same turn - which is why vision
    appeared not to work at all.

    This tool returns ONLY the description and nothing to speak up front. The
    agent calls it, receives what the camera saw, and speaks that. One turn,
    no filler.

    No keyword matching anywhere. The agent decides when looking is called
    for - "what do you see", "how do I look", "is anyone there", "what colour
    is this", "read that sign" - all of it is its judgement, not a string
    match on our side.
    """
    question = (parameters or {}).get("question", "") or "What do you see?"
    print(f"[LOOK] {question}")

    if not ROBOT_AVAILABLE:
        return "My camera isn't available right now."

    try:
        description = g1_robot.describe_view(question)
    except Exception as e:
        print(f"[LOOK] failed: {e}")
        return ("I couldn't get a picture from my camera just now.")

    print(f"[LOOK] -> {description}")
    return description


def main():
    ap = argparse.ArgumentParser(description="ElevenLabs agent pipeline test")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input-device", type=int, default=None,
                    help="override device index (else matched by name)")
    ap.add_argument("--output-device", type=int, default=None)
    ap.add_argument("--mic-name", default=MIC_NAME_MATCH,
                    help="input device name substring (Brio 100 mic)")
    ap.add_argument("--speaker-name", default=SPEAKER_NAME_MATCH,
                    help="output device name substring (AB13X dongle)")
    ap.add_argument("--input-rate", type=int, default=INPUT_RATE)
    ap.add_argument("--output-rate", type=int, default=OUTPUT_RATE,
                    help="must match the agent's configured output rate")
    ap.add_argument("--agent-id", default=os.getenv("ELEVENLABS_AGENT_ID"))
    ap.add_argument("--push-to-talk", action="store_true",
                    help="ENTER-gated capture instead of the default "
                         "always-on wake-word mode")
    ap.add_argument("--duck", type=float, default=0.08, metavar="F",
                    help="mic attenuation while the robot speaks, 0.0-1.0. "
                         "Lower suppresses the robot's own voice harder; too "
                         "low and you cannot interrupt either. 1.0 disables. "
                         "Default 0.08")
    ap.add_argument("--meter", action="store_true",
                    help="print live mic rms - use it to tune --duck")
    ap.add_argument("--echo-guard", action="store_true",
                    help="mute the mic while the robot speaks. OFF by "
                         "default: gating is half duplex, so it would make "
                         "voice interruption impossible. Only worth it if "
                         "the speaker is loud enough to trigger the agent "
                         "on its own voice.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--iface", default="wlan0",
                    help="DDS network interface for the robot")
    ap.add_argument("--no-robot", action="store_true",
                    help="voice only - skip DDS, actions and navigation")
    ap.add_argument("--no-nav", action="store_true",
                    help="skip NavBridge (drops the /slam_info subscription)")
    ap.add_argument("--no-audio-watchdog", action="store_true",
                    help="disable the self-healing audio watchdog. Use it "
                         "to rule the watchdog out if the speaker is silent.")
    ap.add_argument("--guard-mode", choices=["per-walk", "persistent"],
                    default="per-walk",
                    help="per-walk (default): the RealSense is opened at the "
                         "start of each walk and closed at the end. "
                         "persistent: opened once at startup and left "
                         "streaming (released only while pickup_glass / "
                         "measure use the camera). Try persistent if the "
                         "speaker cuts out when a walk starts - see "
                         "g1_audio_diag.py.")
    ap.add_argument("--no-obstacle-guard", action="store_true",
                    help="disable the RealSense-based dodge-and-continue "
                         "obstacle avoidance - navigate() falls back to a "
                         "straight walk with no avoidance, same as before "
                         "it existed. Use this if the guard misbehaves and "
                         "you need navigation back immediately.")
    ap.add_argument("--list-actions", action="store_true",
                    help="print every valid action value and exit")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return
    if args.list_actions:
        if ROBOT_AVAILABLE:
            for a in g1_robot.all_actions():
                print(a)
        else:
            print(f"robot module unavailable: {_ROBOT_ERROR}")
        return

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("ERROR: export ELEVENLABS_API_KEY=...")
        return
    if not args.agent_id:
        print("ERROR: export ELEVENLABS_AGENT_ID=... (or pass --agent-id)")
        return

    # ---- resolve devices ----
    in_dev = args.input_device
    if in_dev is None:
        in_dev, in_name = find_device(args.mic_name, "input")
        if in_dev is None:
            print(f"ERROR: no input device matching '{args.mic_name}'.")
            print("Run --list-devices to see what's connected.")
            return
        print(f"Input : [{in_dev}] {in_name}")
    else:
        print(f"Input : [{in_dev}] (forced)")

    out_dev = args.output_device
    if out_dev is None:
        out_dev, out_name = find_device(args.speaker_name, "output")
        if out_dev is None:
            print(f"WARNING: no output device matching "
                  f"'{args.speaker_name}' - falling back to system default.")
            print("Run --list-devices if you expected the dongle here.")
        else:
            print(f"Output: [{out_dev}] {out_name}")
    else:
        print(f"Output: [{out_dev}] (forced)")
    print(f"Rates : mic hardware {args.input_rate} Hz -> ElevenLabs {ELEVENLABS_RATE} Hz")
    print(f"        ElevenLabs {ELEVENLABS_RATE} Hz -> speaker hardware {args.output_rate} Hz\n")

    audio = SoundDeviceAudioInterface(
        input_device=in_dev,
        output_device=out_dev,
        input_rate=args.input_rate,
        output_rate=args.output_rate,
        agent_rate=ELEVENLABS_RATE,
        verbose=args.verbose,
        push_to_talk=args.push_to_talk,
        echo_guard=args.echo_guard,
        duck_factor=args.duck,
        input_name=args.mic_name if args.input_device is None else None,
        output_name=args.speaker_name if args.output_device is None else None,
        watchdog=not args.no_audio_watchdog,
    )

    client_tools = ClientTools()
    client_tools.register("humanoid_output", humanoid_output)
    client_tools.register("look", look)

    client = ElevenLabs(api_key=api_key)
    conversation = Conversation(
        client,
        args.agent_id,
        requires_auth=True,
        client_tools=client_tools,
        audio_interface=audio,
        callback_user_transcript=on_user_transcript,
        callback_agent_response=lambda r: print(f"Agent: {r}"),
        callback_agent_response_correction=(
            lambda orig, corr: print(f"Agent: {orig} -> {corr}")),
    )

    stopping = threading.Event()

    def shutdown(signum=None, frame=None):
        if stopping.is_set():
            return
        stopping.set()
        print("\nEnding session...")
        try:
            conversation.end_session()
        except Exception as e:
            print(f"  end_session: {e}")

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    audio._meter_enabled = args.meter

    # Persistent guard: the RealSense must start BEFORE any audio stream is
    # open - starting it kills a USB speaker that is already playing (see
    # g1_audio_diag.py). The 2 s settle inside this call is deliberate.
    if (args.guard_mode == "persistent" and ROBOT_AVAILABLE
            and not args.no_robot and not args.no_obstacle_guard):
        g1_robot.preopen_camera_guard()

    print("Session starting...\n")
    t0 = time.time()
    try:
        conversation.start_session()

        # DDS comes up ONLY after the websocket is live. Initialising it
        # first floods the interpreter and the handshake times out.
        if ROBOT_AVAILABLE and not args.no_robot:
            print("\nBringing up robot interfaces...")
            # Lets background work (nav arrival, an RPS result, a grasp
            # confirmation) speak into this session on its own - see
            # g1_robot.announce(). Registered before init() so NavBridge's
            # on_event has somewhere to send its very first message.
            g1_robot.set_conversation(conversation)
            g1_robot.init(args.iface, use_nav=not args.no_nav,
                         use_obstacle_guard=not args.no_obstacle_guard,
                         guard_persistent=(args.guard_mode == "persistent"))
            problem = g1_robot.vision_status()
            if problem:
                print(f"[ROBOT] VISION UNAVAILABLE: {problem}")
            else:
                print("[ROBOT] vision ready (Brio + Claude)")
            print()
        elif args.no_robot:
            print("Robot control disabled (--no-robot).\n")
        else:
            print(f"WARNING: robot module unavailable: {_ROBOT_ERROR}\n")

        if args.push_to_talk:
            ptt = threading.Thread(
                target=push_to_talk_loop, args=(audio, stopping), daemon=True)
            ptt.start()
        else:
            print("=" * 58)
            print("  ALWAYS ON - start with the wake word, e.g. 'hey Tony'")
            if args.echo_guard:
                print("  Echo guard: HARD GATE - cannot interrupt by voice")
            elif args.duck < 1.0:
                print(f"  Ducking: mic at {args.duck:.0%} while speaking "
                      "- interruption still works")
            else:
                print("  Ducking: off")
            print("  Ctrl-C to stop.")
            print("=" * 58 + "\n")

        conv_id = conversation.wait_for_session_end()
        print(f"\nConversation ID: {conv_id}")
    except Exception as e:
        print(f"\nSession error: {e}")
        print("\nCommon causes:")
        print("  - agent not public and requires_auth mismatch")
        print("  - wrong agent_id")
        print("  - no network route to api.elevenlabs.io")
    finally:
        print(f"Duration: {time.time() - t0:.1f}s")
        if ROBOT_AVAILABLE:
            g1_robot.shutdown()
        audio.stop()


if __name__ == "__main__":
    main()