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
                 verbose=False, push_to_talk=True, echo_guard=False):
        self.input_device = input_device
        self.output_device = output_device
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
    def start(self, input_callback):
        self._input_callback = input_callback

        def on_audio(indata, frames, time_info, status):
            if status and self.verbose:
                print(f"[input status] {status}", file=sys.stderr)
            if not self._input_callback:
                return
            muted_by_echo = (self.echo_guard
                             and not self.push_to_talk
                             and self.is_speaking())

            if self.mic_open.is_set() and not muted_by_echo:
                # Hardware mic is 48 kHz; ElevenLabs receives true 16-kHz PCM.
                chunk = self._mic_to_agent_rate(bytes(indata))
                if chunk:
                    self._input_callback(chunk)
            else:
                # Feed silence at the ELEVENLABS rate, not the hardware rate.
                # 960 frames @ 48 kHz = 20 ms = 320 samples @ 16 kHz.
                silence_samples = round(frames * self.agent_rate / self.input_rate)
                self._input_callback(b"\x00" * (silence_samples * 2))

        self._in_stream = sd.RawInputStream(
            samplerate=self.input_rate,
            blocksize=CHUNK_FRAMES,
            device=self.input_device,
            channels=1,
            dtype="int16",
            callback=on_audio,
        )
        self._in_stream.start()

        # ---------- playback ----------
        def on_output(outdata, frames, time_info, status):
            if status and self.verbose:
                print(f"[output status] {status}", file=sys.stderr)
            needed = frames * 2          # int16 mono
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

        self._out_stream = sd.RawOutputStream(
            samplerate=self.output_rate,
            blocksize=CHUNK_FRAMES,
            device=self.output_device,
            channels=1,
            dtype="int16",
            callback=on_output,
        )
        self._out_stream.start()

    def stop(self):
        self._input_callback = None
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = self._out_stream = None
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

def humanoid_output(parameters):
    """The agent's single tool. It SPEAKS `reply` itself; we execute the
    action here and return a short status string back to the agent.
    """
    data = {
        "reply": parameters.get("reply", ""),
        "action": parameters.get("action", "none"),
        "destination": parameters.get("destination", "none"),
    }
    print(json.dumps(data))

    if not ROBOT_AVAILABLE:
        return "Robot control unavailable."

    try:
        result = g1_robot.dispatch(
            data["action"], data["destination"], data["reply"])
    except Exception as e:
        print(f"[ROBOT] dispatch failed: {e}")
        return "Something went wrong doing that."

    print(f"[ROBOT] {data['action']} -> {result}")
    return result

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
    )

    client_tools = ClientTools()
    client_tools.register("humanoid_output", humanoid_output)

    client = ElevenLabs(api_key=api_key)
    conversation = Conversation(
        client,
        args.agent_id,
        requires_auth=True,
        client_tools=client_tools,
        audio_interface=audio,
        callback_user_transcript=lambda t: print(f"You:   {t}"),
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

    print("Session starting...\n")
    t0 = time.time()
    try:
        conversation.start_session()

        # DDS comes up ONLY after the websocket is live. Initialising it
        # first floods the interpreter and the handshake times out.
        if ROBOT_AVAILABLE and not args.no_robot:
            print("\nBringing up robot interfaces...")
            g1_robot.init(args.iface, use_nav=not args.no_nav)
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
            guard = "on (voice interruption disabled)" if args.echo_guard \
                else "off - you can interrupt mid-sentence"
            print(f"  Echo guard: {guard}")
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