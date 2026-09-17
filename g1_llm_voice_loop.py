"""
G1 LLM Voice Loop - Phase 1b (Piper TTS version)
===================================================

Runs ENTIRELY ON THE JETSON. No unitree_sdk2py dependency needed for
this phase - speech goes through Piper + aplay instead of AudioClient
(G1's built-in TtsMaker is Mandarin-only, confirmed not usable for
English output).

Pipeline per turn:
  1. Record N seconds from the USB-C mic (arecord)
  2. Transcribe with whisper.cpp (whisper-cli)
  3. Send transcript to the local llama.cpp server (localhost, same box)
  4. Speak the response via Piper -> aplay, through whichever device
     turned out to be the robot's actual chest speaker

BEFORE RUNNING
--------------
- Confirm MIC_DEVICE (input) and SPEAKER_DEVICE (output) below - they
  may be different ALSA devices even if they seem related.
- Confirm the llama-server is running on this same Jetson (localhost:8080).
- Confirm PIPER_BIN / PIPER_MODEL paths match where you installed them.

RUN (on the Jetson)
-------------------
    python3 g1_llm_voice_loop.py
"""

import subprocess
import requests

# --------------------------------------------------------------------------
# Config - adjust these to match your setup
# --------------------------------------------------------------------------
MIC_DEVICE = "plughw:0,0"          # from `arecord -l` - your USB-C earphones
SPEAKER_DEVICE = "plughw:0,0"          # ADJUST: whichever device you confirmed plays through the robot's chest speaker
RECORD_SECONDS = 5
RECORD_WAV = "/tmp/g1_turn.wav"
TTS_WAV = "/tmp/g1_reply.wav"

WHISPER_BIN = "/home/unitree/models/whisper.cpp-master/build/bin/whisper-cli"  # ADJUST PATH
WHISPER_MODEL = "/home/unitree/models/ggml-base.en.bin"  # ADJUST PATH

PIPER_BIN = "/home/unitree/models/piper/piper"  # ADJUST PATH
PIPER_MODEL = "/home/unitree/models/en_US-lessac-medium.onnx"  # ADJUST PATH

LLAMA_SERVER_URL = "http://localhost:8080"

SYSTEM_PROMPT = (
    "You are G1, a helpful humanoid robot assistant. Keep responses "
    "SHORT (1-3 sentences) and conversational, since they will be "
    "spoken out loud."
)


def record_audio():
    subprocess.run(
        [
            "arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000",
            "-d", str(RECORD_SECONDS), RECORD_WAV,
        ],
        check=True,
    )


def transcribe_audio():
    result = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", RECORD_WAV, "-nt", "-np"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def ask_llm(history):
    payload = {"messages": history, "temperature": 0.7, "max_tokens": 150}
    r = requests.post(f"{LLAMA_SERVER_URL}/v1/chat/completions", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def speak(text):
    # Piper: text -> wav file
    subprocess.run(
        [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", TTS_WAV],
        input=text, text=True, check=True,
    )
    # Play through whichever device is the robot's actual speaker
    subprocess.run(["aplay", "-D", SPEAKER_DEVICE, TTS_WAV], check=True)


def main():
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Ready. Press ENTER to record a 5-second turn, Ctrl+C to quit.")
    try:
        while True:
            input("\n[Press ENTER to talk] ")
            print(f"Recording {RECORD_SECONDS}s...")
            record_audio()

            print("Transcribing...")
            user_text = transcribe_audio()
            if not user_text:
                print("(nothing heard, try again)")
                continue
            print(f"You said: {user_text}")

            history.append({"role": "user", "content": user_text})
            print("Thinking...")
            reply = ask_llm(history)
            print(f"G1: {reply}")
            history.append({"role": "assistant", "content": reply})

            if len(history) > 13:
                history = [history[0]] + history[-12:]

            print("Speaking...")
            speak(reply)

    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
