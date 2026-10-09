#!/usr/bin/env python3
"""
g1_rps_judge.py - Brio-only hand judging for rock-paper-scissors
================================================================
Shared by g1_rps_v2.py (standalone: opens the Brio itself) and g1_robot.py
(under the agent: passes in frames from its own always-open Brio grabber).
Kept in its own module because g1_rps_v2.py installs a SIGTERM handler and
imports the Unitree SDK at import time - neither belongs in the agent.

Never touches the RealSense.
"""
import base64
import json
import os
import time

try:
    import cv2
    import numpy as np
    import anthropic
    VISION_AVAILABLE = True
except ImportError as _e:
    VISION_AVAILABLE = False
    _VISION_IMPORT_ERROR = str(_e)

# BRIO ONLY. The RealSense sits at a very low angle, has a tinted colour
# image and starting it kills the USB speaker - it is never used here.
# The Brio is found by NAME (the RealSense claims six /dev/video nodes).
#
# Two ways this program gets Brio frames:
#   * standalone (no flag): it opens the Brio itself, below.
#   * under the agent (--no-vision): the agent process already holds the Brio
#     open permanently, so this program must NOT touch it. It prints an
#     RPS_REVEAL line at the moment of the throw and the agent grabs the
#     frames and judges them itself (g1_robot.judge_rps_frames).
CAMERA_NAME = "Brio"
VISION_WIDTH = 960
VISION_WARMUP = 10
JUDGE_FRAME_DELAYS = (0.5, 1.3)      # seconds after the reveal
VISION_MODEL = "claude-opus-5-5"     # classifying a hand SHAPE correctly is
                                     # the whole point of this call; Haiku
                                     # hallucinated

JUDGE_PROMPT = """You are looking through a webcam on a humanoid robot's \
head, facing the person it is playing rock-paper-scissors against. The robot \
just threw its sign with its OWN mechanical hand (which may or may not be in \
view). One or two frames follow, taken within about a second of each other, \
just after the robot's throw.

Find a SEPARATE HUMAN hand - real skin, not the robot's mechanical fingers - \
making a rock, paper, or scissors gesture, and identify which sign it shows:
  rock     = a closed fist
  paper    = an open flat hand, fingers extended
  scissors = index and middle fingers extended in a V, others curled
If the frames disagree, trust the one where the hand shape is clearest and \
most deliberate (a hand mid-motion or half-formed does not count).

Reply with ONLY JSON, nothing else:
{"human_hand_visible": <true|false>, "sign": "<rock|paper|scissors|null>", \
"confidence": <0-1>, "note": "<one short sentence>"}

Set "human_hand_visible" to false and "sign" to null if the ONLY hand you \
can see is the robot's own, or if no hand is clearly making one of the three \
shapes. Do not guess a sign you are not fairly confident about."""



def find_brio_device():
    """Resolve the Brio's /dev/videoN by NAME via v4l2-ctl."""
    import subprocess
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
        elif current and CAMERA_NAME.lower() in current.lower():
            dev = line.strip()
            if dev.startswith("/dev/video"):
                return dev
    return None


def capture_brio_frames(delays=JUDGE_FRAME_DELAYS):
    """STANDALONE ONLY: open the Brio, take one frame at each delay (seconds
    from now), close it. Returns a list of BGR frames."""
    dev = find_brio_device()
    if dev is None:
        raise RuntimeError(f"no video device whose name contains "
                           f"'{CAMERA_NAME}'")
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {dev} (is the agent holding "
                           f"the Brio? run with --no-vision under the agent)")
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        for _ in range(VISION_WARMUP):
            cap.read()
        frames, t0 = [], time.time()
        for d in delays:
            while time.time() - t0 < d:
                cap.read()                      # keep draining the buffer
            ok, f = cap.read()
            if ok:
                frames.append(f.copy())
        if not frames:
            raise RuntimeError("no frames from the Brio")
        return frames
    finally:
        cap.release()


def judge_frames(frames):
    """Ask Opus what the human threw, given 1+ BGR frames taken just after
    the robot's reveal. Pure function of the frames - no camera access, so
    the agent process can call it with frames from its own Brio grabber.

    Returns {"visible", "sign", "confidence", "note", "error"}.
    `error` set = camera/API failed; "visible": False with no error = the
    call worked and found no human hand."""
    def _fail(err):
        return {"visible": False, "sign": None, "confidence": 0.0,
                "note": "", "error": err}

    if not VISION_AVAILABLE:
        return _fail(f"vision deps missing: {_VISION_IMPORT_ERROR}")
    if not os.getenv("ANTHROPIC_API_KEY"):
        return _fail("ANTHROPIC_API_KEY not set")
    if not frames:
        return _fail("no frames")

    content = []
    for i, frame in enumerate(frames, 1):
        h, w = frame.shape[:2]
        if w > VISION_WIDTH:
            frame = cv2.resize(frame, (VISION_WIDTH, int(h * VISION_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return _fail("jpeg encode failed")
        if len(frames) > 1:
            content.append({"type": "text", "text": f"Frame {i}:"})
        content.append({"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": base64.b64encode(enc.tobytes()).decode()}})
    content.append({"type": "text", "text": JUDGE_PROMPT})

    try:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model=VISION_MODEL, max_tokens=1024,
            extra_body={"output_config": {"effort": "low"}},
            messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        sign = data.get("sign")
        if sign not in ("rock", "paper", "scissors"):
            sign = None
        return {"visible": bool(data.get("human_hand_visible")) and
                          sign is not None,
                "sign": sign,
                "confidence": float(data.get("confidence", 0.0)),
                "note": data.get("note", ""), "error": None}
    except Exception as e:
        return _fail(f"llm: {e}")
