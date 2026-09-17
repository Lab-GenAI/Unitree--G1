#!/usr/bin/env python3
"""
Vision pipeline test - no agent, no DDS
=======================================

Checks each stage separately so a failure points at one thing:

    1. anthropic package importable
    2. ANTHROPIC_API_KEY present in THIS process
    3. camera device resolvable by name
    4. frame captured, not black
    5. Claude reachable and returning a description

Run:
    python3 test_vision.py
    python3 test_vision.py --save /tmp/look.jpg
    python3 test_vision.py --question "how many people do you see?"
"""

import argparse
import base64
import os
import subprocess
import sys
import time


def stage(n, text):
    print(f"\n[{n}] {text}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="What do you see?")
    ap.add_argument("--save", metavar="PATH",
                    help="also write the captured frame to disk")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--device", help="force /dev/videoN")
    args = ap.parse_args()

    # ---- 1 ----
    stage(1, "importing anthropic")
    try:
        import anthropic
        print("    ok")
    except ImportError as e:
        print(f"    FAIL: {e}")
        print("    fix: pip3 install anthropic")
        return 1

    # ---- 2 ----
    stage(2, "checking ANTHROPIC_API_KEY")
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("    FAIL: not set in this process")
        print("    fix: export ANTHROPIC_API_KEY='sk-ant-...'")
        print("    NOTE: exporting it in one terminal does not carry into")
        print("          another, or into tmux panes started earlier.")
        return 1
    print(f"    ok ({key[:10]}...{key[-4:]}, {len(key)} chars)")

    # ---- 3 ----
    stage(3, "resolving the camera")
    try:
        import cv2
    except ImportError as e:
        print(f"    FAIL: {e}")
        print("    fix: pip3 install opencv-python")
        return 1

    device = args.device
    if device is None:
        try:
            out = subprocess.run(["v4l2-ctl", "--list-devices"],
                                 capture_output=True, text=True,
                                 timeout=5).stdout
        except Exception as e:
            print(f"    FAIL: v4l2-ctl: {e}")
            print("    fix: sudo apt install v4l-utils")
            return 1

        print("    devices seen:")
        current = None
        for line in out.splitlines():
            if line and not line.startswith(("\t", " ")):
                current = line.strip()
                print(f"      {current}")
            elif current and "brio" in current.lower():
                d = line.strip()
                if d.startswith("/dev/video") and device is None:
                    device = d
        if device is None:
            print("    FAIL: no device whose name contains 'Brio'")
            print("    fix: pass --device /dev/videoN explicitly")
            return 1
    print(f"    using {device}")

    # ---- 4 ----
    stage(4, f"capturing ({args.warmup} warmup frames)")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"    FAIL: could not open {device}")
        print("    fix: is another process holding it? sudo fuser -v "
              f"{device}")
        return 1
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for i in range(args.warmup):
            ok, f = cap.read()
            if ok:
                frame = f
                if i in (0, args.warmup // 2, args.warmup - 1):
                    print(f"    frame {i:>2}: mean {f.mean():6.1f}")
            time.sleep(0.02)
    finally:
        cap.release()

    if frame is None:
        print("    FAIL: no frames read")
        return 1

    mean = float(frame.mean())
    print(f"    captured {frame.shape[1]}x{frame.shape[0]}, mean {mean:.1f}")
    if mean < 12:
        print("    FAIL: essentially black - lens cap, or wrong device")
        return 1
    if mean < 40:
        print("    WARN: dim. Raise --warmup or add light.")

    if args.save:
        cv2.imwrite(args.save, frame)
        print(f"    wrote {args.save}")

    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        print("    FAIL: JPEG encode")
        return 1
    b64 = base64.b64encode(enc.tobytes()).decode()
    print(f"    encoded {len(b64) // 1024} KB base64")

    # ---- 5 ----
    stage(5, "asking Claude")
    t0 = time.time()
    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg",
                    "data": b64}},
                {"type": "text", "text": args.question}]}])
        print(f"    ok ({time.time() - t0:.1f}s)")
        print(f"\n    {resp.content[0].text.strip()}\n")
    except Exception as e:
        print(f"    FAIL: {type(e).__name__}: {e}")
        print("    common causes: bad key, no route to api.anthropic.com,")
        print("                   model name not available on this account")
        return 1

    print("Vision pipeline works end to end.")
    print("\nIf it still fails through the agent, the agent is not calling")
    print("the action. Check that 'look' is listed as a valid action value")
    print("in the ElevenLabs system prompt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
