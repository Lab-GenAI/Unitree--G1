#!/usr/bin/env python3
"""
G1 Obstacle Guard - depth-only forward obstacle detection
============================================================

WHY THIS EXISTS
-----------------
slam_operate's native 1102 navigate call was tested across --mode 0/1/2/3
and none of them route around something in the way - the robot walks a
straight line to the goal regardless. obsInfo/ctrl_info's obstacle flag may
or may not even fire for a plain static object (untested at the time of
writing). Either way, nothing in the native service can be relied on for
avoidance, so this builds it independently, in software, on top of the
RealSense that's already on the robot's head.

WHAT IT CHECKS
---------------
Depth only - no colour, no LLM call, no API cost. That matters here in a way
it didn't for the box-grasp or measurement work: this has to run
continuously at several Hz for as long as the robot is walking, not once
per request. Pure geometry is cheap enough for that; a vision-model call per
cycle would not be.

Each depth frame is cut into three vertical strips:

    LEFT third | CENTRE (the path ahead) | RIGHT third

The near edge of whatever's in the centre strip (10th percentile depth, so
a few stray near pixels don't trigger on nothing) is the distance that
matters. Below DANGER_M, something is close enough to need a response.
Below EMERGENCY_M, it's close enough that stepping sideways blindly is
worse than just stopping - that distinction is checked by the caller
(g1_nav_bridge.py's _dodge_step), not here; this module only reports what it
sees. The side with the further median depth is reported as `clear_side` -
which way has more room to step toward.

STANDALONE TEST (do this FIRST, before wiring it into navigation)
---------------------------------------------------------------------
No DDS, no robot motion - just points the RealSense at the room and prints
what it sees, live:

    python3 g1_obstacle_guard.py --watch

Walk into frame, hold up a box, stand at different distances, and check:
  - does `danger` flip at roughly the distance you'd want it to?
  - is `clear_side` actually the side with more room, or does it flip-flop?
  - does a false "danger" ever fire with nothing really in the way (low
    valid-pixel count, a dark surface eating the IR, etc)?

Tune DANGER_M, EMERGENCY_M, ZONE_X, ZONE_Y, MIN_VALID_PX below against what
you actually see before trusting this to steer anything.
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import pyrealsense2 as rs
    RS_AVAILABLE = True
except ImportError:
    # Deliberately NOT sys.exit() here - this module is imported as a
    # library by g1_nav_bridge.py/g1_robot.py, which catch ImportError to
    # fall back to "navigate without avoidance" (see OBSTACLE_GUARD_AVAILABLE
    # in g1_robot.py). sys.exit() raises SystemExit, a BaseException that
    # try/except Exception does NOT catch - that would take the whole voice
    # daemon down instead of degrading gracefully. The error only actually
    # surfaces when open() or main() is called, which is the right time for
    # a CLI run too.
    RS_AVAILABLE = False


# Deliberately small and slow. Forward-obstacle checks don't need 640x480 @
# 30 fps, and the full stream put ~18 MB/s on the USB bus and kept a Python
# thread busy 30x a second - on the Jetson that shares its USB bus (and CPU)
# with the speaker dongle and the Brio, and the speaker went silent after
# walks started. 424x240 @ 15 fps is ~3 MB/s and plenty for something
# walking at 0.3 m/s. All zone/threshold maths below is in fractions or
# metres, so it doesn't care about the resolution.
WIDTH, HEIGHT, FPS = 424, 240, 15
WARMUP = 8

# The "in the path" zone - a centre column of the frame. Narrower than the
# full width because the robot's own body is roughly this wide; something
# off to the side that the frame can still see is not actually blocking it.
ZONE_X = (0.32, 0.68)
# Vertical range to look at. Skips the very top (often ceiling / far wall
# towering over everything close) and the very bottom few percent (floor
# right at the camera, which reads near regardless of anything being there).
ZONE_Y = (0.20, 0.92)

MIN_VALID_PX = 100          # below this, the frame can't be judged - report
                            # "no danger" rather than risk a false trigger on
                            # a reading that's mostly noise
DANGER_M = 0.8              # respond (dodge) below this
EMERGENCY_M = 0.35          # too close to dodge blindly - stop instead


@dataclass
class ObstacleReading:
    danger: bool
    distance_m: Optional[float]
    clear_side: Optional[str]      # "left" | "right" | None
    emergency: bool = False
    valid: bool = True             # False = couldn't judge (low signal)


class ObstacleGuard:
    """Opens the RealSense depth stream and keeps the latest reading warm in
    a background thread. check() is non-blocking - always returns instantly
    with whatever was last computed, so a navigation loop can poll it every
    cycle without ever waiting on camera I/O."""

    def __init__(self, danger_m=DANGER_M, emergency_m=EMERGENCY_M,
                 min_valid_px=MIN_VALID_PX):
        self.danger_m = danger_m
        self.emergency_m = emergency_m
        self.min_valid_px = min_valid_px
        self.pipe = None
        self.scale = 1.0
        self._thread = None
        self._run = False
        self._lock = threading.Lock()
        self._reading = ObstacleReading(False, None, None, valid=False)
        self._err = None
        # persistent=True: close() is a no-op unless force=True, so the camera
        # stays streaming between walks (see g1_robot --guard-mode persistent).
        self.persistent = False

    def open(self):
        if not RS_AVAILABLE:
            raise RuntimeError("pyrealsense2 is not importable - "
                               "pip3 install pyrealsense2")
        if self.pipe is not None:
            return
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        try:
            prof = self.pipe.start(cfg)
        except RuntimeError as e:
            self.pipe = None
            raise RuntimeError(f"obstacle guard couldn't open the "
                               f"RealSense: {e}")
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(WARMUP):
            self.pipe.wait_for_frames()
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def close(self, force=False):
        if self.persistent and not force:
            return
        self._run = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.pipe:
            try:
                self.pipe.stop()
            except Exception:
                pass
            self.pipe = None

    def _loop(self):
        while self._run:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=500)
                depth = frames.get_depth_frame()
                if not depth:
                    continue
                arr = (np.asanyarray(depth.get_data()).astype(np.float32)
                       * self.scale)
                reading = self._analyse(arr)
                with self._lock:
                    self._reading = reading
                self._err = None
            except Exception as e:
                self._err = str(e)

    def _analyse(self, depth):
        h, w = depth.shape
        x0, x1 = int(w * ZONE_X[0]), int(w * ZONE_X[1])
        y0, y1 = int(h * ZONE_Y[0]), int(h * ZONE_Y[1])

        centre = depth[y0:y1, x0:x1]
        centre_v = centre[centre > 0]
        if centre_v.size < self.min_valid_px:
            return ObstacleReading(False, None, None, valid=False)

        near = float(np.percentile(centre_v, 10))

        left = depth[y0:y1, 0:x0]
        right = depth[y0:y1, x1:w]
        left_v = left[left > 0]
        right_v = right[right > 0]
        left_clear = (float(np.median(left_v))
                      if left_v.size >= self.min_valid_px else 0.0)
        right_clear = (float(np.median(right_v))
                       if right_v.size >= self.min_valid_px else 0.0)
        clear_side = "left" if left_clear >= right_clear else "right"

        return ObstacleReading(
            danger=near < self.danger_m,
            distance_m=near,
            clear_side=clear_side,
            emergency=near < self.emergency_m,
            valid=True)

    def check(self):
        with self._lock:
            return self._reading

    def error(self):
        return self._err


def main():
    ap = argparse.ArgumentParser(
        description="Standalone obstacle-guard test - no DDS, no motion")
    ap.add_argument("--watch", action="store_true",
                    help="print live readings until Ctrl-C")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop after this long (0 = until Ctrl-C)")
    ap.add_argument("--danger-m", type=float, default=DANGER_M)
    ap.add_argument("--emergency-m", type=float, default=EMERGENCY_M)
    args = ap.parse_args()

    guard = ObstacleGuard(danger_m=args.danger_m, emergency_m=args.emergency_m)
    print("Opening RealSense (depth only)...")
    try:
        guard.open()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1
    print("ok - walk into frame / hold something up to test\n")

    end = time.time() + args.seconds if args.seconds > 0 else None
    try:
        while end is None or time.time() < end:
            r = guard.check()
            if not r.valid:
                tag = " (no signal) "
            elif r.emergency:
                tag = "!! EMERGENCY !!"
            elif r.danger:
                tag = "   DANGER    "
            else:
                tag = "    clear    "
            dist = f"{r.distance_m:.2f}m" if r.distance_m is not None else "  -  "
            print(f"\r{tag}  dist={dist}  clear_side={r.clear_side or '-':<5}   ",
                  end="", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nClosing...")
        guard.close()
        if guard.error():
            print(f"last error: {guard.error()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
