#!/usr/bin/env python3
"""
G1 Vision - RealSense box detection and alignment
==================================================

Finds a coloured box in the RealSense's view and works out how the robot
should move to reach it: how far to turn, how far to walk, how high to lift.

The camera is mounted in the robot's HEAD, looking out through the face.

WHY A COLOURED BOX
------------------
A deliberately obvious target - red with white stripes - means detection is
classical HSV thresholding plus contours. No model to train, no weights to
load, runs in milliseconds, and fails in ways you can see. For a demo with a
prop you control, that is strictly better than a learned detector.

WHAT THIS DOES AND DOES NOT DO
------------------------------
Does:  find the box, measure its bearing, distance and height relative to the
       camera, and say how far to turn and walk.
Does NOT: plan a grasp. Turning a 3D position into arm joint angles needs the
       fixed transform from camera to arm base (hand-eye calibration), which
       is not measured yet. See CAMERA_TO_ARM below.

USAGE
-----
    python3 g1_vision.py --preview          # live detection, saves frames
    python3 g1_vision.py --once             # one detection, print result
    python3 g1_vision.py --tune             # HSV tuning helper

    from g1_vision import BoxDetector
    det = BoxDetector()
    box = det.detect()
    if box:
        print(box.bearing_deg, box.distance_m, box.height_m)
"""

import argparse
import math
import sys
import time
from dataclasses import dataclass

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: pip3 install opencv-python")
    sys.exit(1)

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP_FRAMES = 20

# Red wraps around 0 in HSV, so it needs two ranges.
RED_LOW_1  = np.array([0, 120, 70])
RED_HIGH_1 = np.array([10, 255, 255])
RED_LOW_2  = np.array([170, 120, 70])
RED_HIGH_2 = np.array([180, 255, 255])

MIN_AREA_PX = 2000          # ignore specks
MIN_FILL     = 0.45         # contour area / bounding box area

# Rough vertical offset from the camera in the head down to the shoulder
# joints, in metres. Used only to report a height relative to the arms rather
# than to the camera. MEASURE THIS on the actual robot.
CAMERA_TO_SHOULDER_DROP = 0.35

# Hand-eye transform placeholder. Converting a camera-frame point into the
# arm's base frame needs this measured. Until it is, grasp planning cannot be
# trusted - the numbers below are for navigation and alignment only.
CAMERA_TO_ARM = None


@dataclass
class BoxObservation:
    """Everything measured about the box in one frame."""
    cx: int                 # centroid in image pixels
    cy: int
    w: int                  # bounding box size in pixels
    h: int
    area: float
    distance_m: float       # straight-line distance from the camera
    bearing_deg: float      # + is to the LEFT of centre
    elevation_deg: float    # + is ABOVE centre
    x_m: float              # camera frame: right
    y_m: float              # camera frame: down
    z_m: float              # camera frame: forward
    height_m: float         # relative to shoulder height, + is above
    width_m: float          # estimated real width

    def turn_direction(self, tolerance_deg=4.0):
        """What the robot should do to centre the box."""
        if abs(self.bearing_deg) <= tolerance_deg:
            return "centred"
        return "turn_left" if self.bearing_deg > 0 else "turn_right"

    def describe(self):
        side = "left" if self.bearing_deg > 0 else "right"
        if abs(self.bearing_deg) <= 4.0:
            where = "straight ahead"
        else:
            where = f"{abs(self.bearing_deg):.0f} degrees to the {side}"
        level = ("about shoulder height" if abs(self.height_m) < 0.15
                 else f"{abs(self.height_m):.2f} m "
                      f"{'above' if self.height_m > 0 else 'below'} shoulder")
        return (f"Box {where}, {self.distance_m:.2f} m away, {level}, "
                f"about {self.width_m * 100:.0f} cm wide.")


class BoxDetector:
    """Opens the RealSense on demand and closes it again.

    Holding the pipeline open would block anything else that wants the
    device, and an uncleanly closed pipeline leaves the USB device claimed -
    which then looks like "camera vanished" on the next attempt.
    """

    def __init__(self, width=WIDTH, height=HEIGHT, fps=FPS,
                 warmup=WARMUP_FRAMES):
        if not REALSENSE_AVAILABLE:
            raise RuntimeError(
                "pyrealsense2 not importable - check PYTHONPATH points at "
                "the librealsense install")
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup = warmup
        self._intrinsics = None

    # ---------- capture ----------
    def _frames(self, pipeline, align):
        for _ in range(self.warmup):
            pipeline.wait_for_frames()
        frames = align.process(pipeline.wait_for_frames())
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("incomplete frame set from RealSense")
        if self._intrinsics is None:
            self._intrinsics = (color.profile.as_video_stream_profile()
                                .intrinsics)
        return np.asanyarray(color.get_data()), depth

    def capture(self):
        """Return (color_image, depth_frame). Caller must not keep the depth
        frame past the pipeline's lifetime."""
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height,
                          rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height,
                          rs.format.z16, self.fps)
        align = rs.align(rs.stream.color)

        started = False
        try:
            pipeline.start(cfg)
            started = True
            return self._frames(pipeline, align)
        finally:
            if started:
                pipeline.stop()

    # ---------- detection ----------
    @staticmethod
    def find_red_mask(bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, RED_LOW_1, RED_HIGH_1),
            cv2.inRange(hsv, RED_LOW_2, RED_HIGH_2))
        # Close first: white stripes split the red into bands, and without
        # this the box is found as several separate blobs.
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def largest_box(mask, min_area=MIN_AREA_PX, min_fill=MIN_FILL):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w * h == 0:
                continue
            # A box is roughly convex, so it should fill most of its bounding
            # rectangle. This rejects scattered red clutter.
            if area / float(w * h) < min_fill:
                continue
            if best is None or area > best[0]:
                best = (area, x, y, w, h)
        return best

    @staticmethod
    def robust_depth(depth_frame, cx, cy, half=5):
        """Median depth over a small patch.

        A single pixel is often a dropout (0.0) on shiny or dark surfaces, so
        reading the centroid alone gives spurious zeros.
        """
        vals = []
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                d = depth_frame.get_distance(cx + dx, cy + dy)
                if d > 0:
                    vals.append(d)
        return float(np.median(vals)) if vals else 0.0

    def detect(self, color=None, depth=None):
        """Find the box. Returns a BoxObservation or None."""
        if color is None or depth is None:
            color, depth = self.capture()

        mask = self.find_red_mask(color)
        found = self.largest_box(mask)
        if found is None:
            return None

        area, x, y, w, h = found
        cx, cy = x + w // 2, y + h // 2

        dist = self.robust_depth(depth, cx, cy)
        if dist <= 0:
            return None

        intr = self._intrinsics
        px, py, pz = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], dist)

        bearing = -math.degrees(math.atan2(px, pz))   # + = left
        elevation = -math.degrees(math.atan2(py, pz))  # + = above

        # Real width from angular width at the measured distance.
        left = rs.rs2_deproject_pixel_to_point(intr, [x, cy], dist)
        right = rs.rs2_deproject_pixel_to_point(intr, [x + w, cy], dist)
        width_m = abs(right[0] - left[0])

        return BoxObservation(
            cx=cx, cy=cy, w=w, h=h, area=area,
            distance_m=dist,
            bearing_deg=bearing,
            elevation_deg=elevation,
            x_m=px, y_m=py, z_m=pz,
            height_m=-py - CAMERA_TO_SHOULDER_DROP,
            width_m=width_m,
        )

    # ---------- debug ----------
    def annotate(self, color, box, mask=None):
        img = color.copy()
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)
        if box:
            x0 = box.cx - box.w // 2
            y0 = box.cy - box.h // 2
            cv2.rectangle(img, (x0, y0), (x0 + box.w, y0 + box.h),
                          (0, 255, 0), 2)
            cv2.circle(img, (box.cx, box.cy), 5, (0, 0, 255), -1)
            for i, line in enumerate([
                    f"dist {box.distance_m:.2f} m",
                    f"bear {box.bearing_deg:+.1f} deg",
                    f"elev {box.elevation_deg:+.1f} deg",
                    f"w    {box.width_m * 100:.0f} cm",
                    box.turn_direction()]):
                cv2.putText(img, line, (10, 25 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(img, "no box", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if mask is not None:
            small = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
                               (w // 4, h // 4))
            img[0:h // 4, w - w // 4:w] = small
        return img


# ============================================================
def cmd_once(det, save):
    color, depth = det.capture()
    mask = det.find_red_mask(color)
    box = det.detect(color, depth)

    if box is None:
        print("No box found.")
        print(f"  red pixels: {int(mask.sum() // 255)}")
        print("  if that is near zero the colour thresholds need tuning:")
        print("    python3 g1_vision.py --tune")
    else:
        print(box.describe())
        print(f"  pixel      ({box.cx}, {box.cy})  {box.w}x{box.h}")
        print(f"  camera xyz ({box.x_m:+.3f}, {box.y_m:+.3f}, "
              f"{box.z_m:+.3f}) m")
        print(f"  bearing    {box.bearing_deg:+.1f} deg  -> "
              f"{box.turn_direction()}")
        print(f"  elevation  {box.elevation_deg:+.1f} deg")
        print(f"  height     {box.height_m:+.2f} m relative to shoulder")

    if save:
        cv2.imwrite(save, det.annotate(color, box, mask))
        print(f"\nwrote {save}")


def cmd_preview(det, save, interval):
    print("Detecting every"
          f" {interval:.1f}s. Ctrl-C to stop.\n")
    n = 0
    try:
        while True:
            color, depth = det.capture()
            mask = det.find_red_mask(color)
            box = det.detect(color, depth)
            n += 1
            if box:
                print(f"[{n:>3}] {box.describe()}  -> "
                      f"{box.turn_direction()}")
            else:
                print(f"[{n:>3}] no box  "
                      f"(red px {int(mask.sum() // 255)})")
            if save:
                cv2.imwrite(save, det.annotate(color, box, mask))
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nstopped")


def cmd_tune(det, save):
    """Report what red is actually present, so thresholds can be set from
    measurements rather than guesses."""
    color, _ = det.capture()
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    mask = det.find_red_mask(color)
    px = int(mask.sum() // 255)
    total = mask.shape[0] * mask.shape[1]
    print(f"red pixels with current thresholds: {px} "
          f"({100.0 * px / total:.1f}% of frame)")

    if px > 0:
        sel = hsv[mask > 0]
        print("\nHSV of the detected region:")
        for i, name in enumerate("HSV"):
            ch = sel[:, i]
            print(f"  {name}: min {ch.min():3d}  "
                  f"p5 {np.percentile(ch, 5):5.1f}  "
                  f"median {np.median(ch):5.1f}  "
                  f"p95 {np.percentile(ch, 95):5.1f}  "
                  f"max {ch.max():3d}")
    else:
        print("\nNothing matched. Dominant hues in frame:")
        hist = cv2.calcHist([hsv], [0], None, [18], [0, 180]).flatten()
        for i, c in enumerate(hist):
            if c > total * 0.02:
                print(f"  hue {i * 10:3d}-{i * 10 + 9:3d}: "
                      f"{100.0 * c / total:5.1f}%")
        print("\nRed is hue 0-10 or 170-180. If the box reads as another")
        print("hue, adjust RED_LOW_*/RED_HIGH_* at the top of this file.")

    if save:
        cv2.imwrite(save, det.annotate(color, det.detect(color, _)
                                       if False else None, mask))
        print(f"\nwrote {save}")


def main():
    ap = argparse.ArgumentParser(description="G1 box detection")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--save", default="/tmp/box_detect.jpg")
    ap.add_argument("--warmup", type=int, default=WARMUP_FRAMES)
    args = ap.parse_args()

    try:
        det = BoxDetector(warmup=args.warmup)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    if args.tune:
        cmd_tune(det, args.save)
    elif args.preview:
        cmd_preview(det, args.save, args.interval)
    else:
        cmd_once(det, args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
