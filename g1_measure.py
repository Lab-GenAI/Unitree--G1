#!/usr/bin/env python3
"""
G1 Measure - object dimensions from the RealSense
==================================================

Point the robot at something and get its size in centimetres.

HOW THE WORK IS SPLIT
---------------------
The LLM says roughly WHERE the object is. Depth geometry says HOW BIG it is.

Asking the LLM for corner pixels and reading depth at those exact points was
the obvious approach and it is the weaker one. Vision models place a loose
bounding box well - Sonnet boxes a cardboard box tightly and reliably - but
they are not trained to regress precise corner coordinates, and a corner that
is 15 px out lands on the background, which reads as a depth metres away.

So the LLM only has to be approximately right, which is what it is good at.
Inside that region there are thousands of depth samples, and the measurement
comes from all of them:

    1. LLM: roughly where is the object
    2. Keep depth points inside that region
    3. Discard background - anything much further than the object's own median
    4. Fit the front face as a plane (RANSAC)
    5. Project the points onto that plane and measure their extent

Averaging over thousands of points is what makes this survive the noise that
kills a four-corner measurement.

WHAT IT CAN AND CANNOT MEASURE
------------------------------
Width and height, seen face-on: good.
Depth, front to back: ONLY if the object is angled enough that a second face
is visible. Face-on, the back simply is not in the data, and the reported
depth is a lower bound - the visible thickness, not the true one.

ACCURACY
--------
The D435i is specified around 2% of range. At 1 m that is ~2 cm, so a 30 cm
box carries roughly 5-8% error on each dimension. Good enough for "about 30
by 20 centimetres", not for anything that needs to be right to the
millimetre. Every result prints its own spread so you can see the confidence
rather than guess at it.

RUN
---
    python3 g1_measure.py                      # measure what is in front
    python3 g1_measure.py --what "the red box"
    python3 g1_measure.py --weight             # also estimate mass
    python3 g1_measure.py --save /tmp/m.jpg
"""

import argparse
import base64
import json
import math
import os
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: pip3 install opencv-python")
    sys.exit(1)

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not importable")
    sys.exit(1)

try:
    import anthropic
except ImportError:
    print("ERROR: pip3 install anthropic")
    sys.exit(1)


WIDTH, HEIGHT, FPS = 640, 480, 30
WARMUP = 30

LOCATE_MODEL = "claude-opus-5-5"      # accuracy matters more than speed
WEIGHT_MODEL = "claude-opus-5-5"

# Depth points further than this beyond the object's own median are wall or
# floor showing through the edges of the bounding box.
BACKGROUND_MARGIN_M = 0.25
MIN_POINTS = 300

PLANE_ITERS = 150
PLANE_TOL_M = 0.012


LOCATE_PROMPT = """You are looking through a camera on a humanoid robot. The \
image is {w} by {h} pixels.

Find {target}.

Reply with ONLY JSON:
{{"found": <true|false>,
  "x": <left>, "y": <top>, "w": <width>, "h": <height>,
  "what": "<what the object is>",
  "material": "<what it appears to be made of>",
  "confidence": <0-1>}}

Box it as tightly as you can - include the whole object and as little \
background as possible. If you cannot see it, set "found" to false."""


WEIGHT_PROMPT = """A robot measured an object with a depth camera:

  object      : {what}
  material    : {material}
  dimensions  : {w:.1f} x {h:.1f} x {d:.1f} cm{depth_note}

Estimate its mass. Reply with ONLY JSON:
{{"grams": <number>, "range": "<low-high in grams>", \
"reasoning": "<one short sentence>", "confidence": <0-1>}}

Be honest about uncertainty - an empty box and a full one of the same size \
differ enormously, and you cannot see inside."""


class Camera:
    def __init__(self):
        self.intr = None

    def capture(self):
        """Aligned colour and depth. Opened per call so nothing else is
        blocked out of the device."""
        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
        cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
        align = rs.align(rs.stream.color)
        started = False
        try:
            prof = pipe.start(cfg)
            started = True
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            frames = None
            for _ in range(WARMUP):
                frames = align.process(pipe.wait_for_frames())
            c, d = frames.get_color_frame(), frames.get_depth_frame()
            if not c or not d:
                raise RuntimeError("incomplete frame set")
            self.intr = c.profile.as_video_stream_profile().intrinsics
            return (np.asanyarray(c.get_data()).copy(),
                    np.asanyarray(d.get_data()).astype(np.float32) * scale)
        except RuntimeError as e:
            if "busy" in str(e).lower():
                raise RuntimeError(
                    "RealSense is busy - videohub_pc4 holds the colour node. "
                    "Disable videohub in the tablet Service Manager.")
            raise
        finally:
            if started:
                try:
                    pipe.stop()
                except Exception:
                    pass

    def points_in(self, depth, rect):
        """Every 3D point inside a bounding box, in camera coordinates."""
        x, y, w, h = rect
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(WIDTH, int(x + w)), min(HEIGHT, int(y + h))
        if x1 <= x0 or y1 <= y0:
            return None

        sub = depth[y0:y1, x0:x1]
        vs, us = np.nonzero(sub > 0)
        if us.size == 0:
            return None
        z = sub[vs, us]
        us = us + x0
        vs = vs + y0
        px = (us - self.intr.ppx) / self.intr.fx * z
        py = (vs - self.intr.ppy) / self.intr.fy * z
        return np.stack([px, py, z], axis=1)


def strip_background(points, margin=BACKGROUND_MARGIN_M):
    """Drop wall and floor visible through the edges of the bounding box.

    The object is the NEAR cluster. Anything well behind the median depth is
    whatever the box happens to overlap.
    """
    z = points[:, 2]
    near = np.median(z[z < np.percentile(z, 60)])
    keep = points[np.abs(z - near) < margin]
    return keep, float(near)


def fit_plane(points, iters=PLANE_ITERS, tol=PLANE_TOL_M):
    """RANSAC. Returns (normal, d, inlier_mask) or None.

    Purely geometric - no colour involved, which matters because the face LED
    tints everything the RealSense sees.
    """
    n = len(points)
    if n < 50:
        return None
    rng = np.random.default_rng(0)
    best = (0, None, None, 0.0)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal = normal / norm
        d = -float(normal @ p0)
        mask = np.abs(points @ normal + d) < tol
        count = int(mask.sum())
        if count > best[0]:
            best = (count, mask, normal, d)
    if best[1] is None:
        return None
    return best[2], best[3], best[1]


def measure_face(points, normal):
    """Extent of the points across the plane, in metres.

    Builds two axes lying in the plane and reports the span along each. Using
    the 2nd and 98th percentiles rather than min and max keeps a few stray
    points from inflating the answer.
    """
    # Any vector not parallel to the normal gives a starting axis.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(normal @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    ax1 = np.cross(normal, ref)
    ax1 /= np.linalg.norm(ax1)
    ax2 = np.cross(normal, ax1)

    a = points @ ax1
    b = points @ ax2
    span_a = float(np.percentile(a, 98) - np.percentile(a, 2))
    span_b = float(np.percentile(b, 98) - np.percentile(b, 2))

    # Whichever axis points more along camera Y is the vertical one, and its
    # span is the height. The other is the width.
    if abs(ax2[1]) > abs(ax1[1]):
        return span_a, span_b, (ax1, ax2)      # ax2 vertical
    return span_b, span_a, (ax1, ax2)          # ax1 vertical


def measure(cam, color, depth, rect):
    """Dimensions from the depth points inside `rect`."""
    points = cam.points_in(depth, rect)
    if points is None or len(points) < MIN_POINTS:
        return None, "not enough depth data in that region"

    obj, near = strip_background(points)
    if len(obj) < MIN_POINTS:
        return None, f"only {len(obj)} points survived background removal"

    plane = fit_plane(obj)
    if plane is None:
        return None, "could not fit a face to the points"
    normal, d, inliers = plane

    face = obj[inliers]
    if len(face) < MIN_POINTS // 2:
        return None, f"front face has only {len(face)} points"

    width, height, _ = measure_face(face, normal)

    # Thickness along the normal. Face-on this is nearly zero and tells you
    # nothing; it only becomes meaningful when a second face is visible.
    along = obj @ normal
    thickness = float(np.percentile(along, 98) - np.percentile(along, 2))

    # How far the face points scatter off the fitted plane - a direct read on
    # the measurement's own noise.
    residual = float(np.std(np.abs(face @ normal + d)))

    return {
        "width_m": width,
        "height_m": height,
        "thickness_m": thickness,
        "distance_m": near,
        "points": len(obj),
        "face_points": len(face),
        "residual_m": residual,
        "face_on": thickness < 0.04,
    }, None


def encode(color, width=768):
    h, w = color.shape[:2]
    img = color
    scale = 1.0
    if w > width:
        scale = width / w
        img = cv2.resize(color, (width, int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return base64.b64encode(enc.tobytes()).decode(), scale


def ask(client, model, image_b64, prompt, max_tokens=1024):
    content = [{"type": "text", "text": prompt}]
    if image_b64:
        content.insert(0, {"type": "image", "source": {
            "type": "base64", "media_type": "image/jpeg",
            "data": image_b64}})
    resp = client.messages.create(model=model, max_tokens=max_tokens,
                                  extra_body={"output_config": {"effort": "low"}},
                                  messages=[{"role": "user",
                                             "content": content}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def locate(client, color, target):
    b64, scale = encode(color)
    t0 = time.time()
    data = ask(client, LOCATE_MODEL, b64,
               LOCATE_PROMPT.format(w=WIDTH, h=HEIGHT, target=target))
    dt = time.time() - t0
    if not data.get("found"):
        return None, dt
    return {
        "rect": (int(data["x"] / scale), int(data["y"] / scale),
                 int(data["w"] / scale), int(data["h"] / scale)),
        "what": data.get("what", "object"),
        "material": data.get("material", "unknown"),
        "confidence": data.get("confidence", 0.0),
    }, dt


def estimate_weight(client, found, dims):
    note = ("  (front-to-back not visible - the robot is looking at it "
            "face-on, so that figure is a lower bound)"
            if dims["face_on"] else "")
    data = ask(client, WEIGHT_MODEL, None, WEIGHT_PROMPT.format(
        what=found["what"], material=found["material"],
        w=dims["width_m"] * 100, h=dims["height_m"] * 100,
        d=dims["thickness_m"] * 100, depth_note=note), max_tokens=1024)
    return data


def annotate(color, found, dims):
    img = color.copy()
    if found:
        x, y, w, h = found["rect"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, found["what"], (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    if dims:
        lines = [f"{dims['width_m'] * 100:.1f} x "
                 f"{dims['height_m'] * 100:.1f} cm",
                 f"{dims['distance_m']:.2f} m away",
                 f"{dims['face_points']} points"]
        for i, line in enumerate(lines):
            cv2.putText(img, line, (10, 25 + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="G1 object measurement")
    ap.add_argument("--what", default="the main object in front of the robot",
                    help="what to measure")
    ap.add_argument("--weight", action="store_true",
                    help="also estimate the mass")
    ap.add_argument("--save", default="/tmp/measure.jpg")
    ap.add_argument("--model", default=LOCATE_MODEL)
    ap.add_argument("--repeat", type=int, default=1,
                    help="measure N times to see the spread")
    ap.add_argument("--json", action="store_true",
                    help="also print one machine-readable result line, "
                         "prefixed MEASURE_RESULT, as the last line of "
                         "output - for a caller to parse instead of "
                         "scraping the human-readable text above it")
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set in this process")
        return 1
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    cam = Camera()

    results = []
    found = dims = color = None

    for run in range(1, args.repeat + 1):
        color, depth = cam.capture()
        found, dt = locate(client, color, args.what)
        if found is None:
            print(f"Could not find {args.what}.")
            return 1

        if run == 1:
            print(f"\n{found['what']}  ({found['material']}, "
                  f"confidence {found['confidence']:.2f}, {dt * 1000:.0f}ms)")
            print(f"region {found['rect']}")

        dims, err = measure(cam, color, depth, found["rect"])
        if dims is None:
            print(f"Could not measure: {err}")
            return 1
        results.append(dims)

        if args.repeat > 1:
            print(f"  run {run}: {dims['width_m'] * 100:6.1f} x "
                  f"{dims['height_m'] * 100:6.1f} cm  "
                  f"at {dims['distance_m']:.2f} m")

    w = [r["width_m"] * 100 for r in results]
    h = [r["height_m"] * 100 for r in results]

    print(f"\n  width      {np.mean(w):6.1f} cm"
          + (f"   (spread {np.std(w):.1f})" if len(w) > 1 else ""))
    print(f"  height     {np.mean(h):6.1f} cm"
          + (f"   (spread {np.std(h):.1f})" if len(h) > 1 else ""))

    if dims["face_on"]:
        print(f"  thickness  {dims['thickness_m'] * 100:6.1f} cm   "
              "LOWER BOUND - looking at it face-on, the back is not visible")
    else:
        print(f"  thickness  {dims['thickness_m'] * 100:6.1f} cm")

    print(f"\n  distance   {dims['distance_m']:.2f} m")
    print(f"  measured from {dims['face_points']} points on the front face")
    print(f"  surface scatter {dims['residual_m'] * 1000:.1f} mm")

    # The D435i is specified around 2% of range; at this distance that is:
    expected = dims["distance_m"] * 0.02 * 100
    print(f"  expect roughly +/- {expected:.1f} cm from sensor noise alone")

    weight_est, weight_err = None, None
    if args.weight:
        print("\n  estimating mass...")
        try:
            weight_est = estimate_weight(client, found, dims)
            print(f"  about {weight_est['grams']} g  (range {weight_est['range']})")
            print(f"  {weight_est['reasoning']}")
            print(f"  confidence {weight_est.get('confidence', 0):.2f} - it "
                  "cannot see inside, so treat this as a guess")
        except Exception as e:
            weight_err = str(e)
            print(f"  weight estimate failed: {weight_err}")

    if args.save:
        cv2.imwrite(args.save, annotate(color, found, dims))
        print(f"\nwrote {args.save}")

    if args.json:
        result = {
            "what": found["what"], "material": found["material"],
            "width_cm": round(float(np.mean(w)), 1),
            "height_cm": round(float(np.mean(h)), 1),
            "thickness_cm": round(dims["thickness_m"] * 100, 1),
            "thickness_is_lower_bound": dims["face_on"],
            "distance_m": round(dims["distance_m"], 2),
        }
        if weight_est is not None:
            result["grams"] = weight_est.get("grams")
            result["weight_range"] = weight_est.get("range")
            result["weight_confidence"] = weight_est.get("confidence")
        elif weight_err is not None:
            result["weight_error"] = weight_err
        print("MEASURE_RESULT " + json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
