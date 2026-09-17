#!/usr/bin/env python3
"""
G1 SLAM Client v5 - native slam_operate service
================================================

All payloads below were CAPTURED FROM THE TABLET'S OWN TRAFFIC on
/api/slam_operate/request during a successful navigation run. They are not
guesses.

THE WORKING SEQUENCE (tablet-verified)
--------------------------------------
  1901  close slam            parameter: ""
  1931  load map, block N     {"data":{"address":"<map>.pcd","blockNum":N}}
                              repeated 1..N until the map is fully loaded
                              (25 blocks for a 721K map)
  1910  set current map       {"data":{"address":"<dir>/currMapInfo.txt",
                                       "params":"<map_stem>"}}
  1804  init pose             {"data":{"x":..,"y":..,"z":0,
                                       "q_x":0,"q_y":0,"q_z":..,"q_w":..,
                                       "address":"<map>.pcd"}}
  1102  navigate              {"data":{"targetPose":{"x":..,"y":..,"z":0,
                                       "q_x":0,"q_y":0,"q_z":0,"q_w":0},
                                       "mode":1}}

Note the asymmetry: 1804 takes a FLAT pose, 1102 nests it under "targetPose"
and adds "mode". Getting this wrong returns errorCode 502 "Json format error".

Poses are effectively 2D: z=0, only q_z/q_w carry yaw. In the captured 1102
call all four quaternion components were 0 - not a valid rotation - which
suggests orientation is ignored when mode=1.

API IDS
-------
  documented:   1801 start mapping | 1802 end mapping | 1804 init pose
                1102 navigate | 1201 pause | 1202 resume | 1901 close slam
  discovered:   1910 set current map | 1911 list files | 1912 read file
                1931 load/download pcd (chunked) | 1934 png (chunked)
                1936 relocalization status -> {"isRelocation": bool}
                1203 unknown

MAP STORAGE
-----------
  /unitree/data/unitree_slam/  on the ROBOT (PC1, .161) - NOT the Jetson.
  Filenames: base64(name)_YYYYMMDD.pcd
    TGFi_20260818.pcd           -> "Lab"
    VXlndXloZ2poZw_20260904.pcd -> "Uyguyhgjhg"

/slam_info IS MULTIPLEXED by a "type" field:
  mapping_info  currentPose as QUATERNION - publishes during MAPPING
  ctrl_info     nav state machine, Euler poses, targetNodeName, is_arrived
  robot_data    motorTemp etc.

USAGE
-----
    source ~/unitree_ros2/setup.sh

    python3 g1_slam_client.py --list-maps
    python3 g1_slam_client.py --load-map TGFi_20260818.pcd
    python3 g1_slam_client.py --status

    python3 g1_slam_client.py --init-pose --x -1.056 --y 0.415 --yaw 2.03
    python3 g1_slam_client.py --goto point1 --i-have-cleared-the-area

    # one-shot: load map, set current, init pose
    python3 g1_slam_client.py --prepare TGFi_20260818.pcd --x -1.056 --y 0.415
"""

import argparse
import base64
import datetime
import json
import math
import os
import sys
import time

import rclpy
from rclpy.node import Node

try:
    from unitree_api.msg import Request, Response
except ImportError:
    print("ERROR: cannot import unitree_api.msg -> source ~/unitree_ros2/setup.sh")
    sys.exit(1)

from std_msgs.msg import String


TOPIC_REQ  = "/api/slam_operate/request"
TOPIC_RES  = "/api/slam_operate/response"
TOPIC_INFO = "/slam_info"

ROBOT_MAP_DIR = "/unitree/data/unitree_slam"
CURR_MAP_INFO = f"{ROBOT_MAP_DIR}/currMapInfo.txt"

WAYPOINT_FILE = os.path.expanduser("~/g1_waypoints.json")
DOWNLOAD_DIR  = os.path.expanduser("~/g1_maps")

API = {
    "start_mapping": 1801, "end_mapping": 1802, "init_pose": 1804,
    "navigate": 1102, "pause_nav": 1201, "resume_nav": 1202,
    "close_slam": 1901, "set_curr_map": 1910, "list_files": 1911,
    "read_file": 1912, "load_pcd": 1931, "load_png": 1934,
    "reloc_status": 1936, "unknown_1203": 1203,
}
API_NAMES = {v: k for k, v in API.items()}

ZERO_EPS = 1e-9


def quat_to_yaw(qw, qx, qy, qz):
    return math.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz))


def yaw_to_quat(yaw):
    """2D pose convention used by this service: only q_z/q_w populated."""
    return math.cos(yaw / 2.0), math.sin(yaw / 2.0)   # q_w, q_z


def encode_map_name(name, date=None):
    d = (date or datetime.date.today()).strftime("%Y%m%d")
    return f"{base64.b64encode(name.encode()).decode().rstrip('=')}_{d}.pcd"


def decode_map_name(filename):
    stem = os.path.basename(filename).rsplit(".", 1)[0]
    if "_" not in stem:
        return filename
    b64 = stem.rsplit("_", 1)[0]
    try:
        return base64.b64decode(b64 + "=" * (-len(b64) % 4)).decode()
    except Exception:
        return filename


def map_stem(path):
    """TGFi_20260818.pcd -> TGFi_20260818  (what 1910 wants in 'params')"""
    return os.path.basename(path).rsplit(".", 1)[0]


def full_map_path(name):
    return name if name.startswith("/") else f"{ROBOT_MAP_DIR}/{name}"


class WaypointStore:
    def __init__(self, path=WAYPOINT_FILE):
        self.path = path
        self.data = {"map": None, "waypoints": {}}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"[WAYPOINTS] {path}: {e}")

    def save(self):
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def set_map(self, p):
        self.data["map"] = p
        self.save()

    def add(self, n, p):
        self.data["waypoints"][n] = p
        self.save()

    def get(self, n):
        return self.data["waypoints"].get(n)

    def names(self):
        return list(self.data["waypoints"].keys())


class G1SlamClient(Node):

    def __init__(self):
        super().__init__("g1_slam_client")
        self.pub = self.create_publisher(Request, TOPIC_REQ, 10)
        self.create_subscription(Response, TOPIC_RES, self._on_res, 10)
        self.create_subscription(String, TOPIC_INFO, self._on_info, 10)
        self.responses = []
        self.pose = None
        self.nav_pose = None
        self.nav = None
        self.types = {}
        self._id = 0

    def _on_res(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except Exception:
            payload = {"_raw": msg.data}
        self.responses.append({
            "payload": payload,
            "binary": bytes(msg.binary) if msg.binary else b"",
            "code": getattr(msg.header.status, "code", None)})

    def _on_info(self, msg):
        try:
            o = json.loads(msg.data)
        except Exception:
            return
        t = o.get("type", "_untyped")
        self.types[t] = self.types.get(t, 0) + 1

        if t == "mapping_info":
            cp = o.get("data", {}).get("currentPose") or {}
            x, y, z = cp.get("x", 0.), cp.get("y", 0.), cp.get("z", 0.)
            qw, qx = cp.get("q_w", 1.), cp.get("q_x", 0.)
            qy, qz = cp.get("q_y", 0.), cp.get("q_z", 0.)
            if all(abs(v) < ZERO_EPS for v in (x, y, z, qx, qy, qz)):
                return
            self.pose = {"x": x, "y": y, "z": z, "q_w": qw, "q_x": qx,
                         "q_y": qy, "q_z": qz,
                         "yaw": quat_to_yaw(qw, qx, qy, qz)}
        elif t == "ctrl_info":
            d = o.get("data", {})
            sm = d.get("stateMachine", {})
            self.nav = {"ctrName": sm.get("ctrName", "?"),
                        "state": sm.get("state", "?"),
                        "is_arrived": d.get("is_arrived", False),
                        "pct": d.get("progress", {}).get(
                            "completion_percentage", 0.0),
                        "targetNodeName": d.get("targetNodeName"),
                        "obstacle": d.get("obsInfo", {}).get("state", False),
                        "info": o.get("info", "")}
            # While RELOCALIZED, ctrl_info carries the live pose in Euler.
            # This is the waypoint source in navigation mode (mapping_info
            # only publishes during a mapping session).
            cp = d.get("currentPose") or {}
            x, y = cp.get("x", 0.0), cp.get("y", 0.0)
            yaw = cp.get("yaw", 0.0)
            if not (abs(x) < ZERO_EPS and abs(y) < ZERO_EPS
                    and abs(yaw) < ZERO_EPS):
                self.nav_pose = {"x": x, "y": y, "z": cp.get("z", 0.0),
                                 "yaw": yaw}

    def call(self, api_id, parameter=None, timeout=15.0, quiet=False):
        self.responses.clear()
        self._id += 1
        r = Request()
        r.header.identity.id = self._id
        r.header.identity.api_id = api_id
        r.header.lease.id = 0
        r.header.policy.priority = 0
        r.header.policy.noreply = False
        r.parameter = json.dumps(parameter) if parameter is not None else ""
        if not quiet:
            print(f"[REQ] {api_id} ({API_NAMES.get(api_id,'?')}) {r.parameter}")
        self.pub.publish(r)

        end = time.time() + timeout
        while time.time() < end and not self.responses:
            rclpy.spin_once(self, timeout_sec=0.1)
        if not self.responses:
            if not quiet:
                print(f"[RES] no response within {timeout}s")
            return None
        res = self.responses[0]
        if not quiet:
            b = f" +{len(res['binary'])}B" if res["binary"] else ""
            print(f"[RES] status={res['code']}{b} {res['payload']}")
        return res

    def spin(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)

    def wait_pose(self, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end and self.pose is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.pose

    def reloc_status(self):
        res = self.call(API["reloc_status"], {"data": {}}, quiet=True)
        if res:
            return res["payload"].get("data", {}).get("isRelocation")
        return None


def fmt(p):
    return (f"x={p['x']:+.3f} y={p['y']:+.3f} "
            f"yaw={math.degrees(p['yaw']):+7.2f}deg")


# ============================================================
# CONFIRMED SEQUENCE
# ============================================================
def load_map(node, map_path, max_blocks=400, save_local=None):
    """1931 - stream the map in blocks. Tablet used 25 for a 721K file.

    This is what makes the map available to the SLAM service. The binary
    coming back is also the file itself, so it doubles as a download.
    """
    print(f"Loading map: {map_path}  ({decode_map_name(map_path)})")
    chunks = []
    for n in range(1, max_blocks + 1):
        res = node.call(API["load_pcd"],
                        {"data": {"address": map_path, "blockNum": n}},
                        timeout=20.0, quiet=True)
        if res is None:
            print(f"  block {n}: no response - stopping")
            break
        if not res["payload"].get("succeed", True):
            info = res["payload"].get("info", "failed")
            print(f"  block {n}: {info} - end of file")
            break
        data = res["binary"]
        if not data:
            d = res["payload"].get("data", {})
            for k in ("data", "content", "block", "buffer"):
                if isinstance(d, dict) and isinstance(d.get(k), str):
                    try:
                        data = base64.b64decode(d[k])
                        break
                    except Exception:
                        pass
        if not data:
            print(f"  block {n}: empty - end of file")
            break
        chunks.append(data)
        print(f"  block {n}: {len(data)}B", end="\r", flush=True)

    total = sum(len(c) for c in chunks)
    print(f"\n  {len(chunks)} blocks, {total} bytes")

    if save_local and chunks:
        os.makedirs(os.path.dirname(save_local) or ".", exist_ok=True)
        with open(save_local, "wb") as f:
            for c in chunks:
                f.write(c)
        print(f"  saved -> {save_local}")
    return len(chunks)


def set_current_map(node, map_path):
    """1910 - tell the service which map is active. Missing this was why
    1804 never took effect."""
    return node.call(API["set_curr_map"], {"data": {
        "address": CURR_MAP_INFO,
        "params": map_stem(map_path)}})


def init_pose(node, map_path, x, y, yaw):
    """1804 - FLAT pose + address. 2D: z=0, only q_z/q_w."""
    qw, qz = yaw_to_quat(yaw)
    return node.call(API["init_pose"], {"data": {
        "x": x, "y": y, "z": 0,
        "q_x": 0, "q_y": 0, "q_z": qz, "q_w": qw,
        "address": map_path}}, timeout=30.0)


def navigate(node, x, y, mode=1, yaw=None):
    """1102 - pose NESTED under targetPose, plus mode.

    The captured tablet call sent all four quaternion components as 0, which
    is not a valid rotation - orientation appeared to be ignored. Passing a
    real yaw here is EXPERIMENTAL: if the robot ends up facing an arbitrary
    direction, orientation is not honoured in this mode.

    `mode` is undocumented. The tablet used 1. Other values may select
    different planner behaviour (e.g. obstacle avoidance vs stop-on-obstacle)
    - probe with --mode 0/2/3.
    """
    if yaw is None:
        q = {"q_x": 0, "q_y": 0, "q_z": 0, "q_w": 0}
    else:
        qw, qz = yaw_to_quat(yaw)
        q = {"q_x": 0, "q_y": 0, "q_z": qz, "q_w": qw}
    return node.call(API["navigate"], {"data": {
        "targetPose": dict({"x": x, "y": y, "z": 0}, **q),
        "mode": mode}}, timeout=20.0)


def relocalize_search(node, map_path, cx, cy, radius, step, yaw_steps):
    """Brute-force the seed pose.

    errorCode 509 ("current location matching degree is low") means the
    service IS scan-matching against the map - the guess was just too far off
    to converge. So sweep candidate poses around an estimate until one takes.
    """
    xs = [cx + dx for dx in frange(-radius, radius, step)]
    ys = [cy + dy for dy in frange(-radius, radius, step)]
    yaws = [i * 2 * math.pi / yaw_steps for i in range(yaw_steps)]
    total = len(xs) * len(ys) * len(yaws)
    print(f"Searching {total} candidate poses "
          f"({len(xs)}x{len(ys)} positions, {yaw_steps} headings)")
    print("Robot must stay STILL for this to be meaningful.\n")

    n = 0
    for x in xs:
        for y in ys:
            for yaw in yaws:
                n += 1
                res = node.call(API["init_pose"], {"data": {
                    "x": x, "y": y, "z": 0, "q_x": 0, "q_y": 0,
                    "q_z": yaw_to_quat(yaw)[1], "q_w": yaw_to_quat(yaw)[0],
                    "address": map_path}}, timeout=20.0, quiet=True)
                ok = res and res["payload"].get("succeed")
                print(f"  [{n}/{total}] x={x:+.2f} y={y:+.2f} "
                      f"yaw={math.degrees(yaw):+6.1f} -> "
                      f"{'OK' if ok else res['payload'].get('info','fail') if res else 'no reply'}")
                if ok:
                    node.spin(2.0)
                    if node.reloc_status():
                        print(f"\nRELOCALIZED at x={x:.3f} y={y:.3f} "
                              f"yaw={math.degrees(yaw):.1f}deg")
                        return {"x": x, "y": y, "yaw": yaw}
    print("\nNo candidate converged. Widen --radius or move the robot to a "
          "more distinctive spot (corners and walls match better than open "
          "floor).")
    return None


def frange(a, b, step):
    out, v = [], a
    while v <= b + 1e-9:
        out.append(round(v, 4))
        v += step
    return out


# ============================================================
# COMMANDS
# ============================================================
def cmd_list_maps(node, directory, ext):
    res = node.call(API["list_files"],
                    {"data": {"address": directory, "extension": ext}})
    if not res:
        return
    paths = res["payload"].get("data", {}).get("paths", [])
    print()
    for p in paths:
        print(f"  {os.path.basename(p):<40} {decode_map_name(p)}")


def cmd_status(node):
    print(f"isRelocation: {node.reloc_status()}")
    node.call(API["read_file"], {"data": {"address": CURR_MAP_INFO}})
    node.spin(2.0)
    if node.nav:
        n = node.nav
        print(f"nav: ctrName={n['ctrName']} state={n['state']} "
              f"arrived={n['is_arrived']} obstacle={n['obstacle']}")
    else:
        print("nav: no ctrl_info received")


def cmd_prepare(node, map_name, x, y, yaw, store, blocks):
    """Full tablet-verified sequence: close -> load -> set current -> init."""
    path = full_map_path(map_name)

    print("=== 1901 close slam ===")
    node.call(API["close_slam"], None)
    time.sleep(1.0)

    print("\n=== 1931 load map ===")
    n = load_map(node, path, max_blocks=blocks)
    if n == 0:
        print("Map load produced nothing - aborting.")
        return

    print("\n=== 1910 set current map ===")
    set_current_map(node, path)

    print(f"\n=== 1804 init pose  x={x} y={y} yaw={yaw} ===")
    init_pose(node, path, x, y, yaw)
    store.set_map(path)

    print("\n=== checking ===")
    node.spin(3.0)
    print(f"isRelocation: {node.reloc_status()}")
    if node.nav:
        print(f"nav: ctrName={node.nav['ctrName']} state={node.nav['state']}")
    print("\nIf isRelocation is True, --goto should work.")


def cmd_goto(node, name, store, mode, face_yaw):
    p = store.get(name)
    if p is None:
        print(f"[ERROR] No waypoint '{name}'. Known: {store.names()}")
        return

    rel = node.reloc_status()
    print(f"isRelocation: {rel}")
    if rel is False:
        print("[WARN] Not relocalized - run --prepare first. Sending anyway.")

    yaw = face_yaw if face_yaw is not None else p.get("yaw")
    print(f"Target '{name}': x={p['x']:+.3f} y={p['y']:+.3f}"
          + (f" facing {math.degrees(yaw):+.1f}deg" if yaw is not None else ""))
    navigate(node, p["x"], p["y"], mode, yaw)

    print("\nMonitoring. Ctrl-C stops WATCHING, not the robot -")
    print("use --pause-nav or the remote to actually stop it.\n")
    stalled = 0
    last_pct = None
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.nav:
                n = node.nav
                here = ""
                if node.nav_pose:
                    here = (f" at x={node.nav_pose['x']:+.2f} "
                            f"y={node.nav_pose['y']:+.2f}")
                print(f"  {n['pct']:5.1f}% arrived={n['is_arrived']} "
                      f"obs={n['obstacle']} {n['state']}{here}    ",
                      end="\r", flush=True)
                if n["obstacle"]:
                    stalled += 1
                    if stalled == 25:
                        print("\n[OBSTACLE] Robot has stopped for an obstacle.")
                        print("  mode=%d does not appear to route around it."
                              % mode)
                        print("  Try a different --mode (0, 2, 3) to see if")
                        print("  another planner mode avoids rather than stops.")
                else:
                    stalled = 0
                if n["is_arrived"] and n["pct"] > 0:
                    print("\n[ARRIVED]")
                    break
                last_pct = n["pct"]
    except KeyboardInterrupt:
        print("\nStopped watching.")


def cmd_watch(node, seconds):
    print("Watching /slam_info. Ctrl-C to stop.\n")
    try:
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.2)
            line = ""
            if node.pose:
                line += f"POSE {fmt(node.pose)}  "
            if node.nav:
                n = node.nav
                line += (f"| NAV {n['ctrName']}/{n['state']} {n['pct']:.0f}% "
                         f"arrived={n['is_arrived']} obs={n['obstacle']}")
            if line:
                print(f"  {line[:160]}", end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    print("\n\nMessage types:")
    for t, c in sorted(node.types.items()):
        print(f"  {t:<16} {c}")


def cmd_save_waypoint(node, name, store):
    """Pose comes from mapping_info while MAPPING, ctrl_info while
    RELOCALIZED. Record waypoints in the SAME mode you'll navigate in -
    a fresh mapping session starts its own frame, which will not match a
    saved map's frame."""
    end = time.time() + 10.0
    while time.time() < end and node.pose is None and node.nav_pose is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    if node.pose:
        p, src = node.pose, "mapping_info"
    elif node.nav_pose:
        yaw = node.nav_pose["yaw"]
        qw, qz = yaw_to_quat(yaw)
        p = {"x": node.nav_pose["x"], "y": node.nav_pose["y"],
             "z": node.nav_pose["z"], "yaw": yaw,
             "q_w": qw, "q_x": 0.0, "q_y": 0.0, "q_z": qz}
        src = "ctrl_info (relocalized)"
    else:
        print("[ERROR] No pose in 10s.")
        print("  Start a mapping session, or relocalize with --prepare.")
        return

    print(f"[POSE from {src}] {fmt(p)}")
    store.add(name, p)
    print(f"[SAVED] '{name}'")


def cmd_list_waypoints(store):
    print(f"Map: {store.data.get('map')}")
    for n, p in store.data["waypoints"].items():
        print(f"  {n:<16} " + (fmt(p) if "yaw" in p else "(old - re-record)"))
    if not store.names():
        print("  (none)")


# ============================================================
def main():
    ap = argparse.ArgumentParser(description="G1 native SLAM client v5")

    ap.add_argument("--list-maps", action="store_true")
    ap.add_argument("--dir", default=ROBOT_MAP_DIR)
    ap.add_argument("--ext", default=".pcd")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--read-file", metavar="PATH")

    ap.add_argument("--load-map", metavar="NAME",
                    help="1931 - stream map blocks to the service")
    ap.add_argument("--download", metavar="NAME",
                    help="same as --load-map but also writes the file locally")
    ap.add_argument("--blocks", type=int, default=400)
    ap.add_argument("--out-dir", default=DOWNLOAD_DIR)

    ap.add_argument("--prepare", metavar="MAP",
                    help="close + load + set-current + init-pose")
    ap.add_argument("--x", type=float, default=0.0)
    ap.add_argument("--y", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=None, help="radians")
    ap.add_argument("--yaw-deg", type=float, default=None,
                    help="degrees (--watch prints degrees)")
    ap.add_argument("--from-waypoint", metavar="NAME",
                    help="use a saved waypoint as the init pose")

    ap.add_argument("--relocalize-search", metavar="MAP",
                    help="sweep seed poses until relocalization converges")
    ap.add_argument("--radius", type=float, default=1.0,
                    help="search half-width in metres")
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--yaw-steps", type=int, default=8)

    ap.add_argument("--init-pose", action="store_true")
    ap.add_argument("--set-current-map", metavar="MAP")
    ap.add_argument("--map", metavar="MAP", default="")

    ap.add_argument("--start-mapping", action="store_true")
    ap.add_argument("--stop-mapping", metavar="NAME")
    ap.add_argument("--close-slam", action="store_true")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--watch-seconds", type=float, default=600.0)
    ap.add_argument("--save-waypoint", metavar="NAME")
    ap.add_argument("--list-waypoints", action="store_true")

    ap.add_argument("--goto", metavar="NAME")
    ap.add_argument("--mode", type=int, default=1,
                    help="1102 planner mode; tablet used 1. Probe 0/2/3 for "
                         "obstacle-avoidance behaviour.")
    ap.add_argument("--face-yaw-deg", type=float, default=None,
                    help="EXPERIMENTAL: target heading in degrees. The "
                         "tablet sent a zero quaternion, so orientation may "
                         "be ignored.")
    ap.add_argument("--pause-nav", action="store_true")
    ap.add_argument("--resume-nav", action="store_true")
    ap.add_argument("--i-have-cleared-the-area", action="store_true")

    ap.add_argument("--probe", type=int, metavar="API_ID")
    ap.add_argument("--param", metavar="JSON")
    args = ap.parse_args()

    store = WaypointStore()

    if args.list_waypoints:
        cmd_list_waypoints(store)
        return

    if args.goto and not args.i_have_cleared_the_area:
        print("REFUSING: --goto makes the robot WALK.")
        print("Re-run with --i-have-cleared-the-area, area clear, "
              "remote in hand.")
        return

    x, y = args.x, args.y
    if args.yaw_deg is not None:
        yaw = math.radians(args.yaw_deg)
    elif args.yaw is not None:
        yaw = args.yaw
    else:
        yaw = 0.0
    if args.from_waypoint:
        wp = store.get(args.from_waypoint)
        if wp is None:
            print(f"No waypoint '{args.from_waypoint}'")
            return
        x, y, yaw = wp["x"], wp["y"], wp.get("yaw", 0.0)

    face_yaw = (math.radians(args.face_yaw_deg)
                if args.face_yaw_deg is not None else None)

    rclpy.init()
    node = G1SlamClient()
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    try:
        if args.list_maps:
            cmd_list_maps(node, args.dir, args.ext)
        elif args.status:
            cmd_status(node)
        elif args.read_file:
            r = node.call(API["read_file"], {"data": {"address": args.read_file}})
            if r and r["binary"]:
                print(r["binary"].decode("utf-8", errors="replace"))
        elif args.load_map:
            load_map(node, full_map_path(args.load_map), args.blocks)
        elif args.download:
            p = full_map_path(args.download)
            load_map(node, p, args.blocks,
                     save_local=os.path.join(args.out_dir,
                                             os.path.basename(p)))
        elif args.prepare:
            cmd_prepare(node, args.prepare, x, y, yaw, store, args.blocks)
        elif args.set_current_map:
            set_current_map(node, full_map_path(args.set_current_map))
        elif args.init_pose:
            m = args.map or store.data.get("map")
            if not m:
                print("--init-pose needs --map")
                return
            init_pose(node, full_map_path(m), x, y, yaw)
            node.spin(3.0)
            print(f"isRelocation: {node.reloc_status()}")
        elif args.start_mapping:
            node.call(API["start_mapping"], {"data": {"slam_type": "indoor"}})
        elif args.stop_mapping:
            fn = args.stop_mapping
            path = fn if fn.startswith("/") \
                else f"{ROBOT_MAP_DIR}/{encode_map_name(fn)}"
            print(f"Saving as: {path}")
            if node.call(API["end_mapping"], {"data": {"address": path}},
                         timeout=60.0):
                store.set_map(path)
        elif args.close_slam:
            node.call(API["close_slam"], None)
        elif args.watch:
            cmd_watch(node, args.watch_seconds)
        elif args.save_waypoint:
            cmd_save_waypoint(node, args.save_waypoint, store)
        elif args.relocalize_search:
            p = full_map_path(args.relocalize_search)
            print("=== loading map first ===")
            load_map(node, p, args.blocks)
            set_current_map(node, p)
            found = relocalize_search(node, p, x, y,
                                      args.radius, args.step, args.yaw_steps)
            if found:
                store.set_map(p)
        elif args.goto:
            cmd_goto(node, args.goto, store, args.mode, face_yaw)
        elif args.pause_nav:
            node.call(API["pause_nav"], {})
        elif args.resume_nav:
            node.call(API["resume_nav"], {})
        elif args.probe is not None:
            p = json.loads(args.param) if args.param else {"data": {}}
            node.call(args.probe, p, timeout=20.0)
        else:
            ap.print_help()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
