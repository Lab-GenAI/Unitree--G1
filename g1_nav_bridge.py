#!/usr/bin/env python3
"""
G1 Navigation Bridge - slam_operate over unitree_sdk2py
========================================================

Navigation for the voice daemon. Deliberately uses unitree_sdk2py (NOT rclpy)
because g1_voice_daemon_v8.py already initialises that DDS stack for
LowState_. Running rclpy alongside it would put two DDS participants in one
process, which is a good way to reproduce the kind of fault that cost a day
earlier in this project.

Assumes preparation was done MANUALLY beforehand with g1_slam_client_v6.py:

    python3 g1_slam_client_v6.py --prepare <map>.pcd --from-waypoint home
    python3 g1_slam_client_v6.py --status        # expect isRelocation: True

This module only NAVIGATES. It does not map, save, or relocalise.

VERIFIED PAYLOADS (captured from tablet traffic)
------------------------------------------------
  1102 navigate  {"data":{"targetPose":{"x":..,"y":..,"z":0,
                          "q_x":0,"q_y":0,"q_z":0,"q_w":0},"mode":1}}
  1201 pause     {}
  1936 status    {"data":{}} -> {"data":{"isRelocation": bool}}

/slam_info carries type="ctrl_info" with is_arrived, progress, obsInfo -
that is how arrival is detected.

IMPORT NOTE
-----------
The unitree_sdk2py IDL path for Request_/Response_ can vary by SDK version.
If the import below fails, find the real path with:

    grep -rn "class Request_" ~/models/unitree_sdk2_python/
"""

import json
import math
import os
import threading
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber

try:
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import (
        Request_, Response_, RequestHeader_, RequestIdentity_,
        RequestLease_, RequestPolicy_)
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
    IDL_OK = True
except ImportError as e:
    IDL_OK = False
    _IMPORT_ERROR = str(e)


def make_request(api_id, parameter="", req_id=0, priority=0, noreply=False):
    """Build a Request_.

    These IDL types are frozen dataclasses - every field is required at
    construction. `Request_()` followed by attribute assignment raises
    "missing 3 required positional arguments".
    """
    return Request_(
        header=RequestHeader_(
            identity=RequestIdentity_(id=req_id, api_id=api_id),
            lease=RequestLease_(id=0),
            policy=RequestPolicy_(priority=priority, noreply=noreply),
        ),
        parameter=parameter,
        binary=[],
    )


TOPIC_REQ  = "rt/api/slam_operate/request"
TOPIC_RES  = "rt/api/slam_operate/response"
TOPIC_INFO = "rt/slam_info"

WAYPOINT_FILE = os.path.expanduser("~/g1_waypoints.json")

# Named multi-stop routes. Edit here, or add a "routes" object to
# ~/g1_waypoints.json which takes precedence:
#     {"map": ..., "waypoints": {...},
#      "routes": {"tour": ["newstance1", "newstance2", "newstance3"]}}
DEFAULT_ROUTES = {
    "tour": ["newstance1", "newstance2", "newstance3"],
}

# Pause at each stop so the robot settles and the arrival line is audible
# before it sets off again.
STOP_DWELL_SECONDS = 3.0

# ctrl_info keeps reporting the PREVIOUS goal's is_arrived for a moment after
# a new goal is accepted. Ignore arrival reports for this long after sending.
ARRIVAL_GRACE_SECONDS = 3.0

API_NAVIGATE     = 1102
API_PAUSE_NAV    = 1201
API_RESUME_NAV   = 1202
API_RELOC_STATUS = 1936

ZERO_EPS = 1e-9


def norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def clock_direction(rel):
    deg = math.degrees(norm_angle(rel))
    if abs(deg) < 15:
        return "straight ahead"
    if abs(deg) > 165:
        return "directly behind me"
    side = "left" if deg > 0 else "right"
    if abs(deg) < 60:
        return f"slightly to my {side}"
    if abs(deg) < 120:
        return f"to my {side}"
    return f"behind me to the {side}"


class NavBridge:
    """Background navigation the voice loop can start, poll and interrupt."""

    def __init__(self, waypoint_file=WAYPOINT_FILE, on_event=None):
        if not IDL_OK:
            raise RuntimeError(
                f"unitree_sdk2py IDL import failed: {_IMPORT_ERROR}\n"
                "Find the real path with:\n"
                "  grep -rn 'class Request_' ~/models/unitree_sdk2_python/")

        self.on_event = on_event or (lambda msg: None)

        self._pub = ChannelPublisher(TOPIC_REQ, Request_)
        self._pub.Init()

        self._res_sub = ChannelSubscriber(TOPIC_RES, Response_)
        self._res_sub.Init(self._on_response, 10)

        self._info_sub = ChannelSubscriber(TOPIC_INFO, String_)
        self._info_sub.Init(self._on_info, 10)

        self._responses = []
        self._res_lock = threading.Lock()
        self._call_lock = threading.Lock()
        self._req_id = 0

        self.nav = None          # latest ctrl_info snapshot
        self.pose = None         # live pose while navigating

        self._thread = None
        self._stop = threading.Event()
        self.current_destination = None
        self.current_leg = None

        self.waypoints = {}
        self.routes = {}
        self.load_waypoints(waypoint_file)

    # ---------- waypoints ----------
    def load_waypoints(self, path=WAYPOINT_FILE):
        self.routes = dict(DEFAULT_ROUTES)
        try:
            with open(path) as f:
                d = json.load(f)
            self.waypoints = d.get("waypoints", {})
            if isinstance(d.get("routes"), dict):
                self.routes.update(d["routes"])
        except Exception:
            self.waypoints = {}

        # Drop any route referencing a waypoint that no longer exists, rather
        # than failing halfway through a demo.
        valid = {}
        for name, stops in self.routes.items():
            resolved = [self.resolve(p) for p in stops]
            if all(resolved):
                valid[name] = resolved
            else:
                missing = [p for p, r in zip(stops, resolved) if not r]
                print(f"[NAV] route '{name}' disabled - unknown stop(s): "
                      f"{missing}")
        self.routes = valid
        return self.waypoints

    def route_names(self):
        return list(self.routes.keys())

    def places(self):
        return list(self.waypoints.keys())

    def resolve(self, spoken):
        """Loosely match a spoken place name to a waypoint key.

        Whisper will produce 'the kitchen' or 'Point A', not 'kitchen' or
        'pointA', so exact matching would fail constantly.
        """
        if not spoken:
            return None
        s = spoken.lower().strip()
        for junk in ("the ", "to ", "a ", "my "):
            if s.startswith(junk):
                s = s[len(junk):]
        norm = s.replace(" ", "").replace("_", "").replace("-", "")

        for k in self.waypoints:
            if k.lower().replace("_", "").replace("-", "") == norm:
                return k
        for k in self.waypoints:
            kn = k.lower().replace("_", "").replace("-", "")
            if norm in kn or kn in norm:
                return k
        return None

    # ---------- DDS callbacks ----------
    def _on_response(self, msg):
        try:
            payload = json.loads(msg.data) if msg.data else {}
        except Exception:
            payload = {"_raw": getattr(msg, "data", "")}
        with self._res_lock:
            self._responses.append(payload)

    def _on_info(self, msg):
        # /slam_info is multiplexed and high volume. Only ctrl_info is used
        # here, so reject everything else with a substring test before
        # paying for json.loads - that parse was a measurable share of CPU.
        data = msg.data
        if "ctrl_info" not in data:
            return
        try:
            o = json.loads(data)
        except Exception:
            return
        if o.get("type") != "ctrl_info":
            return
        d = o.get("data", {})
        sm = d.get("stateMachine", {})
        self.nav = {
            "state": sm.get("state", "?"),
            "is_arrived": d.get("is_arrived", False),
            "pct": d.get("progress", {}).get("completion_percentage", 0.0),
            "obstacle": d.get("obsInfo", {}).get("state", False),
        }
        cp = d.get("currentPose") or {}
        x, y, yaw = cp.get("x", 0.), cp.get("y", 0.), cp.get("yaw", 0.)
        if not (abs(x) < ZERO_EPS and abs(y) < ZERO_EPS
                and abs(yaw) < ZERO_EPS):
            self.pose = {"x": x, "y": y, "yaw": yaw}

    # ---------- raw call ----------
    def call(self, api_id, parameter=None, timeout=10.0):
        with self._call_lock:
            with self._res_lock:
                self._responses.clear()
            self._req_id += 1
            req = make_request(
                api_id,
                json.dumps(parameter) if parameter is not None else "",
                req_id=self._req_id)
            self._pub.Write(req)

            end = time.time() + timeout
            while time.time() < end:
                with self._res_lock:
                    if self._responses:
                        return self._responses[0]
                time.sleep(0.05)
            return None

    def is_ready(self):
        """True only if the robot is relocalised against a map."""
        r = self.call(API_RELOC_STATUS, {"data": {}}, timeout=5.0)
        if r is None:
            return None
        return r.get("data", {}).get("isRelocation")

    # ---------- navigation ----------
    def is_busy(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, place, mode=1, timeout=180.0):
        """Kick off navigation in the background. Returns (ok, message)."""
        key = self.resolve(place)
        if key is None:
            known = ", ".join(p.replace("_", " ") for p in self.places())
            return False, (f"I don't know where {place} is. "
                           f"I know: {known}." if known
                           else "I don't have any places saved yet.")

        if self.is_busy():
            self.stop()
            time.sleep(0.5)

        self._stop = threading.Event()
        self.current_destination = key
        self._thread = threading.Thread(
            target=self._run, args=(key, mode, timeout), daemon=True)
        self._thread.start()
        return True, f"Heading to {key.replace('_', ' ')}."

    def _navigate_leg(self, key, mode, timeout):
        """Walk to one waypoint. Blocks. Returns (ok, message)."""
        wp = self.waypoints[key]

        # Clear stale arrival state from the previous leg, otherwise the next
        # leg reports success instantly.
        self.nav = None

        res = self.call(API_NAVIGATE, {"data": {
            "targetPose": {"x": wp["x"], "y": wp["y"], "z": 0,
                           "q_x": 0, "q_y": 0, "q_z": 0, "q_w": 0},
            "mode": mode}}, timeout=15.0)

        if res is None:
            return False, "Navigation service didn't respond."
        if not res.get("succeed"):
            return False, res.get("info", "I couldn't start moving.")

        # Give the controller a moment to pick up the new goal before we
        # start believing whatever ctrl_info says. Without this the leftover
        # is_arrived=True from the previous goal is read as instant success.
        time.sleep(ARRIVAL_GRACE_SECONDS)
        self.nav = None

        end = time.time() + timeout
        obstacle_since = None
        while time.time() < end:
            if self._stop.is_set():
                self.call(API_PAUSE_NAV, {}, timeout=5.0)
                return False, "Stopped."

            n = self.nav
            if n:
                # ctrl_info reports completion_percentage 0.0 even when it has
                # genuinely arrived - observed as "0.0% arrived=True
                # state=FINISHED". So do NOT require pct > 0, or this loops
                # until timeout and blocks every later command.
                if n["is_arrived"] or str(n["state"]).upper() == "FINISHED":
                    return True, f"I've arrived at {key.replace('_', ' ')}."
                if n["obstacle"]:
                    obstacle_since = obstacle_since or time.time()
                    if time.time() - obstacle_since > 20.0:
                        self.call(API_PAUSE_NAV, {}, timeout=5.0)
                        return False, "Something's in my way. I've stopped."
                else:
                    obstacle_since = None
            time.sleep(0.2)

        self.call(API_PAUSE_NAV, {}, timeout=5.0)
        return False, f"I gave up trying to reach {key.replace('_', ' ')}."

    def _run(self, key, mode, timeout):
        ok, msg = self._navigate_leg(key, mode, timeout)
        self.on_event(msg)
        self.current_destination = None

    def _run_sequence(self, stops, mode, timeout):
        total = len(stops)
        for i, key in enumerate(stops, 1):
            if self._stop.is_set():
                self.on_event("Stopped.")
                break

            self.current_destination = key
            self.current_leg = (i, total)

            ok, msg = self._navigate_leg(key, mode, timeout)
            if not ok:
                # Abandon the rest of the route rather than blindly pressing
                # on past an obstacle or a failure.
                self.on_event(msg if i == 1 else
                              f"{msg} I was on stop {i} of {total}.")
                break

            if i < total:
                self.on_event(f"{msg} Stop {i} of {total}.")
                time.sleep(STOP_DWELL_SECONDS)
            else:
                self.on_event(f"{msg} That's the last stop.")

        self.current_destination = None
        self.current_leg = None

    def start_route(self, route_name, mode=1, timeout=180.0):
        """Walk a named multi-stop route in order. Returns (ok, message)."""
        name = None
        rn = (route_name or "").lower().strip().replace(" ", "")
        for k in self.routes:
            if k.lower().replace(" ", "") == rn:
                name = k
                break
        if name is None:
            known = ", ".join(self.route_names())
            return False, (f"I don't know a route called {route_name}. "
                           f"I know: {known}." if known
                           else "I don't have any routes set up.")

        stops = self.routes[name]
        if self.is_busy():
            self.stop()
            time.sleep(0.5)

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run_sequence, args=(stops, mode, timeout),
            daemon=True)
        self._thread.start()
        first = stops[0].replace("_", " ")
        return True, (f"On my way. {len(stops)} stops, starting with "
                      f"{first}.")

    def stop(self):
        if not self.is_busy():
            return False, "I'm not going anywhere."
        dest = self.current_destination
        self._stop.set()
        self.call(API_PAUSE_NAV, {}, timeout=5.0)
        if self._thread:
            self._thread.join(timeout=3.0)
        return True, f"Stopped heading to {(dest or '').replace('_', ' ')}."

    def status_text(self):
        if not self.is_busy():
            return "I'm not moving right now."
        d = (self.current_destination or "").replace("_", " ")
        pct = self.nav["pct"] if self.nav else 0.0
        if self.current_leg:
            i, total = self.current_leg
            return (f"I'm on my way to {d}, stop {i} of {total}, "
                    f"about {pct:.0f} percent there.")
        return f"I'm on my way to {d}, about {pct:.0f} percent there."

    # ---------- spatial ----------
    def where_is(self, place):
        key = self.resolve(place)
        if key is None:
            return f"I don't know where {place} is."
        if self.pose is None:
            return ("I know that place, but I'm not sure where I am right "
                    "now.")
        wp = self.waypoints[key]
        dx, dy = wp["x"] - self.pose["x"], wp["y"] - self.pose["y"]
        dist = math.hypot(dx, dy)
        rel = norm_angle(math.atan2(dy, dx) - self.pose["yaw"])
        return (f"{key.replace('_', ' ')} is {clock_direction(rel)}, "
                f"about {dist:.1f} metres away.")
