"""
Sensing — ArUco ground truth + camera capture.  Owner: Benji.

THE GROUND-TRUTH CHANNEL. master_reference.md section 5 splits the cameras
into two roles that must never merge:

    overhead ArUco  -> the CRITIC only. Numeric. Cannot hallucinate.
    wrist camera    -> the PLANNER only. Scene context, never a verdict.

This file owns the first. Everything it returns is a measurement in
centimetres; there is no model anywhere in it, and there must never be.

Marker IDs (tape these down and do not renumber mid-event):
    ID 0   the block
    ID 1   the target zone
    ID 2   the gripper / wrist
    ID 3   the table-origin reference

SCALE. Pixel distances become centimetres via the known physical edge
length of the markers themselves (MARKER_CM). Every detected marker gives
an independent estimate of the scale, and we take the median. That is
noticeably more robust than trusting one reference marker: if the origin
marker is half-occluded by the arm — which happens constantly, because the
arm moves over the table — a single-reference scale silently goes wrong and
takes every measurement with it.

Assumes a roughly top-down camera and a flat table. No lens calibration, no
homography. For a 60cm workspace and a 3cm success threshold that is
comfortably good enough, and it needs no calibration step on the day.
"""
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

MARKER_CM = float(os.getenv("ARUCO_MARKER_CM", "4.0"))

# Team Yellow's names first (so101_yellow/configs/cameras.env), then ours.
#
# THE OVERHEAD CAMERA MUST BE FIXED, AND IT MUST NOT BE THE WRIST CAMERA.
# Their cameras.env defines CAM_WRIST_INDEX=0 and says in as many words
# that it is "mounted on the end effector, not a fixed workspace view" —
# while this file used to default OVERHEAD_CAM to 0. On that hardware the
# critic's ground truth would have been measured through a camera bolted
# to the moving arm: every marker appears to move when the arm moves, so
# the block's "position" changes without the block going anywhere.
#
# That is not a degraded measurement, it is a meaningless one, and it
# would be measuring the one thing in this system that is not allowed to
# be wrong. Hence the guard in _check_camera_roles() rather than a default
# that quietly does the wrong thing.
WRIST_INDEX = int(os.getenv("WRIST_CAM")
                  or os.getenv("CAM_WRIST_INDEX") or "1")
_OVERHEAD_RAW = os.getenv("OVERHEAD_CAM") or os.getenv("CAM_OVERHEAD_INDEX")
OVERHEAD_INDEX = int(_OVERHEAD_RAW) if _OVERHEAD_RAW else 0

DICT_NAME = os.getenv("ARUCO_DICT", "DICT_4X4_50")


def _check_camera_roles() -> Optional[str]:
    """Why the ground-truth camera cannot be trusted, or None if it can.

    Called on every read_scene() that opens a real camera. Cheap, and the
    failure it catches is silent and total.
    """
    if OVERHEAD_INDEX == WRIST_INDEX:
        return (f"the overhead and wrist cameras are both index "
                f"{OVERHEAD_INDEX}. The critic would measure ground truth "
                f"through a camera mounted on the moving arm, which makes "
                f"every measurement meaningless. Set CAM_OVERHEAD_INDEX to "
                f"a SEPARATE, FIXED camera looking down at the workspace.")
    if _OVERHEAD_RAW is None and os.getenv("CAM_WRIST_INDEX") == "0":
        return ("no CAM_OVERHEAD_INDEX is set, and CAM_WRIST_INDEX=0 — so "
                "the default overhead index 0 IS the wrist camera. Add a "
                "second, fixed camera and set CAM_OVERHEAD_INDEX, or run "
                "on the mock. Do not measure ground truth from the wrist.")
    return None

ID_BLOCK, ID_ZONE, ID_GRIPPER, ID_ORIGIN = 0, 1, 2, 3

_caps: Dict[str, object] = {}
_detector = None
_last_good: Dict[str, Tuple[float, float]] = {}


# ---------------- camera ----------------

def _open(which: str):
    """Lazily open a camera and keep it open. Reopening per frame costs
    ~300ms on macOS, which would dominate the trial loop."""
    import cv2

    if which in _caps:
        return _caps[which]
    index = OVERHEAD_INDEX if which == "overhead" else WRIST_INDEX
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[vision] camera {which} (index {index}) did not open")
        cap = None
    _caps[which] = cap
    return cap


_warned_roles = False


def capture(which: str = "overhead") -> Optional[np.ndarray]:
    """One BGR frame, or None. Never raises."""
    global _warned_roles
    if which == "overhead" and not _warned_roles:
        problem = _check_camera_roles()
        if problem:
            print(f"\n[vision] ⚠ GROUND TRUTH IS NOT TRUSTWORTHY: {problem}\n")
        _warned_roles = True
    try:
        cap = _open(which)
        if cap is None:
            return None
        # Grab twice: the first read is often a stale buffered frame, and a
        # stale frame in the critic would measure the PREVIOUS attempt.
        cap.read()
        ok, frame = cap.read()
        return frame if ok else None
    except Exception as e:                                  # noqa: BLE001
        print(f"[vision] capture failed: {e}")
        return None


def release() -> None:
    for cap in _caps.values():
        try:
            if cap is not None:
                cap.release()
        except Exception:                                   # noqa: BLE001
            pass
    _caps.clear()


# ---------------- ArUco ----------------

def _get_detector():
    """OpenCV moved the ArUco API in 4.7. Support both, because whichever
    version is on the demo laptop is not something to discover at T+0:30."""
    global _detector
    import cv2

    if _detector is not None:
        return _detector

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    if hasattr(aruco, "ArucoDetector"):                     # >= 4.7
        _detector = ("new", aruco.ArucoDetector(
            dictionary, aruco.DetectorParameters()))
    else:                                                   # legacy
        _detector = ("old", (dictionary, aruco.DetectorParameters_create()))
    return _detector


def detect(frame: np.ndarray) -> Dict[int, Dict]:
    """{marker_id: {'centre_px', 'edge_px'}} for every marker seen."""
    if frame is None:
        return {}
    try:
        import cv2

        kind, det = _get_detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if kind == "new":
            corners, ids, _ = det.detectMarkers(gray)
        else:
            dictionary, params = det
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=params)

        if ids is None:
            return {}

        out = {}
        for marker_id, quad in zip(ids.flatten(), corners):
            pts = quad.reshape(4, 2)
            edges = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4]))
                     for i in range(4)]
            out[int(marker_id)] = {
                "centre_px": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "edge_px": float(np.median(edges)),
            }
        return out
    except Exception as e:                                  # noqa: BLE001
        print(f"[vision] detect failed: {e}")
        return {}


def _scale_cm_per_px(markers: Dict[int, Dict]) -> Optional[float]:
    """Median over every visible marker. See the module docstring for why
    this is not taken from a single reference marker."""
    edges = [m["edge_px"] for m in markers.values() if m["edge_px"] > 1.0]
    if not edges:
        return None
    return MARKER_CM / float(np.median(edges))


def read_scene(frame: Optional[np.ndarray] = None) -> Dict:
    """Ground truth in centimetres, measured from the overhead camera.

    Origin is the ID 3 reference marker when visible, otherwise the image
    centre. Coordinates are (x right, y up) in cm.

    A marker that is missing this frame falls back to its LAST KNOWN
    position, and the scene is tagged 'aruco_partial'. The arm occludes
    markers constantly, and one dropped detection must not read as "the
    block teleported to the origin" — which the critic would score as a
    huge error and the memory would then learn from as if it were real.
    """
    if frame is None:
        frame = capture("overhead")
    markers = detect(frame)

    if not markers:
        return {"block_cm": _last_good.get("block", (0.0, 0.0)),
                "zone_cm": _last_good.get("zone", (0.0, 0.0)),
                "gripper_cm": _last_good.get("gripper", (0.0, 0.0)) + (0.0,),
                "source": "aruco_none", "seen": []}

    scale = _scale_cm_per_px(markers)
    if scale is None:
        return {"source": "aruco_none", "seen": list(markers)}

    if ID_ORIGIN in markers:
        ox, oy = markers[ID_ORIGIN]["centre_px"]
    elif frame is not None:
        oy, ox = frame.shape[0] / 2.0, frame.shape[1] / 2.0
    else:
        ox = oy = 0.0

    def to_cm(marker_id: int, key: str):
        if marker_id not in markers:
            return _last_good.get(key)
        px, py = markers[marker_id]["centre_px"]
        # y is negated: image rows increase downward, the table frame is
        # y-up, and getting this backwards flips every dy correction the
        # planner emits into exactly the wrong direction.
        pos = (round((px - ox) * scale, 1), round(-(py - oy) * scale, 1))
        _last_good[key] = pos
        return pos

    block = to_cm(ID_BLOCK, "block")
    zone = to_cm(ID_ZONE, "zone")
    gripper = to_cm(ID_GRIPPER, "gripper")
    complete = all(i in markers for i in (ID_BLOCK, ID_ZONE))

    return {
        "block_cm": block or (0.0, 0.0),
        "zone_cm": zone or (0.0, 0.0),
        "gripper_cm": (gripper or (0.0, 0.0)) + (0.0,),
        "source": "aruco" if complete else "aruco_partial",
        "seen": sorted(markers),
        "scale_cm_per_px": round(scale, 5),
    }


def annotate(frame: np.ndarray, scene: Dict) -> np.ndarray:
    """Draw the measurement onto the frame — for the dashboard, and so a
    human can see at a glance whether tracking is actually working."""
    try:
        import cv2

        out = frame.copy()
        for marker_id, m in detect(frame).items():
            c = tuple(int(v) for v in m["centre_px"])
            cv2.circle(out, c, 6, (0, 255, 0), -1)
            cv2.putText(out, str(marker_id), (c[0] + 8, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(out, f"{scene.get('source')} block={scene.get('block_cm')} "
                         f"zone={scene.get('zone_cm')}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return out
    except Exception:                                       # noqa: BLE001
        return frame


# ---------------- standalone check ----------------

if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="ArUco ground-truth check")
    ap.add_argument("--live", action="store_true",
                    help="read the real overhead camera")
    ap.add_argument("--save", default="", help="write an annotated frame here")
    args = ap.parse_args()

    if args.live:
        print(f"reading overhead camera index {OVERHEAD_INDEX}…")
        frame = capture("overhead")
        if frame is None:
            raise SystemExit("no frame — check OVERHEAD_CAM and permissions")
        scene = read_scene(frame)
        print(json.dumps(scene, indent=2, default=str))
        if scene["source"] == "aruco_none":
            print("\nNo markers found. Check: printed size matches "
                  f"ARUCO_MARKER_CM={MARKER_CM}, dictionary is {DICT_NAME}, "
                  "and the markers are lit and unoccluded.")
        if args.save:
            import cv2
            cv2.imwrite(args.save, annotate(frame, scene))
            print(f"wrote {args.save}")
        release()
        raise SystemExit(0)

    # ---- synthetic test: render markers, then measure them back ----
    import cv2

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, DICT_NAME))
    canvas = np.full((600, 800, 3), 255, dtype=np.uint8)

    # 40px marker == 4cm  ->  0.1 cm/px. Placed at known pixel offsets so
    # the expected centimetre answer is known in advance.
    # Kept well apart: at 40px wide, markers closer than ~60px overlap and
    # the detector silently returns only some of them.
    placements = {ID_ORIGIN: (400, 300), ID_BLOCK: (600, 200),
                  ID_ZONE: (200, 400), ID_GRIPPER: (400, 120)}
    size = 40
    for marker_id, (cx, cy) in placements.items():
        draw = aruco.generateImageMarker if hasattr(aruco, "generateImageMarker") \
            else aruco.drawMarker
        img = draw(dictionary, marker_id, size)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        canvas[cy - size // 2:cy + size // 2,
               cx - size // 2:cx + size // 2] = img

    found = detect(canvas)
    print(f"detected markers: {sorted(found)}")
    assert set(found) == set(placements), f"expected all 4, got {sorted(found)}"

    scale = _scale_cm_per_px(found)
    print(f"scale: {scale:.4f} cm/px (expected ~{MARKER_CM / size:.4f})")
    assert abs(scale - MARKER_CM / size) < 0.005, "scale estimate is off"

    scene = read_scene(canvas)
    print(json.dumps(scene, indent=2, default=str))

    # block is 200px right and 100px UP from origin -> (+20cm, +10cm)
    assert abs(scene["block_cm"][0] - 20.0) < 0.6, scene["block_cm"]
    assert abs(scene["block_cm"][1] - 10.0) < 0.6, scene["block_cm"]
    # zone is 200px left and 100px DOWN -> (-20cm, -10cm)
    assert abs(scene["zone_cm"][0] + 20.0) < 0.6, scene["zone_cm"]
    assert abs(scene["zone_cm"][1] + 10.0) < 0.6, scene["zone_cm"]
    assert scene["source"] == "aruco"

    # Occlusion: hide the block, and it must hold its last position rather
    # than collapsing to the origin.
    occluded = canvas.copy()
    occluded[150:250, 550:650] = 255
    partial = read_scene(occluded)
    print(f"\nblock occluded -> source={partial['source']} "
          f"block_cm={partial['block_cm']}")
    assert partial["source"] == "aruco_partial", partial["source"]
    assert abs(partial["block_cm"][0] - 20.0) < 0.6, \
        "an occluded marker must hold its last known position"

    if args.save:
        cv2.imwrite(args.save, annotate(canvas, scene))
        print(f"wrote {args.save}")

    print("\n=== the camera-role guard ===")
    import importlib
    for env, expect_problem, label in (
        ({"OVERHEAD_CAM": "0", "WRIST_CAM": "0"}, True,
         "same index for both"),
        ({"CAM_WRIST_INDEX": "0"}, True,
         "Team Yellow's config with no overhead camera"),
        ({"CAM_OVERHEAD_INDEX": "2", "CAM_WRIST_INDEX": "0"}, False,
         "a separate fixed overhead camera"),
    ):
        saved = {k: os.environ.get(k) for k in
                 ("OVERHEAD_CAM", "WRIST_CAM", "CAM_WRIST_INDEX",
                  "CAM_OVERHEAD_INDEX")}
        for k in saved:
            os.environ.pop(k, None)
        os.environ.update(env)
        import vision as v
        importlib.reload(v)
        problem = v._check_camera_roles()
        print(f"  {label:<42} -> "
              f"{'REFUSED' if problem else 'ok'}")
        assert bool(problem) == expect_problem, (label, problem)
        for k, val in saved.items():
            os.environ.pop(k, None)
            if val is not None:
                os.environ[k] = val
    importlib.reload(v)

    print("\nvision smoke test passed (synthetic — run --live before the demo)")
