"""
Glasses-eye view of the workspace.  Owner: Aaryan (RED team).

Feeds a human-captured image of the scene into reason.diagnose(), so the
VLM can see what the fixed camera cannot — the arm from the operator's
point of view, the block that rolled off the table, the human's hand
pointing at the problem.

    import glasses, reason
    frame = glasses.request_view("I keep missing the grasp.")
    result = reason.diagnose(task, verdict, frame_after=frame)

=== WHAT ACTUALLY WORKS, AND WHY THIS IS A FOLDER WATCHER ===

Ray-Ban Meta glasses do NOT stream video to a Mac. They are not a UVC
camera, so they never appear in the system camera list, and Bluetooth has
nowhere near the bandwidth for a video feed. Meta's Wearables Device Access
Toolkit does expose the camera, but it is a developer-preview binding for
iOS/Android apps — there is no macOS/Python path, and no wake word either.

What the glasses DO reliably give you: press capture, the photo lands in the
Meta AI app on the phone, and from there it reaches a Mac folder in seconds
(AirDrop, iCloud, a shared folder). That is the ingest below. It is
unglamorous and it works with zero dependencies, which is the right trade
for something on a demo's critical path.

BACKENDS (GLASSES_BACKEND):
    folder  default. Any image dropped into GLASSES_DIR. Works today, works
            with any phone, works if the glasses are flat.
    aria    Project Aria research glasses, which DO have a real streaming
            Python SDK. Only useful if those are the glasses you have.
    off     disabled; every call returns None.

Returns PATHS, never decoded arrays — reason._encode_frame() reads a JPEG
straight to base64, so nothing here needs OpenCV.
"""
import glob
import os
import time
from typing import Dict, List, Optional

try:
    import narrate

    def _say(text: str, block: bool = False) -> None:
        narrate.speak(text, block=block)
except Exception as _e:                                     # noqa: BLE001
    print(f"[glasses] narrate unavailable ({_e}); prompts print only")

    def _say(text: str, block: bool = False) -> None:
        print(f"  🗣  {text}")


BACKEND = os.getenv("GLASSES_BACKEND", "folder")    # folder | aria | off
GLASSES_DIR = os.getenv("GLASSES_DIR", "glasses_frames")

# What a vision model will actually accept. HEIC is deliberately excluded:
# it is the iPhone default and no mainstream vision endpoint decodes it, so
# silently forwarding one produces a confusing API error rather than a
# diagnosis. See _reject_reason().
USABLE_EXT = {".jpg", ".jpeg", ".png", ".webp"}

# Files still being written by AirDrop/iCloud. Never hand one to the model.
PARTIAL_EXT = {".part", ".download", ".crdownload", ".tmp", ".icloud"}

SETTLE_S = float(os.getenv("GLASSES_SETTLE_S", "0.6"))
POLL_S = 0.4


# ---------------- capability reporting ----------------

def capabilities() -> Dict:
    """What this machine can do right now. Print it at bring-up."""
    caps = {
        "backend": BACKEND,
        "folder": os.path.isdir(GLASSES_DIR),
        "folder_path": os.path.abspath(GLASSES_DIR),
        "usable_frames": len(list_frames()),
        "aria_sdk": _have_aria(),
        "live_stream": BACKEND == "aria" and _have_aria(),
    }
    return caps


def _have_aria() -> bool:
    for mod in ("projectaria_client_sdk", "projectaria_tools"):
        try:
            __import__(mod)
            return True
        except Exception:                                   # noqa: BLE001
            continue
    return False


# ---------------- folder backend ----------------

def _reject_reason(path: str) -> Optional[str]:
    """Why this file cannot be sent to the model, or None if it can."""
    name = os.path.basename(path)
    if name.startswith("."):
        return "hidden file"
    ext = os.path.splitext(name)[1].lower()
    if ext in PARTIAL_EXT:
        return "still transferring"
    if ext == ".heic":
        return ("HEIC is not decodable by the vision endpoint — set the "
                "iPhone camera to 'Most Compatible', or share as JPEG")
    if ext not in USABLE_EXT:
        return f"unsupported extension {ext or '(none)'}"
    return None


def _is_settled(path: str) -> bool:
    """True once the file has stopped growing.

    A photo arriving over AirDrop or iCloud is readable long before it is
    complete. Handing a half-written JPEG to the model gives a decode error
    that looks like a model problem and is really a race.
    """
    try:
        first = os.path.getsize(path)
        if first == 0:
            return False
        time.sleep(SETTLE_S)
        return os.path.getsize(path) == first
    except OSError:
        return False


def list_frames() -> List[str]:
    """Usable image paths in GLASSES_DIR, newest first."""
    if not os.path.isdir(GLASSES_DIR):
        return []
    paths = [p for p in glob.glob(os.path.join(GLASSES_DIR, "*"))
             if os.path.isfile(p) and _reject_reason(p) is None]
    return sorted(paths, key=os.path.getmtime, reverse=True)


def latest_frame(max_age_s: Optional[float] = None) -> Optional[str]:
    """Newest usable frame, or None. `max_age_s` rejects stale images —
    a photo from twenty minutes ago describes a world that has moved."""
    if BACKEND == "off":
        return None
    for path in list_frames():
        if max_age_s is not None and time.time() - os.path.getmtime(path) > max_age_s:
            return None                 # newest is already too old
        if _is_settled(path):
            return path
    return None


def wait_for_frame(timeout_s: float = 45.0) -> Optional[str]:
    """Block until a NEW frame arrives, or timeout.

    Only counts files that were not present when the wait started, so an
    old photo left in the folder cannot satisfy a fresh request for help.
    """
    if BACKEND == "off":
        return None

    os.makedirs(GLASSES_DIR, exist_ok=True)
    before = set(glob.glob(os.path.join(GLASSES_DIR, "*")))
    deadline = time.time() + timeout_s
    print(f"  📷 waiting for a glasses photo in ./{GLASSES_DIR}/ "
          f"({timeout_s:.0f}s)")

    warned = set()
    while time.time() < deadline:
        for path in sorted(set(glob.glob(os.path.join(GLASSES_DIR, "*"))) - before,
                           key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0,
                           reverse=True):
            why = _reject_reason(path)
            if why:
                if path not in warned and "transferring" not in why:
                    print(f"[glasses] ignoring {os.path.basename(path)}: {why}")
                    warned.add(path)
                continue
            if _is_settled(path):
                return path
        time.sleep(POLL_S)
    return None


# ---------------- aria backend ----------------

def _aria_frame() -> Optional[str]:
    """Grab one frame from Project Aria glasses.

    UNVERIFIED — no Aria SDK is installed here, so this path has never run.
    It exists because Aria is the ONE Meta eyewear product with a real
    Python streaming SDK; if those are the glasses on the table this is the
    hook to finish. Ray-Ban Meta consumer glasses cannot do this at all.
    """
    if not _have_aria():
        raise RuntimeError(
            "no Aria SDK installed (pip install projectaria-client-sdk). "
            "NOTE: this only applies to Project Aria research glasses. "
            "Ray-Ban Meta consumer glasses have no Python streaming path — "
            "use GLASSES_BACKEND=folder.")
    raise NotImplementedError(
        "Aria streaming hook is a stub. Fill in device connect + "
        "image_data callback, save the frame to GLASSES_DIR, return its "
        "path. Everything downstream already takes a path.")


# ---------------- public API ----------------

def request_view(reason_text: str = "", timeout_s: float = 45.0,
                 speak: bool = True) -> Optional[str]:
    """Ask a human for a glasses photo of the scene, and return its path.

    Returns None if none arrived — the caller must carry on without it. A
    missing photo degrades the diagnosis; it must never stop the run.
    """
    if BACKEND == "off":
        return None

    if BACKEND == "aria":
        try:
            return _aria_frame()
        except Exception as e:                              # noqa: BLE001
            print(f"[glasses] aria unavailable, falling back to folder: {e}")

    if speak:
        _say(f"{reason_text} Can you show me the scene?".strip())

    path = wait_for_frame(timeout_s)
    if path:
        print(f"[glasses] got {os.path.basename(path)}")
        if speak:
            _say("Thank you. Let me look.")
    else:
        print("[glasses] no photo arrived")
        if speak:
            _say("No picture arrived. I will work it out myself.")
    return path


def diagnose_with_view(task: str, verdict: Dict, frame_before=None,
                       ask: bool = True, timeout_s: float = 45.0,
                       **kwargs) -> Optional[Dict]:
    """reason.diagnose(), with the glasses' view of the aftermath attached.

    Prefers a frame that already arrived; only interrupts a human to ask if
    there isn't one. Falls straight through to a frameless diagnosis when no
    photo is available, so this is always safe to call.
    """
    try:
        import reason
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] reason unavailable: {e}")
        return None

    frame_after = latest_frame(max_age_s=kwargs.pop("max_age_s", 120.0))
    if frame_after is None and ask:
        frame_after = request_view(timeout_s=timeout_s)
    if frame_after:
        print(f"[glasses] diagnosing with {os.path.basename(frame_after)}")
    else:
        print("[glasses] diagnosing without a glasses view")

    return reason.diagnose(task, verdict, frame_before=frame_before,
                           frame_after=frame_after, **kwargs)


if __name__ == "__main__":
    import json
    import shutil
    import tempfile

    GLASSES_DIR = os.path.join(tempfile.gettempdir(), "glasses_smoke")
    shutil.rmtree(GLASSES_DIR, ignore_errors=True)
    os.makedirs(GLASSES_DIR, exist_ok=True)
    print("capabilities:", json.dumps(capabilities(), indent=2))

    def _drop(name: str, data: bytes = b"\xff\xd8\xff\xe0stub\xff\xd9"):
        p = os.path.join(GLASSES_DIR, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    # ---- what must be refused, and why ----
    assert latest_frame() is None, "empty folder must yield nothing"
    heic = _drop("IMG_0421.HEIC")
    assert "HEIC" in (_reject_reason(heic) or ""), "HEIC must be refused"
    assert latest_frame() is None, "an unusable file is not a frame"
    print("PASS  HEIC refused with an actionable message")

    for bad, label in ((".DS_Store", "hidden file"),
                       ("photo.jpg.part", "still transferring"),
                       ("notes.txt", "unsupported extension")):
        p = _drop(bad)
        assert _reject_reason(p) is not None, f"{label} must be refused"
    assert latest_frame() is None
    print("PASS  hidden, partial and non-image files refused")

    # ---- a real photo is picked up ----
    good = _drop("glasses_capture_1.jpg")
    got = latest_frame()
    assert got == good, f"expected {good}, got {got}"
    print(f"PASS  picked up {os.path.basename(got)} "
          f"(ignoring {len(os.listdir(GLASSES_DIR)) - 1} unusable files)")

    # ---- newest wins, and staleness is enforced ----
    time.sleep(0.05)
    newest = _drop("glasses_capture_2.jpg")
    assert latest_frame() == newest, "newest frame must win"
    old = time.time() - 3600
    os.utime(newest, (old, old))
    os.utime(good, (old, old))
    assert latest_frame(max_age_s=60) is None, "stale frames must be rejected"
    print("PASS  newest frame wins; stale frames rejected")

    # ---- the frame actually reaches the model as a data URL ----
    import reason
    url = reason._encode_frame(newest)
    assert url and url.startswith("data:image/jpeg;base64,"), url
    print(f"PASS  frame encodes for the VLM ({len(url)} chars, no OpenCV)")
    assert reason._encode_frame(heic) is None, \
        "HEIC must not be forwarded mislabelled as JPEG"
    print("PASS  HEIC is not forwarded with a wrong mime type")

    # ---- end to end: glasses view -> diagnosis -> spoken ----
    os.utime(newest, None)              # make it fresh again
    verdict = {"passed": False, "error_cm": 11.0, "arm_moved_cm": 21.0,
               "block_moved_cm": 0.1}
    result = diagnose_with_view("pick up the red block and place it in the "
                                "target zone", verdict, ask=False)
    assert result and result["failure_mode"] in reason.FAILURE_MODES
    print(f"\ndiagnosis: {result['failure_mode']} — {result['diagnosis']}")
    print(f"prompt_update: {result['prompt_update']}")

    # ---- a passing verdict is still never diagnosed ----
    assert diagnose_with_view("t", {"passed": True}, ask=False) is None
    print("\nPASS  a passing verdict is never diagnosed, glasses or not")

    shutil.rmtree(GLASSES_DIR, ignore_errors=True)
    print("\nall glasses.py assertions passed")
