"""
The human interface — Meta Ray-Ban glasses.  Owner: Aaryan (RED team).

This is the "Input (human interp)" box in the diagram, and it is two
signals with VERY different risk profiles. Keeping them apart is the whole
design of this file.

    AUDIO — core, works today, zero SDK.
        The glasses pair as an ordinary Bluetooth headset. narrate.py routes
        speech OUT to them via `say -a`; voice.py records IN from them by
        input-device index. Nothing here needs Meta's toolkit.

    VIDEO — stretch goal, gated OFF by default.
        Camera access needs a native iOS/Android app with the Meta AI app
        bridging (Wearables Device Access Toolkit), 720p/30fps over
        Bluetooth, 2-4 hours of work. master_reference §9 puts a hard gate
        at T-1:00: if it is not streaming a real frame by then, CUT IT.

        So capture_frame() returns None unless a bridge is explicitly
        configured, and EVERY caller must treat None as normal. If a
        missing Meta video feed can break the run, the gate cannot actually
        be exercised and someone will be soldering at T-0:10.

Nothing in this file raises. Worst case it returns None or "".

    import glasses
    glasses.pair()                      # route audio both ways
    task = glasses.listen_for_task()    # human speaks the goal
    note = glasses.listen_for_feedback()# human corrects mid-run
    frame = glasses.capture_frame()     # None unless the bridge is up
"""
import glob
import os
import time
from typing import Dict, List, Optional

DEVICE_HINT = os.getenv("GLASSES_DEVICE", "Ray-Ban")

# Video stays off unless someone deliberately turns it on. Default-off is
# the gate: it means "no Meta video" is the tested path, not the surprise.
VIDEO_ENABLED = os.getenv("GLASSES_VIDEO", "0") == "1"
VIDEO_URL = os.getenv("GLASSES_VIDEO_URL", "")      # MJPEG/RTSP from the app
FRAME_DIR = os.getenv("GLASSES_FRAME_DIR", "glasses_frames")

_paired = {"audio_out": False, "audio_in": False}
_video_cap = None


# ---------------- audio: the core path ----------------

def pair(name_substring: str = None) -> Dict[str, bool]:
    """Route speech OUT to the glasses and recording IN from them.

    Returns what actually succeeded. A False anywhere is survivable — that
    half falls back to the laptop's default device and the demo continues,
    just without the glasses. Call this once at startup and print it, so
    nobody spends the morning wondering which device is live.
    """
    hint = name_substring or DEVICE_HINT

    try:
        import narrate
        _paired["audio_out"] = narrate.set_output_device(hint)
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] speech routing unavailable: {e}")
        _paired["audio_out"] = False

    try:
        import voice
        _paired["audio_in"] = voice.set_input_device(hint)
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] microphone routing unavailable: {e}")
        _paired["audio_in"] = False

    if not any(_paired.values()):
        print(f"[glasses] no device matching {hint!r}; using laptop audio. "
              f"This is fine — the loop does not depend on the glasses.")
    return dict(_paired)


def listen_for_task(timeout_s: int = 8) -> str:
    """The human speaks the goal. "" if nothing was captured."""
    try:
        import voice
        return voice.listen_for_task(timeout_s)
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] listen_for_task failed: {e}")
        return ""


def listen_for_feedback(timeout_s: int = 8) -> str:
    """A mid-run spoken correction. "" if nothing was captured.

    This is the arrow that makes the loop genuinely multi-modal: the human
    says "it's further left than you think" and those exact words go into
    the next planner prompt.
    """
    try:
        import voice
        return voice.listen_for_feedback(timeout_s)
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] listen_for_feedback failed: {e}")
        return ""


def say(text: str, block: bool = False) -> None:
    """Speak through the glasses, if they are the routed output."""
    try:
        import narrate
        narrate.speak(text, block=block)
    except Exception as e:                                  # noqa: BLE001
        print(f"  🗣  {text}   [{e}]")


# ---------------- video: the gated path ----------------

def video_available() -> bool:
    """Is a Meta video bridge actually configured AND reachable?

    This is the T-1:00 gate check. Run it, look at the answer, and make the
    cut decision from it rather than from optimism.
    """
    if not VIDEO_ENABLED:
        return False
    if VIDEO_URL:
        return _open_stream() is not None
    return os.path.isdir(FRAME_DIR)


def _open_stream():
    """Open the app's video stream once and keep it."""
    global _video_cap
    if _video_cap is not None:
        return _video_cap
    if not VIDEO_URL:
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(VIDEO_URL)
        if not cap.isOpened():
            print(f"[glasses] video url did not open: {VIDEO_URL}")
            return None
        _video_cap = cap
        return cap
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] video stream unavailable: {e}")
        return None


def capture_frame(timeout_s: int = 20):
    """One BGR frame from the glasses, or None.

    None is a NORMAL return value, not an error — see the module docstring.
    Two bridge shapes are supported, both fed by the native app:

      GLASSES_VIDEO_URL   an MJPEG/RTSP stream the app serves
      GLASSES_FRAME_DIR   a folder the app drops stills into

    The folder path is the pragmatic one. A phone shortcut that saves a
    photo into a synced folder is twenty minutes of work and gives the
    planner a real human-perspective frame, versus 2-4 hours for the full
    toolkit integration.
    """
    if not VIDEO_ENABLED:
        return None

    if VIDEO_URL:
        cap = _open_stream()
        if cap is not None:
            try:
                cap.read()                  # drop the stale buffered frame
                ok, frame = cap.read()
                if ok:
                    return frame
            except Exception as e:                          # noqa: BLE001
                print(f"[glasses] stream read failed: {e}")

    return _wait_for_dropped_frame(timeout_s)


def _wait_for_dropped_frame(timeout_s: int):
    """Poll FRAME_DIR for a newly arrived image."""
    try:
        import cv2
    except Exception as e:                                  # noqa: BLE001
        print(f"[glasses] no OpenCV, cannot read dropped frames: {e}")
        return None

    os.makedirs(FRAME_DIR, exist_ok=True)
    before = set(glob.glob(f"{FRAME_DIR}/*"))
    deadline = time.time() + timeout_s
    print(f"  📷 waiting for a frame in ./{FRAME_DIR}/ ({timeout_s}s)")

    while time.time() < deadline:
        new = set(glob.glob(f"{FRAME_DIR}/*")) - before
        if new:
            path = max(new, key=os.path.getmtime)
            time.sleep(0.4)                 # let the write land
            img = cv2.imread(path)
            if img is not None:
                print(f"  ✓ got {path}")
                return img
        time.sleep(0.5)
    return None


def release() -> None:
    global _video_cap
    try:
        if _video_cap is not None:
            _video_cap.release()
    except Exception:                                       # noqa: BLE001
        pass
    _video_cap = None


# ---------------- status ----------------

def status() -> Dict:
    """Everything the T-1:00 gate needs, in one dict."""
    try:
        import voice
        caps = voice.capabilities()
    except Exception:                                       # noqa: BLE001
        caps = {}
    return {
        "audio_out_routed": _paired["audio_out"],
        "audio_in_routed": _paired["audio_in"],
        "mic_stack": caps.get("mic", False),
        "transcription": ("nebius" if caps.get("nebius") else
                          "whisper" if caps.get("whisper") else "typed"),
        "video_enabled": VIDEO_ENABLED,
        "video_available": video_available(),
        "video_source": VIDEO_URL or (FRAME_DIR if VIDEO_ENABLED else None),
    }


if __name__ == "__main__":
    import json

    print("=== glasses status (this is the T-1:00 gate check) ===")
    st = status()
    print(json.dumps(st, indent=2))

    print("\n=== pairing audio ===")
    result = pair()
    print(f"  {result}")
    assert isinstance(result, dict) and set(result) == {"audio_out", "audio_in"}

    print("\n=== video is gated OFF by default ===")
    print(f"  GLASSES_VIDEO={os.getenv('GLASSES_VIDEO', '0')} "
          f"-> video_available={video_available()}")
    frame = capture_frame(timeout_s=1)
    print(f"  capture_frame() -> {frame if frame is None else frame.shape}")
    if not VIDEO_ENABLED:
        assert frame is None, "video must stay off until explicitly enabled"

    print("\n=== every caller survives no video ===")
    # This is the property the T-1:00 gate depends on: cutting Meta video
    # must be a config change, not a code change.
    for value in ("0", "1"):
        os.environ["GLASSES_VIDEO"] = value
        import importlib
        import glasses as g
        importlib.reload(g)
        f = g.capture_frame(timeout_s=1)
        print(f"  GLASSES_VIDEO={value} -> {f if f is None else f.shape} "
              f"(no exception)")
    os.environ["GLASSES_VIDEO"] = "0"

    print("\n=== speech + listen return the right types, unattended ===")
    say("Glasses check complete.", block=True)
    t0 = time.monotonic()
    task = listen_for_task(timeout_s=2)
    note = listen_for_feedback(timeout_s=2)
    elapsed = time.monotonic() - t0
    print(f"  task={task!r} feedback={note!r} in {elapsed:.1f}s")
    assert isinstance(task, str) and isinstance(note, str)

    release()
    print("\nglasses smoke test passed")
