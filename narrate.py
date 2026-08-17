"""
Speech out.  Owner: Aaryan (RED team).

The arm says its own diagnosis out loud. This is the highest
impact-per-minute feature in the build — it is the moment the judges hear
the loop close, rather than being told it exists.

Audio goes to the Meta Ray-Ban glasses, which pair as an ordinary
Bluetooth headset. No Meta SDK, no wake word: Meta excludes "Hey Meta"
from the developer toolkit, so push-to-talk lives in voice.py instead.

ROUTING: macOS `say` takes `-a <device>`, and `say -a '?'` enumerates
output devices with their IDs. That routes our speech to the glasses
WITHOUT changing the system default output — so notifications, Zoom and
everything else stay on the laptop, and unplugging the glasses does not
leave the demo mute. Falls back silently to the default device.

Nothing here ever raises. Worst case it prints.
"""
import glob
import os
import re
import subprocess
import threading
import time
from typing import Dict, List, Optional

HELP_DIR = os.getenv("HELP_DIR", "help_frames")
BACKEND = os.getenv("TTS_BACKEND", "auto")      # auto | say | pyttsx3 | print
ESCALATE = os.getenv("ESCALATE_BACKEND", "phone")   # phone | glasses | off
RATE = os.getenv("TTS_RATE", "180")

_IS_MAC = os.uname().sysname == "Darwin"

# Resolved output device id/name for `say -a`. None = system default.
_device: Optional[str] = None

# Speech serialises through this. Two overlapping `say` processes talk over
# each other and the diagnosis becomes unintelligible, which defeats the
# entire point of speaking it.
_speech_lock = threading.Lock()

_engine = None


# ---------------- device discovery ----------------

def list_audio_devices() -> List[Dict]:
    """Available TTS output devices as [{"id", "name"}].

    Parsed from `say -a '?'`, which prints lines like "  105 AirPods Pro".
    Returns [] off macOS or if `say` is unavailable — callers must treat an
    empty list as "just use the default", not as an error.
    """
    if not _IS_MAC:
        return []
    try:
        out = subprocess.run(["say", "-a", "?"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception as e:                                  # noqa: BLE001
        print(f"[tts] could not list audio devices: {e}")
        return []

    devices = []
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(.+?)\s*$", line)
        if m:
            devices.append({"id": m.group(1), "name": m.group(2)})
    return devices


def set_output_device(name_substring: str) -> bool:
    """Route speech to the first device whose name contains the substring.

    Pass something like "Ray-Ban" or "Meta". Matching is case-insensitive
    and ignores the curly apostrophes macOS puts in device names.

    Returns True if a device matched. On False, speech keeps working on the
    system default — a missing headset must never silence the demo.
    """
    global _device

    if not name_substring:
        _device = None
        print("[tts] output device reset to system default")
        return True

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    want = _norm(name_substring)
    for dev in list_audio_devices():
        if want in _norm(dev["name"]):
            _device = dev["id"]
            print(f"[tts] routing speech to {dev['name']} (id {dev['id']})")
            return True

    names = [d["name"] for d in list_audio_devices()]
    print(f"[tts] no audio device matching {name_substring!r}; "
          f"staying on system default. Available: {names}")
    return False


# ---------------- speech ----------------

def _pyttsx3_speak(text: str) -> None:
    """Fallback engine. NOTE: cannot target a specific output device — it
    follows the system default, so glasses routing needs `say`."""
    global _engine
    import pyttsx3
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
    _engine.say(text)
    _engine.runAndWait()


def speak(text: str, block: bool = False) -> None:
    """Say it out loud. Never raises; worst case it only prints."""
    print(f"  🗣  {text}")

    def _run():
        with _speech_lock:
            try:
                if BACKEND in ("auto", "say") and _IS_MAC:
                    cmd = ["say", "-r", str(RATE)]
                    if _device:
                        cmd += ["-a", _device]
                    subprocess.run(cmd + [text], check=False, timeout=30)
                elif BACKEND in ("auto", "pyttsx3"):
                    _pyttsx3_speak(text)
            except Exception as e:                           # noqa: BLE001
                print(f"[tts] silent: {e}")

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


def announce_attempt(trial: int, task: str) -> None:
    speak(f"Attempt {trial}. {task}")


def announce_failure(critique: dict) -> None:
    """Speak the VLM's diagnosis of WHY the attempt failed.

    The camera already decided that it failed; this only explains it. There
    is deliberately no path here that reads a success boolean from a model.
    """
    speak((critique or {}).get("diagnosis", "That attempt did not work."))


def announce_success(trial: int, error_cm: float) -> None:
    try:
        margin = f", within {int(round(float(error_cm)))} centimetres"
    except (TypeError, ValueError):
        margin = ""
    speak(f"Done on attempt {trial}{margin}.")


def announce_learned(skill_name: str) -> None:
    """Kept for the existing loop, which calls it after a taught skill."""
    speak(f"I have learned a new skill: {str(skill_name).replace('_', ' ')}.")


# ---------------- escalation (legacy) ----------------
# Under the current architecture human input arrives as a leader-arm
# demonstration (teach.py) or as speech (voice.py). This photo path is kept
# only because the existing loop still calls it; it is not part of the Red
# team plan. Do not build on it.

def request_help(reason: str, timeout_s: int = 45):
    """Ask a human to drop a photo of the scene into HELP_DIR."""
    if ESCALATE == "off":
        return None

    speak(f"I have failed twice. {reason} Can someone show me the scene?")

    if ESCALATE == "glasses":
        try:
            from glasses_source import capture_frame
            return capture_frame(timeout_s)
        except Exception as e:                              # noqa: BLE001
            print(f"[escalate] glasses unavailable, using phone: {e}")

    return _wait_for_photo(timeout_s)


def _wait_for_photo(timeout_s: int):
    # cv2 imported lazily: at module scope it made `import narrate` — and
    # therefore ALL speech — fail on any machine without OpenCV.
    import cv2

    os.makedirs(HELP_DIR, exist_ok=True)
    before = set(glob.glob(f"{HELP_DIR}/*"))
    deadline = time.time() + timeout_s
    print(f"  📷 waiting for a photo in ./{HELP_DIR}/ ({timeout_s}s)")
    while time.time() < deadline:
        new = set(glob.glob(f"{HELP_DIR}/*")) - before
        if new:
            path = max(new, key=os.path.getmtime)
            time.sleep(0.4)                     # let the write finish
            img = cv2.imread(path)
            if img is not None:
                speak("Thank you. Let me try again.")
                return img
        time.sleep(0.5)
    speak("No help arrived. I will try on my own.")
    return None


if __name__ == "__main__":
    print("audio output devices:")
    devices = list_audio_devices()
    for d in devices or []:
        print(f"  [{d['id']:>4}] {d['name']}")
    if not devices:
        print("  (none found — speech will use the system default)")

    # The glasses will not be paired during development; the point is that
    # a miss is reported and speech still works afterwards.
    print(f"\nset_output_device('Ray-Ban') -> {set_output_device('Ray-Ban')}")
    if devices:
        first = devices[0]["name"]
        print(f"set_output_device({first!r}) -> {set_output_device(first)}")
    print(f"set_output_device('') -> {set_output_device('')}")

    print()
    announce_attempt(1, "pick up the red block and place it in the target zone")
    announce_failure({"diagnosis": "I closed the gripper two centimetres "
                                   "short of the block."})
    announce_failure({})                    # missing diagnosis must not crash
    announce_failure(None)                  # nor must a missing critique
    announce_success(2, 3.4)
    announce_success(2, None)               # nor a missing measurement
    announce_learned("human_taught_grasp")
    speak("Narration check complete.", block=True)
    print("\nnarrate smoke test passed")
