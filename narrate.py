"""
TTS narration + human escalation.  Owner: Aaryan.

Two jobs:
  1. The robot says its diagnosis out loud. This is the single highest
     impact-per-minute feature in the build — judges hear the loop close.
  2. After 2 consecutive failures the robot ASKS FOR HELP: a human supplies
     a photo of the scene, which goes into the next plan's context.

Escalation backend is pluggable. Phone photo is the default and works.
Glasses are a stretch goal behind a hard gate at T-1:00.
"""
import glob
import os
import subprocess
import threading
import time
from typing import Optional

# cv2 is imported lazily inside _wait_for_photo. It is only needed to decode
# a help photo, but at module scope it made `import narrate` — and therefore
# ALL speech — fail on any machine without OpenCV. Speech is the highest
# impact-per-minute feature in the build; it must not depend on the camera.

HELP_DIR = os.getenv("HELP_DIR", "help_frames")
BACKEND = os.getenv("TTS_BACKEND", "auto")     # auto | say | pyttsx3 | print
ESCALATE = os.getenv("ESCALATE_BACKEND", "phone")   # phone | glasses | off


# ---------------- speech ----------------

_engine = None


def _pyttsx3_speak(text: str) -> None:
    global _engine
    import pyttsx3
    if _engine is None:
        _engine = pyttsx3.init()
        _engine.setProperty("rate", 165)
    _engine.say(text)
    _engine.runAndWait()


def speak(text: str, block: bool = False) -> None:
    """Say it. Never let TTS crash the demo — worst case it prints."""
    print(f"  🗣  {text}")

    def _run():
        try:
            if BACKEND in ("auto", "say") and os.uname().sysname == "Darwin":
                subprocess.run(["say", "-r", "180", text], check=False)
            elif BACKEND in ("auto", "pyttsx3"):
                _pyttsx3_speak(text)
        except Exception as e:                      # noqa: BLE001
            print(f"[tts] silent: {e}")

    if block:
        _run()
    else:
        threading.Thread(target=_run, daemon=True).start()


def announce_attempt(trial: int, task: str) -> None:
    speak(f"Attempt {trial}. {task}")


def announce_failure(critique: dict) -> None:
    speak(critique.get("diagnosis", "That attempt did not work."))


def announce_success(trial: int, error_cm: float) -> None:
    speak(f"Target reached on attempt {trial}, "
          f"within {int(error_cm)} centimetres.")


def announce_learned(skill_name: str) -> None:
    speak(f"I have added a new skill: {skill_name.replace('_', ' ')}.")


# ---------------- escalation ----------------

def request_help(reason: str, timeout_s: int = 45) -> Optional["cv2.Mat"]:
    """Robot asks a human for a view of the scene.

    phone   : human drops a photo into HELP_DIR (AirDrop / shared folder).
              Newest file wins. Poll until timeout.
    glasses : Meta DAT frame. ONLY if the T-1:00 gate passed.
    off     : skip entirely.
    """
    if ESCALATE == "off":
        return None

    speak(f"I have failed twice. {reason} Can someone show me the scene?")

    if ESCALATE == "glasses":
        try:
            from glasses_source import capture_frame   # stretch goal
            return capture_frame(timeout_s)
        except Exception as e:                      # noqa: BLE001
            print(f"[escalate] glasses unavailable, using phone: {e}")

    return _wait_for_photo(timeout_s)


def _wait_for_photo(timeout_s: int):
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
    speak("I overshot the target by nine centimetres. "
          "The carpet is grippier than I estimated.", block=True)
