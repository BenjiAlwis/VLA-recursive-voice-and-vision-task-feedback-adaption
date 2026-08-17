"""
Speech in.  Owner: Aaryan (RED team).

The human speaks the task, and speaks corrections mid-run. Audio comes
from the Meta Ray-Ban glasses, which pair as an ordinary Bluetooth headset
(mic + speaker only, no Meta SDK).

PUSH-TO-TALK, NOT A WAKE WORD. Meta excludes "Hey Meta" from the developer
toolkit, so there is no way to trigger on it. Recording starts on a
keypress and stops on the next keypress or at the timeout.

THE FALLBACK CHAIN, in order:
    1. mic + Nebius transcription
    2. mic + local whisper
    3. typed input at the terminal

Step 3 must always work — this is on the demo's critical path, and a
missing microphone, an unpaired headset or a dead network must degrade to
someone typing the task rather than stopping the run. Every layer above it
is optional.

When there is no terminal either (piped stdin, CI, smoke test), listen()
returns "" promptly instead of blocking forever.
"""
import json
import os
import sys
import tempfile
import threading
import time
import wave
from typing import Optional

try:
    import narrate

    def _say(text: str, block: bool = True) -> None:
        narrate.speak(text, block=block)
except Exception as _e:                                     # noqa: BLE001
    print(f"[voice] narrate unavailable ({_e}); prompts print only")

    def _say(text: str, block: bool = True) -> None:
        print(f"  🗣  {text}")


SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
SPEECH_MODEL = os.getenv("NEBIUS_SPEECH_MODEL", "whisper-large-v3")
NEBIUS_BASE = "https://api.studio.nebius.com/v1/"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


# ---------------- capability probing ----------------

def _have_mic() -> bool:
    try:
        import sounddevice        # noqa: F401
        import numpy              # noqa: F401
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _have_nebius() -> bool:
    if not os.getenv("NEBIUS_API_KEY"):
        return False
    try:
        import openai             # noqa: F401
        return True
    except Exception:                                        # noqa: BLE001
        return False


def _have_whisper() -> bool:
    try:
        import whisper            # noqa: F401
        return True
    except Exception:                                        # noqa: BLE001
        return False


def capabilities() -> dict:
    """What this machine can actually do right now. Useful at bring-up:
    run `python voice.py` and see which layer you are on."""
    return {
        "mic": _have_mic(),
        "nebius": _have_nebius(),
        "whisper": _have_whisper(),
        "terminal": sys.stdin.isatty(),
    }


# ---------------- recording ----------------

def _record_wav(max_s: float) -> Optional[str]:
    """Record from the default input device until Enter or max_s.

    The default input is the glasses when they are paired — that is the
    whole reason this takes no device argument. Returns a WAV path, or
    None if anything at all goes wrong.
    """
    try:
        import numpy as np
        import sounddevice as sd
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] no microphone stack: {e}")
        return None

    chunks = []

    def _cb(indata, frames, time_info, status):             # noqa: ANN001
        if status:
            print(f"[voice] audio status: {status}")
        chunks.append(indata.copy())

    stop = threading.Event()

    def _wait_for_key():
        try:
            input()
        except Exception:                                   # noqa: BLE001
            pass
        stop.set()

    if sys.stdin.isatty():
        threading.Thread(target=_wait_for_key, daemon=True).start()

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                            dtype="int16", callback=_cb):
            deadline = time.monotonic() + max_s
            while not stop.is_set() and time.monotonic() < deadline:
                time.sleep(0.05)
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] recording failed: {e}")
        return None

    if not chunks:
        print("[voice] recorded nothing")
        return None

    try:
        audio = np.concatenate(chunks, axis=0)
        path = os.path.join(tempfile.gettempdir(),
                            f"voice_{int(time.time())}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)                # int16
            w.setframerate(SAMPLE_RATE)
            w.writeframes(audio.tobytes())
        return path
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] could not write wav: {e}")
        return None


# ---------------- transcription ----------------

def _transcribe_nebius(path: str) -> Optional[str]:
    """UNVERIFIED: Nebius is OpenAI-compatible for chat, but whether this
    account is served a speech-to-text model is not confirmed — it could
    not be checked here (no openai package, no key). If the endpoint 404s
    the caller falls through to whisper, so a wrong model id costs one
    failed request, not the demo. Override with NEBIUS_SPEECH_MODEL.
    """
    try:
        from openai import OpenAI
        client = OpenAI(base_url=NEBIUS_BASE,
                        api_key=os.environ["NEBIUS_API_KEY"])
        with open(path, "rb") as f:
            resp = client.audio.transcriptions.create(model=SPEECH_MODEL,
                                                      file=f)
        text = getattr(resp, "text", None) or ""
        return text.strip() or None
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] nebius transcription unavailable: {e}")
        return None


def _transcribe_whisper(path: str) -> Optional[str]:
    """Local whisper. Slower, but works with no network."""
    try:
        import whisper
        model = whisper.load_model(WHISPER_MODEL)
        text = (model.transcribe(path).get("text") or "").strip()
        return text or None
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] local whisper unavailable: {e}")
        return None


def _typed(prompt: str) -> str:
    """Last resort. Returns "" rather than blocking when there is no tty,
    so unattended runs and smoke tests cannot hang here."""
    if not sys.stdin.isatty():
        print(f"[voice] no tty; returning empty for: {prompt}")
        return ""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


# ---------------- public API ----------------

def listen(timeout_s: int = 8, prompt: str = "🎤 ") -> str:
    """Capture one spoken utterance and return it as text.

    Walks the fallback chain and returns "" if every layer fails. Never
    raises.
    """
    try:
        if _have_mic():
            if sys.stdin.isatty():
                print("  ⏺  press Enter to start talking…", end="", flush=True)
                input()
                print("  ⏺  recording — press Enter when done "
                      f"(max {timeout_s}s)")
            wav = _record_wav(timeout_s)
            if wav:
                try:
                    for name, fn in (("nebius", _transcribe_nebius),
                                     ("whisper", _transcribe_whisper)):
                        if (name == "nebius" and not _have_nebius()) or \
                           (name == "whisper" and not _have_whisper()):
                            continue
                        text = fn(wav)
                        if text:
                            print(f"  ✓ heard ({name}): {text}")
                            return text
                    print("[voice] recorded audio but could not transcribe it")
                finally:
                    try:
                        os.remove(wav)
                    except OSError:
                        pass
        return _typed(prompt)
    except Exception as e:                                  # noqa: BLE001
        print(f"[voice] listen failed: {e}")
        return ""


def listen_for_task(timeout_s: int = 8) -> str:
    """Ask what to do, and return the spoken task description.

    Returns "" if nothing was captured. Deliberately does NOT substitute a
    default task: quietly inventing a goal the human did not ask for is
    worse than an empty string the caller can handle.
    """
    _say("What should I do?")
    task = listen(timeout_s, prompt="task> ")
    if not task:
        print("[voice] no task captured")
    return task


def listen_for_feedback(timeout_s: int = 8) -> str:
    """Mid-run human correction, e.g. 'it is further left than you think'."""
    _say("What did I get wrong?")
    feedback = listen(timeout_s, prompt="feedback> ")
    if not feedback:
        print("[voice] no feedback captured")
    return feedback


if __name__ == "__main__":
    caps = capabilities()
    print("voice capabilities:", json.dumps(caps, indent=2))

    layer = ("mic + nebius" if caps["mic"] and caps["nebius"] else
             "mic + whisper" if caps["mic"] and caps["whisper"] else
             "mic only (no transcription — will fall through)"
             if caps["mic"] else
             "typed terminal input" if caps["terminal"] else
             "nothing available — listen() returns ''")
    print(f"active layer: {layer}\n")

    # Must return promptly and must be a str on every path. Piped stdin
    # (no tty) is the case that would otherwise hang forever on input().
    t0 = time.monotonic()
    task = listen_for_task(timeout_s=3)
    feedback = listen_for_feedback(timeout_s=3)
    elapsed = time.monotonic() - t0

    print(f"\ntask     = {task!r}")
    print(f"feedback = {feedback!r}")
    print(f"elapsed  = {elapsed:.1f}s")

    assert isinstance(task, str) and isinstance(feedback, str), \
        "listen_* must always return a str"
    if not caps["terminal"] and elapsed > 15:
        raise SystemExit("FAIL: blocked without a tty — this would hang the demo")
    print("\nvoice smoke test passed "
          f"({'interactive' if caps['terminal'] else 'unattended'})")
