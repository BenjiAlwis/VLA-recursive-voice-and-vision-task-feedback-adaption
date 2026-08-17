"""
Receiver for glasses frames pushed from the iOS app.  Owner: Aaryan (RED).

The Meta Wearables Device Access Toolkit is a SWIFT/iOS SDK. It has no
macOS or Python binding, so the glasses cannot be opened from this repo
directly — a phone app holds the session and we receive what it sends. That
is the whole reason this file exists.

    ┌─ iPhone ──────────────────────────┐        ┌─ Mac (this repo) ────────┐
    │ Shadow app                        │        │ glasses_bridge.py        │
    │  Wearables.shared / DeviceKit     │        │  POST /frame  ──────┐    │
    │  StreamSession                    │  HTTP  │                     v    │
    │  capturePhoto(format: .jpeg) ─────┼───────>│  glasses_frames/*.jpg    │
    └───────────────────────────────────┘        │        │                 │
                                                 │        v                 │
                                                 │  glasses.latest_frame()  │
                                                 │  reason.diagnose(...)    │
                                                 └──────────────────────────┘

Standard library only — no Flask, no FastAPI. This sits on the demo's
critical path and a dependency that fails to install is a worse outcome
than fifty lines of http.server.

    python3 glasses_bridge.py          # prints the URL to put in the app

=== SWIFT SIDE (add to StreamSessionViewModel.handlePhotoData) ===

The sample already turns a capture into bytes; forward those same bytes:

    // in handlePhotoData(_ data: ...) alongside UIImage(data: data.data)
    var req = URLRequest(url: URL(string: "http://<MAC-IP>:8765/frame")!)
    req.httpMethod = "POST"
    req.setValue("image/jpeg", forHTTPHeaderField: "Content-Type")
    req.setValue(token, forHTTPHeaderField: "X-Glasses-Token")  // optional
    req.httpBody = data.data
    URLSession.shared.dataTask(with: req).resume()

Video frames work the same way — `frame.makeUIImage()` then
`img.jpegData(compressionQuality: 0.7)` as the body. Do NOT push every
frame at 30fps; one on capture, or one every second or two while streaming,
is plenty. The VLM only needs to see the aftermath.
"""
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

FRAME_DIR = os.getenv("GLASSES_DIR", "glasses_frames")
PORT = int(os.getenv("GLASSES_PORT", "8765"))
HOST = os.getenv("GLASSES_HOST", "0.0.0.0")     # phone must reach us

# Shared secret, optional. Hackathon Wi-Fi is a shared network and this
# endpoint writes files to disk, so a token costs nothing and closes the
# obvious hole. Unset = accept anything (fine on a hotspot you control).
TOKEN = os.getenv("GLASSES_TOKEN", "")

MAX_BYTES = int(os.getenv("GLASSES_MAX_BYTES", str(12 * 1024 * 1024)))
KEEP_LAST = int(os.getenv("GLASSES_KEEP_LAST", "40"))

_stats = {"received": 0, "rejected": 0, "bytes": 0, "last_ts": None,
          "last_file": None}


def lan_ip() -> str:
    """Best guess at the address the phone should POST to."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no packet is actually sent
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:                                       # noqa: BLE001
        return "127.0.0.1"


def _sniff(body: bytes) -> Optional[str]:
    """File extension from magic bytes, or None if it is not an image.

    Trusting the Content-Type header would let a mislabelled HEIC through,
    and HEIC is the one format the vision endpoint cannot decode. The bytes
    are the only honest source.
    """
    if body[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if body[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if body[:4] == b"RIFF" and body[8:12] == b"WEBP":
        return ".webp"
    if body[4:12] in (b"ftypheic", b"ftypheix", b"ftyphevc", b"ftypmif1"):
        return "heic"                       # recognised, deliberately refused
    return None


def save_frame(body: bytes) -> Tuple[bool, str]:
    """Persist one pushed frame. Returns (ok, message)."""
    if not body:
        return False, "empty body"
    if len(body) > MAX_BYTES:
        return False, f"too large ({len(body)} > {MAX_BYTES})"

    ext = _sniff(body)
    if ext is None:
        return False, "body is not a JPEG, PNG or WebP"
    if ext == "heic":
        return False, ("HEIC cannot be decoded by the vision endpoint — "
                       "send capturePhoto(format: .jpeg) or re-encode with "
                       "jpegData(compressionQuality:)")

    try:
        os.makedirs(FRAME_DIR, exist_ok=True)
        name = f"glasses_{time.strftime('%H%M%S')}_{int(time.time()*1000)%1000:03d}{ext}"
        path = os.path.join(FRAME_DIR, name)
        # Write then rename: glasses.py polls this directory and must never
        # see a partially written frame.
        tmp = f"{path}.part"
        with open(tmp, "wb") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception as e:                                  # noqa: BLE001
        return False, f"could not write frame: {e}"

    _stats["received"] += 1
    _stats["bytes"] += len(body)
    _stats["last_ts"] = time.time()
    _stats["last_file"] = name
    _prune()
    return True, name


def _prune() -> None:
    """Keep the directory bounded. A long demo should not fill the disk."""
    try:
        files = sorted(
            (os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR)
             if not f.startswith(".") and not f.endswith(".part")),
            key=os.path.getmtime, reverse=True)
        for old in files[KEEP_LAST:]:
            os.remove(old)
    except Exception:                                       # noqa: BLE001
        pass            # pruning is housekeeping; never let it break intake


class _Handler(BaseHTTPRequestHandler):
    server_version = "GlassesBridge/1.0"

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorised(self) -> bool:
        if not TOKEN:
            return True
        return self.headers.get("X-Glasses-Token", "") == TOKEN

    def do_GET(self):                                       # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._json(200, {"ok": True, "dir": os.path.abspath(FRAME_DIR),
                             **_stats})
        else:
            self._json(404, {"ok": False, "error": "try POST /frame"})

    def do_POST(self):                                      # noqa: N802
        if self.path.rstrip("/") != "/frame":
            self._json(404, {"ok": False, "error": "POST /frame"})
            return
        if not self._authorised():
            _stats["rejected"] += 1
            self._json(401, {"ok": False, "error": "bad or missing "
                                                   "X-Glasses-Token"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            # Counted: /health is how the phone side gets debugged, so every
            # refusal has to show up there, not just the interesting ones.
            _stats["rejected"] += 1
            self._json(411, {"ok": False, "error": "Content-Length required"})
            return
        if length > MAX_BYTES:
            _stats["rejected"] += 1
            self._json(413, {"ok": False, "error": "too large"})
            return

        body = self.rfile.read(length)
        ok, msg = save_frame(body)
        if ok:
            print(f"[bridge] +{len(body)//1024}KB  {msg}")
            self._json(200, {"ok": True, "saved": msg})
        else:
            _stats["rejected"] += 1
            print(f"[bridge] rejected: {msg}")
            self._json(415, {"ok": False, "error": msg})

    def log_message(self, *args):
        pass                    # our own prints are more useful


def serve_forever(port: int = PORT) -> None:
    srv = ThreadingHTTPServer((HOST, port), _Handler)
    url = f"http://{lan_ip()}:{port}/frame"
    print(f"[bridge] listening on {HOST}:{port}")
    print(f"[bridge] frames -> {os.path.abspath(FRAME_DIR)}")
    print(f"[bridge] token  -> {'set' if TOKEN else 'NONE (open on the LAN)'}")
    print(f"\n  POST glasses JPEGs to:  {url}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] stopped")
    finally:
        srv.server_close()


def serve_in_background(port: int = PORT) -> ThreadingHTTPServer:
    """Start the receiver inside another process (e.g. Benji's red loop).
    Returns the server so it can be shut down."""
    srv = ThreadingHTTPServer((HOST, port), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[bridge] background receiver on {lan_ip()}:{port} "
          f"-> {os.path.abspath(FRAME_DIR)}")
    return srv


if __name__ == "__main__":
    import argparse
    import shutil
    import tempfile
    import urllib.error
    import urllib.request

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="exercise the endpoint and exit")
    args = ap.parse_args()

    if not args.selftest:
        serve_forever()
        raise SystemExit(0)

    # ---- self test: real HTTP against a real server, then exit ----
    FRAME_DIR = os.path.join(tempfile.gettempdir(), "glasses_bridge_smoke")
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    port = 8799
    srv = serve_in_background(port)
    base = f"http://127.0.0.1:{port}"

    def post(body: bytes, ctype: str = "image/jpeg", token: str = ""):
        req = urllib.request.Request(f"{base}/frame", data=body, method="POST")
        req.add_header("Content-Type", ctype)
        if token:
            req.add_header("X-Glasses-Token", token)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64 + b"\xff\xd9"
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
    HEIC = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 32

    code, body = post(JPEG)
    assert code == 200 and body["ok"], body
    print(f"PASS  JPEG accepted -> {body['saved']}")

    code, body = post(PNG)
    assert code == 200, body
    print("PASS  PNG accepted")

    # A JPEG content-type on HEIC bytes is exactly the mislabelling that
    # would otherwise reach the model and fail confusingly.
    code, body = post(HEIC, ctype="image/jpeg")
    assert code == 415 and "HEIC" in body["error"], body
    print("PASS  HEIC refused despite an image/jpeg header")

    code, body = post(b"<html>not an image</html>")
    assert code == 415, body
    print("PASS  non-image refused")

    code, body = post(b"")
    assert code == 411, body
    print("PASS  empty body refused")

    with urllib.request.urlopen(f"{base}/health", timeout=5) as r:
        health = json.loads(r.read())
    assert health["received"] == 2 and health["rejected"] == 3, health
    print(f"PASS  /health reports {health['received']} received, "
          f"{health['rejected']} rejected")

    # ---- the frames are immediately consumable downstream ----
    os.environ["GLASSES_DIR"] = FRAME_DIR
    import glasses
    import reason
    glasses.GLASSES_DIR = FRAME_DIR
    frame = glasses.latest_frame()
    assert frame is not None, "bridge output must be visible to glasses.py"
    assert not frame.endswith(".part"), "partial writes must never surface"
    url = reason._encode_frame(frame)
    assert url and url.startswith("data:image/"), url
    print(f"PASS  pushed frame reaches the VLM "
          f"({os.path.basename(frame)}, {len(url)} chars)")

    srv.shutdown()
    shutil.rmtree(FRAME_DIR, ignore_errors=True)
    print("\nall glasses_bridge.py assertions passed")
