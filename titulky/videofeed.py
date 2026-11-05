"""Upload a serving videí pre /video stránku.

Samostatný stdlib HTTP server na :8766 — websockets handler na :8765
nečíta request body, takže upload cez neho nejde. CORS povolený pre
stránky z :8765. ffmpeg extrahuje 16 kHz mono WAV pre pipeline.
"""
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPLOADS = pathlib.Path.home() / ".config" / "titulky" / "uploads"
PORT = int(os.environ.get("TITULKY_VIDEO_PORT", "8766"))
MAX_BYTES = 2 * 1024 ** 3   # 2 GB strop

_CT = {".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
       ".m4v": "video/x-m4v", ".mkv": "video/x-matroska"}


def extract_wav(src: pathlib.Path, dst: pathlib.Path) -> float:
    """ffmpeg → 16 kHz mono WAV; vráti trvanie v sekundách."""
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1",
         "-ar", "16000", "-f", "wav", str(dst)],
        capture_output=True, timeout=600)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError(
            f"ffmpeg zlyhal: {r.stderr.decode(errors='replace')[-300:]}")
    with wave.open(str(dst), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):   # ticho
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers",
                         "X-Filename, Content-Type")
        self.send_header("Access-Control-Allow-Methods",
                         "POST, GET, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _json(self, code: int, data: dict):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/upload":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BYTES:
            return self._json(413, {"error": f"veľkosť {n} mimo limitu 2 GB"})
        name = self.headers.get("X-Filename", "video.mp4")
        ext = pathlib.Path(name).suffix.lower() or ".mp4"
        if ext not in _CT:
            return self._json(415, {"error": f"nepodporovaný formát {ext}"})
        vid = time.strftime("%Y%m%d-%H%M%S")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        dst = UPLOADS / f"{vid}{ext}"
        remaining = n
        with open(dst, "wb") as f:
            while remaining:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        try:
            dur = extract_wav(dst, UPLOADS / f"{vid}.wav")
        except Exception as e:
            dst.unlink(missing_ok=True)
            return self._json(422, {"error": str(e)[:300]})
        self._json(200, {"id": vid, "media": f"/media/{vid}{ext}",
                         "duration_s": round(dur, 1)})

    def do_GET(self):
        m = re.fullmatch(r"/media/([\w.-]+)", self.path)
        if not m:
            return self._json(404, {"error": "not found"})
        p = UPLOADS / os.path.basename(m.group(1))
        if not p.is_file():
            return self._json(404, {"error": "not found"})
        size = p.stat().st_size
        ct = _CT.get(p.suffix.lower(), "application/octet-stream")
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            mm = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if mm:
                if mm.group(1):
                    start = int(mm.group(1))
                    if mm.group(2):
                        end = min(int(mm.group(2)), size - 1)
                elif mm.group(2):          # bytes=-N (suffix)
                    start = max(0, size - int(mm.group(2)))
                code = 206
        if start > end or start >= size:
            self.send_response(416)
            self._cors()
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        length = end - start + 1
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", ct)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if code == 206:
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return   # prehrávač zrušil range request — normálne
                left -= len(chunk)


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
