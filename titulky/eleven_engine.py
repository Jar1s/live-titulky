"""ElevenLabs Scribe v2 Realtime engine — tretí prepínateľný režim.

TITULKY_ENGINE=eleven → mikrofón aj video súbor streamuje do
wss://api.elevenlabs.io/v1/speech-to-text/realtime (scribe_v2_realtime):
  partial_transcript                     → šedý text (pending)
  committed_transcript                   → biely text (committed)
  committed_transcript_with_timestamps   → vwords (slová s časmi pre video)

Kľúč (v tomto poradí): ELEVENLABS_API_KEY env, .elevenlabs_key v koreni
repa, ~/.config/grindcast/elevenlabs.env (KEY=... — zdieľané s grindcast).
Nikdy sa neloguje.

Audio: 16 kHz mono PCM16 — natívne (bez resamplingu, na rozdiel od OpenAI).
Video súbor sa posiela 2× rýchlejšie (feedpos eventy pre buffer na klientovi).
"""
import asyncio
import base64
import json
import os
import pathlib
import threading
import time

import numpy as np
import websockets

try:
    import certifi
    import ssl
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:          # pragma: no cover
    _SSL_CTX = None

SAMPLE_RATE = 16000
CHUNK_S = 0.25              # 250 ms chunk
FILE_SPEED = 2.0            # súborový feed rýchlejšie než real-time
WS_BASE = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
MODEL_ID = "scribe_v2_realtime"


def _api_key() -> str:
    k = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if k:
        return k
    for p in (pathlib.Path(__file__).parent.parent / ".elevenlabs_key",
              pathlib.Path.home() / ".config" / "grindcast" /
              "elevenlabs.env"):
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    lhs, _, rhs = line.partition("=")
                    if lhs.strip() in ("ELEVENLABS_API_KEY", "KEY") \
                            and rhs.strip():
                        return rhs.strip()
                else:
                    return line      # holý kľúč v .elevenlabs_key
    raise RuntimeError("ElevenLabs kľúč nie je — env ELEVENLABS_API_KEY, "
                       ".elevenlabs_key alebo ~/.config/grindcast/"
                       "elevenlabs.env")


class ElevenPipeline:
    """Rovnaké rozhranie, aké main.py používa z TwoTierPipeline."""

    def __init__(self, language="sk", on_update=None, on_event=None):
        self.language = language
        self.on_update = on_update or (lambda d: None)
        self.on_event = on_event or (lambda e: None)
        self.paused = False
        self.running = False
        self.final_text: list[str] = []
        self.draft_text = ""
        self.display_from = 0
        self.draft_latencies: list[float] = []
        self.final_latencies: list[float] = []
        self.compute_s = 0.0
        self._audio_sent_s = 0.0
        self._t_last_voice = None
        self._vw_prev_end = 0.0     # monotónnosť časov slov medzi commitmi
        self._thread = None

    # -- verejné API (zrkadlí TwoTierPipeline) ---------------------------
    def start_mic(self, device=None):
        self._start(("mic", device))

    def start_file(self, path):
        self._start(("file", path))

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)

    def _start(self, source):
        self.running = True
        self._thread = threading.Thread(
            target=lambda: asyncio.run(self._run(source)), daemon=True)
        self._thread.start()

    # -- jadro -----------------------------------------------------------
    async def _run(self, source):
        params = (f"?model_id={MODEL_ID}"
                  f"&language_code={self.language}"
                  f"&include_timestamps=true"
                  f"&commit_strategy=vad")
        headers = {"xi-api-key": _api_key()}
        try:
            async with websockets.connect(
                    WS_BASE + params, additional_headers=headers,
                    ssl=_SSL_CTX, max_size=1 << 24) as ws:
                # počkaj na session_started pred posielaním audia
                first = json.loads(await ws.recv())
                if first.get("message_type") != "session_started":
                    raise RuntimeError(f"nečakaná 1. správa: "
                                       f"{str(first)[:120]}")
                sender = asyncio.create_task(self._send_audio(ws, source))
                try:
                    await self._receive(ws)
                finally:
                    sender.cancel()
        except Exception as e:
            self.on_event({"t": "cloud_err", "msg": str(e)[:200]})
            print(f"[eleven] CHYBA: {e}")
        self.running = False

    async def _push(self, ws, chunk: np.ndarray, commit=False):
        pcm = (np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes()
        await ws.send(json.dumps({
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(pcm).decode(),
            "commit": commit,
            "sample_rate": SAMPLE_RATE,
        }))
        self._audio_sent_s += len(chunk) / SAMPLE_RATE
        if len(chunk) and float(np.abs(chunk).max()) > 0.01:
            self._t_last_voice = time.monotonic()
        now = time.monotonic()
        if now - getattr(self, "_t_audio_evt", 0) > 0.3:
            self._t_audio_evt = now
            rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
            self.on_event({"t": "audio", "rms": rms, "thr": 0.01})

    async def _send_audio(self, ws, source):
        kind, arg = source
        chunk_n = int(SAMPLE_RATE * CHUNK_S)
        if kind == "file":
            import soundfile as sf
            audio, sr = sf.read(arg, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                idx = np.linspace(0, len(audio) - 1,
                                  int(len(audio) * SAMPLE_RATE / sr))
                audio = audio[idx.astype(int)]
            last_pos = 0.0
            for i in range(0, len(audio), chunk_n):
                if not self.running:
                    return
                await self._push(ws, audio[i:i + chunk_n])
                pos = (i + chunk_n) / SAMPLE_RATE
                if pos - last_pos >= 0.5:
                    last_pos = pos
                    self.on_event({"t": "feedpos", "s": round(pos, 2)})
                await asyncio.sleep(CHUNK_S / FILE_SPEED)
            # koniec súboru: finálny commit + čas na posledné odpovede
            await self._push(ws, np.zeros(0, dtype=np.float32), commit=True)
            await asyncio.sleep(3)
            self.on_event({"t": "video_eof"})
            await asyncio.sleep(2)
            self.running = False
        else:
            import sounddevice as sd
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def cb(indata, frames, t, status):
                loop.call_soon_threadsafe(q.put_nowait, indata[:, 0].copy())

            kw = {"device": arg} if arg else {}
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                                blocksize=chunk_n, callback=cb, **kw):
                while self.running:
                    chunk = await q.get()
                    if not self.paused:
                        await self._push(ws, chunk)

    async def _receive(self, ws):
        async for raw in ws:
            if not self.running:
                return
            ev = json.loads(raw)
            mt = ev.get("message_type", "")
            if mt == "partial_transcript":
                self.draft_text = ev.get("text", "")
                if self._t_last_voice:
                    self.draft_latencies.append(
                        time.monotonic() - self._t_last_voice)
                if time.monotonic() - getattr(self, "_t_draft_evt", 0) > 0.3:
                    self._t_draft_evt = time.monotonic()
                    self.on_event({"t": "draft", "text": self.draft_text,
                                   "dur": 0})
                self._emit()
            elif mt == "committed_transcript":
                txt = ev.get("text", "").strip()
                if txt:
                    self.final_text.extend(txt.split())
                    lat = (time.monotonic() - self._t_last_voice) \
                        if self._t_last_voice else None
                    if lat is not None:
                        self.final_latencies.append(lat)
                    self.on_event({"t": "final", "text": txt,
                                   "lat": lat, "dur": 0})
                self.draft_text = ""
                self._emit()
            elif mt == "committed_transcript_with_timestamps":
                items = []
                for w in ev.get("words") or []:
                    if w.get("type") == "spacing":
                        continue
                    txt = str(w.get("text", "")).strip()
                    if not txt:
                        continue
                    s = float(w.get("start", 0.0))
                    e = float(w.get("end", s))
                    items.append({"w": txt, "s": round(s, 2),
                                  "e": round(e, 2)})
                if items:
                    # heuristika: časy relatívne k segmentu → posuň na
                    # absolútne (ak prvé slovo začína pred koncom
                    # predošlého commitu)
                    if items[0]["s"] + 0.3 < self._vw_prev_end:
                        off = self._vw_prev_end
                        for it in items:
                            it["s"] = round(it["s"] + off, 2)
                            it["e"] = round(it["e"] + off, 2)
                    self._vw_prev_end = items[-1]["e"]
                    self.on_event({"t": "vwords", "items": items})
            elif mt in ("input_error", "error", "transcriber_error",
                        "auth_error", "quota_exceeded", "rate_limited",
                        "invalid_request", "chunk_size_exceeded",
                        "session_time_limit_exceeded"):
                self.on_event({"t": "cloud_err",
                               "msg": f"{mt}: {ev.get('error', '')}"[:200]})
                print(f"[eleven] API error: {mt}: {ev.get('error')}")

    def _emit(self):
        self.on_update({
            "committed": " ".join(self.final_text[self.display_from:]),
            "base": self.display_from,
            "pending": self.draft_text,
            "speaking": bool(self.draft_text),
            "draft_latency_s": self.draft_latencies[-1]
            if self.draft_latencies else None,
            "avg_draft_latency_s": (sum(self.draft_latencies)
                                    / len(self.draft_latencies))
            if self.draft_latencies else None,
            "final_latency_s": self.final_latencies[-1]
            if self.final_latencies else None,
            "avg_final_latency_s": (sum(self.final_latencies)
                                    / len(self.final_latencies))
            if self.final_latencies else None,
            "flicker_rate": 0.0,
            "rtf": 0.0,
            "audio_s": round(self._audio_sent_s, 1),
        })
