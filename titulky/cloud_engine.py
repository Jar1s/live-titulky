"""Cloud STT engine — OpenAI Realtime transcription ako prepínateľný režim.

TITULKY_ENGINE=cloud → namiesto lokálnej dvojvrstvovej pipeline streamuje
mikrofón (alebo WAV v reálnom čase) do OpenAI Realtime API:
  delta eventy  → šedý text (pending)
  completed     → biely text (committed)

Lokálna pipeline ostáva default (semestrálka beží bez cloudu).
Kľúč: OPENAI_API_KEY env alebo .openai_key v koreni repa (mimo gitu).
Model: TITULKY_CLOUD_MODEL (default gpt-4o-mini-transcribe, $0.003/min).
"""
import asyncio
import base64
import json
import os
import pathlib
import re
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
CLOUD_RATE = 24000   # GA realtime API berie len 24 kHz PCM
CHUNK_S = 0.10          # 100 ms chunk → plynulý stream
CLOUD_MODEL = os.environ.get("TITULKY_CLOUD_MODEL", "gpt-4o-mini-transcribe")
WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"


def _api_key() -> str:
    k = os.environ.get("OPENAI_API_KEY", "").strip()
    if not k:
        p = pathlib.Path(__file__).parent.parent / ".openai_key"
        if p.exists():
            k = p.read_text().strip()
    if not k:
        raise RuntimeError("OPENAI_API_KEY nie je — env alebo .openai_key")
    return k


class CloudPipeline:
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
        self._audio_sent_s = 0.0      # koľko sekúnd audia odoslaných
        self._t_speech_stop = None    # wall čas posledného speech_stopped
        self._t_last_voice = None     # wall čas posledného chunku s rečou
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
        headers = {"Authorization": f"Bearer {_api_key()}"}
        try:
            async with websockets.connect(
                    WS_URL, additional_headers=headers,
                    ssl=_SSL_CTX, max_size=1 << 24) as ws:
                await ws.send(json.dumps({
                    "type": "session.update",
                    "session": {
                        "type": "transcription",
                        "audio": {
                            "input": {
                                "format": {"type": "audio/pcm",
                                           "rate": CLOUD_RATE},
                                "transcription": {
                                    "model": CLOUD_MODEL,
                                    "language": self.language,
                                },
                                # gpt-live-transcribe ma vlastny
                                # streaming — turn_detection nepodporuje
                                "turn_detection": None
                                if CLOUD_MODEL == "gpt-live-transcribe" else {
                                    "type": "server_vad",
                                    "threshold": 0.25,
                                    "silence_duration_ms": 350,
                                    "prefix_padding_ms": 300,
                                },
                            },
                        },
                    },
                }))
                sender = asyncio.create_task(self._send_audio(ws, source))
                committer = asyncio.create_task(self._auto_commit())
                try:
                    await self._receive(ws)
                finally:
                    sender.cancel()
                    committer.cancel()
        except Exception as e:
            self.on_event({"t": "cloud_err", "msg": str(e)[:200]})
            print(f"[cloud] CHYBA: {e}")
        self.running = False

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
            t0 = time.monotonic()
            for i in range(0, len(audio), chunk_n):
                if not self.running:
                    return
                # reálny čas — simulácia live vstupu
                target = i / SAMPLE_RATE
                wait = t0 + target - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                await self._push(ws, audio[i:i + chunk_n])
            # dohraté — commitni zvyšok buffera a nechaj serveru čas
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            await asyncio.sleep(8)
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

    async def _push(self, ws, chunk: np.ndarray):
        # 16 kHz -> 24 kHz (GA API berie len 24 kHz)
        n24 = int(len(chunk) * CLOUD_RATE / SAMPLE_RATE)
        idx = np.linspace(0, len(chunk) - 1, n24)
        up = np.interp(idx, np.arange(len(chunk)), chunk)
        pcm = (np.clip(up, -1, 1) * 32767).astype("<i2").tobytes()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode(),
        }))
        self._audio_sent_s += len(chunk) / SAMPLE_RATE
        if float(np.abs(chunk).max()) > 0.01:
            self._t_last_voice = time.monotonic()

    async def _receive(self, ws):
        async for raw in ws:
            if not self.running:
                return
            ev = json.loads(raw)
            t = ev.get("type", "")
            if t == "conversation.item.input_audio_transcription.delta":
                item = ev.get("item_id")
                if getattr(self, "_cur_item", None) not in (None, item):
                    # novy item — predosly sedy text commitni (live-transcribe
                    # bez VAD nie vzdy posle completed)
                    if self.draft_text.strip():
                        self.final_text.extend(self.draft_text.split())
                        if self._t_last_voice:
                            self.final_latencies.append(
                                time.monotonic() - self._t_last_voice)
                    self.draft_text = ""
                self._cur_item = item
                self.draft_text = (self.draft_text + ev.get("delta", "")) \
                    if self.draft_text else ev.get("delta", "")
                self._t_last_delta = time.monotonic()
                # uzavrete vety commitni HNED, ked uz bezi dalsia veta —
                # deltas su append-only, revizia nehrozi
                m = None
                for m in re.finditer(r"[.!?…]\s+", self.draft_text):
                    pass
                if m and len(self.draft_text) > m.end() + 1:
                    head = self.draft_text[:m.end()].strip()
                    self.final_text.extend(head.split())
                    if self._t_last_voice:
                        self.final_latencies.append(
                            time.monotonic() - self._t_last_voice)
                    self.draft_text = self.draft_text[m.end():].lstrip()
                if self._t_last_voice:
                    self.draft_latencies.append(
                        time.monotonic() - self._t_last_voice)
                self._emit()
            elif t == "conversation.item.input_audio_transcription.completed":
                self._cur_item = None
                txt = ev.get("transcript", "").strip()
                if txt:
                    self.final_text.extend(txt.split())
                self.draft_text = ""
                ref = self._t_speech_stop or self._t_last_voice
                if ref:
                    self.final_latencies.append(time.monotonic() - ref)
                self._emit()
            elif t == "input_audio_buffer.speech_stopped":
                self._t_speech_stop = time.monotonic()
            elif t == "error":
                self.on_event({"t": "cloud_err",
                               "msg": str(ev.get("error"))[:200]})
                print(f"[cloud] API error: {ev.get('error')}")

    async def _auto_commit(self):
        """gpt-live-transcribe nedava completed — commituj dokoncene vety,
        ked sa delta tok na chvilu odmlci (hranica vety + 0.8 s ticha)."""
        SENT_END = re.compile(r"[.!?…]\s")
        while self.running:
            await asyncio.sleep(0.1)
            d = self.draft_text
            t_last = getattr(self, "_t_last_delta", None)
            if not d or t_last is None:
                continue
            idle = time.monotonic() - t_last
            cut = 0
            if idle > 0.45:
                ends = [m.end() for m in SENT_END.finditer(d + " ")]
                if ends:
                    cut = ends[-1]
            if idle > 2.0:          # dlhe ticho — commitni vsetko
                cut = len(d)
            if cut:
                head = d[:cut].strip()
                if head:
                    self.final_text.extend(head.split())
                    if self._t_last_voice:
                        self.final_latencies.append(
                            time.monotonic() - self._t_last_voice)
                self.draft_text = d[cut:].lstrip()
                self._emit()

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
