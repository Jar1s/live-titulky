"""Live pipeline: mikrofón (alebo WAV súbor ako simulovaný live vstup)
→ kĺzavé okno → ASR → LocalAgreement → callback s titulkami + metrikami.
"""
import queue
import threading
import time
import numpy as np

from .asr import ASR, LocalAgreement, SAMPLE_RATE

WINDOW_S = 8.0      # kĺzavé okno posielané do ASR
STEP_S = 2.0        # ako často spúšťame ASR


class LivePipeline:
    def __init__(self, model_size="small", language="sk", step_s=STEP_S,
                 window_s=WINDOW_S, on_update=None):
        self.asr = ASR(model_size)
        self.agree = LocalAgreement()
        self.language = language
        self.step_s = step_s
        self.window_s = window_s
        self.on_update = on_update or (lambda d: None)
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.buffer = np.zeros(0, dtype=np.float32)
        self.buffer_offset_s = 0.0   # čas začiatku bufferu od štartu
        self.t0 = None               # wall-clock štart audia
        self.latencies: list[float] = []
        self.committed_text: list[str] = []
        self.running = False

    # ---- vstupy -------------------------------------------------------
    def feed(self, chunk: np.ndarray):
        """float32 mono 16 kHz chunk (z mikrofónu alebo súboru)."""
        self.audio_q.put(chunk)

    def start_mic(self):
        import sounddevice as sd
        self.t0 = time.monotonic()

        def cb(indata, frames, t, status):
            self.feed(indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1), callback=cb)
        self.stream.start()

    def play_file(self, path, realtime=True):
        """Simulovaný live vstup zo súboru (WAV 16 kHz mono, na testy)."""
        import wave
        wf = wave.open(path, "rb")
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
        self.t0 = time.monotonic()

        def pump():
            chunk_frames = int(SAMPLE_RATE * 0.1)
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
                self.feed(a)
                if realtime:
                    time.sleep(0.1)
            self.feed(None)  # sentinel: koniec vstupu

        threading.Thread(target=pump, daemon=True).start()

    # ---- hlavná slučka ------------------------------------------------
    def run(self):
        self.running = True
        eof = False
        last_asr = 0.0
        while self.running:
            try:
                chunk = self.audio_q.get(timeout=0.2)
                if chunk is None:
                    eof = True
                else:
                    self.buffer = np.concatenate([self.buffer, chunk])
            except queue.Empty:
                pass

            now = time.monotonic()
            audio_s = self.buffer_offset_s + len(self.buffer) / SAMPLE_RATE
            if (now - last_asr >= self.step_s and len(self.buffer) > SAMPLE_RATE * 0.5) or eof:
                last_asr = now
                # orež okno
                max_len = int(self.window_s * SAMPLE_RATE)
                if len(self.buffer) > max_len:
                    cut = len(self.buffer) - max_len
                    # posunúť možno len ak je začiatok potvrdený — zjednodušene
                    # režeme a resetujeme agreement na committed hranici
                    self.buffer = self.buffer[cut:]
                    self.buffer_offset_s += cut / SAMPLE_RATE
                    self.agree.prev_words = []
                    self.agree.committed = []

                words = self.asr.transcribe(self.buffer, self.language)
                new, pending = self.agree.update(words)
                if new or pending:
                    # latencia: wall-clock teraz − čas konca posl. potvrdeného slova v audiu
                    if new and words:
                        idx = len(self.agree.committed) - 1
                        if idx < len(words):
                            w_end = self.buffer_offset_s + words[idx]["end"]
                            lat = (time.monotonic() - self.t0) - w_end
                            self.latencies.append(lat)
                    self.committed_text.extend(new)
                    self.on_update({
                        "committed": " ".join(self.committed_text),
                        "pending": " ".join(pending),
                        "latency_s": self.latencies[-1] if self.latencies else None,
                        "avg_latency_s": (sum(self.latencies) / len(self.latencies))
                                         if self.latencies else None,
                        "flicker_rate": self.agree.flicker_rate(),
                        "audio_s": round(audio_s, 1),
                    })
                if eof and self.audio_q.empty():
                    # koniec vstupu: potvrď aj zvyšok
                    if pending:
                        self.committed_text.extend(pending)
                        self.on_update({
                            "committed": " ".join(self.committed_text),
                            "pending": "",
                            "latency_s": None,
                            "avg_latency_s": (sum(self.latencies) / len(self.latencies))
                                             if self.latencies else None,
                            "flicker_rate": self.agree.flicker_rate(),
                            "audio_s": round(audio_s, 1),
                        })
                    break
        self.running = False

    def stop(self):
        self.running = False
