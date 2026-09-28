"""Dvojvrstvová live pipeline (draft + final rewrite).

mikrofón/WAV → spoločný buffer
  ├─ DRAFT vlákno:  small-mlx, krok 0.5 s, okno 6 s → okamžité šedé titulky
  └─ FINAL vlákno:  large-v3-turbo, krok 2 s, okno 12 s → prepíše na biele

Latencia sa meria pre draft (čas zobrazenia prvej hypotézy slova)
aj pre final (čas potvrdenia). Vzor: produkčné partial/final captioning
(Google/Zoom), ufal/whisper_streaming, WhisperLiveKit.
"""
import queue
import threading
import time
import numpy as np

from .asr import ASR, LocalAgreement, SAMPLE_RATE, DRAFT_REPO, FINAL_REPO

DRAFT_STEP_S = 0.25
DRAFT_WINDOW_S = 3.0
FINAL_STEP_S = 2.0   # sweet spot z experiments.py: 97.6 % presnosť, flicker 6.7 %
FINAL_WINDOW_S = 10.0
SILENCE_RMS = 0.008       # pod týmto = ticho
SILENCE_CLEAR_S = 0.6     # po toľkých sekundách ticha vyčisti draft


def dedupe_repeats(words, max_rep=2):
    """Kolaps halucinačných slučiek (slovo opakované dookola)."""
    out, rep = [], 0
    for w in words:
        k = w["text"].strip().lower().strip(".,!?…")
        if out and k == out[-1]["text"].strip().lower().strip(".,!?…"):
            rep += 1
            if rep >= max_rep:
                continue
        else:
            rep = 0
        out.append(w)
    return out


class SharedBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.audio = np.zeros(0, dtype=np.float32)
        self.eof = False
        self.t0: float = 0.0  # wall-clock štart audia

    def append(self, chunk):
        with self.lock:
            self.audio = np.concatenate([self.audio, chunk])

    def snapshot(self, window_s):
        with self.lock:
            n = len(self.audio)
            start = max(0, n - int(window_s * SAMPLE_RATE))
            return self.audio[start:].copy(), start / SAMPLE_RATE, n / SAMPLE_RATE

    def snapshot_from(self, start_s, max_s):
        """Audio od start_s, max max_s sekúnd."""
        with self.lock:
            n = len(self.audio)
            start = min(int(start_s * SAMPLE_RATE), n)
            end = min(start + int(max_s * SAMPLE_RATE), n)
            return self.audio[start:end].copy(), start / SAMPLE_RATE, n / SAMPLE_RATE


class TwoTierPipeline:
    def __init__(self, language="sk", on_update=None,
                 draft_repo=DRAFT_REPO, final_repo=FINAL_REPO):
        self.language = language
        self.on_update = on_update or (lambda d: None)
        self.buf = SharedBuffer()
        self.draft_asr = ASR(draft_repo)
        # prompt v streamingu NEPOUŽÍVAŤ: pri krátkych oknách ho Whisper
        # halucinuje do výstupu (zisk promptu bol len ~0.3 p.b., škoda veľká)
        self.final_asr = ASR(final_repo)
        self.final_agree = LocalAgreement()
        self.final_text: list[str] = []      # potvrdený presný text (celý)
        self.display_from = 0                # index prvého slova aktuálneho bloku
        self.block_start_s = 0.0             # čas audia, kde začína aktuálny blok
        self.final_upto_s = 0.0              # po aký čas audia je final potvrdený
        self.draft_text = ""                 # aktuálny draft za final hranicou
        self.draft_latencies: list[float] = []
        self.final_latencies: list[float] = []
        self._seen_draft_words: dict[tuple, float] = {}
        self.running = False

    # ---- vstupy -------------------------------------------------------
    def start_mic(self):
        import sounddevice as sd
        self.buf.t0 = time.monotonic()

        def cb(indata, frames, t, status):
            self.buf.append(indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=int(SAMPLE_RATE * 0.1), callback=cb)
        self.stream.start()

    def play_file(self, path, realtime=True):
        import wave
        wf = wave.open(path, "rb")
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1
        self.buf.t0 = time.monotonic()

        def pump():
            chunk_frames = int(SAMPLE_RATE * 0.1)
            while True:
                raw = wf.readframes(chunk_frames)
                if not raw:
                    break
                a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768
                self.buf.append(a)
                if realtime:
                    time.sleep(0.1)
            self.buf.eof = True

        threading.Thread(target=pump, daemon=True).start()

    # ---- vlákna -------------------------------------------------------
    @staticmethod
    def _norm_key(w: str) -> str:
        return w.strip().lower().strip(".,!?…")

    def _emit(self):
        self.on_update({
            "committed": " ".join(self.final_text[self.display_from:]),
            "pending": self.draft_text,
            "draft_latency_s": self.draft_latencies[-1] if self.draft_latencies else None,
            "avg_draft_latency_s": (sum(self.draft_latencies) / len(self.draft_latencies))
                                   if self.draft_latencies else None,
            "final_latency_s": self.final_latencies[-1] if self.final_latencies else None,
            "avg_final_latency_s": (sum(self.final_latencies) / len(self.final_latencies))
                                   if self.final_latencies else None,
            "flicker_rate": self.final_agree.flicker_rate(),
            "audio_s": round(len(self.buf.audio) / SAMPLE_RATE, 1),
        })

    def _draft_loop(self):
        while self.running:
            t_start = time.monotonic()
            audio, off_s, total_s = self.buf.snapshot(DRAFT_WINDOW_S)
            if len(audio) < SAMPLE_RATE * 0.3:
                if self.buf.eof:
                    break
                time.sleep(0.1)
                continue
            # VAD gate: posledných SILENCE_CLEAR_S ticho → koniec bloku;
            # vyčisti draft aj zobrazenú históriu (stará reč sa už nevracia)
            tail_a = audio[-int(SILENCE_CLEAR_S * SAMPLE_RATE):]
            if float(np.sqrt(np.mean(tail_a ** 2))) < SILENCE_RMS:
                # koniec bloku: všetko dopovedané pred týmto tichom patrí
                # do starého bloku — nech final dobehne, ale už NEZOBRAZUJE
                self.block_start_s = max(self.block_start_s,
                                         total_s - SILENCE_CLEAR_S)
                final_caught_up = self.final_upto_s >= self.block_start_s - 0.3
                if self.draft_text or (final_caught_up
                                       and self.display_from < len(self.final_text)):
                    self.draft_text = ""
                    if final_caught_up:
                        self.display_from = len(self.final_text)
                    self._emit()
                time.sleep(DRAFT_STEP_S)
                continue
            words = self.draft_asr.transcribe(audio, self.language)
            now = time.monotonic()
            # draft = slová za final hranicou
            tail = [w for w in words if off_s + w["end"] > self.final_upto_s]
            self.draft_text = " ".join(w["text"] for w in tail)
            # draft latencia: prvé zobrazenie nového slova
            # kľúč toleruje jitter timestampov (zaokrúhlenie na 1 s)
            for w in tail:
                key = (round(off_s + w["start"]), self._norm_key(w["text"]))
                if key not in self._seen_draft_words:
                    self._seen_draft_words[key] = now
                    lat = (now - self.buf.t0) - (off_s + w["end"])
                    if 0 < lat < 30:
                        self.draft_latencies.append(lat)
            self._emit()
            if self.buf.eof and total_s - off_s - len(audio) / SAMPLE_RATE <= 0 \
               and self.buf.audio is not None and self.buf.eof:
                # súbor dočítaný a spracovaný — draft končí až keď final dobehne
                if self.final_upto_s >= total_s - 0.2:
                    break
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, DRAFT_STEP_S - elapsed))

    def _final_loop(self):
        while self.running:
            t_start = time.monotonic()
            # vždy od potvrdenej hranice — okno sa neposúva pod agreementom
            audio, off_s, total_s = self.buf.snapshot_from(
                self.final_upto_s, FINAL_WINDOW_S)
            if len(audio) < SAMPLE_RATE * 1.0 and not self.buf.eof:
                time.sleep(0.2)
                continue
            # VAD gate: celé okno ticho → posuň hranicu, nedekóduj (halucinácie!)
            if len(audio) and float(np.sqrt(np.mean(audio ** 2))) < SILENCE_RMS:
                self.final_upto_s = total_s
                self.final_agree.reset()
                if self.buf.eof:
                    break
                time.sleep(FINAL_STEP_S)
                continue
            words = self.final_asr.transcribe(audio, self.language)
            words = dedupe_repeats(words)
            now = time.monotonic()
            new, pending = self.final_agree.update(words)
            if new:
                belongs_to_old_block = False
                self.final_text.extend(new)
                idx = len(self.final_agree.committed) - 1
                if idx < len(words):
                    self.final_upto_s = off_s + words[idx]["end"]
                    lat = (now - self.buf.t0) - self.final_upto_s
                    if 0 < lat < 60:
                        self.final_latencies.append(lat)
                    belongs_to_old_block = self.final_upto_s <= self.block_start_s
                # oneskorené slová zo starého bloku (pred pauzou) nezobrazuj —
                # do bežiacej novej reči nesmie „dotiecť minulosť"
                if belongs_to_old_block:
                    self.display_from = len(self.final_text)
                # hranica sa posunula → ďalší beh začína inde; zarovnaj agreement
                self.final_agree.reset()
                self.final_agree.prev_words = pending
                self._emit()
            if self.buf.eof:
                # koniec: potvrď zvyšok z posledného behu
                if pending:
                    self.final_text.extend(pending)
                    self.final_upto_s = total_s
                    self.draft_text = ""
                    self._emit()
                break
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, FINAL_STEP_S - elapsed))

    # ---- riadenie -----------------------------------------------------
    def warmup(self):
        self.draft_asr.warmup()
        self.final_asr.warmup()

    def run(self):
        self.running = True
        td = threading.Thread(target=self._draft_loop, daemon=True)
        tf = threading.Thread(target=self._final_loop, daemon=True)
        td.start()
        tf.start()
        tf.join()
        self.running = False
        td.join(timeout=2)

    def stop(self):
        self.running = False


# spätná kompatibilita pre test_offline.py
LivePipeline = TwoTierPipeline
