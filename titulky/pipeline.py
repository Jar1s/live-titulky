"""Dvojvrstvová live pipeline (draft + final rewrite).

mikrofón/WAV → spoločný buffer
  ├─ DRAFT vlákno:  small-mlx, krok 0.5 s, okno 6 s → okamžité šedé titulky
  └─ FINAL vlákno:  large-v3-turbo, krok 2 s, okno 12 s → prepíše na biele

Latencia sa meria pre draft (čas zobrazenia prvej hypotézy slova)
aj pre final (čas potvrdenia). Vzor: produkčné partial/final captioning
(Google/Zoom), ufal/whisper_streaming, WhisperLiveKit.
"""
import threading
import time

import numpy as np

from .asr import ASR, DRAFT_REPO, FINAL_REPO, SAMPLE_RATE, LocalAgreement

DRAFT_STEP_S = 0.25
DRAFT_WINDOW_S = 3.0
DRAFT_HOLD_S = 0.8        # min interval prekreslenia draftu (stabilita očí)
FINAL_STEP_S = 2.0   # sweet spot z experiments.py: 97.6 % presnosť, flicker 6.7 %
FINAL_WINDOW_S = 10.0
SILENCE_RMS = 0.008       # horná hranica prahu ticha (hlasný mik)
SILENCE_RMS_MIN = 3e-4    # dolná hranica (tichý MacBook mik: reč ~1e-3)
SILENCE_CLEAR_S = 0.6     # po toľkých sekundách ticha vyčisti draft

class SpeechLevel:
    """Bežiaci odhad úrovne reči (rýchly nábeh, pomalý pokles).
    Inštancia per pipeline — nesmie prežívať medzi behmi/nahrávkami."""

    def __init__(self):
        # štart na úrovni bežnej reči cez MacBook mik — prah je od prvého
        # okna rozumný; skutočná úroveň sa doučí za pár sekúnd
        self.level = 0.004

    def update(self, whole: np.ndarray) -> None:
        w = float(np.sqrt(np.mean(whole ** 2))) if len(whole) else 0.0
        if w > self.level:
            self.level = w
        else:
            self.level += 0.02 * (w - self.level)

    def thr(self) -> float:
        """Prah ticha = 25 % úrovne reči, ohraničený zdola aj zhora."""
        return float(np.clip(0.25 * self.level, SILENCE_RMS_MIN, SILENCE_RMS))

    def is_silent(self, tail: np.ndarray, whole: np.ndarray) -> bool:
        self.update(whole)
        t = float(np.sqrt(np.mean(tail ** 2))) if len(tail) else 0.0
        p = float(np.max(np.abs(tail))) if len(tail) else 0.0
        thr = self.thr()
        return t < thr and p < thr * 3


def word_energy_ok(audio: np.ndarray, w: dict, ref_rms: float) -> bool:
    """Slovo musí ležať v úseku s energiou aspoň 15 % referenčnej úrovne
    reči — inak je to halucinácia z ticha/dychu."""
    seg = audio[int(w["start"] * SAMPLE_RATE):int(w["end"] * SAMPLE_RATE)]
    if not len(seg):
        return False
    return float(np.sqrt(np.mean(seg ** 2))) >= 0.15 * ref_rms


def dedup_boundary(words, committed_tail, max_ngram=5):
    """ufal/whisper_streaming insert() vzor: ak nový dekód začína n-gramom
    zhodným s koncom už potvrdeného textu (posun timestampov medzi oknami),
    prekryv odstráň — inak by agreement videl slová dvakrát."""
    if not words or not committed_tail:
        return words

    def norm(w):
        return w.strip().lower().strip(".,!?…")

    for n in range(min(len(words), len(committed_tail), max_ngram), 0, -1):
        if [norm(w["text"]) for w in words[:n]] == \
           [norm(t) for t in committed_tail[-n:]]:
            return words[n:]
    return words


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
        self.offset_s = 0.0   # koľko sekúnd audia bolo orezaných zo začiatku

    def append(self, chunk):
        with self.lock:
            self.audio = np.concatenate([self.audio, chunk])

    def trim_before(self, keep_from_s: float, pre_context_s: float = 0.5):
        """Orež audio pred keep_from_s (ufal vzor): pamäť O(1) pri dlhých
        sessions a žiadny mel-padding drift. Časové súradnice navonok
        ostávajú absolútne (offset_s)."""
        with self.lock:
            cut_s = keep_from_s - pre_context_s - self.offset_s
            cut = int(cut_s * SAMPLE_RATE)
            if cut > 0:
                self.audio = self.audio[cut:]
                self.offset_s += cut / SAMPLE_RATE

    def snapshot(self, window_s):
        with self.lock:
            n = len(self.audio)
            start = max(0, n - int(window_s * SAMPLE_RATE))
            return (self.audio[start:].copy(),
                    self.offset_s + start / SAMPLE_RATE,
                    self.offset_s + n / SAMPLE_RATE)

    def snapshot_from(self, start_s, max_s):
        """Audio od start_s (absolútne sekundy), max max_s sekúnd."""
        with self.lock:
            n = len(self.audio)
            start = min(max(0, int((start_s - self.offset_s) * SAMPLE_RATE)), n)
            end = min(start + int(max_s * SAMPLE_RATE), n)
            return (self.audio[start:end].copy(),
                    self.offset_s + start / SAMPLE_RATE,
                    self.offset_s + n / SAMPLE_RATE)


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
        self.slevel = SpeechLevel()
        self.final_text: list[str] = []      # potvrdený presný text (celý)
        self.display_from = 0                # index prvého slova aktuálneho bloku
        self.block_start_s = 0.0             # čas audia, kde začína aktuálny blok
        self.final_upto_s = 0.0              # po aký čas audia je final potvrdený
        self.draft_text = ""                 # aktuálny draft za final hranicou
        self.draft_latencies: list[float] = []
        self.compute_s = 0.0                 # súčet dekódovacieho času (RTF)
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
            "rtf": round(self.compute_s / max(
                self.buf.offset_s + len(self.buf.audio) / SAMPLE_RATE, 0.1), 2),
            "audio_s": round(
                self.buf.offset_s + len(self.buf.audio) / SAMPLE_RATE, 1),
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
            if self.slevel.is_silent(tail_a, audio):
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
            words = self.draft_asr.transcribe(audio, self.language,
                                              skip_if_busy=True)
            if words is None:      # GPU drží final dekód — preskoč kolo
                time.sleep(0.1)
                continue
            self.compute_s += time.monotonic() - t_start
            # energia gate aj na draft: referencia = naučená úroveň reči
            # (dych/šum nad prahom ticha inak preblikáva ako „Ďakujem!")
            ref = max(self.slevel.level, float(np.sqrt(np.mean(audio ** 2))))
            words = [w for w in words if word_energy_ok(audio, w, ref)]
            now = time.monotonic()
            # draft = slová za final hranicou
            tail = [w for w in words if off_s + w["end"] > self.final_upto_s]
            new_text = " ".join(w["text"] for w in tail)
            # stabilizácia zobrazenia: prekresli len keď text zmysluplne
            # narástol, alebo raz za DRAFT_HOLD_S (inak drafty „skáču")
            grew = new_text.startswith(self.draft_text) or \
                self.draft_text.startswith(new_text[:max(len(new_text) - 12, 0)])
            if not hasattr(self, "_last_draft_redraw"):
                self._last_draft_redraw = 0.0
            if grew or (now - self._last_draft_redraw) >= DRAFT_HOLD_S:
                self.draft_text = new_text
                self._last_draft_redraw = now
                changed = True
            else:
                changed = False
            # draft latencia: prvé zobrazenie nového slova
            # kľúč toleruje jitter timestampov (zaokrúhlenie na 1 s)
            for w in tail:
                key = (round(off_s + w["start"]), self._norm_key(w["text"]))
                if key not in self._seen_draft_words:
                    self._seen_draft_words[key] = now
                    lat = (now - self.buf.t0) - (off_s + w["end"])
                    if 0 < lat < 30:
                        self.draft_latencies.append(lat)
            if changed:
                self._emit()
            if self.buf.eof and total_s - off_s - len(audio) / SAMPLE_RATE <= 0 \
               and self.buf.audio is not None and self.buf.eof:
                # súbor dočítaný a spracovaný — draft končí až keď final dobehne
                if self.final_upto_s >= total_s - 0.2:
                    break
            elapsed = time.monotonic() - t_start
            time.sleep(max(0.0, DRAFT_STEP_S - elapsed))

    def _final_loop(self):
        eof_retries = 0
        while self.running:
            t_start = time.monotonic()
            # vždy od potvrdenej hranice — okno sa neposúva pod agreementom
            audio, off_s, total_s = self.buf.snapshot_from(
                self.final_upto_s, FINAL_WINDOW_S)
            if len(audio) < SAMPLE_RATE * 1.0 and not self.buf.eof:
                time.sleep(0.2)
                continue
            # VAD gate: celé okno ticho → posuň hranicu, nedekóduj (halucinácie!)
            if len(audio) and self.slevel.is_silent(audio, audio):
                self.final_upto_s = total_s
                self.final_agree.reset()
                if self.buf.eof:
                    break
                time.sleep(FINAL_STEP_S)
                continue
            words = self.final_asr.transcribe(audio, self.language)
            self.compute_s += time.monotonic() - t_start
            words = dedupe_repeats(words)
            words = dedup_boundary(words, self.final_text)
            # energia gate: referencia = max(RMS okna, naučená úroveň reči) —
            # v prevažne tichom okne by samotný RMS okna prepustil
            # halucinácie z ticha („Ďakujem za pozornosť")
            ref = max(float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0,
                      self.slevel.level)
            words = [w for w in words if word_energy_ok(audio, w, ref)]
            # okno plné šumu/ticha, filtre zahodili všetko → posuň hranicu,
            # inak final navždy melie to isté okno a nikdy nedobehne reč
            if not words:
                if len(audio) >= FINAL_WINDOW_S * SAMPLE_RATE * 0.95:
                    self.final_upto_s = total_s - 1.0
                    self.final_agree.reset()
                if self.buf.eof:
                    break
                time.sleep(FINAL_STEP_S)
                continue
            now = time.monotonic()
            new, pending = self.final_agree.update(words)
            # anti-loop: nekomituj presne tú istú sekvenciu, aká už je na
            # konci prepisu (Whisper v slučke opakuje „Ďakujem za pozornosť")
            if new and self.final_text[-len(new):] == new:
                new = []
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
                # dlhé sessions: audio pred potvrdenou hranicou už netreba
                # (ufal vzor) — pamäť O(1). Trim len pri veľkom buffri:
                # na krátkych behoch drobný posun okien zvyšoval flicker.
                if len(self.buf.audio) > 60 * SAMPLE_RATE:
                    self.buf.trim_before(self.final_upto_s, pre_context_s=2.0)
                self._emit()
            if self.buf.eof:
                if self.final_upto_s < total_s - 0.5 and eof_retries < 5:
                    # ešte je nespracované audio — skús ďalší beh
                    eof_retries += 1
                    continue
                # koniec: potvrď zvyšok — ale len slová ležiace v reálnej
                # reči (Whisper z koncového ticha halucinuje „Ďakujem…").
                # pending zodpovedá chvostu words → timestampy z words.
                tail = words[-len(pending):] if pending else []
                kept = []
                for w in tail:
                    s = int((off_s + w["start"] - self.buf.offset_s) * SAMPLE_RATE)
                    e = int((off_s + w["end"] - self.buf.offset_s) * SAMPLE_RATE)
                    seg = self.buf.audio[s:e]
                    if len(seg) and not self.slevel.is_silent(seg, seg):
                        kept.append(w["text"])
                self.final_text.extend(kept)
                if pending:
                    self.final_upto_s = total_s
                    self.draft_text = ""
                    self._emit()
                break
            elapsed = time.monotonic() - t_start
            # dynamický krok: ak sme commitli a stále je backlog reči,
            # pokračuj hneď (final dobieha). Bez commitu čakaj plný krok —
            # hammering nezhodného (noisy) okna len zvyšuje flicker.
            backlog = total_s - self.final_upto_s
            step = 0.0 if (new and backlog > FINAL_WINDOW_S * 0.5) \
                else FINAL_STEP_S
            time.sleep(max(0.0, step - elapsed))

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
