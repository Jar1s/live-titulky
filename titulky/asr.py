"""ASR jadro: MLX Whisper (Apple GPU/ANE) + LocalAgreement stabilizácia.

Dvojvrstvová architektúra (podľa auditu produkčných riešení —
Google/Zoom/Kuaishou pattern, ufal/whisper_streaming, WhisperLiveKit):
  - DRAFT:  whisper-small-mlx — rýchly, zobrazuje sa okamžite (šedé)
  - FINAL:  whisper-large-v3-turbo — presný, prepíše draft (biele)
"""
import os
import threading
import time

import mlx_whisper
import numpy as np
from scipy.signal import butter, sosfilt

SAMPLE_RATE = 16000
# 2. rád Butterworth highpass 80 Hz — plochý passband, žiadny ripple
# (predtým moving-average: 6 dB/okt, ripple pri cutoffe)
_HP_SOS = butter(2, 80.0, btype="high", fs=SAMPLE_RATE, output="sos")
DRAFT_REPO = os.environ.get("TITULKY_DRAFT_REPO", "mlx-community/whisper-tiny-mlx")
# ^ exp_grid 22.10.: base dekóduje 1.5s okno za ~40ms (small ~240ms) →
#   draft p50 0.73s a nižší flicker; tiny je na SK nepoužiteľný.
FINAL_REPO = "mlx-community/whisper-large-v3-turbo"
# kinit/whisper-large-v3-turbo-sk (SK fine-tune, rovnaká architektúra):
# FLEURS SK test 120 viet: kinit 5.81% vs turbo 10.54% corpus WER;
# naše čisté testy 0.0% vs 3.2%. Konverzia: mlx-examples/whisper
# convert.py --dtype float16 (+ ln -s model.safetensors weights.safetensors).
_KINIT_LOCAL = "/Volumes/GRINDCASTER/AI-MODELS/mlx/kinit-whisper-turbo-sk"
if os.path.isdir(_KINIT_LOCAL):
    FINAL_REPO = _KINIT_LOCAL
FINAL_REPO = os.environ.get("TITULKY_FINAL_REPO", FINAL_REPO)

# mlx_whisper (Numba workqueue vnútri) NIE JE thread-safe — súbežné volanie
# z draft aj final vlákna spôsobí crash celej pipeline („Concurrent access
# has been detected"). Všetky transcribe volania serializujeme.
_MLX_LOCK = threading.Lock()


SK_PROMPT = "Toto je presný prepis hovorovej slovenčiny s diakritikou."


def preprocess(audio: np.ndarray) -> np.ndarray:
    """Butterworth highpass 80 Hz + RMS normalizácia na cieľovú úroveň.

    Podľa researchu (WebRTC AGC pattern): RMS normalizácia namiesto peak —
    jediný tranzient (klik, kašeľ) inak stlačí gain celého okna; slabý
    MacBook mik (reč RMS ~0.001) potrebuje ~40 dB, peak norm dá len ~20 dB.
    Denoising (RNNoise/DeepFilterNet) zámerne NIE JE: literatúra
    (arXiv:2512.17562, 2603.04710) ukazuje, že Whisperu škodí.
    """
    if audio.size == 0:
        return audio.astype(np.float32)  # prázdne okno (race po flush wake)
    out = sosfilt(_HP_SOS, audio).astype(np.float32)
    # Normalizácia: peak s vyšším stropom gainu (40 dB kvôli slabému
    # MacBook miku). RMS varianty testované (suite16/17): plné-RMS zlepšilo
    # noisy 12.9→9.7, ale rozbilo slow 0→12.5; active-RMS rozbilo oboje.
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 1e-4:
        out *= min(0.95 / peak, 100.0)
    return np.clip(out, -0.99, 0.99)


class ASR:
    def __init__(self, repo=FINAL_REPO, beam=1, use_prompt=False):
        # lokálna cesta raz — inak mlx_whisper kontaktuje HF hub pri KAŽDOM volaní
        if os.path.isdir(repo):
            self.repo = repo
        else:
            from huggingface_hub import snapshot_download
            self.repo = snapshot_download(repo)
        self.beam = beam
        self.use_prompt = use_prompt

    def warmup(self):
        # plné okno šumu, nie 1 s ticha — skompiluje Metal kernely na
        # reálnej veľkosti vstupu, inak má prvý živý dekód „rozbeh"
        rng = np.random.default_rng(0)
        self.transcribe(rng.normal(0, 0.01, SAMPLE_RATE * 10)
                        .astype(np.float32))

    def transcribe(self, audio: np.ndarray, language="sk", skip_if_busy=False):
        """audio: float32 mono 16 kHz. Vráti list slov (text, start, end).

        skip_if_busy: nečakaj na _MLX_LOCK — vráť None (draft vrstva radšej
        preskočí kolo, než by čakala sekundy na final dekód).

        Filtre halucinácií (vzor whisper_streaming + whisper-guard):
        - segment s no_speech_prob > 0.8 → zahodiť celý
        - segment s avg_logprob < -1.5 → zahodiť celý
        - slovo s probability < 0.5 A trvaním < 0.15 s → zahodiť
          (krátke nízko-isté vsuvky typu „No" na hraniciach okien)
        """
        kw = {}
        if self.use_prompt:
            kw["initial_prompt"] = SK_PROMPT
        t_req = time.monotonic()
        if skip_if_busy:
            if not _MLX_LOCK.acquire(blocking=False):
                return None
        else:
            _MLX_LOCK.acquire()
        t_lock = time.monotonic()
        try:
            r = mlx_whisper.transcribe(
                preprocess(audio), path_or_hf_repo=self.repo, language=language,
                word_timestamps=True, condition_on_previous_text=False,
                no_speech_threshold=0.6, temperature=0.0, fp16=True, verbose=None,
                **kw,
            )
        finally:
            _MLX_LOCK.release()
        words = []
        drops = []   # (dôvod, text) — do trace/studia, nech gaty nie sú tiché
        for seg in r["segments"]:
            if seg.get("no_speech_prob", 0.0) > 0.8:
                drops.append(("no_speech>0.8", seg.get("text", "").strip()))
                continue
            if seg.get("avg_logprob", 0.0) < -1.5:
                drops.append(("logprob<-1.5", seg.get("text", "").strip()))
                continue
            for w in seg.get("words", []):
                dur = w["end"] - w["start"]
                if w.get("probability", 1.0) < 0.5 and dur < 0.15:
                    drops.append(("p<0.5∧<0.15s", w["word"].strip()))
                    continue
                words.append({"text": w["word"].strip(),
                              "start": w["start"], "end": w["end"]})
        # profilovanie posledného behu — čítajú si ho pipeline loops
        self.last_stats = {"lock_wait_s": round(t_lock - t_req, 3),
                           "decode_s": round(time.monotonic() - t_lock, 3),
                           "drops": drops}
        return words


class CpuDraftASR:
    """Draft na CPU (faster-whisper/ctranslate2, tiny int8) — NEzdieľa GPU
    s final vrstvou, takže draft nehladuje počas final dekódov.

    Merané 22.10.: dekód 1,2 s okna ~0,18 s na CPU (GPU tiny 0,12–0,19 s,
    ale s až 2 s výpadkami počas final dekódu). Rovnaké rozhranie a gaty
    ako ASR.transcribe.
    """

    def __init__(self, model="base", threads=4):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu",
                                  compute_type="int8", cpu_threads=threads)
        self.last_stats = {}

    def warmup(self):
        import numpy as np
        a = np.random.normal(0, 0.01, SAMPLE_RATE * 2).astype(np.float32)
        self.transcribe(a)

    def transcribe(self, audio, language="sk", skip_if_busy=False):
        # skip_if_busy je irelevantné — žiadny zdieľaný lock
        t0 = time.monotonic()
        segs, _ = self.model.transcribe(
            preprocess(audio), language=language, beam_size=1,
            word_timestamps=True, condition_on_previous_text=False,
            temperature=0.0, vad_filter=False)
        words = []
        drops = []
        for seg in segs:
            if getattr(seg, "no_speech_prob", 0.0) > 0.8:
                drops.append(("no_speech>0.8", seg.text.strip()))
                continue
            if getattr(seg, "avg_logprob", 0.0) < -1.5:
                drops.append(("logprob<-1.5", seg.text.strip()))
                continue
            for w in seg.words or []:
                dur = w.end - w.start
                if w.probability < 0.5 and dur < 0.15:
                    drops.append(("p<0.5∧<0.15s", w.word.strip()))
                    continue
                words.append({"text": w.word.strip(),
                              "start": w.start, "end": w.end})
        self.last_stats = {"lock_wait_s": 0.0,
                           "decode_s": round(time.monotonic() - t0, 3),
                           "drops": drops}
        return words


class AneDraftASR:
    """Draft cez whisper.cpp + CoreML — encoder na Apple Neural Engine.

    Merané 22.10. na reálnej reči (1,2 s okno): ANE 0,13–0,17 s vs
    faster-whisper CPU int8 1,7–1,9 s (!) vs GPU tiny 0,12–0,19 s ale
    s ~2 s výpadkami počas final dekódu. ANE = base kvalita, tiny
    rýchlosť, nulová kolízia s GPU (MLX final) aj CPU.
    Build: CMAKE_ARGS="-DWHISPER_COREML=1 -DGGML_METAL=OFF" pip install
    --no-binary :all: pywhispercpp  (+ symlink libwhisper.1.dylib).
    """

    MODEL = "/Volumes/GRINDCASTER/AI-MODELS/whispercpp/base/ggml-base.bin"

    def __init__(self, model=None, threads=4):
        from pywhispercpp.model import Model
        self.model = Model(model or self.MODEL, n_threads=threads,
                           print_progress=False, print_realtime=False)
        self.last_stats = {}

    def warmup(self):
        a = np.random.normal(0, 0.01, SAMPLE_RATE * 2).astype(np.float32)
        self.transcribe(a)

    def transcribe(self, audio, language="sk", skip_if_busy=False):
        t0 = time.monotonic()
        segs = self.model.transcribe(
            preprocess(audio), language=language, token_timestamps=True,
            max_len=1, split_on_word=True)
        words = []
        drops = []
        for s in segs:
            txt = s.text.strip()
            if not txt:
                continue
            # whisper.cpp halucinuje anotácie na šum: (Ktuk), [MUZYKA],
            # *klop*, ♪ — žiadne probability polia, filter textovo
            if (txt[0] in "([*♪" or txt[-1] in ")]*♪"
                    or txt.isupper() and len(txt) > 2):
                drops.append(("non-speech-anotacia", txt))
                continue
            # t0/t1 sú v centisekundách
            words.append({"text": txt, "start": s.t0 / 100.0,
                          "end": s.t1 / 100.0})
        self.last_stats = {"lock_wait_s": 0.0,
                           "decode_s": round(time.monotonic() - t0, 3),
                           "drops": drops}
        return words


class LocalAgreement:
    """Potvrdí len prefix slov, na ktorom sa zhodli dva po sebe idúce behy.

    Zabraňuje 'skákaniu' titulkov: potvrdený text sa už nemení.
    """

    def __init__(self):
        self.prev_words: list[str] = []
        self.committed: list[str] = []
        self.flicker = 0   # koľko slov sa zmenilo medzi behmi (metrika)
        self.total = 0

    @staticmethod
    def _norm(w: str) -> str:
        return w.strip().lower().strip(".,!?…")

    def update(self, words: list[dict]) -> tuple[list[str], list[str]]:
        """Vráti (novo potvrdené slová, nepotvrdený zvyšok)."""
        cur = [w["text"].strip() for w in words]
        n_committed = len(self.committed)
        cur_tail = cur[n_committed:]
        prev_tail = self.prev_words[n_committed:]

        agree = []
        for a, b in zip(cur_tail, prev_tail):
            if self._norm(a) == self._norm(b):
                agree.append(a)
            else:
                break
        self.total += len(cur_tail)
        self.flicker += sum(
            1 for a, b in zip(cur_tail, prev_tail) if self._norm(a) != self._norm(b)
        )
        self.committed.extend(agree)
        self.prev_words = cur
        pending = cur[len(self.committed):]
        return agree, pending

    def reset(self):
        self.prev_words = []
        self.committed = []

    def flicker_rate(self) -> float:
        return self.flicker / self.total if self.total else 0.0
