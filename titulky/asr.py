"""ASR jadro: MLX Whisper (Apple GPU/ANE) + LocalAgreement stabilizácia.

Dvojvrstvová architektúra (podľa auditu produkčných riešení —
Google/Zoom/Kuaishou pattern, ufal/whisper_streaming, WhisperLiveKit):
  - DRAFT:  whisper-small-mlx — rýchly, zobrazuje sa okamžite (šedé)
  - FINAL:  whisper-large-v3-turbo — presný, prepíše draft (biele)
"""
import numpy as np
import mlx_whisper

SAMPLE_RATE = 16000
DRAFT_REPO = "mlx-community/whisper-tiny"
FINAL_REPO = "mlx-community/whisper-large-v3-turbo"


class ASR:
    def __init__(self, repo=FINAL_REPO):
        # lokálna cesta raz — inak mlx_whisper kontaktuje HF hub pri KAŽDOM volaní
        from huggingface_hub import snapshot_download
        self.repo = snapshot_download(repo)

    def warmup(self):
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, language="sk"):
        """audio: float32 mono 16 kHz. Vráti list slov (text, start, end)."""
        r = mlx_whisper.transcribe(
            audio, path_or_hf_repo=self.repo, language=language,
            word_timestamps=True, condition_on_previous_text=False,
            no_speech_threshold=0.6, fp16=True, verbose=None,
        )
        words = []
        for seg in r["segments"]:
            for w in seg.get("words", []):
                words.append({"text": w["word"].strip(),
                              "start": w["start"], "end": w["end"]})
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
