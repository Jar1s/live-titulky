"""ASR jadro: faster-whisper + LocalAgreement stabilizácia."""
import time
import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000


class ASR:
    def __init__(self, model_size="small", device="cpu", compute_type="int8"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, language="sk"):
        """audio: float32 mono 16 kHz. Vráti list slov (text, start, end)."""
        segments, _ = self.model.transcribe(
            audio, language=language, word_timestamps=True,
            vad_filter=True, beam_size=1,
            condition_on_previous_text=False,
        )
        words = []
        for seg in segments:
            for w in seg.words or []:
                words.append({"text": w.word, "start": w.start, "end": w.end})
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

    def flicker_rate(self) -> float:
        return self.flicker / self.total if self.total else 0.0
