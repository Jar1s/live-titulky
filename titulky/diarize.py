"""Rozpoznanie hovoriacich (diarizácia) — podfarbenie textu podľa rečníka.

Tvrdé podmienky (JBOP-100): nulový dopad na WER a latenciu.
- beží v samostatnom daemon vlákne na CPU (voľnom — draft je na ANE)
- final slučka len vloží segment do fronty (neblokuje)
- výsledok príde ako event {"t": "spk", "a": word_idx, "n": počet, "spk": k}
  ~0.2-0.4 s po zobrazení textu; text sa NIKDY nemení, len farba
- max 3 rečníci (online centroid clustering, kosínusová podobnosť)
- fail-safe: akákoľvek chyba vlákno ukončí, titulky bežia ďalej bez farieb
"""
import queue
import threading

import numpy as np

MAX_SPEAKERS = 3
SIM_THRESHOLD = 0.82   # merané na tests/two_speakers.wav: rovnaký hlas 0.89-0.92, iný 0.67-0.76
MIN_SEG_S = 0.6        # kratšie segmenty: zdedia posledného rečníka
SAMPLE_RATE = 16000


class SpeakerTagger:
    """Online clustering speaker embeddingov (resemblyzer, CPU)."""

    def __init__(self, on_event):
        self.on_event = on_event
        self.q = queue.Queue(maxsize=50)
        self.centroids = []     # [np.ndarray] — bežiaci priemer na rečníka
        self.counts = []
        self.last_spk = 0
        self.enabled = True
        self._t = threading.Thread(target=self._run, daemon=True,
                                   name="diarize")
        self._t.start()

    def submit(self, audio: np.ndarray, word_idx: int, n_words: int):
        """Z final slučky — nikdy neblokuje (plná fronta = segment preskočí)."""
        if not self.enabled:
            return
        try:
            self.q.put_nowait((audio.copy(), word_idx, n_words))
        except queue.Full:
            pass

    # -- interné ----------------------------------------------------------
    def _assign(self, emb: np.ndarray) -> int:
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        if not self.centroids:
            self.centroids.append(emb)
            self.counts.append(1)
            return 0
        sims = [float(np.dot(emb, c / (np.linalg.norm(c) + 1e-9)))
                for c in self.centroids]
        best = int(np.argmax(sims))
        if sims[best] < SIM_THRESHOLD and len(self.centroids) < MAX_SPEAKERS:
            self.centroids.append(emb)
            self.counts.append(1)
            return len(self.centroids) - 1
        # bežiaci priemer centroidu (stabilizuje sa s dátami)
        n = self.counts[best]
        self.centroids[best] = (self.centroids[best] * n + emb) / (n + 1)
        self.counts[best] = n + 1
        return best

    def _run(self):
        try:
            from resemblyzer import VoiceEncoder
            encoder = VoiceEncoder(device="cpu", verbose=False)
        except Exception as e:  # noqa: BLE001 — fail-safe podmienka JBOP-100
            self.enabled = False
            self.on_event({"t": "spk_off", "msg": f"diarizácia vypnutá: {e}"})
            return
        while True:
            audio, widx, n = self.q.get()
            try:
                if len(audio) < MIN_SEG_S * SAMPLE_RATE:
                    spk = self.last_spk
                else:
                    emb = encoder.embed_utterance(audio.astype(np.float32))
                    spk = self._assign(emb)
                    self.last_spk = spk
                self.on_event({"t": "spk", "a": widx, "n": n, "spk": spk})
            except Exception as e:  # noqa: BLE001 — jeden zlý segment ≠ koniec
                self.on_event({"t": "spk_off", "msg": f"diarize chyba: {e}"})
