"""Asynchrónna slovná korekcia committed titulkov (ensemble second opinion).

Vzor: grindcast_auto/asr_ensemble.py (merge 1:1, hunspell gate), sprísnený
podľa researchu (ROVER literatúra + CHI 2023):

  - Azure (iná architektúra) prepíše ten istý zvuk; chyby sú nekorelované
    s Whisperom, preto vie opraviť OOV halucinácie ("nežňašam"→"neznášam").
  - Mení sa LEN 1:1 rozdiel v difflib replace bloku s rovnakým počtom slov.
  - Gaty (všetky musia platiť):
      1. Whisper slovo je mimo sk_SK slovníka (hunspell) — nikdy nemeníme
         platné slovo (anglicizmy "hater" sú OOV, ale gate 2 ich ochráni),
      2. Azure slovo v slovníku JE (nikdy platné→neplatné),
      3. Levenshtein(whisper, azure) <= 2 — fonetická blízkosť; Azure
         halucinácia, čo náhodou padne do slovníka, neprejde,
      4. žiadne propriá (veľké písmeno v strede vety) ani čísla.
  - Okno: slovo staršie než CORRECTION_WINDOW_S sa už nemení (UX, CHI 2023).
  - Bez kľúča / siete / SDK → modul sa ticho vypne, pipeline beží lokálne.

Kľúč: .azure_key v koreňi projektu alebo ~/.config/grindcast/azure_speech.env.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import unicodedata
import wave

import numpy as np

SAMPLE_RATE = 16000
CORRECTION_WINDOW_S = 10.0   # staršie slová sa nemenia (UX)
MIN_BATCH_S = 4.0            # Azure volanie až keď je aspoň toľko audia
AZURE_TIMEOUT_S = 8          # max čakanie na Azure/hunspell (bolo 30 s; znižuje latency pri fallback)
HUNSPELL_DICT = "sk_SK"
_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)

KEY_PATHS = [
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".azure_key"),
    os.path.expanduser("~/.config/grindcast/azure_speech.env"),
]


def load_azure_key() -> tuple[str, str] | None:
    for path in KEY_PATHS:
        if not os.path.exists(path):
            continue
        cfg = {}
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
        key = cfg.get("KEY") or cfg.get("AZURE_SPEECH_KEY")
        if key:
            return key, (cfg.get("REGION") or cfg.get("AZURE_SPEECH_REGION")
                         or "westeurope")
    return None


def norm_token(w: str) -> str:
    w = unicodedata.normalize("NFC", w)
    return _WORD_RE.sub("", w.lower())


def _deaccent(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if not unicodedata.combining(c))


def levenshtein(a: str, b: str) -> int:
    """Vzdialenosť bez diakritiky — ž/z, ň/n sú fonetické dvojice, nie
    rozdielne písmená ('nežňašam' vs 'neznášam' = 0, nie 3)."""
    a, b = _deaccent(a), _deaccent(b)
    if abs(len(a) - len(b)) > 2:
        return 3  # early out — nad gate limitom
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def hunspell_unknown(words: list[str], dic: str = HUNSPELL_DICT) -> set[str] | None:
    """Množina normalizovaných slov mimo slovníka. None = hunspell nie je."""
    exe = shutil.which("hunspell")
    if not exe:
        return None
    toks = sorted({norm_token(w) for w in words
                   if norm_token(w) and not norm_token(w).isdigit()})
    if not toks:
        return set()
    try:
        out = subprocess.run([exe, "-d", dic, "-l"],
                             input="\n".join(toks) + "\n",
                             capture_output=True, text=True, timeout=AZURE_TIMEOUT_S,
                             check=False).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return {norm_token(x) for x in out.split() if norm_token(x)}


def _case_like(src: str, new: str) -> str:
    if src[:1].isupper() and not new[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


def _carry_punct(src: str, new: str) -> str:
    lead = re.match(r"^[^\w]*", src, re.UNICODE).group(0)
    trail = re.search(r"[^\w]*$", src, re.UNICODE).group(0)
    return lead + new + trail


def merge_words(whisper_words: list[str], azure_text: str,
                unknown: set[str]) -> list[dict]:
    """Vráti zoznam zmien [{'i', 'from', 'to'}]. Počet slov sa NEMENÍ.

    Gaty podľa docstringu modulu — konzervatívne: pri pochybnosti nemeniť.
    """
    import difflib
    az = [t for t in azure_text.split() if norm_token(t)]
    a = [norm_token(w) for w in whisper_words]
    b = [norm_token(t) for t in az]
    changes = []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        for k in range(i2 - i1):
            wi, zj = i1 + k, j1 + k
            wn, zn = a[wi], b[zj]
            if not wn or not zn or wn == zn:
                continue
            if wn.isdigit() or zn.isdigit():
                continue
            src = whisper_words[wi]
            # gate 4: propriá — slovo s veľkým písmenom nechaj tak
            if src[:1].isupper():
                continue
            # gate 1+2: whisper OOV a azure v slovníku
            if wn not in unknown or zn in unknown:
                continue
            # gate 3: fonetická blízkosť
            if levenshtein(wn, zn) > 2:
                continue
            new = _carry_punct(src, _case_like(src, az[zj].strip(".,!?;:\"'()")))
            if new != src:
                changes.append({"i": wi, "from": src, "to": new})
    return changes


def _to_wav(audio: np.ndarray, path: str):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


class CorrectionWorker:
    """Fronta committed dávok → Azure → gated merge → callback do pipeline.

    apply_cb(start_idx, changes) voláva pipeline pod vlastným lockom.
    """

    def __init__(self, apply_cb, log=print):
        self.apply_cb = apply_cb
        self.log = log
        self.enabled = False
        self._queue: list[tuple[int, list[str], np.ndarray, float]] = []
        self._lock = threading.Lock()
        self._thread = None
        self.corrections_total = 0
        creds = load_azure_key()
        if not creds:
            log("[korekcia] bez Azure kľúča — vypnutá, beží čisto lokálne")
            return
        try:
            import azure.cognitiveservices.speech  # noqa: F401
        except ImportError:
            log("[korekcia] Azure SDK chýba — vypnutá")
            return
        if hunspell_unknown(["test"]) is None:
            log("[korekcia] hunspell chýba — vypnutá (brew install hunspell)")
            return
        self.key, self.region = creds
        self.enabled = True
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log(f"[korekcia] aktívna (Azure {self.region}, okno "
            f"{CORRECTION_WINDOW_S:.0f}s)")

    def submit(self, start_idx: int, words: list[str], audio: np.ndarray):
        """Pipeline sem pošle committed dávku + jej audio segment."""
        if not self.enabled or not words or not len(audio):
            return
        with self._lock:
            self._queue.append((start_idx, words, audio, time.monotonic()))

    def stop(self):
        self.running = False

    def _loop(self):
        import tempfile
        while self.running:
            batch = None
            with self._lock:
                # spoj susedné dávky do jedného Azure volania (šetrí F0 kvótu)
                while self._queue:
                    item = self._queue.pop(0)
                    if batch is None:
                        batch = list(item)
                    elif batch[0] + len(batch[1]) == item[0]:
                        batch[1] = batch[1] + item[1]
                        batch[2] = np.concatenate([batch[2], item[2]])
                    else:
                        self._queue.insert(0, item)
                        break
            if batch is None:
                time.sleep(0.3)
                continue
            start_idx, words, audio, t_commit = batch
            if len(audio) < MIN_BATCH_S * SAMPLE_RATE:
                # krátka dávka: chvíľu počkaj, či nepríde pokračovanie
                if time.monotonic() - t_commit < 2.0:
                    with self._lock:
                        self._queue.insert(0, tuple(batch))
                    time.sleep(0.5)
                    continue
            try:
                azure_text = self._azure(audio, tempfile.gettempdir())
            except Exception as e:  # noqa: BLE001 — sieť/kvóta, best-effort
                self.log(f"[korekcia] azure zlyhalo: {str(e)[:120]}")
                continue
            if time.monotonic() - t_commit > CORRECTION_WINDOW_S:
                continue  # neskoro — slová už nemeníme (UX)
            if not azure_text:
                continue
            unknown = hunspell_unknown(words + azure_text.split())
            if unknown is None or not unknown:
                continue  # hunspell odpadol alebo niet čo opravovať
            changes = merge_words(words, azure_text, unknown)
            if changes:
                self.corrections_total += len(changes)
                for ch in changes:
                    self.log(f"[korekcia] {ch['from']} → {ch['to']}")
                self.apply_cb(start_idx, changes)

    def _azure(self, audio: np.ndarray, tmpdir: str) -> str:
        import azure.cognitiveservices.speech as speechsdk
        path = os.path.join(tmpdir, f"corr_{os.getpid()}_{time.time_ns()}.wav")
        _to_wav(audio, path)
        try:
            cfg = speechsdk.SpeechConfig(subscription=self.key,
                                         region=self.region)
            cfg.speech_recognition_language = "sk-SK"
            rec = speechsdk.SpeechRecognizer(
                speech_config=cfg,
                audio_config=speechsdk.audio.AudioConfig(filename=path))
            parts: list[str] = []
            done = {"v": False}
            rec.recognized.connect(lambda e: parts.append(e.result.text))
            rec.session_stopped.connect(lambda e: done.update(v=True))
            rec.canceled.connect(lambda e: done.update(v=True))
            t0 = time.time()
            rec.start_continuous_recognition()
            while not done["v"] and time.time() - t0 < AZURE_TIMEOUT_S:
                time.sleep(0.1)
            rec.stop_continuous_recognition()
            return " ".join(p for p in parts if p).strip()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
