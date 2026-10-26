"""Asynchrónna LLM korekcia committed titulkov cez lokálnu ollama.

Rovnaký kanál a gaty ako Azure ensemble (correction.merge_words):
LLM prepíše text, zmení sa LEN 1:1 náhrada kde whisper slovo je OOV,
LLM slovo je v slovníku, levenshtein<=2, žiadne propriá. Pri
pochybnosti nemeniť. Negatívny výsledok z 23.10.: qwen2.5:3b WER 84%
(nedodrží inštrukciu) — preto min 8B a tvrdý word-count guard.

Beh na GPU (CPU qwen3:8b nestíhal 25s timeout aj pre think tokeny);
kontencia s final dekódom je bursty — merané suite-om.

Env: TITULKY_LLM_CORR=1 zapne, TITULKY_LLM_MODEL (default qwen3:8b),
OLLAMA_URL (default http://localhost:11434).
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request

from .correction import (CORRECTION_WINDOW_S, hunspell_unknown,
                         merge_words, norm_token)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("TITULKY_LLM_MODEL", "qwen3:8b")
MIN_BATCH_W = 6          # slov; menšie dávky LLM len kazí
TIMEOUT_S = 25

PROMPT = (
    "/no_think Oprav preklepy a chybne rozpoznané slová v slovenskom "
    "prepise prednášky o telekomunikáciách. Pravidlá: vráť IBA opravený "
    "text, presne rovnaký počet slov, žiadne komentáre, žiadne "
    "úvodzovky, nemeň slovosled ani správne slová.\n\n"
    "Prepis: {text}\n\nOpravený:")


def _ollama_fix(text: str) -> str | None:
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=json.dumps({
            "model": MODEL,
            "prompt": PROMPT.format(text=text),
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 160},
            "think": False,
        }).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        out = json.loads(r.read())["response"].strip()
    # qwen3 think tagy / prefixy odstráň
    if "</think>" in out:
        out = out.split("</think>", 1)[1].strip()
    for pref in ("Opravený:", "Opraveny:", "Prepis:"):
        if out.startswith(pref):
            out = out[len(pref):].strip()
    return out or None


class LlmCorrectionWorker:
    """submit() kompatibilný s CorrectionWorker — druhý názor z LLM."""

    def __init__(self, apply_cb, log=print):
        self.apply_cb = apply_cb
        self.log = log
        self.enabled = os.environ.get("TITULKY_LLM_CORR", "0") == "1"
        self._q: list[tuple[float, int, list[str]]] = []
        self._lock = threading.Lock()
        if not self.enabled:
            return
        try:
            urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        except Exception:
            self.enabled = False
            self.log("[llm-korekcia] ollama nedostupná — vypínam")
            return
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, start_idx: int, words: list[str], audio=None):
        if not self.enabled or not words:
            return
        with self._lock:
            self._q.append((time.monotonic(), start_idx, list(words)))

    def _run(self):
        while True:
            time.sleep(0.5)
            with self._lock:
                if not self._q:
                    continue
                # spoj čakajúce dávky (súvislý text má viac kontextu)
                batches = self._q[:]
                total = sum(len(w) for _, _, w in batches)
                if total < MIN_BATCH_W and \
                        time.monotonic() - batches[0][0] < 3.0:
                    continue
                self._q.clear()
            for t0, idx, words in batches:
                if time.monotonic() - t0 > CORRECTION_WINDOW_S:
                    continue   # už je po okne, UX nemeniť
                try:
                    t1 = time.monotonic()
                    fixed = _ollama_fix(" ".join(words))
                    dt = time.monotonic() - t1
                    if not fixed:
                        continue
                    unknown = hunspell_unknown(
                        [norm_token(w) for w in words])
                    if unknown is None:
                        continue
                    changes = merge_words(words, fixed, unknown)
                    if changes:
                        self.apply_cb(idx, changes, src="llm")
                        self.log(f"[korekcia] llm {dt:.1f}s: " + ", ".join(
                            f"{c['from']}→{c['to']}" for c in changes))
                except Exception as e:
                    self.log(f"[llm-korekcia] chyba: {e}")
