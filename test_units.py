"""Unit testy čistých funkcií — bežia bez modelov a bez audia (CI)."""
import sys
import types

import numpy as np

# mlx_whisper je macOS/Apple Silicon only — pre unit testy stačí stub,
# testujeme čistú logiku (agreement, dedup, VAD), nie dekódovanie
sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from titulky.asr import LocalAgreement, preprocess
from titulky.pipeline import (
    SpeechLevel,
    dedup_boundary,
    dedupe_repeats,
    word_energy_ok,
)


def w(text, start=0.0, end=0.5):
    return {"text": text, "start": start, "end": end}


def test_local_agreement_commits_on_two_runs():
    la = LocalAgreement()
    new, pending = la.update([w("ahoj"), w("svet")])
    assert new == [] and pending == ["ahoj", "svet"]
    new, _ = la.update([w("ahoj"), w("svet")])
    assert new == ["ahoj", "svet"]


def test_local_agreement_no_commit_on_disagreement():
    la = LocalAgreement()
    la.update([w("ahoj")])
    new, _ = la.update([w("čau")])
    assert new == []


def test_local_agreement_norm_ignores_case_punct():
    la = LocalAgreement()
    la.update([w("Ahoj,")])
    new, _ = la.update([w("ahoj")])
    assert len(new) == 1


def test_dedupe_repeats_collapses_loops():
    words = [w("kakáči") for _ in range(6)]
    assert len(dedupe_repeats(words, max_rep=2)) == 2


def test_dedupe_repeats_keeps_normal_text():
    words = [w("ahoj"), w("svet"), w("ahoj")]
    assert len(dedupe_repeats(words)) == 3


def test_dedup_boundary_removes_overlap():
    committed = ["ahoj", "svet"]
    words = [w("svet"), w("nový")]
    out = dedup_boundary(words, committed)
    assert [x["text"] for x in out] == ["nový"]


def test_dedup_boundary_no_match_keeps_all():
    out = dedup_boundary([w("iné"), w("slová")], ["ahoj", "svet"])
    assert len(out) == 2


def test_speech_level_learns_and_thresholds():
    sl = SpeechLevel()
    loud = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
    sl.update(loud)
    assert sl.level > 0.004          # naučil sa vyššiu úroveň
    silence = np.zeros(4800, dtype=np.float32)
    assert sl.is_silent(silence, loud)
    assert not sl.is_silent(loud[:4800], loud)


def test_word_energy_ok_rejects_silence_word():
    audio = np.zeros(16000, dtype=np.float32)
    audio[:8000] = 0.1               # reč v prvej polovici
    assert word_energy_ok(audio, w("x", 0.0, 0.4), ref_rms=0.05)
    assert not word_energy_ok(audio, w("x", 0.6, 0.9), ref_rms=0.05)


def test_preprocess_normalizes_weak_signal():
    weak = (np.random.default_rng(1).normal(0, 0.001, 16000)
            .astype(np.float32))
    out = preprocess(weak)
    assert float(np.max(np.abs(out))) > 0.3   # gain (strop 100×) zosilnil
    assert float(np.max(np.abs(out))) <= 0.99
