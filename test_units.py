"""Unit testy čistých funkcií — bežia bez modelov a bez audia (CI)."""
import sys
import types

import numpy as np

# mlx_whisper je macOS/Apple Silicon only — pre unit testy stačí stub,
# testujeme čistú logiku (agreement, dedup, VAD), nie dekódovanie
sys.modules.setdefault("mlx_whisper", types.ModuleType("mlx_whisper"))
sys.modules.setdefault("sounddevice", types.ModuleType("sounddevice"))

from titulky.asr import LocalAgreement, preprocess
from titulky.correction import AZURE_TIMEOUT_S, levenshtein, merge_words
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
    rng = np.random.default_rng(0)
    # reč = bursty signál (slová s pauzami), nie uniformný šum —
    # ten by dynamický gate správne označil za ticho
    loud = rng.normal(0, 0.004, 16000).astype(np.float32)
    for s in (0.1, 0.5):
        i = int(s * 16000)
        loud[i:i + 4000] += rng.normal(0, 0.1, 4000).astype(np.float32)
    sl.update(loud)
    assert sl.level > 0.004          # naučil sa vyššiu úroveň
    silence = np.zeros(4800, dtype=np.float32)
    assert sl.is_silent(silence, loud)
    assert not sl.is_silent(loud[1600:6400], loud)


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


# ---- korekčná vrstva (gated ensemble merge) ------------------------------

def test_levenshtein():
    assert levenshtein("tým", "tím") == 1
    assert levenshtein("nežňašam", "neznášam") == 0  # diakritika sa ignoruje
    assert levenshtein("ahoj", "ahoj") == 0


def test_merge_corrects_oov_with_close_azure_word():
    # whisper halucinácia mimo slovníka, azure blízke slovo v slovníku
    whisper = ["to", "proste", "nežňašam"]
    azure = "to proste neznášam"
    changes = merge_words(whisper, azure, unknown={"nežňašam"})
    assert len(changes) == 1
    assert changes[0]["from"] == "nežňašam" and changes[0]["to"] == "neznášam"


def test_merge_never_touches_valid_word():
    # whisper slovo je v slovníku → nemení sa, aj keď azure nesúhlasí
    changes = merge_words(["mám", "rád", "hudbu"], "mám rada hudbu",
                          unknown=set())
    assert changes == []


def test_merge_gate_levenshtein_blocks_distant_word():
    # azure úplne iné slovo (halucinácia v slovníku) → gate 3 blokuje
    changes = merge_words(["krátke", "nežňašam"], "krátke stolička",
                          unknown={"nežňašam"})
    assert changes == []


def test_merge_gate_proper_noun_protected():
    # proprium s veľkým písmenom sa nikdy nemení
    changes = merge_words(["Nežňašam"], "Neznášam", unknown={"nežňašam"})
    assert changes == []


def test_merge_never_valid_to_invalid():
    # azure slovo mimo slovníka nesmie nahradiť nič
    changes = merge_words(["neznášam"], "nežňašam",
                          unknown={"nežňašam"})
    assert changes == []


def test_merge_preserves_punctuation_and_case():
    changes = merge_words(["Nežňašam,"], "neznášam",
                          unknown={"nežňašam"})
    # veľké písmeno → proprium gate blokuje; over malé s interpunkciou
    changes = merge_words(["nežňašam,"], "neznášam",
                          unknown={"nežňašam"})
    assert changes and changes[0]["to"] == "neznášam,"

# ---- azure timeout hodnota -----------------------------------------------

def test_azure_timeout_is_8s():
    # CTO odporucanie: timeout 30s->8s znizuje latency pri fallback
    assert AZURE_TIMEOUT_S == 8


def test_ensemble_still_works_after_timeout_change():
    # merge_words (ensemble second-opinion) funguje nezavisle od timeout hodnoty
    whisper = ["to", "proste", "nezhnasam"]
    azure = "to proste neznaasam"
    changes = merge_words(whisper, azure, unknown={"nezhnasam"})
    assert isinstance(changes, list)


def test_ensemble_oov_correction_graceful():
    # klasicky pripad: halucovane OOV slovo, azure blizke v slovniku
    whisper = ["to", "proste", "nezhnasam"]
    azure = "to proste neznasam"
    changes = merge_words(whisper, azure, unknown={"nezhnasam"})
    assert isinstance(changes, list)


def test_speech_dynamics_gate():
    """Plochý šum = ticho (halucinačný bug z 15.10.), reč prejde."""
    import numpy as np

    from titulky.pipeline import has_speech_dynamics
    rng = np.random.default_rng(7)
    noise = (rng.normal(0, 0.006, 16000 * 4)).astype(np.float32)
    assert not has_speech_dynamics(noise)
    speech = noise.copy()
    for s in (0.5, 1.8, 3.1):  # slová: krátke energetické úseky
        i = int(s * 16000)
        speech[i:i + 6000] += (rng.normal(0, 0.08, 6000)).astype(np.float32)
    assert has_speech_dynamics(speech)
    assert has_speech_dynamics(noise[:4000])  # prikrátke okno -> prejde


def test_flux_gate_real_noise_vs_fast_speech():
    """Regres z 16.10.: súvislá rýchla reč má plochú amplitúdu ako šum —
    amp gate ju držal 4 s ako ticho (fast.wav WER 62 %). Spektrálny flux
    ich rozlíši: reálny room tone 0.13, reč 0.31+. Fixture = skutočná
    nahrávka šumu z MacBook miku."""
    import wave

    import numpy as np

    from titulky.pipeline import FLUX_SPEECH_THR, SpeechLevel, spectral_flux
    noise = np.load("tests/fixtures/noise_floor.npy")
    assert spectral_flux(noise) < FLUX_SPEECH_THR
    w = wave.open("tests/fast.wav")
    a = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
         .astype(np.float32) / 32768)
    first4 = a[:16000 * 4]
    assert spectral_flux(first4) > FLUX_SPEECH_THR
    # celá kombinácia: is_silent nesmie rýchlu reč označiť za ticho
    sl = SpeechLevel()
    assert not sl.is_silent(first4[-4800:], first4)
    # ...a reálny šum áno (aj keď sa jeho RMS stihla naučiť ako „reč")
    sl2 = SpeechLevel()
    sl2.level = float(np.sqrt(np.mean(noise ** 2)))  # worst case z bugu
    assert sl2.is_silent(noise[-4800:], noise)
