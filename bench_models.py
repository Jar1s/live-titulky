"""Offline A/B: final modely na našich referenčných nahrávkach.

Izoluje kvalitu MODELU od streamingu — celý súbor naraz, beam 5.
Použitie: .venv/bin/python bench_models.py
"""
import json
import time
import wave

import mlx_whisper
import numpy as np

from bench import wer

MODELS = {
    "turbo (baseline)": "mlx-community/whisper-large-v3-turbo",
    "kinit-sk": "/Volumes/GRINDCASTER/AI-MODELS/mlx/kinit-whisper-turbo-sk",
}

FILES = ["tests/normal.wav", "tests/fast.wav", "tests/quiet.wav",
         "tests/noisy.wav", "tests/pauses.wav", "mic_test.wav"]


def load(p):
    w = wave.open(p)
    return (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            .astype(np.float32) / 32768)


def main():
    refs = {**json.load(open("tests/refs.json")),
            **json.load(open("bench_refs.json"))}
    rows = {}
    for mname, repo in MODELS.items():
        rows[mname] = {}
        for f in FILES:
            a = load(f)
            t0 = time.monotonic()
            r = mlx_whisper.transcribe(
                a, path_or_hf_repo=repo, language="sk",
                temperature=0.0, condition_on_previous_text=False)
            dt = time.monotonic() - t0
            w_ = wer(refs[f], r["text"])
            rows[mname][f] = {"wer": w_, "dt": dt, "hyp": r["text"].strip()}
            print(f"{mname:18} {f:18} WER {w_:6.1%}  {dt:5.1f}s")
    with open("bench_models_results.json", "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)

    print("\n| nahrávka | " + " | ".join(MODELS) + " |")
    print("|---|" + "---|" * len(MODELS))
    for f in FILES:
        cells = " | ".join(f"{rows[m][f]['wer']:.1%}" for m in MODELS)
        print(f"| {f} | {cells} |")
    for m in MODELS:
        avg = sum(rows[m][f]["wer"] for f in FILES) / len(FILES)
        rtfx = sum(load_dur(f) for f in FILES) / sum(
            rows[m][f]["dt"] for f in FILES)
        print(f"{m}: avg WER {avg:.1%}, RTFx {rtfx:.0f}")


def load_dur(p):
    w = wave.open(p)
    return w.getnframes() / w.getframerate()


if __name__ == "__main__":
    main()
