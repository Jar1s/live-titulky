"""FLEURS SK test (120 viet) — offline corpus WER: turbo vs kinit-sk.

Korpusový WER = (S+D+I)/N cez všetky vety (jiwer na zoznamoch).
Použitie: .venv/bin/python bench_fleurs.py
"""
import json
import time

import jiwer
import mlx_whisper

from bench import _norm_text, load

MODELS = {
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "kinit-sk": "/Volumes/GRINDCASTER/AI-MODELS/mlx/kinit-whisper-turbo-sk",
}


def main():
    refs = json.load(open("fleurs_sk/refs.json"))
    paths = sorted(refs)
    out = {}
    for mname, repo in MODELS.items():
        hyps = []
        t0 = time.monotonic()
        for i, p in enumerate(paths):
            r = mlx_whisper.transcribe(
                load(p), path_or_hf_repo=repo, language="sk",
                temperature=0.0, condition_on_previous_text=False)
            hyps.append(r["text"])
            if (i + 1) % 30 == 0:
                print(f"{mname}: {i+1}/{len(paths)}")
        dt = time.monotonic() - t0
        w = jiwer.wer([_norm_text(refs[p]) for p in paths],
                      [_norm_text(h) for h in hyps])
        out[mname] = {"wer": w, "dt_s": round(dt, 1),
                      "hyps": dict(zip(paths, hyps))}
        print(f"== {mname}: corpus WER {w:.2%}  ({dt:.0f}s)")
    json.dump(out, open("bench_fleurs_results.json", "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
