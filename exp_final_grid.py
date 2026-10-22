"""Grid na FINAL vrstvu: krok slučky × veľkosť okna.

Meria final p50 latenciu, WER a flicker cez test_suite machinériu.
Append-only JSONL (exp_final_results.jsonl) — resume preskočí hotové.
Nikdy nepúšťať súbežne s iným meraním (GPU contention).
"""
import json
import os
import statistics
import threading

import titulky.pipeline as pl
from bench import wer

RESULTS = "exp_final_results.jsonl"
TESTS = ["tests/normal.wav", "tests/fast.wav", "tests/pauses.wav"]

CONFIGS = [
    # name, final_step_s, final_window_s
    ("step2.0 w10 (baseline)", 2.0, 10.0),
    ("step1.0 w10", 1.0, 10.0),
    ("step1.0 w8", 1.0, 8.0),
    ("step1.0 w6", 1.0, 6.0),
    ("step0.5 w6", 0.5, 6.0),
    ("step1.5 w8", 1.5, 8.0),
]


def run_config(name, step, window, refs):
    pl.FINAL_STEP_S = step
    pl.FINAL_WINDOW_S = window
    fl, wers, flicks = [], [], []
    for path in TESTS:
        pipe = pl.TwoTierPipeline()
        pipe.warmup()
        pipe.play_file(path)
        t = threading.Thread(target=pipe.run)
        t.start()
        t.join(timeout=240)
        if pipe.errors:
            return {"config": name, "crash": pipe.errors[0][:200]}
        hyp = " ".join(pipe.final_text)
        wers.append(wer(refs[path], hyp))
        if pipe.final_latencies:
            fl.append(statistics.median(pipe.final_latencies))
        flicks.append(pipe.final_agree.flicker_rate())
    return {"config": name, "step_s": step, "window_s": window,
            "final_p50": round(statistics.median(fl), 2) if fl else None,
            "wer_avg": round(sum(wers) / len(wers), 4),
            "wer_max": round(max(wers), 4),
            "flicker_avg": round(sum(flicks) / len(flicks), 3)}


def main():
    refs = json.load(open("tests/refs.json"))
    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS):
            if line.strip():
                done.add(json.loads(line)["config"])
    for name, step, window in CONFIGS:
        if name in done:
            print(f"== {name}: už zmerané, preskakujem")
            continue
        print(f"== {name}")
        r = run_config(name, step, window, refs)
        print(r)
        with open(RESULTS, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
