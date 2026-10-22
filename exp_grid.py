"""Experiment grid: draft vrstva — model × okno × krok → latencia/WER/flicker.

Cieľ: nájsť config s vnímanou draft latenciou ≤ 0.4 s bez rozbitia presnosti.
Každý config sa meria na viacerých nahrávkach (syntetické + reálny mik).
Výsledky sa priebežne appendujú do exp_grid_results.jsonl (odolné voči timeoutu),
na konci sa vypíše markdown tabuľka.

Spustenie:  .venv/bin/python exp_grid.py [--quick]
"""
import json
import statistics
import sys
import threading

from bench import wer
from titulky import pipeline as pl
from titulky.pipeline import TwoTierPipeline

TESTS = {
    "tests/normal.wav": None,   # ref z tests/refs.json
    "tests/fast.wav": None,
    "tests/pauses.wav": None,
    "mic_test.wav": None,       # ref z bench_refs.json (reálny mik, spontánna)
}

CONFIGS = [
    # name, draft_repo, window_s, step_s, hold_s
    ("small w3.0 s0.25 (baseline)", "mlx-community/whisper-small-mlx", 3.0, 0.25, 0.8),
    ("small w2.0 s0.25", "mlx-community/whisper-small-mlx", 2.0, 0.25, 0.8),
    ("small w1.5 s0.25", "mlx-community/whisper-small-mlx", 1.5, 0.25, 0.8),
    ("small w1.5 s0.15", "mlx-community/whisper-small-mlx", 1.5, 0.15, 0.8),
    ("small w2.0 s0.15", "mlx-community/whisper-small-mlx", 2.0, 0.15, 0.8),
    ("base w2.0 s0.25", "mlx-community/whisper-base-mlx", 2.0, 0.25, 0.8),
    ("base w1.5 s0.15", "mlx-community/whisper-base-mlx", 1.5, 0.15, 0.8),
    # v2: zobrazovacia latencia — hold dole, krok dole
    ("base w1.5 s0.10 h0.2", "mlx-community/whisper-base-mlx", 1.5, 0.10, 0.2),
    ("base w2.0 s0.10 h0.2", "mlx-community/whisper-base-mlx", 2.0, 0.10, 0.2),
    ("small w1.5 s0.10 h0.2", "mlx-community/whisper-small-mlx", 1.5, 0.10, 0.2),
    ("base w1.5 s0.15 h0.3", "mlx-community/whisper-base-mlx", 1.5, 0.15, 0.3),
    # v3: útok na p50 <= 0.42 s — hold je najväčšia zložka, dole s ním
    ("base w1.5 s0.10 h0.10", "mlx-community/whisper-base-mlx", 1.5, 0.10, 0.10),
    ("base w1.5 s0.10 h0.15", "mlx-community/whisper-base-mlx", 1.5, 0.10, 0.15),
    ("base w1.2 s0.10 h0.10", "mlx-community/whisper-base-mlx", 1.2, 0.10, 0.10),
    ("base w1.0 s0.10 h0.10", "mlx-community/whisper-base-mlx", 1.0, 0.10, 0.10),
    ("tiny w1.2 s0.08 h0.10", "mlx-community/whisper-tiny-mlx", 1.2, 0.08, 0.10),
]


def load_refs():
    refs = json.load(open("tests/refs.json"))
    bench = json.load(open("bench_refs.json"))
    out = {}
    for path in TESTS:
        out[path] = refs.get(path) or bench.get(path)
        if not out[path]:
            raise SystemExit(f"chýba referencia pre {path}")
    return out


def run_config(name, repo, window, step, hold, refs):
    pl.DRAFT_WINDOW_S = window
    pl.DRAFT_STEP_S = step
    pl.DRAFT_HOLD_S = hold
    agg = {"draft_lats": [], "final_lats": [], "wers": [], "flickers": []}
    hyps = {}
    for path, ref in refs.items():
        pipe = TwoTierPipeline(draft_repo=repo)
        pipe.warmup()
        pipe.play_file(path)
        t = threading.Thread(target=pipe.run)
        t.start()
        t.join(timeout=180)
        pipe.stop()
        hyp = " ".join(pipe.final_text)
        hyps[path] = hyp
        agg["draft_lats"] += pipe.draft_latencies
        agg["final_lats"] += pipe.final_latencies
        agg["wers"].append(wer(ref, hyp))
        agg["flickers"].append(pipe.final_agree.flicker_rate())
    dl = agg["draft_lats"]
    row = {
        "config": name, "repo": repo.split("/")[-1],
        "window_s": window, "step_s": step,
        "draft_p50": round(statistics.median(dl), 3) if dl else None,
        "draft_min": round(min(dl), 3) if dl else None,
        "draft_p90": round(statistics.quantiles(dl, n=10)[8], 3)
                     if len(dl) >= 10 else None,
        "final_p50": round(statistics.median(agg["final_lats"]), 2)
                     if agg["final_lats"] else None,
        "wer_avg": round(statistics.mean(agg["wers"]), 4),
        "wer_max": round(max(agg["wers"]), 4),
        "flicker_avg": round(statistics.mean(agg["flickers"]), 4),
        "hyps": hyps,
    }
    return row


def main():
    refs = load_refs()
    configs = CONFIGS[:3] if "--quick" in sys.argv else CONFIGS
    done = set()
    try:
        for line in open("exp_grid_results.jsonl"):
            done.add(json.loads(line)["config"])
    except FileNotFoundError:
        pass
    for name, repo, window, step, hold in configs:
        if name in done:
            print(f"== {name}: už zmerané, preskakujem")
            continue
        print(f"== {name}")
        row = run_config(name, repo, window, step, hold, refs)
        row["hold_s"] = hold
        with open("exp_grid_results.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print({k: v for k, v in row.items() if k != "hyps"})

    rows = [json.loads(line) for line in open("exp_grid_results.jsonl")]
    print("\n| config | draft p50 | draft min | draft p90 | final p50 "
          "| WER avg | WER max | flicker |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['draft_p50']} | {r['draft_min']} "
              f"| {r['draft_p90']} | {r['final_p50']} | {r['wer_avg']:.1%} "
              f"| {r['wer_max']:.1%} | {r['flicker_avg']:.1%} |")


if __name__ == "__main__":
    main()
