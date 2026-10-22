"""Test suite: prežene testovaciu sadu cez TwoTierPipeline a vyhodnotí
WER, latencie, flicker + špecifické kontroly (halucinácie na tichu).
"""
import json
import statistics
import sys
import threading

from titulky.pipeline import TwoTierPipeline
from bench import wer


def run(path):
    pipe = TwoTierPipeline()
    pipe.warmup()
    pipe.play_file(path)
    t = threading.Thread(target=pipe.run)
    t.start()
    t.join(timeout=240)
    return pipe


def main():
    refs = json.load(open("tests/refs.json"))
    rows = []
    crashes = []
    for path, ref in refs.items():
        pipe = run(path)
        if pipe.errors:
            crashes.append({"test": path, "errors": pipe.errors})
            print(f"\n== {path}: CRASH vlákna pipeline!")
            for e in pipe.errors:
                print(e)
        hyp = " ".join(pipe.final_text)
        w = wer(ref, hyp) if ref else None
        halluc = len(hyp.split()) if not ref else None
        rows.append({
            "test": path.split("/")[-1],
            "wer": w,
            "hallucinated_words": halluc,
            "draft_p50": statistics.median(pipe.draft_latencies)
                         if pipe.draft_latencies else None,
            "final_p50": statistics.median(pipe.final_latencies)
                         if pipe.final_latencies else None,
            "flicker": pipe.final_agree.flicker_rate(),
            "hyp": hyp,
        })
        r = rows[-1]
        wtxt = f"{r['wer']:.1%}" if r["wer"] is not None else f"halluc={r['hallucinated_words']}"
        print(f"\n== {r['test']}: WER={wtxt} "
              f"draft_p50={r['draft_p50']} final_p50={r['final_p50']}")
        print("   HYP:", hyp[:160])

    print("\n===== SÚHRN =====")
    print(f"{'test':14} {'WER':>8} {'draft p50':>10} {'final p50':>10} {'flicker':>8}")
    for r in rows:
        wtxt = f"{r['wer']:.1%}" if r["wer"] is not None else f"halluc={r['hallucinated_words']}sl"
        d = f"{r['draft_p50']:.2f}s" if r["draft_p50"] else "–"
        f_ = f"{r['final_p50']:.2f}s" if r["final_p50"] else "–"
        print(f"{r['test']:14} {wtxt:>8} {d:>10} {f_:>10} {r['flicker']:>7.1%}")
    json.dump(rows, open("tests/results.json", "w"), ensure_ascii=False, indent=1)
    if crashes:
        print(f"\nZLYHANIE: {len(crashes)} test(ov) s pádom vlákna — exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
