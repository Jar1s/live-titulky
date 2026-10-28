"""Test suite: prežene testovaciu sadu cez TwoTierPipeline a vyhodnotí
WER, latencie, flicker + špecifické kontroly (halucinácie na tichu).
"""
import json
import os
import statistics
import sys
import threading

from titulky.pipeline import TwoTierPipeline
from bench import wer


def draft_acc(pipe):
    """Recall: koľko bielych slov bolo predtým zobrazených sivé (±2 s)."""
    seen = pipe._seen_draft_words
    tot = hit = 0
    for sec, norm in pipe.final_word_keys:
        tot += 1
        if any((sec + d, norm) in seen for d in (-2, -1, 0, 1, 2)):
            hit += 1
    return hit / tot if tot else None


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
    # reálne nahrávky (prednáška) s PSEUDO-refs (kinit-turbo offline):
    # WER na nich = streamingová degradácia voči offline stropu
    if os.path.exists("tests/refs_real.json"):
        refs.update(json.load(open("tests/refs_real.json")))
    only = os.environ.get("TITULKY_SUITE_ONLY")
    if only:
        refs = {p: r for p, r in refs.items() if only in p}
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
            "draft_acc": draft_acc(pipe) if ref else None,
            "retract": pipe.draft_retractions,
            "hyp": hyp,
        })
        r = rows[-1]
        wtxt = f"{r['wer']:.1%}" if r["wer"] is not None else f"halluc={r['hallucinated_words']}"
        dw = f"{r['draft_acc']:.1%}" if r.get("draft_acc") is not None else "–"
        print(f"\n== {r['test']}: WER={wtxt} draftACC={dw} "
              f"draft_p50={r['draft_p50']} final_p50={r['final_p50']}")
        print("   HYP:", hyp[:160])

    print("\n===== SÚHRN =====")
    print(f"{'test':14} {'WER':>8} {'draftACC':>9} {'draft p50':>10} {'final p50':>10} {'flicker':>8} {'retr':>5}")
    for r in rows:
        wtxt = f"{r['wer']:.1%}" if r["wer"] is not None else f"halluc={r['hallucinated_words']}sl"
        d = f"{r['draft_p50']:.2f}s" if r["draft_p50"] else "–"
        f_ = f"{r['final_p50']:.2f}s" if r["final_p50"] else "–"
        dw = f"{r['draft_acc']:.1%}" if r.get("draft_acc") is not None else "–"
        print(f"{r['test']:14} {wtxt:>8} {dw:>9} {d:>10} {f_:>10} {r['flicker']:>7.1%} {r.get('retract', 0):>5}")
    json.dump(rows, open("tests/results.json", "w"), ensure_ascii=False, indent=1)
    if crashes:
        print(f"\nZLYHANIE: {len(crashes)} test(ov) s pádom vlákna — exit 1")
        sys.exit(1)


if __name__ == "__main__":
    main()
