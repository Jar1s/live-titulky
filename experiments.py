"""Trade-off experiment: veľkosť kroku final vrstvy × latencia × WER.

Spustí TwoTierPipeline na demo_sk.wav pre rôzne FINAL_STEP_S
a vypíše tabuľku: krok → draft/final latencia, WER, flicker.
Výstup: results.md (tabuľka pre dokumentáciu).
"""
import json
import statistics
import threading

from titulky import pipeline as pl
from titulky.pipeline import TwoTierPipeline
from bench import wer

REF = ("Vitajte pri demonštrácii automatických titulkov v reálnom čase. Tento systém "
       "rozpoznáva slovenskú reč a zobrazuje titulky s minimálnym oneskorením. "
       "Architektúra má dve vrstvy. Rýchla vrstva používa malý model a zobrazuje "
       "pracovný prepis okamžite. Presná vrstva používa veľký model a text následne "
       "opravuje. Meriame latenciu aj presnosť rozpoznávania. Latencia je čas od "
       "dopovedania slova po jeho zobrazenie. Presnosť meriame ako podiel správne "
       "rozpoznaných slov oproti referenčnému prepisu. Systém beží lokálne na "
       "počítači bez pripojenia na internet. Rozpoznávanie prebieha priebežne, "
       "segment po segmente. Ďakujeme za pozornosť.")


def run_once(final_step, draft_step):
    pl.FINAL_STEP_S = final_step
    pl.DRAFT_STEP_S = draft_step
    pipe = TwoTierPipeline()
    pipe.warmup()
    pipe.play_file("demo_sk.wav")
    t = threading.Thread(target=pipe.run)
    t.start()
    t.join(timeout=300)
    hyp = " ".join(pipe.final_text)
    return {
        "final_step": final_step,
        "draft_step": draft_step,
        "draft_lat_p50": statistics.median(pipe.draft_latencies) if pipe.draft_latencies else None,
        "draft_lat_min": min(pipe.draft_latencies) if pipe.draft_latencies else None,
        "final_lat_p50": statistics.median(pipe.final_latencies) if pipe.final_latencies else None,
        "wer": wer(REF, hyp),
        "flicker": pipe.final_agree.flicker_rate(),
        "hyp": hyp,
    }


def main():
    rows = []
    for fs, ds in [(1.0, 0.25), (1.5, 0.25), (2.0, 0.5), (3.0, 0.5)]:
        print(f"--- final_step={fs} draft_step={ds}")
        r = run_once(fs, ds)
        print({k: v for k, v in r.items() if k != "hyp"})
        rows.append(r)

    with open("results.md", "w") as f:
        f.write("# Trade-off: krok final vrstvy × latencia × presnosť\n\n")
        f.write("Vstup: demo_sk.wav (47.6 s, syntetická SK reč), MacBook Apple Silicon.\n\n")
        f.write("| final krok | draft krok | draft lat p50 | draft lat min "
                "| final lat p50 | WER | presnosť | flicker |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['final_step']} s | {r['draft_step']} s "
                    f"| {r['draft_lat_p50']:.2f} s | {r['draft_lat_min']:.2f} s "
                    f"| {r['final_lat_p50']:.2f} s | {r['wer']:.1%} "
                    f"| {1 - r['wer']:.1%} | {r['flicker']:.1%} |\n")
    json.dump(rows, open("results.json", "w"), indent=1, ensure_ascii=False)
    print("\nZapísané: results.md, results.json")


if __name__ == "__main__":
    main()
