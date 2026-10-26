"""Cloud bench: prežene reálne fixtures cez CloudPipeline, WER + latencie.

Použitie: TITULKY_CLOUD_MODEL=gpt-4o-mini-transcribe .venv/bin/python bench_cloud.py
"""
import json
import os
import statistics
import time

from bench import wer
from titulky.cloud_engine import CloudPipeline, CLOUD_MODEL

refs = json.load(open("tests/refs_real.json"))

print(f"model: {CLOUD_MODEL}\n")
rows = []
for path, ref in sorted(refs.items()):
    pipe = CloudPipeline(language="sk")
    pipe.start_file(path)
    t0 = time.monotonic()
    while pipe.running and time.monotonic() - t0 < 60:
        time.sleep(0.5)
    pipe.stop()
    hyp = " ".join(pipe.final_text)
    w = wer(ref, hyp)
    dl = sorted(pipe.draft_latencies)
    fl = sorted(pipe.final_latencies)
    d50 = dl[len(dl) // 2] if dl else None
    f50 = fl[len(fl) // 2] if fl else None
    rows.append((os.path.basename(path), w, d50, f50))
    print(f"== {os.path.basename(path)}: WER={w:.1%} "
          f"draft_p50={d50 and round(d50, 2)} final_p50={f50 and round(f50, 2)}")
    print(f"   HYP: {hyp[:160]}")

print("\n===== SÚHRN (cloud) =====")
for name, w, d50, f50 in rows:
    print(f"{name:14s} WER={w:6.1%} draft={d50 and round(d50,2)} "
          f"final={f50 and round(f50,2)}")
ws = [r[1] for r in rows]
print(f"WER priemer: {statistics.mean(ws):.1%}")
