"""Test bez mikrofónu: prehrá WAV ako simulovaný live vstup a vypíše metriky."""
import sys, threading, statistics
from titulky.pipeline import TwoTierPipeline

def show(d):
    print(f"[{d['audio_s']:5.1f}s draft={d['draft_latency_s'] and round(d['draft_latency_s'],2)} "
          f"final={d['final_latency_s'] and round(d['final_latency_s'],2)}] "
          f"{d['committed'][-60:]} | {d['pending'][:40]}")

pipe = TwoTierPipeline(on_update=show)
print("Warmup modelov…")
pipe.warmup()
pipe.play_file(sys.argv[1] if len(sys.argv) > 1 else "test_sk.wav")
t = threading.Thread(target=pipe.run); t.start(); t.join(timeout=180)

def stats(name, xs):
    if xs:
        print(f"{name}: avg={statistics.mean(xs):.2f}s p50={statistics.median(xs):.2f}s "
              f"min={min(xs):.2f}s max={max(xs):.2f}s n={len(xs)}")

print()
stats("DRAFT latencia", pipe.draft_latencies)
stats("FINAL latencia", pipe.final_latencies)
print(f"Flicker: {pipe.final_agree.flicker_rate():.1%}")
print("PREPIS:", " ".join(pipe.final_text))
