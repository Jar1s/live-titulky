"""Test bez mikrofónu: prehrá WAV ako simulovaný live vstup a vypíše metriky."""
import sys, threading, time
from titulky.pipeline import LivePipeline

pipe = LivePipeline(model_size="small", on_update=lambda d: print(
    f"[{d['audio_s']:5.1f}s lat={d['latency_s'] and round(d['latency_s'],2)}] "
    f"{d['committed'][-70:]} | {d['pending'][:40]}"))
print("Načítavam model…")
pipe.play_file(sys.argv[1] if len(sys.argv) > 1 else "test_sk.wav")
t = threading.Thread(target=pipe.run); t.start(); t.join(timeout=120)
import statistics
if pipe.latencies:
    print(f"\nLatencia avg={statistics.mean(pipe.latencies):.2f}s max={max(pipe.latencies):.2f}s")
print(f"Flicker: {pipe.agree.flicker_rate():.1%}")
print("PREPIS:", " ".join(pipe.committed_text))
