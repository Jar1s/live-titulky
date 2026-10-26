"""Parakeet TDT 0.6B v3 (MLX) — offline kvalita + rýchlosť na test wavoch.

Meria strop kvality (celý súbor naraz) a dekód 8s okna — porovnanie
s kinit whisper finalmi. Spúšťať SERIÁLNE s inými GPU úlohami!
"""
import json
import time

import numpy as np
import soundfile as sf
from parakeet_mlx import from_pretrained

from bench import wer

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def main():
    refs = json.load(open("tests/refs.json"))
    model = from_pretrained(MODEL)
    # warmup
    model.transcribe("tests/normal.wav")

    print(f"{'test':<18}{'WER':>7}{'audio':>7}{'dekod':>7}")
    for path, ref in refs.items():
        if "silence" in path:
            continue
        a, sr = sf.read(path)
        t0 = time.monotonic()
        r = model.transcribe(path)
        dt = time.monotonic() - t0
        w = wer(ref, r.text)
        print(f"{path.split('/')[-1]:<18}{w*100:>6.1f}%"
              f"{len(a)/sr:>6.1f}s{dt:>6.2f}s")

    # latencia: 8s okno (ako final) a 3s okno (ako draft)
    a, _ = sf.read("tests/normal.wav", dtype="float32")
    for wl in (8, 3):
        seg = a[:16000 * wl]
        sf.write("/tmp/pk_win.wav", seg, 16000)
        ts = []
        for _ in range(4):
            t0 = time.monotonic()
            r = model.transcribe("/tmp/pk_win.wav")
            ts.append(time.monotonic() - t0)
        ts.sort()
        print(f"okno {wl}s: p50 {ts[1]:.2f}s | {r.text[:70]}")
    # token timestampy k dispozícii?
    s = r.sentences[0] if r.sentences else None
    if s and s.tokens:
        print("tokeny:", [(t.text, round(t.start, 2), round(t.end, 2))
                          for t in s.tokens[:6]])


if __name__ == "__main__":
    main()
