"""Vyrobí reálne test fixtures z .runs nahrávok (prednáška 26.10.).

Zošije po sebe idúce commit segmenty do ~20s súvislých úsekov,
vyrobí pseudo-referencie (kinit-turbo OFFLINE = strop pipeline) a
zapíše tests/real/*.wav + tests/refs_real.json.

POZOR: refs sú PSEUDO (model, nie človek) — WER proti nim meria
streamingovú degradáciu pipeline voči offline stropu, nie absolútnu
pravdu. Na absolútnu pravdu treba ľudské ucho (TODO Jaro).
"""
import glob
import json
import os
import re

import numpy as np
import soundfile as sf

RUN = ".runs/titulky-2026-10-26T09:18:09-6073/audio"
OUT = "tests/real"
TARGET_S = 20.0
N_FIX = 6


def main():
    files = sorted(glob.glob(f"{RUN}/*.wav"))
    offs = [float(re.search(r"(\d+\.\d+)s\.wav$", f).group(1)) for f in files]
    total = offs[-1]
    os.makedirs(OUT, exist_ok=True)

    # kandidátne štarty husto po prednáške; ber len úseky s obsahom
    starts = [total * i / 14 for i in range(1, 13)]
    refs = {}
    import mlx_whisper
    OPT = dict(path_or_hf_repo="/Volumes/GRINDCASTER/AI-MODELS/mlx/"
               "kinit-whisper-turbo-sk", language="sk",
               condition_on_previous_text=False, temperature=0.0,
               fp16=True, verbose=None)

    k = 0
    for st in starts:
        if k >= N_FIX:
            break
        i = min(range(len(offs)), key=lambda j: abs(offs[j] - st))
        chunks, dur = [], 0.0
        j = i
        while j < len(files) and dur < TARGET_S:
            a, sr = sf.read(files[j], dtype="float32")
            # kontinuita: ďalší segment musí nadväzovať (<1.5s diera)
            if j > i and offs[j] - (offs[j - 1] + len(chunks[-1]) / sr) > 1.5:
                break
            chunks.append(a)
            dur += len(a) / sr
            j += 1
        if dur < 12.0:
            continue
        audio = np.concatenate(chunks)
        r = mlx_whisper.transcribe(audio, **OPT)
        txt = r["text"].strip()
        if len(txt.split()) < 15:
            continue   # hudba/šum/prázdno — nepoužiteľný fixture
        k += 1
        path = f"{OUT}/real{k}.wav"
        sf.write(path, audio, sr)
        refs[path] = txt
        print(f"{path}: {dur:.1f}s | {txt[:80]}")

    json.dump(refs, open("tests/refs_real.json", "w"),
              ensure_ascii=False, indent=1)
    print("refs_real.json zapisany (PSEUDO-REFS, kinit-turbo offline)")


if __name__ == "__main__":
    main()
