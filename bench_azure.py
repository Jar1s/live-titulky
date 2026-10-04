"""Benchmark: Azure AI Speech (cloud) vs náš lokálny pipeline.

Posiela testovacie WAV cez Azure Speech SDK (sk-SK) a meria WER proti
rovnakým referenciám ako bench.py. Kľúč v .azure_key (KEY=..., REGION=...),
súbor je v .gitignore.

Použitie: .venv/bin/python bench_azure.py
"""
import json
import time
import wave


def load_key():
    cfg = {}
    with open(".azure_key") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                cfg[k] = v
    return cfg["KEY"], cfg.get("REGION", "westeurope")


def wer(ref: str, hyp: str) -> float:
    from bench import wer as _w
    return _w(ref, hyp)


def azure_transcribe(path: str, key: str, region: str) -> tuple[str, float]:
    """Vráti (prepis, čas_spracovania_s). Continuous recognition zo súboru."""
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=key, region=region)
    cfg.speech_recognition_language = "sk-SK"
    audio_cfg = speechsdk.audio.AudioConfig(filename=path)
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio_cfg)

    parts: list[str] = []
    done = {"v": False}
    rec.recognized.connect(lambda e: parts.append(e.result.text))
    rec.session_stopped.connect(lambda e: done.update(v=True))
    rec.canceled.connect(lambda e: done.update(v=True))

    t0 = time.time()
    rec.start_continuous_recognition()
    while not done["v"] and time.time() - t0 < 120:
        time.sleep(0.1)
    rec.stop_continuous_recognition()
    return " ".join(parts).strip(), time.time() - t0


def main():
    key, region = load_key()
    refs = json.load(open("bench_refs.json"))
    print(f"Azure Speech ({region}, sk-SK) vs lokálne výsledky z bench.py\n")
    rows = []
    for wav_path, ref in refs.items():
        with wave.open(wav_path) as wf:
            dur = wf.getnframes() / wf.getframerate()
        hyp, dt = azure_transcribe(wav_path, key, region)
        w = wer(ref, hyp)
        rows.append((wav_path, w, dt, dur))
        print(f"=== {wav_path} ({dur:.1f}s)")
        print(f"REF: {ref}")
        print(f"AZURE: {hyp}")
        print(f"WER={w:.1%} čas={dt:.1f}s (RTF={dt/dur:.2f})\n")

    print("== SÚHRN ==")
    for p, w, dt, dur in rows:
        print(f"{p:20s} WER={w:.1%} RTF={dt/dur:.2f}")


if __name__ == "__main__":
    main()
