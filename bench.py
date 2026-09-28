"""Benchmark ASR variantov na SK: WER + čas.

Použitie:
  python bench.py                # obe testovacie sady (syntetická + mic)
Referencie: bench_refs.json  {wav: "referenčný prepis"}
Varianty: turbo (baseline), turbo+prompt+preprocess, SK fine-tuned modely,
          turbo + LLM korekcia (ollama qwen2.5:3b).
"""
import json
import time
import wave

import numpy as np
import jiwer

import re

SAMPLE_RATE = 16000


def _norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^\w\sáäčďéíĺľňóôŕšťúýž]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(ref, hyp):
    return jiwer.wer(_norm_text(ref), _norm_text(hyp))


def load(path):
    wf = wave.open(path, "rb")
    return np.frombuffer(wf.readframes(wf.getnframes()),
                         dtype=np.int16).astype(np.float32) / 32768


# ---- varianty -----------------------------------------------------------
def v_mlx(repo, prompt=False, pre=False):
    import mlx_whisper
    from titulky.asr import preprocess, SK_PROMPT
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo)

    def run(a):
        kw = {"initial_prompt": SK_PROMPT} if prompt else {}
        a2 = preprocess(a) if pre else a
        return mlx_whisper.transcribe(
            a2, path_or_hf_repo=path, language="sk",
            temperature=0.0, verbose=None, **kw)["text"].strip()
    return run


def v_hf(repo):
    """SK fine-tuned Whisper cez transformers (CPU/MPS)."""
    import torch
    from transformers import pipeline as hf_pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = hf_pipeline("automatic-speech-recognition", model=repo, device=device)
    # starý generation config: jazyk vnútime priamo do configu modelu
    forced = pipe.tokenizer.get_decoder_prompt_ids(language="slovak", task="transcribe")
    pipe.model.generation_config.forced_decoder_ids = forced
    pipe.model.config.forced_decoder_ids = forced

    def run(a):
        return pipe(a)["text"].strip()
    return run


def llm_fix(text, model="qwen2.5:3b"):
    import urllib.request
    prompt = ("Oprav preklepy a komoleniny v tomto slovenskom prepise reči. "
              "Nemeň význam, nepridávaj nič, vráť IBA opravený text.\n\n" + text)
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        json.dumps({"model": model, "prompt": prompt, "stream": False,
                    "options": {"temperature": 0}}).encode(),
        {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["response"].strip()


def main():
    refs = json.load(open("bench_refs.json"))
    variants = {
        "turbo (baseline)": v_mlx("mlx-community/whisper-large-v3-turbo"),
        "turbo+prompt+pre": v_mlx("mlx-community/whisper-large-v3-turbo",
                                  prompt=True, pre=True),
        "small-mlx (draft)": v_mlx("mlx-community/whisper-small-mlx"),
        "sk-ft medium cv11": v_hf("mikr/whisper-medium-sk-cv11"),
        "sk-ft small cv11": v_hf("mikr/whisper-small-sk-cv11"),
    }
    results = {}
    for wav, ref in refs.items():
        a = load(wav)
        print(f"\n=== {wav} ({len(a)/SAMPLE_RATE:.1f}s) ===\nREF: {ref}")
        for name, fn in variants.items():
            try:
                t = time.time()
                hyp = fn(a)
                dt = time.time() - t
                w = wer(ref, hyp)
                results.setdefault(name, []).append((w, dt))
                print(f"\n[{name}] WER={w:.1%} t={dt:.1f}s\n  {hyp}")
                # LLM korekcia len na baseline+prompt variante
                if name == "turbo+prompt+pre":
                    try:
                        t = time.time()
                        fixed = llm_fix(hyp)
                        dt2 = time.time() - t
                        w2 = wer(ref, fixed)
                        results.setdefault("turbo+prompt+LLM", []).append((w2, dt + dt2))
                        print(f"\n[turbo+prompt+LLM] WER={w2:.1%} t=+{dt2:.1f}s\n  {fixed}")
                    except Exception as e:
                        print("  LLM korekcia zlyhala:", e)
            except Exception as e:
                print(f"\n[{name}] ZLYHAL: {e}")

    print("\n\n===== SÚHRN (priemer cez sady) =====")
    print(f"{'variant':26} {'WER':>8} {'čas':>7}")
    for name, xs in sorted(results.items(), key=lambda kv: np.mean([x[0] for x in kv[1]])):
        print(f"{name:26} {np.mean([x[0] for x in xs]):7.1%} "
              f"{np.mean([x[1] for x in xs]):6.1f}s")
    json.dump({k: [(float(a), float(b)) for a, b in v] for k, v in results.items()},
              open("bench_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
