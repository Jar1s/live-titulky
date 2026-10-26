"""Parakeet v3 STREAMING benchmark — simulovaný živý vstup.

Kŕmi test wav po CHUNK_S sekundách ako mikrofón a meria presne to,
čo suite: latenciu prvého zobrazenia slova (draft), latenciu
finalizácie, WER finálneho textu a flicker draftu.
Spúšťať SERIÁLNE s inými GPU úlohami!
"""
import json
import time

import mlx.core as mx
import soundfile as sf
from parakeet_mlx import from_pretrained

from bench import wer

MODEL = "mlx-community/parakeet-tdt-0.6b-v3"
CHUNK_S = 0.25
SR = 16000


def norm(t):
    return "".join(c for c in t.lower() if c.isalnum())


def run_file(model, path, ref, ctx=(256, 256), depth=1):
    a, _ = sf.read(path, dtype="float32")
    draft_lat, final_lat = [], []
    seen = {}        # norm slovo+idx -> čas prvého draft zobrazenia
    fin_seen = set()
    flicker_changes = 0
    last_draft = []
    wall0 = time.monotonic()
    compute = 0.0

    with model.transcribe_stream(context_size=ctx, depth=depth) as st:
        n = 0
        while n < len(a):
            chunk = a[n:n + int(CHUNK_S * SR)]
            n += len(chunk)
            t_feed = n / SR          # koľko audia je "odohraté"
            t0 = time.monotonic()
            st.add_audio(mx.array(chunk))
            compute += time.monotonic() - t0
            res = st.result
            fin_words, dr_words = [], []
            toks = st.finalized_tokens + st.draft_tokens
            nfin = len(st.finalized_tokens)
            # tokeny -> slová (nová medzera = nové slovo)
            words, cur, cur_end, is_fin = [], "", 0.0, True
            for i, tk in enumerate(toks):
                if tk.text.startswith(" ") and cur:
                    words.append((cur, cur_end, is_fin))
                    cur, is_fin = "", True
                cur += tk.text.strip() if not cur else tk.text
                cur_end = tk.end
                if i >= nfin:
                    is_fin = False
            if cur:
                words.append((cur, cur_end, is_fin))
            # metriky
            dr_txt = [w for w, _, f in words if not f]
            for wi, (w, wend, fin) in enumerate(words):
                key = (wi, norm(w))
                if key not in seen:
                    seen[key] = (t_feed, wend)
                if fin and key not in fin_seen:
                    fin_seen.add(key)
                    lat = t_feed - wend
                    if 0 < lat < 30:
                        final_lat.append(lat)
            if dr_txt != last_draft:
                flicker_changes += 1
                last_draft = dr_txt
        final_text = st.result.text

    for (wi, w), (t_feed, wend) in seen.items():
        lat = t_feed - wend
        if 0 < lat < 30:
            draft_lat.append(lat)
    draft_lat.sort()
    final_lat.sort()
    rtf = compute / (len(a) / SR)
    return {
        "wer": wer(ref, final_text),
        "draft_p50": draft_lat[len(draft_lat) // 2] if draft_lat else -1,
        "final_p50": final_lat[len(final_lat) // 2] if final_lat else -1,
        "rtf": rtf,
        "text": final_text,
    }


def main():
    refs = json.load(open("tests/refs.json"))
    model = from_pretrained(MODEL)
    model.transcribe("tests/normal.wav")   # warmup
    print(f"{'test':<18}{'WER':>7}{'draft p50':>11}{'final p50':>11}{'RTF':>6}")
    for path, ref in refs.items():
        if "silence" in path:
            continue
        r = run_file(model, path, ref)
        print(f"{path.split('/')[-1]:<18}{r['wer']*100:>6.1f}%"
              f"{r['draft_p50']:>10.2f}s{r['final_p50']:>10.2f}s"
              f"{r['rtf']:>6.2f}")
    print("\nposledny text:", r["text"][:160])


if __name__ == "__main__":
    main()
