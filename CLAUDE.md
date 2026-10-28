# live-titulky — AI coding agent context

Semestrálny projekt (FEI STU, AI v spracovaní multimédií, téma 01):
živé automatické slovenské titulky na Apple Silicon, bez cloudu, bez GPU servera.

## Čo to robí

Mikrofón (alebo WAV) → dvojvrstvová pipeline → WebSocket → HTML overlay v prehliadači.

```
mikrofón/WAV → zdieľaný audio buffer (16 kHz mono)
   ├─ DRAFT vlákno:  whisper-tiny-mlx (ANE), krok 0.08 s, okno 3.0 s
   │                 → šedý text, p50 ~0.4–0.9 s od dopovedania slova
   └─ FINAL vlákno:  kinit/whisper-large-v3-turbo-sk, krok 1.0 s, okno 8.0 s
                     → LocalAgreement-2 → biely potvrdený text, p50 ~3.5–4.0 s
                → WebSocket ws://localhost:8765
                → overlay.html (vhodné ako OBS browser source)
```

## Stack

| Vrstva | Čo |
|---|---|
| ASR inference | `mlx-whisper 0.4.3` + `MLX 0.32.2` (Apple GPU/Metal, float16) |
| Draft model | `mlx-community/whisper-tiny-mlx` (ANE) / `whisper-base-mlx` (CPU fallback) |
| Final model | `kinit/whisper-large-v3-turbo-sk` (SK fine-tune, MLX) |
| VAD | vlastný RMS gate + spektrálny flux (bez silero) |
| Agreement | LocalAgreement-2 (vzor: ufal/whisper_streaming) |
| Diarizácia | resemblyzer embeddingy + online centroid clustering (opt-in) |
| Korekcia | lexikon.json (deterministický) + voliteľná Azure Speech STT async |
| Transport | websockets 17.1 (WS + HTTP v jednom) |
| Audio | sounddevice (CoreAudio) |
| Python | 3.12, `.venv/` |
| Testy | pytest, test_suite.py (E2E streaming), test_units.py (unit) |

## Štruktúra

```
titulky/
  main.py        — vstupný bod, WS server, CLI, run-trace
  pipeline.py    — TwoTierPipeline, VAD, metriky, draft/final slučky
  asr.py         — ASR wrapper (mlx-whisper), LocalAgreement, preprocessing
  diarize.py     — diarizácia (resemblyzer), voliteľná
  lexicon.py     — slovníková korekcia (lexikon.json)
  correction.py  — Azure Speech korekcia (asynchrónna, voliteľná)
  trace.py       — run-trace JSONL do .runs/
  overlay.html   — titulkový overlay (TITULKY/DIKTOVANIE režim, OBS-ready)
  flow.html      — debug flow UI
tests/           — WAV fixtures pre regresný suite
docs/
  BENCHMARK.md   — všetky namerané čísla (záväzný zdroj pravdy)
  ROADMAP.md     — plánované featury
bench.py         — WER benchmark modelov
test_suite.py    — E2E streaming regresná sada (7 scenárov)
test_units.py    — unit testy (lexikon, agreement, VAD)
exp_grid.py      — grid search hyper-parametrov
lexikon.json     — editovateľné korekcie anglicizmov/proprii
```

## Kľúčové príkazy

```bash
# Aktivácia venv (vždy najprv)
source .venv/bin/activate

# Spustenie — živý mikrofón
python -m titulky.main
# Overlay: http://localhost:8765

# Spustenie — simulácia zo súboru (bez mikrofónu, demo/debug)
python -m titulky.main --file demo_sk.wav

# Rýchly offline test (bez WS servera, výpis do terminálu)
python test_offline.py

# Regresný suite (E2E, ~2 min, potrebuje tests/*.wav)
python test_suite.py

# Unit testy
pytest test_units.py -v

# WER benchmark modelov
python bench.py

# FLEURS SK benchmark (stiahne dataset z HF)
python bench_fleurs.py
```

## ENV premenné (runtime ladenie)

| Premenná | Default | Popis |
|---|---|---|
| `TITULKY_DEVICE` | `default` | CoreAudio zariadenie — akceptuje substring názvu (nie index! indexy sa posúvajú medzi restartmi) |
| `TITULKY_DRAFT_BACKEND` | `ane` | `ane` / `cpu` / `gpu` — backend pre draft vrstvu |
| `TITULKY_DRAFT_WIN` | `3.0` | Draft okno v sekundách |
| `TITULKY_FINAL_STEP` | `1.0` | Final krok v sekundách |
| `TITULKY_FINAL_WIN` | `8.0` | Final okno v sekundách |
| `TITULKY_DRAFT_REPO` | `mlx-community/whisper-tiny-mlx` | HF repo pre draft model |
| `TITULKY_FINAL_REPO` | `kinit/whisper-large-v3-turbo-sk` | HF repo pre final model |
| `TITULKY_ANE_MODEL` | (base-sk cesta) | Cesta ku CoreML modelu pre ANE draft |
| `TITULKY_XAGREE` | `1` | Cross-model agreement (`0` = vypnúť, ušetrí ~1.5–2 s latency) |
| `TITULKY_FAST_FINAL` | `0` | Experimentálny fast final (`1` = zapnúť; WER 6.5–12.9 %, zamietnutý) |
| `TITULKY_WORD_TS` | `exact` | `approx` šetrí 1.1 s ale rozbíja WER 3→15 %, nepoužívať |
| `TITULKY_SPK` | `1` | Diarizácia rečníkov (`0` = vypnúť) |
| `TITULKY_NO_AZURE` | (unset) | Nastaviť na `1` pre vypnutie Azure korekcie |

## Hardvérové požiadavky

- **Apple Silicon M-chip** — povinné (MLX, CoreML ANE, Metal)
- Testované na: MacBook Pro M3 Pro 36 GB, macOS 27.0.1
- Python 3.12
- RAM: ~2–4 GB pre oba modely v pamäti
- Na Intel Mac / Windows / Linux: nespustí sa (mlx je Apple-only)

## Výsledky meraní (22. 10. 2026)

| scenár | WER | draft p50 | final p50 |
|---|---|---|---|
| normal | 6.5 % | 1.8 s | 3.8 s |
| fast (280 wpm) | 0 % | 0.9 s | 3.6 s |
| slow (120 wpm) | 0 % | 1.2 s | 3.5 s |
| quiet (5 % hlasitosti) | 6.5 % | 1.5 s | 3.8 s |
| noisy (SNR 10 dB) | 12.9 % | 1.8 s | 5.1 s |
| pauses (3 s medzery) | 3.2 % | 1.4 s | 4.0 s |
| silence (15 s) | **0 halucinácií** | — | — |

Požiadavka ≥ 91 % presnosti **splnená** (93.8 %) na kombinovanej sade.
Detaily: `docs/BENCHMARK.md`.

## Gotchas a pasce

### Zariadenie (CoreAudio)
- **Nepoužívaj numerický index** v `TITULKY_DEVICE` — indexy sa posúvajú pri reštartoch
- Zadávaj substring názvu: `TITULKY_DEVICE="MacBook Pro Microphone"`
- Ak zvuk nejde, spusti `python -c "import sounddevice; print(sounddevice.query_devices())"` a skontroluj mená

### Modely
- `mlx_whisper 0.4.3` **nepodporuje beam search** — parameter `beam_size` sa ticho ignoruje (greedy dekód). Starší benchmark zápis „beam 5\" bol omyl.
- `TITULKY_WORD_TS=approx` šetrí 1.1 s dekódu, ale rozbíja agreement → WER 3→15 %. **Nepoužívať.**
- `FAST_FINAL=1`: WER 35–55 % na ANE retime, zamietnutý v benchmarku. Nepoužívať.
- init_prompt v streamingu halucinuje do výstupu na krátkych oknách (−30 p. b. WER) — používa sa **len v offline benchmarku**, nie v pipeline

### Slovenčina
- SK fine-tuned modely (Common Voice) na **spontánnej reči zlyhávajú** — fine-tune na čítanom korpuse negeneralizuje na hovorovú reč. Výnimka: `kinit/whisper-large-v3-turbo-sk` (produkčný fine-tune)
- Čínske modely (SenseVoice, Paraformer, Qwen-Audio, Kimi-Audio, Dolphin) slovenčinu **nepodporujú**
- Denoising pred Whisperom **nepoužívať** — zvyšuje WER o 3–17 p. b. (distribution shift, arXiv:2512.17562)

### LLM post-korekcia
- `qwen2.5:3b` → WER 84 % — nedodrží inštrukciu, pridáva komentáre. Negatívny výsledok.
- Pre LLM korekciu treba minimálne 7B model (mimo rozsah projektu)

### Testy
- `test_suite.py` potrebuje `tests/*.wav` — generuje ich `make_tests.py` (macOS `say` + sox)
- Suite je deterministická (SHA-256 WAV v run-trace), výsledky sa logujú do `.runs/`
- Jeden beh na konfiguráciu (limitácia benchmarku) — trend potvrdený dvoma nezávislými sériami

### Azure korekcia
- Voliteľná, asynchrónna — nesmie blokovať pipeline
- Kľúč v `.azure_key` (nie v gite)
- Vypnutie: `TITULKY_NO_AZURE=1`

## Negatívne výsledky (zdokumentované, neopakuj)

1. **ANE retime (FAST_FINAL)**: WER 35–55 % — zamietnutý
2. **LLM korekcia 3B**: WER 84 % — zamietnutý
3. **RMS AGC** (WebRTC vzor): noisy lepší, slow horšší — zamietnutý
4. **Denoising** pred ASR: WER +3–17 p. b. — zámerne nie je
5. **Parakeet-TDT v3**: streaming WER 12.9–29 % SK — zamietnutý
   - ako final engine: latencia 1.3–2.8 s super, ale jazykový drift (PL/HR/EN, bez language forcingu) na reálnej reči — zamietnutý
   - ako draft backend: GPU kontencia s MLX finalom (draft 3.8–8.2 s) — zamietnutý
6. **Nemotron-3.5 streaming**: WER 9.7–62.5 % SK — zamietnutý
7. **word_timestamps=approx**: −1.1 s dekódu ale WER 3→15 % — zamietnutý
8. **CONF_COMMIT** (p ≥ prah + SK slovník, commit z 1. čítania): WER +2–4 p. b., latencia +0.4–0.6 s — Whisper p nekalibrované, slovník nechráni pred validnými homofónmi; čaká na kalibrovaný judge (Jev)
9. **DRAFT_STABLE** (append-only šedý text à la Realtime delty): retr 0, ale draftACC 90→35 % — Whisper okno jitteruje, strict prefix zobrazí polovicu slov; opt-in flag ostáva
10. **SIL_CLEAR 0.45** (flush eagerness à la semantic_vad): final p50 −1.7 s, ale WER +1.5–5 p. b. (reže výroky v nádychu) — energetický VAD nevie, či veta skončila; default 0.6

## Overlay (UI)

- **http://localhost:8765** — titulkový overlay (HTML/CSS/JS, žiaden framework)
- Dva režimy: `TITULKY` (2 stabilné riadky, roll-up ako TV) / `DIKTOVANIE` (rýchly draft tok)
- Prepínač: tlačidlo M alebo klavesa M
- Pause/Resume: tlačidlo alebo klavesa K; WS správa `{t: "pause"}`
- Vhodné ako OBS browser source (chroma/dark téma v settings)
- `/flow` — debug flow UI (run-trace vizualizácia)
- `/audio/<name>.wav` — prístup k uloženým WAV segmentom
- `/replay?wav=<name>` — A/B porovnanie modelov na konkrétnom segmente

## Run-trace

Každý beh sa loguje do `.runs/` (JSONL, append-only). Obsahuje:
- SHA-256 vstupných WAV
- časy krokov (warmup, vstup, finaly, korekcie, koniec)
- uložené WAV segmenty pre každý final commit (`audio/`)
- metriky (flicker, avg/p50/max latency)

Pozri `titulky/trace.py` pre formát.
