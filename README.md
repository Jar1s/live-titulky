# Live titulky (SK) — automatické titulky v reálnom čase

Semestrálny projekt (AI v spracovaní multimédií, téma 01): živá reč →
titulky s minimálnym oneskorením, slovenčina, meranie latencie a presnosti.

Beží **lokálne na Apple Silicon** (MLX, žiadny cloud).

## Architektúra: dvojvrstvové titulkovanie (draft + final)

Vzor z produkčných systémov (Google/Zoom live captions — partial/final):

```
mikrofón/WAV → zdieľaný audio buffer
   ├─ DRAFT vlákno:  whisper-small-mlx, krok 0.25 s, okno 3 s
   │                 → okamžitý pracovný prepis (šedý text)
   └─ FINAL vlákno:  whisper-large-v3-turbo, krok 1.5 s, okno od potvrdenej hranice
                     → LocalAgreement-2 → potvrdený presný text (biely)
                → WebSocket → HTML overlay (blokové zobrazenie, fade-out po pauze)
```

Kľúčové mechanizmy:
- **LocalAgreement-2** (ufal/whisper_streaming): slovo sa potvrdí, až keď sa
  na ňom zhodnú dva po sebe idúce behy ASR → potvrdený text „neskáče".
- **VAD gate** (RMS): ticho sa nedekóduje → žiadne halucinácie z ticha,
  po pauze sa blok titulkov uzavrie a obrazovka vyčistí.
- **dedupe filter**: kolaps halucinačných slučiek opakovaného slova.
- **SK init_prompt + preprocessing** (highpass, normalizácia): −4 p. b. WER
  na spontánnej reči.
- **Filtre halucinácií podľa OSS praxe** (ufal/whisper_streaming,
  whisper-guard, WhisperLiveKit): segment gate (no_speech_prob > 0.8,
  avg_logprob < −1.5), word gate (probability < 0.5 ∧ < 0.15 s) a
  relatívny energia gate (slovo v úseku < 15 % RMS okna = halucinácia
  z ticha). Nameraný efekt: pauses 19,4 → 6,5 % WER, noisy 22,6 → 12,9 %.
- final okno začína vždy na potvrdenej hranici → agreement nestráca kontext.

### Regresná sada (make_tests.py + test_suite.py)

| scenár | WER | draft p50 | final p50 |
|---|---|---|---|
| normal | 6,5 % | 1,8 s | 3,8 s |
| fast (280 wpm) | 0 % | 0,9 s | 3,6 s |
| slow (120 wpm) | 0 % | 1,2 s | 3,5 s |
| quiet (5 % hlasitosti) | 6,5 % | 1,5 s | 3,8 s |
| noisy (SNR 10 dB) | 12,9 % | 1,8 s | 5,1 s |
| pauses (3 s medzery) | 3,2 % | 1,4 s | 4,0 s |
| silence (15 s ticha) | 0 halucinovaných slov | — | — |

### Metriky v štandardnej terminológii (pre porovnateľnosť)

- draft latencia = *Word First Correct* (Baumann et al., NAACL 2009)
- final latencia = *Word First Final* (tamtiež)
- flicker ≈ *Unstable Partial Word Ratio* (Shangguan et al., Interspeech 2020)
- RTF (real-time factor) meraný live v overlayi; < 1,0 = beží v reálnom čase

### Negatívne výsledky (vlastné merania)

1. **initial_prompt v streamingu**: na krátkych oknách prompt halucinuje do
   výstupu (−30 p. b.); ufal ho používa len s orezaným bufferom (text mimo
   okna). Nechané len v offline benchmarku.
2. **LLM korekcia (qwen2.5:3b)**: WER 84 % — 3B model nezvláda inštrukcie.
3. **RMS AGC namiesto peak normalizácie** (WebRTC vzor): noisy 12,9 → 9,7 %,
   ale slow 0 → 12,5 % (gain riadený pomerom reč/ticho deformuje pomalú
   reč). Variant „RMS z aktívnych vzoriek" zhoršil oboje. Peak norm ostáva.
4. **Denoising pred Whisperom zámerne nie je**: literatúra (arXiv:2512.17562,
   arXiv:2603.04710) — enhancement zvyšuje WER Whispera o 3–17 p. b.
   (distribution shift; Whisper je trénovaný na šumených dátach).
5. **SloPal SK fine-tuny (NaiveNeuron, 2 806 h parlamentu, LREC 2026)**:
   na čítanej reči small-sk výborný (3,7 %), ale na spontánnej reči
   turbo-sk 39,1 % vs vanilla turbo 13,0 % — doménový fine-tune mimo
   domény prehráva s multitask pretrainingom. Čínske modely (SenseVoice,
   Paraformer, Qwen-Audio, Kimi-Audio, Dolphin…) slovenčinu nepodporujú
   vôbec (overené v model cards a zdrojákoch).

## Inštalácia a spustenie

```bash
python3 -m venv .venv
.venv/bin/pip install mlx-whisper sounddevice websockets numpy jiwer

.venv/bin/python -m titulky.main                    # živý mikrofón
.venv/bin/python -m titulky.main --file demo_sk.wav # simulovaný stream
```

Overlay: **http://localhost:8765** (live metriky vpravo hore; vhodné ako
OBS browser source). Test bez servera: `.venv/bin/python test_offline.py`.

## Merania

- **draft latencia** = čas prvého zobrazenia slova − čas dopovedania slova
- **final latencia** = čas potvrdenia slova presnou vrstvou
- **presnosť** = 1 − WER (jiwer, normalizácia: lowercase, bez interpunkcie)
- **flicker rate** = podiel slov zmenených medzi behmi (stabilita)

### Benchmark modelov (bench.py, 2 sady: čítaná syntetická + spontánna reč z mikrofónu)

| variant | WER Ø | presnosť | čas/10 s |
|---|---|---|---|
| **large-v3-turbo + prompt + preprocess (nasadené)** | **6,2 %** | **93,8 %** | 1,2 s |
| large-v3-turbo baseline | 6,5 % | 93,5 % | 4,1 s |
| SK fine-tune medium (mikr/whisper-medium-sk-cv11) | 26,1 % | 73,9 % | 25,1 s |
| whisper-small-mlx (draft vrstva) | 31,3 % | 68,7 % | 3,2 s |
| SK fine-tune small (mikr/whisper-small-sk-cv11) | 49,4 % | 50,6 % | 4,4 s |
| turbo + LLM korekcia (qwen2.5:3b) | 84,1 % | — | 18,6 s |

Zistenia:
1. Požiadavka ≥91 % presnosti je splnená (93,8 %) na kombinovanej sade;
   na čítanej reči ~96–100 %, na spontánnej hovorovej ~91 %.
2. **SK fine-tuned modely (Common Voice) na spontánnej reči zlyhali** —
   fine-tune na čítanom korpuse negeneralizuje na hovorovú reč.
3. **LLM post-korekcia s 3B modelom je negatívny výsledok** — model
   nedodrží inštrukciu, pridáva komentáre. Vyžadoval by silnejší model.

### Trade-off latencia × presnosť (experiments.py)

Streaming beh na 47,6 s demo nahrávke (výsledky aj v `results.md`):

| final krok | draft lat p50/min | final lat p50 | presnosť | flicker |
|---|---|---|---|---|
| 1,0 s | 3,11 / 0,33 s | 4,07 s | 96,4 % | 19,7 % |
| 1,5 s | 2,77 / 0,29 s | 3,90 s | 95,2 % | 20,7 % |
| **2,0 s (nasadené)** | 2,08 / 0,35 s | 4,24 s | **97,6 %** | **6,7 %** |
| 3,0 s | 2,00 / 0,22 s | 5,89 s | 97,6 % | 5,9 % |

Draft vrstva zobrazuje najnovšie slová typicky v stovkách ms (min ~0,2 s);
final potvrdenie prichádza ~4 s po dopovedaní. To zodpovedá produkčnému
vzoru partial (rýchle) / final (presné) titulkov.

**Metodologické zistenie:** init_prompt, ktorý v offline benchmarku
zlepšoval WER, v streamingu **škodí** — pri krátkych oknách s málo rečou
Whisper text promptu halucinuje do výstupu (nameraný prepad presnosti
o ~30 p. b.). V streamingu preto prompt nepoužívame; offline vs. streaming
správanie parametra je samostatný výsledok merania.

## Demo (min. 2 minúty)

1. `.venv/bin/python -m titulky.main` + otvor http://localhost:8765
2. scenár: čítaný text (vysoká presnosť) → spontánna reč → pauzy
   (ukážka čistenia blokov) → záver so štatistikami v termináli (Ctrl+C)
3. záložné demo bez mikrofónu: `--file demo_sk.wav`

## Súbory

| | |
|---|---|
| `titulky/asr.py` | MLX Whisper wrapper, LocalAgreement, preprocessing |
| `titulky/pipeline.py` | dvojvrstvová pipeline, VAD gate, metriky |
| `titulky/main.py` | WebSocket server + CLI |
| `titulky/overlay.html` | titulkový overlay |
| `bench.py` + `bench_refs.json` | benchmark modelov (WER) |
| `experiments.py` | trade-off experiment |
| `test_offline.py` | rýchly test bez mikrofónu |

## Limity a ďalšia práca

- Spontánna tichá/rýchla reč cez laptop mikrofón zostáva hranicou SOTA
  pre slovenčinu (žiadny lokálny model nedal < 8 % WER).
- Fine-tuning (LoRA) na spontánnom SK korpuse — mimo rozsah 30 h.
- LLM korekcia so silnejším modelom (7B+) alebo cloud ASR porovnanie.
