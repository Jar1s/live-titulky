# Live titulky (SK) — automatické titulky v reálnom čase

Semestrálny projekt: živá reč → titulky s minimálnym oneskorením.
Stack: faster-whisper (int8, CPU/Apple Silicon) + vlastná LocalAgreement
stabilizácia + WebSocket overlay.

## Inštalácia

```bash
python3 -m venv .venv
.venv/bin/pip install faster-whisper sounddevice websockets numpy
```

## Spustenie

```bash
# živý mikrofón (predvolené: model small, slovenčina)
.venv/bin/python -m titulky.main

# simulovaný live vstup zo súboru (WAV 16 kHz mono)
.venv/bin/python -m titulky.main --file test_sk.wav

# iný model / krok ASR
.venv/bin/python -m titulky.main --model medium --step 2.0
```

Overlay s titulkami: **http://localhost:8765** (vpravo hore live metriky:
latencia, flicker rate). Dá sa vložiť do OBS ako browser source.

Rýchly test bez servera:

```bash
.venv/bin/python test_offline.py [subor.wav]
```

Testovací WAV sa dá vyrobiť macOS syntézou:

```bash
say -v Laura -o /tmp/t.aiff "ľubovoľný slovenský text"
afconvert -f WAVE -d LEI16@16000 -c 1 /tmp/t.aiff test_sk.wav
```

## Ako to funguje

```
mikrofón/WAV → buffer → kĺzavé okno 8 s (ASR každé 2 s, faster-whisper int8)
             → LocalAgreement (potvrdí len slová zhodné v 2 po sebe idúcich behoch)
             → potvrdený text (nemení sa) + pending (šedý) → WebSocket → overlay
```

Merané metriky:
- **latencia** = wall-clock čas zobrazenia − čas konca slova v audiu (end-to-end)
- **flicker rate** = podiel slov, ktoré sa medzi behmi zmenili (stabilita titulkov)

Východiskové čísla (MacBook, CPU int8, model small, krok 2 s):
latencia avg ≈ 4 s, flicker ≈ 2 %.

## Kam ďalej (plán experimentov)

- porovnanie modelov small/medium/large-v3 + Vosk SK: WER × latencia (SK sada — Common Voice)
- trade-off krivka: veľkosť kroku/okna vs. latencia vs. presnosť
- WER skript proti referenčnému prepisu
