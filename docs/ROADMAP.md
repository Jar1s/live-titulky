# Live Titulky — Produktová roadmapa modalít a funkcií

*Verzia: 1.0 | Dátum: október 2026 | Jazyk: SK | Autor: agent*

---

## Kontext a konkurencia

### Čo robia špičky

| Nástroj | Kľúčová vlastnosť | Slabina |
|---|---|---|
| **YouTube Live Captions** | Automatické CC v 15+ jazykoch, server-side, nulová latencia pre diváka | len vlastný stream, žiadne lokálne API |
| **Zoom AI Companion** | Preklad titulkov do 36 jazykov, AI post-meeting súhrn, speaker ID | cloud-only, GDPR riziká, latencia 1–2 s |
| **Wordly** | Preklad 60+ jazykov, export transkriptu, AI súhrn, divák si vyberá jazyk cez QR | drahý ($$$), event-focused, nie pre solo obsah |
| **Interprefy** | Simultánny preklad 45→74 jazykov, ľudskí tlmočníci + AI fallback | enterprise, nie real-time lokálne |
| **OBS LocalVocal** (royshil) | Whisper lokálne v OBS, preklad, filter na reč, chroma overlay | len Windows/Linux binárky, latencia 3–8 s, no Slovak fine-tune |
| **WhisperLive / faster-whisper** | Streamingový ASR server, WebSocket, SRT export | CPU-heavy, no MLX, no lokálny preklad |
| **Otter.ai / Fireflies** | Real-time prepis + súhrn + action items po skončení | cloud, angličtina primárne, platené |

### Náš stack — výhody oproti špičkám

- **Najlepší WER na SK** zo všetkých testovaných (kinit-turbo-sk: 0–6 % vs Azure 47 %, large-v3 30 %)
- **Plne lokálny** — žiadne audio neopúšťa Mac
- **Apple Silicon optimizovaný** — ANE draft + GPU final
- **Dvojvrstvový pipeline** — draft <1 s + final 3–4 s + Azure korekcia

---

## ⭐ TOP 3 Odporúčania (vysoký dopad, nízka námaha)

### 🥇 1. Export SRT/VTT + automatické ukladanie transkriptu session

**Hodnota:** Každá prednáška, podcast, stream = hotový prepis. Zákazník/škola to chcú ako výstup.  
Wordly a Zoom to robia automaticky — my nie.

**Implementácia:**
```python
# titulky/export.py
# Final commits majú timestamp → priamy SRT/VTT export
# Ukladanie: .runs/<run>/transcript.srt + transcript.vtt + transcript.txt
```
- `_emit()` už posiela `{t:'final', text, ts_start, ts_end}` cez WebSocket
- Stačí pridať `TranscriptWriter` triedu, ktorá append-uje SRT bloky do súboru
- Na konci session (`SIGINT` / `POST /stop`) flush + zapis `session_YYYYMMDD_HHMMSS.srt`
- Overlay UI: tlačidlo **⬇ Stiahnuť prepis** (fetch `/transcript/latest.srt`)

**GPU/ANE/CPU:** 0 — iba I/O  
**Latencia dopad:** Žiadna  
**Prácnosť:** ~4 h

---

### 🥈 2. OBS Browser Source integrácia (transparentný overlay)

**Hodnota:** Každý streamer to chce. OBS browser source = titulky na streame bez kompozitingu.  
LocalVocal to robí natívne, ale nemá SK fine-tune ani náš pipeline.

**Implementácia:**
- Server už beží na `localhost:8765` — browser source to vie otvoriť priamo
- Problém: OBS browser source má bielu background; treba `background: transparent`
- Pridať URL parameter: `http://localhost:8765/?obs=1` → CSS `body { background: transparent !important }`
- Chroma key alternatíva: `?chroma=green` → `body { background: #00ff00 }` pre OBS Chroma Key filter
- Export `OBS_SETUP.md` s návodnými screenshotmi (URL, zaškrtávacia možnosť „Shutdown source when not visible")
- Voliteľne: dedikovaná `/obs` route s väčším fontom, tieňom textu optimalizovaným pre video

**GPU/ANE/CPU:** 0  
**Latencia dopad:** Žiadna  
**Prácnosť:** ~2 h

---

### 🥉 3. AI súhrn po skončení session (kapitoly + action items)

**Hodnota:** Killer feature pre prednášky, podcasty, meetingy. Zoom/Otter/Fireflies to predávajú ako premium.  
Po 1-hodinovej prednáške FEI = hotové poznámky, body z prednášky, TODOs.

**Implementácia:**
- Transkript session (feature #1 vyššie) → lokálny LLM cez Ollama (`llama3.2:3b` alebo `mistral:7b`)
- Prompt: `"Vytvor: 1) Kapitoly s časovými značkami 2) Kľúčové body 3) Action items. Jazyk: slovenský."`
- API call na `http://localhost:11434/api/generate` (Ollama) — bez cloudu
- Fallback: Claude API ak Ollama nie je k dispozícii (`TITULKY_SUMMARY_API=claude`)
- Výstup: `session_YYYYMMDD_HHMMSS_summary.md` + zobrazenie v `/flow` po skončení
- Trigger: automaticky po `POST /stop`, alebo manuálne `POST /summarize`

**GPU/ANE/CPU:** Ollama 3B model ≈ 2–5 s na generáciu (CPU/Metal), bez vplyvu na live pipeline  
**Prácnosť:** ~6 h

---

## Kompletný roadmap — všetky features

### Priorita A — Implementovať v najbližšom sporte (< 1 týždeň každá)

| # | Feature | Hodnota | Prácnosť | Latencia dopad |
|---|---|---|---|---|
| A1 | **Export SRT/VTT + transcript** | ⭐⭐⭐ | 4 h | 0 |
| A2 | **OBS transparent overlay** | ⭐⭐⭐ | 2 h | 0 |
| A3 | **AI súhrn session** | ⭐⭐⭐ | 6 h | 0 |
| A4 | **Klávesové skratky** | ⭐⭐ | 3 h | 0 |
| A5 | **Fullscreen prezentačný mód** | ⭐⭐ | 4 h | 0 |

---

### A4. Klávesové skratky

**Hodnota:** Počas prednášky/streamu ovládanie bez myši — profesionálny workflow.  
Zoom, Wordly, OBS — všetky majú skratky.

**Implementácia:**
- Frontend overlay: `document.addEventListener('keydown', ...)` 
- Backend: WebSocket správy `{t:'cmd', cmd:'pause'|'clear'|'save'|'mode_toggle'}`
- Navrhované skratky:

| Skratka | Akcia |
|---|---|
| `Space` | Pauza/Obnoviť titulkovanie |
| `C` | Vyčistiť obrazovku |
| `S` | Uložiť transkript hneď |
| `M` | Prepnúť TITULKY ↔ DIKTOVANIE mód |
| `F` | Fullscreen mód |
| `Esc` | Zavrieť fullscreen |
| `T` | Toggle priehľadnosť pozadia |

- Konfigurovateľné v `config.json` (user-definable)

**Prácnosť:** 3 h

---

### A5. Fullscreen prezentačný mód

**Hodnota:** Prednáška v aule — titulky na projektor plným displejom, veľký font, kontrast.  
YouTube Live a Wordly majú dedikovaný "audience view".

**Implementácia:**
- Nová route `/fullscreen` — čierné pozadie, biely text, font 48–72px, centrovane
- Parametre: `?font=64&lines=3&theme=dark|light|solarized`
- JavaScript: `document.documentElement.requestFullscreen()` cez skratku F
- Automatická zmena veľkosti textu podľa dĺžky riadku (CSS `clamp()`)
- Dual-screen podpora: overlay na externý displej, `/flow` debug na MacBook

**Prácnosť:** 4 h

---

### Priorita B — Strednodobé (1–3 týždne implementácie)

| # | Feature | Hodnota | Prácnosť | Technická zložitosť |
|---|---|---|---|---|
| B1 | **Živý preklad SK→EN (lokálne, NLLB)** | ⭐⭐⭐ | 2–3 dni | Stredná |
| B2 | **Živé zvýraznenie kľúčových slov** | ⭐⭐ | 1 deň | Nízka |
| B3 | **Viacjazyčný výstup (WebSocket multi-channel)** | ⭐⭐ | 2 dni | Stredná |

---

### B1. Živý preklad SK→EN (lokálne NLLB vs API)

**Hodnota:** Stream/prednáška pre medzinárodnú audienciu. Interprefy to predáva za tisíce eur.  
FEI seminár s medzinárodnými hosťami = okamžitá aplikácia.

**Porovnanie prístupov:**

| Prístup | Model | Latencia | Kvalita SK→EN | Pamäť | Odporúčanie |
|---|---|---|---|---|---|
| **NLLB-200 lokálne** | `facebook/nllb-200-distilled-600M` | 200–500 ms/vetu | BLEU ~30–35 | 1.2 GB RAM | ✅ Produkcia |
| NLLB-200 veľký | `nllb-200-1.3B` | 800 ms–1.5 s | BLEU ~38 | 2.6 GB RAM | Pre finálny prepis |
| **Helsinki-NLP** | `opus-mt-sk-en` | 100–200 ms | BLEU ~28 | 300 MB | Draft preklad |
| DeepL API | — | 300–600 ms (sieť) | BLEU ~45+ | 0 (cloud) | Ako premium opt-in |
| Claude/GPT API | — | 500 ms–2 s | Najlepší | 0 (cloud) | Len post-session |

**Odporúčaná architektúra:**
```
Final commit → TranslationWorker (fronta)
    → Helsinki opus-mt (draft preklad, <200ms, ANE možné cez CoreML)
    → NLLB-200-distilled (final preklad, 300–500ms, CPU/MPS)
    → emit {t:'translation', lang:'en', text:'...'}
```

**Implementácia:**
```python
# titulky/translation.py
from transformers import pipeline
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-sk-en", device="mps")
# alebo NLLB: model="facebook/nllb-200-distilled-600M", src_lang="slk_Latn", tgt_lang="eng_Latn"
```

**Overlay:** Druhý riadok textu, menší font, šedá farba  
**OBS:** Separátny WebSocket channel `?lang=en` → vlastný browser source  
**GPU/ANE budget:** Helsinki na CPU: ~15 ms/vetu; NLLB na MPS: ~300 ms — mimo final dekódu, bez konfliktu  
**Prácnosť:** 2–3 dni (+ testovanie kvality na reálnych vzorkách)

---

### B2. Živé zvýraznenie kľúčových slov

**Hodnota:** Pre prednášky — publiku sa zvýraznia pojmy, mená, čísla. Pedagogická hodnota.

**Implementácia:**
- `lexikon.json` už existuje → rozšír o `"keywords": ["FEI", "STU", "algorithm", ...]`
- Regex match pri `_emit()` → wrap do `<mark>` tagu v HTML
- CSS animácia: `@keyframes highlight { 0% { background: yellow } 100% { background: transparent } }`
- Dynamické pridávanie slov cez `POST /keywords {"add": ["neuronová sieť", "backpropagation"]}`
- Farby podľa kategórie: vlastné mená (modrá), technické pojmy (žltá), čísla/dátumy (zelená)

**Prácnosť:** 1 deň

---

### Priorita C — Dlhodobé (týždne–mesiace, vyššia zložitosť)

| # | Feature | Hodnota | Prácnosť | Prekážky |
|---|---|---|---|---|
| C1 | **Speaker diarization** | ⭐⭐⭐ | 1–2 týždne | Real-time latencia pyannote |
| C2 | **Mobilná companion app** (QR → titulky na telefóne) | ⭐⭐ | 2–3 týždne | iOS/Android |
| C3 | **Zvuková vizualizácia / waveform** v overlay | ⭐ | 3 dni | GPU overlap |
| C4 | **Cloudová sync + zdieľanie session** | ⭐⭐ | 2 týždne | Infraštruktura |
| C5 | **Whisper fine-tune na Jarových dátach** | ⭐⭐⭐ | 1–2 mesiace | Dátová sada |

---

### C1. Speaker Diarization (kto hovorí)

**Hodnota:** Podcast, debata, konferencia — „Jaro: ...", „Hosť: ..."  
Zoom to má. Otter.ai na tom postavil celý produkt.

**Implementácia:**
```python
# pyannote/audio 3.x — real-time diarization
from pyannote.audio import Pipeline
diarize = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
# PROBLÉM: pyannote nie je real-time optimalizovaný
```

**Realistická architektúra pre náš stack:**
1. **Offline diarizácia** (post-session): pyannote na celej nahrávke → výstup RTTM → prepis obohatený o speaker labels
2. **Real-time aproximácia**: 
   - pyannote `SpeakerSegmentation` na kĺzavom okne 10–30 s
   - Latencia: 2–5 s naviac → pipeline musí byť na separátnom CPU vlákne
   - Speaker embedding cache — keď raz identifikujeme „Jaro" (enroll fáza), ostáva stabilný
3. **Overlay**: `[Jaro]` prefix pred textom, rôzna farba na meno

**Obmedzenia:**
- pyannote model = 1–2 GB VRAM, konflikt s MLX final dekódom
- Real-time: maxálne 2 rečníci spoľahlivo; 3+ degraduje
- Enroll fáza: 10–30 s čistej reči každého rečníka pred spustením

**GPU/ANE budget:** Pyannote na CPU ≈ 30–50 ms/s audio → zvládnuteľné na M-chip ako background vlákno  
**Prácnosť:** 1–2 týždne (real-time aproximácia), ďalší týždeň stabilizácia

---

### C5. Fine-tune whisper na Jarových dátach

**Hodnota:** WER 0% na vlastnom hlase, žargóne (Grindcast, DJ), skratkách (FEI, STU, MFCC)

**Dátová sada:**
- `podcast_test/` v repe — dobrý začiatok
- FLEURS SK: 120 viet (už použité v bench)
- Cieľ: 500–2000 viet z reálnych nahrávok

**Implementácia:**
```bash
# Hugging Face transformers fine-tune
# alebo mlx-examples/whisper fine-tune (priamo na Apple Silicon)
python mlx-examples/whisper/finetune.py \
  --model whisper-large-v3-turbo \
  --data jaro_train.jsonl \
  --epochs 5
```

**Prácnosť:** Príprava dát 2 týždne + tréning 1–2 dni na M-chip  
**Poznámka:** kinit-whisper-turbo-sk ukázal doménový posun na hovorovej reči — vlastný fine-tune by to vyriešil definitívne

---

## Implementačný plán (sprint-based)

### Sprint 1 — „Deployment ready" (2 týždne)
1. ✅ **A2 OBS overlay** (2 h) — okamžitý unlock pre streamery
2. ✅ **A1 SRT/VTT export** (4 h) — každá session = hotový prepis
3. ✅ **A4 Klávesové skratky** (3 h) — profesionálny UX
4. ✅ **A5 Fullscreen mód** (4 h) — prednáška v aule

**Výstup:** Systém nasaditeľný pre reálne prednášky a streamy

### Sprint 2 — „AI enhanced" (2 týždne)
5. **A3 AI súhrn** (6 h) — killer feature pre školu
6. **B2 Zvýraznenie kľúčových slov** (1 deň)
7. **B1 Preklad SK→EN draft** (Helsinki, 1 deň) — rýchla hodnota
8. **B1 Preklad SK→EN final** (NLLB, 1 deň) — kvalita

**Výstup:** Plnohodnotný AI asistent pre prednášky

### Sprint 3 — „Professional" (mesiac)
9. **C1 Speaker diarization offline** (1 týždeň)
10. **B3 Multi-channel WebSocket** (2 dni)
11. Dokumentácia, demo video, FEI prezentácia

---

## Technické úvahy pre Apple Silicon

### GPU/ANE/CPU budget (M-chip)

```
┌─────────────────────────────────────────────────────────┐
│                  Apple Silicon M-chip                   │
│                                                         │
│  ANE (16 TOPS)     GPU (Metal/MPS)     CPU (P+E cores)  │
│  ──────────────    ───────────────     ──────────────   │
│  Draft whisper     Final whisper       Translation      │
│  (base, ~0.5s)     (turbo-sk, ~3s)    worker (NLLB)    │
│                                        Diarization      │
│                    ← NLLB 600M →       (pyannote CPU)   │
│                    (300ms, MPS)         Transcript I/O  │
│                                        Ollama (LLM)     │
│                    Ollama Metal        (3B: 2-5s)       │
│                    (opt-in)                             │
└─────────────────────────────────────────────────────────┘
```

**Kľúčové pravidlo:** Translation worker a LLM súhrn bežia **mimo** final dekódu (fronta, async). Nikdy súbežný GPU prístup — latenčná kontaminácia (pozri PITFALLS v skille).

### Knižnice a závislosti

| Feature | Knižnica | Veľkosť | Poznámka |
|---|---|---|---|
| Preklad (draft) | `Helsinki-NLP/opus-mt-sk-en` | 300 MB | transformers, MPS |
| Preklad (final) | `facebook/nllb-200-distilled-600M` | 1.2 GB | transformers, MPS |
| LLM súhrn | Ollama (`llama3.2:3b`) | 2 GB | lokálny server |
| Diarization | `pyannote/speaker-diarization-3.1` | 1.5 GB | CPU vlákno |
| SRT export | stdlib `datetime` | — | žiadna závislosť |
| Klávesové skratky | vanilla JS | — | žiadna závislosť |

---

## Diferenciátory oproti OBS LocalVocal

OBS LocalVocal (royshil) je najbližší open-source konkurent. Naše výhody:

| Kritérium | OBS LocalVocal | Live Titulky (náš stack) |
|---|---|---|
| Slovenčina WER | ~15–25% (generic turbo) | **0–6%** (kinit-turbo-sk fine-tune) |
| Platforma | Windows/Linux | **macOS Apple Silicon natívne** |
| Korekčná vrstva | Nie | **Azure + lexikon.json gate** |
| Debug nástroje | Nie | **`/flow` studio, A/B replay** |
| Preklad | Áno (whisper built-in) | NLLB lokálne (roadmap B1) |
| OBS integrácia | Natívny plugin | **Browser source (roadmap A2)** |
| Latencia (SK) | 3–8 s | **draft 0.7 s / final 3.4 s** |

---

## Záver

**Tri kroky s najvyšším dopadom v poradí:**

1. **OBS Browser Source** (2 h) — otvorí celý streamer trh, nulová latencia dopad
2. **SRT/VTT export** (4 h) — každá session = profesionálny deliverable
3. **AI súhrn po session** (6 h) — differenciátor od všetkej konkurencie pre vzdelávacie použitie

Spolu ~12 hodín práce = systém schopný riešiť reálne produkčné scenáre: FEI prednášky, Grindcast podcasty, LEADER X CAST streamy, konferencie.
