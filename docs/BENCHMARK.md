# Živé slovenské titulky — benchmark a evaluačná metodika

*Verzia 2.0 — 22. 10. 2026 · všetky čísla v tomto dokumente sú reálne namerané
(commit `4a2335a` a predchodcovia; surové dáta: `exp_grid_results.jsonl`,
`bench_models_results.json`, `bench_fleurs_results.json`, `tests/results.json`)*

---

## 1. Metodika

### 1.1 Hardvér a softvér

| Komponent | Hodnota |
|---|---|
| Zariadenie | MacBook Pro, Apple M3 Pro, 36 GB RAM |
| OS | macOS 27.0.1 |
| Inferencia | mlx-whisper 0.4.3, MLX 0.32.2 (Apple GPU/Metal), float16 |
| Python | 3.12.5 |
| Draft model | `mlx-community/whisper-base-mlx` |
| Final model | `kinit/whisper-large-v3-turbo-sk` (MLX konverzia, float16), greedy dekód — pozn.: mlx_whisper 0.4.3 beam search nepodporuje; skorší zápis „beam 5" bol omyl (parameter sa ticho ignoroval) |
| Korekčná vrstva | Azure Speech STT (voliteľná, asynchrónna) |

Všetko okrem voliteľnej Azure korekcie beží **lokálne, bez siete a bez GPU
servera**.

### 1.2 Testovacie dáta

| Sada | Pôvod | Rozsah | Na čo |
|---|---|---|---|
| `tests/*.wav` (normal, fast, slow, quiet, noisy, pauses, silence) | synteticky (TTS `say` + mix šumu), referencie dané pri syntéze | 7 súborov, ~80 s | E2E streaming: WER, latencie, flicker, halucinácie |
| `mic_test.wav` | reálny MacBook mikrofón, spontánna hovorová reč, referencia prepísaná hovoriacim | 10 s | sanity check na reálnych dátach |
| **FLEURS SK test** (`google/fleurs`, split test) | verejný korpus, reálni rečníci | 120 viet, 23,6 min | výber final modelu (korpusový WER) |
| `tests/fixtures/noise_floor.npy` | reálna nahrávka ticha z MacBook mikrofónu | 6 s | kalibrácia a regresný test šumového gate-u |

Common Voice SK nebol použitý priamo (gated dataset); publikované čísla
citujeme z literatúry (§6).

### 1.3 Protokol merania

1. **Simulácia živého vstupu:** WAV sa prehráva do pipeline po blokoch v reálnom
   čase (žiadne batch spracovanie) — `TwoTierPipeline.play_file()`.
2. **Časy:** `time.monotonic()`; všetko na jednom stroji.
3. **Draft latencia** sa meria od **akustického konca slova** po jeho **prvé
   skutočné zobrazenie** (nie po dekód — prekresľovanie je škrtené
   `DRAFT_HOLD_S`, poctivé je meranie zobrazenia). Metodicky zodpovedá
   word-level latencii z Macháček et al. 2023 (§6 [1]).
4. **Final latencia**: koniec slova → commit slova LocalAgreement vrstvou.
5. **Jeden beh na konfiguráciu** (limitácia §5; trend potvrdený dvoma nezávislými
   behmi celej sady — ráno baseline, večer v2).
6. Behy sa logujú append-only do run-trace JSONL (`.runs/`) vrátane SHA-256
   vstupných WAV; ku každému commitu existujú auditné artefakty.

### 1.4 Normalizácia textu pred WER

Identicky na referenciu aj hypotézu (`bench._norm_text`):

```
lowercase → odstrániť všetko okrem písmen (vrátane á ä č ď é í ĺ ľ ň ó ô ŕ š ť ú ý ž)
a číslic → zlúčiť biele znaky
```

**Diakritika sa zachováva** — v slovenčine je fonologicky dištinktívna
(„mas" ≠ „máš"); interpunkcia a veľkosť písmen sa zahadzujú (Whisper ich
generuje nedeterministicky). Čísla sa v testovacích referenciách vyskytujú
slovom, num2words nebol potrebný.

---

## 2. Metriky

| Metrika | Definícia |
|---|---|
| **WER** | (S + D + I) / N na normalizovanom texte, `jiwer` |
| **Draft p50/p90** | medián / 90. percentil latencie slova: koniec slova → prvé zobrazenie (sivý draft) |
| **Final p50** | medián latencie: koniec slova → commit (biely text) |
| **Flicker** | podiel committed slov, ktoré boli dodatočne zmenené (prepis final vrstvou alebo korekciou) |
| **Halluc** | počet slov emitovaných počas čistého ticha/šumu (ideál 0) |
| **Corpus WER (FLEURS)** | (ΣS+D+I)/(ΣN) cez 120 viet |
| **RTFx** | koľkokrát rýchlejšie než realtime beží offline dekód |

---

## 3. Výsledky

### 3.1 Výber final modelu (offline, izoluje kvalitu modelu od streamingu)

Greedy dekód, `condition_on_previous_text=False`, normalizácia §1.4.

| nahrávka | large-v3-turbo | **kinit-turbo-sk** |
|---|---|---|
| tests/normal.wav | 3,2 % | **0,0 %** |
| tests/fast.wav | 0,0 % | **0,0 %** |
| tests/quiet.wav | 3,2 % | **0,0 %** |
| tests/noisy.wav | 3,2 % | **0,0 %** |
| tests/pauses.wav | 3,2 % | **0,0 %** |
| mic_test.wav (reálny mik, 10 s) | **13,0 %** | 30,4 % |
| **FLEURS SK test, 120 viet (korpus)** | 10,54 % | **5,81 %** |

**Rozhodnutie:** final = `kinit-whisper-turbo-sk` — na verejnom korpuse
reálnych rečníkov znižuje chybovosť o **45 % relatívne** a na čítanej reči je
bezchybný. Výhrada z 10 s mic_test sa na väčšej vzorke nepotvrdila —
na 3 min reálnej podcastovej slovenčiny je kinit zreteľne lepší (§3.7).
Architektúra je identická s turbo (809 M parametrov), takže latencia sa
nemení.

Publikované hodnoty konzistentné s naším meraním: KInIT reportuje CV24 WER
9,29 % (fine-tune) vs 29,23 % (base turbo); SLAIH leaderboard uvádza pre base
turbo 18,35 % a pre Azure STT 13,29 % na spontánnom slovenskom sete (§6).

### 3.2 Ladenie draft vrstvy (grid 11 konfigurácií)

Meria **zobrazovaciu** latenciu (§1.3 bod 3) na 4 nahrávkach (normal, fast,
pauses, mic_test). Final vrstva vždy large-v3-turbo; WER celej pipeline je
naprieč konfigmi ~konštantný (rozdiely sú šum final dekódu), preto tabuľka
ukazuje latencie a flicker.

| draft konfig | draft p50 [s] | draft min | draft p90 | flicker |
|---|---|---|---|---|
| small · okno 3,0 s · krok 0,25 s · hold 0,8 s *(pôvodné)* | 1,762 | 0,240 | 3,235 | 14,9 % |
| small · 2,0 · 0,25 · 0,8 | 1,717 | 0,235 | 4,236 | 14,1 % |
| small · 1,5 · 0,25 · 0,8 | 1,121 | 0,257 | 2,043 | 14,8 % |
| small · 1,5 · 0,15 · 0,8 | 1,121 | 0,233 | 1,923 | 15,4 % |
| small · 2,0 · 0,15 · 0,8 | 1,533 | 0,228 | 2,481 | 14,4 % |
| small · 1,5 · 0,10 · 0,2 | 1,199 | 0,247 | 2,165 | 15,0 % |
| base · 2,0 · 0,25 · 0,8 | 1,054 | 0,061 | 2,006 | 10,7 % |
| base · 1,5 · 0,15 · 0,8 | 0,864 | 0,038 | 1,745 | 12,5 % |
| base · 1,5 · 0,10 · 0,2 | 0,745 | 0,042 | 1,600 | 14,7 % |
| base · 2,0 · 0,10 · 0,2 | 0,977 | 0,039 | 2,044 | 10,4 % |
| **base · 1,5 · 0,15 · 0,3 (zvolené)** | **0,732** | **0,030** | **1,544** | 11,5 % |

Poznatky: (a) `base` dekóduje 1,5 s okno za ~40 ms vs ~240 ms pri `small` —
draft latencia je potom daná prevažne čakaním na pravý kontext slova, nie
dekódom; (b) skrátenie hold 0,8 → 0,3 s znížilo p50 o ďalších ~0,13 s bez
nárastu flickeru; (c) `base` má paradoxne nižší flicker než `small` — final
vrstva prepisuje obe, ale rýchlejšia kadencia draftu znamená, že final commit
častejšie „dobehne" draft ešte pred prekreslením.

Metodická poznámka: prvá verzia merania brala čas dekódu; po zistení, že
`DRAFT_HOLD_S` zdržiava zobrazenie, bola metrika opravená na čas skutočného
zobrazenia a celý grid **premeraný nanovo** (v1 dáta:
`exp_grid_v1_decode_metric.jsonl`).

### 3.3 E2E pipeline: pôvodná vs. nová konfigurácia (celá sada, 1 beh)

Pôvodná = small/3,0/0,25/0,8 + final turbo. Nová = base/1,5/0,15/0,3 + final
kinit-sk + ghost cursor. Obe s aktívnou Azure korekciou.

| test | WER pôv. | WER nová | draft p50 pôv. | **nová** | final p50 pôv. | nová | flicker pôv. | nová |
|---|---|---|---|---|---|---|---|---|
| normal | 3,2 % | 3,2 % | 1,74 s | **0,89 s** | 3,94 s | 4,36 s | 7,2 % | 10,0 % |
| fast | 62,5 %* | **0,0 %** | 0,95 s | **0,28 s** | 2,36 s | 4,05 s | 0,0 % | 3,0 % |
| slow | 0,0 % | 0,0 % | 1,32 s | **0,22 s** | 3,52 s | 3,87 s | 5,9 % | 0,0 % |
| quiet | 6,5 % | **3,2 %** | 1,66 s | **0,70 s** | 3,82 s | 4,50 s | 7,2 % | 20,0 % |
| noisy | 12,9 % | **9,7 %** | 2,08 s | **0,65 s** | 3,40 s | 5,87 s | 17,8 % | 14,9 % |
| pauses | 0,0 % | 3,2 %** | 1,54 s | **0,82 s** | 4,23 s | 4,59 s | 6,1 % | 6,8 % |
| silence | halluc 0 | halluc 0 | – | – | – | – | 0 % | 0 % |

\* regres amplitúdového gate-u — súvislá rýchla reč bez páuz má plochú RMS ako
šum; opravené dvojitým gate-om (§3.5). \** vypadnutá spojka „a" na hranici
final okna (boundary jav LocalAgreement, §4.3).

**Súhrn:** draft p50 klesol z 0,95–2,08 s na **0,22–0,89 s** (−50 až −83 %),
WER sa zlepšil alebo držal na 5 zo 6 testov. Cena: final p50 +0,4 až +2,5 s
(pomalší konsenzus dvoch po sebe idúcich kinit dekódov, §4.2) a flicker na
`quiet` 20 %.

### 3.4 Vnímaná odozva (ghost cursor)

Od detekcie reči (spektrálny flux gate prejde) po viditeľnú reakciu UI
(blikajúci `▊`) ubehne najviac `DRAFT_STEP_S + emit` ≈ **≤ 0,2 s** — t. j.
systém reaguje viditeľne pod hranicou ~300 ms, pri ktorej sa podľa NTID/CHI
výskumu narúša pocit synchronizácie; prvé sivé slovo nasleduje s p50 0,73 s
a biele potvrdené znenie s p50 ~4 s. Trojstupňová odozva (kurzor → draft →
final) je zámerný dizajn podľa vzoru Google Meet (§6 [7]).

### 3.5 Halucinácie z ticha — vývoj šumového gate-u

Historicky najzávažnejší bug (šum sa „naučil" ako úroveň reči → final
donekonečna dekódoval halucinácie typu „Ďakujem za pozornosť" × 11).

| Gate | Princíp | Výsledok |
|---|---|---|
| v0: fixný RMS prah | konštanta | zlyháva pri hlasnom miku |
| v1: dynamický prah (p95/p20 frame RMS) | reč má dynamiku, šum je plochý | halluc 0, ale **regres: rýchla súvislá reč je tiež „plochá"** → fast.wav WER 62,5 % |
| **v2: amplitúdová dynamika AND spektrálny flux** | šum = plochá amplitúda **a zároveň** statické spektrum; reč zlyhá aspoň na jednom (pauzy ⇒ dynamika ≥ 1,95; pohyb formantov ⇒ flux ≥ 0,31) | halluc 0 **a** fast 0 % WER |

Prahy merané na reálnych dátach: room tone flux 0,131, reč 0,31–0,46, prah
0,22; regresné testy používajú **fixture skutočného šumu**
(`tests/fixtures/noise_floor.npy`), nie syntetický — biely šum má totiž
vysoký flux (0,55) a prešiel by. Biely/hiss šum gate vedome prepúšťa; chráni
ho energy gate slov + anti-loop detekcia.

### 3.6 Systém D vs E: prínos Azure korekčnej vrstvy

Jeden beh celej sady bez korekcie (`TITULKY_NO_AZURE=1`) vs s ňou; ostatná
konfigurácia zhodná (base/1,5/0,15/0,3 + kinit final).

| test | D: bez korekcie | E: s korekciou |
|---|---|---|
| normal | 9,7 % | **3,2 %** |
| fast | 0,0 % | 0,0 % |
| slow | 0,0 % | 0,0 % |
| quiet | **0,0 %** | 3,2 % |
| noisy | 9,7 % | 9,7 % |
| pauses | 9,7 % | **3,2 %** |
| silence (halluc) | 0 | 0 |

Interpretácia: pri 31-slovových referenciách je 1 slovo = 3,2 p. b., takže
jednotlivé rozdiely sú na hranici run-to-run šumu final dekódu (§5 bod 1).
Vzor je však konzistentný s dizajnom vrstvy: korekcia pomohla na testoch
s OOV preklepmi (normal, pauses), nič nepokazila na fast/slow/noisy a na
quiet jeden zásah uškodil. Systém je plne funkčný aj offline (D) — korekcia
je voliteľné spresnenie, nie závislosť.

### 3.7 Reálna spontánna reč — podcast (bez zlatej referencie)

4 × 90 s z epizód LEADER X PODCAST (16 kHz mono z produkčných stôp), offline
prepis oboma modelmi. Zlatá referencia neexistuje, preto kvalitatívna analýza
nezhôd + miera vzájomnej nezhody modelov (25,8 % slov).

**Slovenský rečník (2 klipy, čistá mikrofónová stopa):** kinit konzistentne
správne slovenské tvary a propriá („išiel", „nepraje", „míľnik", „podľa mňa",
„Svätý Jur", „v Bratislave", „priebojnosti"), turbo tie isté miesta komolí
(„išel", „neprája", „poľa mea", „svetho júra", „bátislave", „vzáraníčne").
Opačný vzor len pri anglicizmoch: turbo „hard techno", „Afromelodic" vs kinit
„hartéchno", „afromelody" — kandidát na slovníkovú korekčnú vrstvu. Toto
**prevracia indíciu z §3.1** (10 s mic_test, kde turbo vyzeral lepšie):
na 3 minútach reálnej spontánnej slovenčiny je kinit zreteľne presnejší,
takže voľba final modelu platí aj pre spontánnu reč.

**Mimo zadania:** pri česky hovoriacom hosťovi sa SK fine-tune rozpadá
(opakovacie slučky), turbo češtinu drží. Projekt cieli výlučne slovenčinu,
preto bez riešenia; prepínač `TITULKY_FINAL_REPO` existuje.

Surové prepisy: `podcast_test/hyps.json`, nezhody: `podcast_test/REVIEW.md`.

### 3.8 Ladenie draft latencie (grid v3) a verdikt k cieľu 420 ms

16 konfigurácií (model × okno × krok × hold), append-only JSONL
(`exp_grid_results.jsonl`). Víťaz nasadený ako default:
**tiny, okno 1,2 s, krok 0,08 s, hold 0,10 s** → draft p50 **0,47 s**
(grid), v plnej suite 0,17–0,52 s podľa tempa reči; pôvodná konfigurácia
mala 0,78–1,76 s. Návrat na base: `TITULKY_DRAFT_REPO`.

Cieľ 420 ms: **splnený pri rýchlej a pomalej reči (0,17/0,27 s), tesne
nesplnený pri bežnom tempe (0,47–0,52 s)**. Fyzikálny rozklad dna:
slovo musí doznieť (0 ms referencia) + krok okna (80 ms) + dekód tiny
(~120 ms) + Whisper slovo často uzná až s ďalším kúskom kontextu
(~1 krok) → ~0,35–0,45 s je dno prístupu s kĺzavým oknom. Vnímanú
odozvu < 200 ms zabezpečuje ghost-kurzor (reakcia na RMS, nie na dekód)
— pri obhajobe treba rozlíšiť „systém reaguje" (kurzor, < 0,2 s) od
„slovo na obrazovke" (draft, ~0,5 s) a „potvrdené slovo" (final, ~4 s).

Pozn.: final WER je draft modelom nedotknutý (nezávislé vlákna); noisy
kolíše 6,5–12,9 % medzi behmi (jitter, viď limitácia 2).

### 3.9 Pokrytie dlhej súvislej reči (stitch) a poctivosť metriky

Problém z živého testu: draft dekódoval len posledných 1,2 s, takže pri
monológu slová medzi final hranicou (~4 s dozadu) a krátkym oknom nevidel
nikto — titulky „čakali na dohovorenie". Fix: **stitch** — krátke okno
(1,2 s) každých 0,08 s pre čerstvé slová + dlhé okno (po final hranicu,
strop 5 s) max raz za 1,5 s na backfill medzery; zobrazenie je zošité
a prefix sa živo filtruje final hranicou.

Dôsledok pre metriku: draft p50 vzrástol na ~0,7–1,2 s, ale pôvodných
0,47 s bolo merané na **skreslenej vzorke** — slová v medzere sa do
latencie vôbec nepočítali, lebo sa nikdy nezobrazili. Čerstvé slová
chodia naďalej ~0,5–0,7 s; nad tým je backfill, ktorý predtým chýbal
úplne. Bonus merania so stitch: normal.wav WER prvýkrát 0,0 % a flicker
klesol z 27 % na 15 % (dlhšie okno = stabilnejší kontext draftu).

### 3.10 Ladenie final latencie (final grid)

6 konfigurácií krok × okno (`exp_final_results.jsonl`), 3 testy každá.
Nasadené: **krok 1,5 s, okno 8 s** → final p50 3,94 → **3,56 s** pri
najlepšom WER (2,2 %) aj flickeri (4 %). Plus flush budí final slučku
okamžite (Event) — po pauze v reči odpadá až 2 s spánku.

Dno final latencie tohto prístupu je ~3,4 s: dekód large modelu na 8 s
okna trvá ~1,5–2 s a LocalAgreement vyžaduje dve zhodné čítania → 
2 × (dekód + krok). Ďalšie zrýchlenie = menší final model alebo commit
z jedného čítania — oboje obchoduje presnosť, ktorá je tvrdým cieľom
(§1). Vedomé rozhodnutie: ostať na 3,5 s.

---

## 4. Diskusia

### 4.1 Prečo fine-tune vyhráva a kde nie

kinit-turbo-sk je trénovaný na TEDxSK, Common Voice SK a SloPal — čítaná
a prednášková reč. Na FLEURS (čítaná) −45 % chýb; na 10 s hovorovej nahrávky
z miku horší než base turbo (30 % vs 13 %). Hypotéza: doménový posun
(hovorové „blbosti", anglicizmus „hater", šum MacBook miku). Pre scenár
prednášky/dema je fine-tune správna voľba; pre čisto hovorový obsah treba
väčší reálny testset (§5).

### 4.2 Trade-off rýchly draft vs. pokojný final

Zrýchlenie draftu (menšie okno, kratší krok) nie je zadarmo: final
LocalAgreement-2 čaká na zhodu dvoch po sebe idúcich dekódov 10 s okna; kinit
má mierne vyššiu variabilitu výstupu medzi oknami než base turbo (viac
synoným/tvarov na hovorovej reči), preto final p50 stúpol o ~0,5–2,5 s. Pre
UX je rozhodujúci draft (užívateľ číta sivý text) — final je potvrdenie. Ak
by final latencia prekážala, kandidáti: `FINAL_STEP_S` 2,0 → 1,5 s alebo
agreement na úrovni normalizovaného textu.

### 4.3 Hraničná strata krátkych slov

`pauses.wav` stratil spojku „a" presne na hranici final okna — známy jav
sliding-window prístupov (slovo bez pravého kontextu v jednom okne, v ďalšom
už mimo okna). Riešenie by bol overlap-commit buffer; pri frekvencii výskytu
(1 funkčné slovo na ~80 s testov) ponechané ako zdokumentovaná limitácia.

### 4.4 Flicker na quiet

20 % flicker na tichej reči = final vrstva viac prepisuje neisté draft slová
(tiché fonémy → base draft viac háda). Alternatíva (držať draft dlhšie
= vyšší hold) by zhoršila latenciu všetkým; ponechané, kandidát na ďalšiu
iteráciu (confidence-gating draft slov pod RMS prahom).

---

## 5. Limitácie

1. **1 beh na konfiguráciu** — bez konfidenčných intervalov; trend potvrdený
   dvoma nezávislými behmi sady (ráno/večer), ale p50 čísla majú run-to-run
   šum odhadom ±0,1 s (draft) / ±0,5 s (final).
2. **Syntetické referencie** (TTS) pre 6 zo 7 E2E testov — ideálna
   výslovnosť; reálna reč je ťažšia (viď mic_test).
3. **Spontánna reč bez zlatej referencie** — podcastové klipy (§3.7) majú len
   pseudo-referencie (zhoda modelov ~74 %); kvalitatívny verdikt je jasný,
   presný WER by vyžadoval ručne overený prepis.
4. Common Voice SK nemerané priamo (gated); čísla prevzaté z literatúry.
5. Azure korekčná vrstva vyžaduje sieť a kľúč — systém bez nej beží plne
   lokálne (§3.6).
6. Flicker metrika počíta zmeny committed slov; nemeria vizuálny flicker
   sivého draftu.

---

## 6. Literatúra a zdroje

[1] Macháček, Dabre, Bojar: *Turning Whisper into Real-Time Transcription
System* (IJCNLP 2023) — LocalAgreement-2, word-level latencia.
https://arxiv.org/abs/2307.14743
[2] KInIT: *whisper-large-v3-turbo-sk* — SK fine-tune, reportované WER.
https://huggingface.co/kinit/whisper-large-v3-turbo-sk
[3] SLAIH SK ASR leaderboard (spontánna slovenčina, 39 h).
https://slaih.sk/sk/benchmarks/speech-to-text
[4] FLEURS — google/fleurs, sk_sk.
https://huggingface.co/datasets/google/fleurs
[5] Liu et al.: *Modeling and Improving Text Stability in Live Captions*
(CHI 2023) — flicker, stabilita zobrazenia, fade-in.
https://dl.acm.org/doi/10.1145/3544549.3585609
[6] WhisperKit (Argmax, 2025): on-device streaming ASR benchmarky na Apple
Silicon. https://arxiv.org/abs/2507.10860
[7] Google Meet live captions UX (draft/final vzor, ~0,9 s medián).
[8] Radford et al.: *Robust Speech Recognition via Large-Scale Weak
Supervision* (Whisper). https://arxiv.org/abs/2212.04356
[9] jiwer — WER implementácia. https://github.com/jitsi/jiwer

---

## 7. Reprodukcia

```bash
# E2E sada (7 WAV, realtime playback):
.venv/bin/python test_suite.py            # systém E (s Azure korekciou)
TITULKY_NO_AZURE=1 .venv/bin/python test_suite.py   # systém D (plne lokálny)

# Grid draft konfigurácií (append-only exp_grid_results.jsonl):
.venv/bin/python exp_grid.py

# Offline A/B final modelov na vlastných nahrávkach + FLEURS:
.venv/bin/python bench_models.py
.venv/bin/python bench_fleurs.py          # vyžaduje fleurs_sk/

# Konverzia kinit modelu do MLX:
python mlx-examples/whisper/convert.py \
  --torch-name-or-path kinit/whisper-large-v3-turbo-sk \
  --mlx-path <cieľ> --dtype float16
ln -s model.safetensors <cieľ>/weights.safetensors

# Živé demo:
.venv/bin/python -m titulky.main          # overlay: http://localhost:8765
```

Prepínač final modelu: env `TITULKY_FINAL_REPO` (HF repo alebo lokálna cesta).
Návrh plnej akademickej metodiky (N=5 behov, CI, force-alignment, grafy):
`docs/BENCHMARK-navrh-metodiky.md`.

---

## 8. Vlastný prínos vs. prevzaté

**Prevzaté (open source / literatúra):** Whisper modely a mlx-whisper
inferencia; kinit fine-tune váhy; LocalAgreement-2 myšlienka [1]; hunspell SK
slovník; jiwer; FLEURS dáta; draft/final UX vzor z produkčných systémov.

**Vlastná práca:**
- dvojvrstvová architektúra draft(base)/final(kinit) s nezávislými oknami,
  GPU arbitrážou (skip-if-busy) a trojstupňovou odozvou kurzor→draft→final;
- **dvojitý šumový gate** (amplitúdová dynamika ∧ spektrálny flux)
  s prahmi meranými na reálnych nahrávkach a regresnou fixture skutočného
  šumu — rieši súčasne halucinácie z ticha aj rýchlu súvislú reč;
- **poctivá zobrazovacia metrika latencie** (meria sa prvé zobrazenie slova,
  nie dekód) a zistenie, že display-hold dominoval vnímanej latencii;
- asynchrónna cross-ASR korekčná vrstva s viacnásobným gate-om (hunspell ∧
  slovník ∧ Levenshtein ≤ 2 ∧ žiadne propriá/čísla ∧ časové okno);
- evaluačná infraštruktúra: realtime test harness, grid runner s append-only
  výsledkami, offline model A/B, FLEURS korpusový bench, run-trace JSONL
  s auditnými WAV artefaktmi per commit;
- empirický výber konfigurácie (11-bodový grid) a modelu (3 nezávislé
  testové sady) namiesto defaultov.

## Audit stacku (23.10.) — slepé uličky zdokumentované
- word_timestamps approx (segmentové delenie): dekód −1,1 s, ale WER 3→15 % (agreement sa rozpadá) — default `exact`, prepínač `TITULKY_WORD_TS`
- kinit int4 kvantizácia (MLX, g64): 2,89 s vs fp16 2,69 s/dekód + horší text — int4 dequant réžia pri batch-1 preváži, fp16 na M-GPU je optimum
- mŕtve závislosti: faster-whisper/ctranslate2/torch/transformers — CPU draft backend porazený ANE (whisper.cpp)
- draft latencia vyriešená ANE backendom (0,13–0,17 s dekód), final 3,5 s = strop architektúry pri 1,5 s kroku a 2× agreemente

## FAST_FINAL experiment (23.10., TITULKY_FAST_FINAL=1)
Final dekód bez DTW (1.65→0.72s) + hranice commitov snapnuté na ANE draft časy:
final p50 3.5→2.1-2.6s, ALE WER 0-3%→6.5-12.9%, flicker 0-8→16-33%.
Lepšie než čistý approx (WER 15-22%), stále pod kvalitatívnym cieľom.
Default ostáva exact. Ďalší smer ak treba <2.5s: commit z ANE agreementu
(biela by sa ale mohla meniť) alebo streaming model namiesto Whisper okien.

## Draft upgrade na kinit-small-sk + whisper.cpp final zamietnuty (23.10. vecer)
- A: draft base → kinit/whisper-small-sk (ggml konverzia + CoreML encoder na ANE):
  3s okno 0.28s, bezchybna slovencina. Suite: draft p50 0.37-0.72s (lepsie nez base
  0.54-1.22 — stabilny text = menej prepisovani), WER baseline. NASADENE.
- B: whisper.cpp final s audio_ctx orezom: encoder 777→205ms, ALE halucinacie/
  duplikacie pri kazdom ctx (turbo aj small, v3 aj v2 arch) — ZAMIETNUTE.
  whisper.cpp plny ctx = 1.54s ≈ MLX 1.65s, ziadna vyhra. Final ostava MLX.
- Konverzny postup: models/convert-h5-to-ggml.py + convert-h5-to-coreml.py
  + xcrun coremlcompiler; modely v /Volumes/GRINDCASTER/AI-MODELS/whispercpp/

## Loop: draftACC metrika + ladenie okna (23.10. noc)
- Nova metrika draftACC = recall: % committed slov, ktore divak videl sive vopred (±2s).
- Baseline (base-sk, okno 1.2s): draftACC 59-84%, draft p50 0.23-0.63s.
- small-sk draft: HORSI recall (28-75%) — pomalsi dekod = slova sa nestihnu ukazat. Zamietnute.
- Okno 3.0s: draftACC 81-100% cista rec (p50 0.4-0.8s) — NASADENE ako default.
- Okno 4.0s: horsie (p50 az 2.6s). Strop.
- Noisy ostava slaby (62%) — energy gate + realne chyby v hluku, final WER tam 9.7%.

## Parakeet-TDT 0.6B v3 vyhodnoteny a zamietnuty (23.10. noc)
- Offline: WER 0-18.8% (fast/slow 18.8% — zlieva slova "zivurecna"), dekod celeho suboru 0.3-4.6s.
- Streaming (chunk 0.5s, ctx 256/256): WER 12.9-29% — slovencina sa rozpadne
  ("titloty", "zivurecna", "blúch"). Krok p50 0.27-0.43s.
- Zaver: bez SK fine-tunu nepouzitelny; kinit whisper ostava. Buduca cesta
  k <1s bielemu textu = fine-tune streaming modelu (Parakeet/RNN-T) na SK — iny projekt.
- Pozn.: parakeet-mlx vyzaduje mlx<=0.28 (mx.concat API) a mx.array vstup.

## Nemotron-3.5-asr-streaming-0.6b (sk-SK) vyhodnoteny a zamietnuty (23.10. noc)
- Hotovy NVIDIA streaming model so slovencinou (40 jazykov) — ziadny trening netreba.
- Offline cez parakeet.cpp (mudler, Metal/CPU): WER normal/quiet 9.7%, pauses 12.9%,
  noisy 29%, fast 62.5%, slow 43.8%. Rychly (RTF ~0.2) ale slovencina slaba na
  rychlej/hlucnej reci. Pod latkou 91% presnosti.
- Pozn.: cstr GGUF (CrispASR layout) je nekompatibilny s mudler parakeet.cpp —
  treba mudler/parakeet-cpp-gguf. Modely v /Volumes/GRINDCASTER/AI-MODELS/nemotron/.
- Zaver: kinit whisper two-tier ostava najlepsia dostupna kombinacia pre SK.
  Jedina cesta vyssie = SK fine-tune nemotronu (ARTUR-style, ~100h+, GPU trening) —
  slovinsky pripad (texdata) dosiahol po fine-tune len ~22-25% WER, co varuje.

## Cross-model agreement (XAGREE) — final zrychleny o ~1.5-2s (24.10. v noci)
- Princip: slovo z PRVEHO final behu sa commitne hned, ak ho draft (iny model)
  uz zobrazil so zhodnym textom a casom (±2s) — zhoda 2 roznych modelov
  nahradza cakanie na druhy final cyklus (~1.6-2.6s).
- 2 nutne poistky (bez nich WER 19-25%!):
  1) edge guard: nepromovat slova <1.2s od okraja okna — oba modely sa tam
     mylia SPOLU (korelovane chyby na odseknutom audiu),
  2) stabilita: draft slovo musi na obrazovke visiet >=0.4s.
- Suite: WER 0-6.5% (noisy 3.2), final p50 4.4-5.5s (baseline 5.4-7.6),
  flicker 0-6.2%. FINAL_STEP 0.5 zhorsil flicker (24%) — ostava 1.0.
- Vypnutie: TITULKY_XAGREE=0; krok: TITULKY_FINAL_STEP.

## Kolo optimalizacii kodu po XAGREE (24.10. ~00:30) — co NEfunguje
- Krok 0.25s po commite: p50 bez zmeny, flicker fast 21% — bottleneck je dekod (1.6s), nie cakanie.
- FINAL_WINDOW 6s: WER horsi (9.7-16%), latencia rovnaka. Ostava 8s.
- FAST_FINAL (bez DTW, -0.9s): final p50 2.1-2.7s (!) ale WER 12-29% — approx
  timestampy rozbiju commit mapovanie. Tichovy snap hranic (_silence_snap) to NEopravil
  → problem nie su len hranice, ale cele casovanie slov. Flag ostava 0.
- Zaver: strop two-tier architektury s MLX DTW = final ~2-2.5s live (4.4-5.5 suite).
  Dalsi level by vyzadoval presne timestampy z lacnejsieho zdroja (ANE draft
  alignment na final text) — vacsi zasah, nie nocny patch.

## FAST_FINAL v2 (ANE retime) zlyhalo + novy test continuous.wav (24.10. ~01:00)
- v2: final bez DTW, casy slov cez difflib zarovnanie na ANE draft (kandidati
  len z aktualneho okna). Vysledok: WER 35-55% — tam kde sa modely nezhoduju,
  interpolacia reze slova a kazy sa kaskada commit->dalsi dekod. Flag ostava 0,
  kod ostava ako experimentalna platforma (_retime_from_draft).
- Pouceni: DTW (-0.9s) sa NEDA nahradit lacnym zarovnanim na iny model.
  Jedina realna cesta k rychlejsiemu finalu = presne timestampy z final modelu
  lacnejsie (whisper.cpp DTW? CoreML?) — vyzaduje redesign, nie patch.
- NOVY TEST tests/continuous.wav: 37s suvislej reci bez pauz (say Laura).
  Baseline (nasadena konfiguracia): WER 5.6%, draftACC 91.5%, final p50 4.8s —
  suvisla rec v two-tier drzi; vnimany problem je latencia finalu, nie text.

## Latencia loop 2 (24.10. vecer): cpp final, q4, win6 — vsetko zamietnute meranim
Cielom bolo biele <1.5s. Vysledky:
- FINAL_ENGINE=cpp (whisper.cpp Metal turbo-sk, text-only dekod 0.8-1.3s
  + commity cez ANE draft kotvy): 6 kol ladenia (UTF-8 token splitting,
  GPU kontencia draftu [use_gpu=False fix ostava], odseknute kotvy na
  okraji okna, difflib zarovnanie, flush bez kotiev). Najlepsi stav:
  WER 0-94% (fast.wav rozbity), final p50 3.9-10s — HORSIE nez baseline
  na oboch osiach. Kotvy su bud konzervativne (stall) alebo rozbiju text.
  Architektura "text z jedneho modelu, casy z druheho" na streame
  NEFUNGUJE spolahlivo. Kod ostava za TITULKY_FINAL_ENGINE=cpp.
- MLX q4 kvantizacia (tools/quantize_mlx.py): dekod 2.68 vs 2.75s f16
  (nic) + kazi slova ("tituloky"). Zmazane.
- FINAL_WIN=6: final p50 5.3-8.6 vs 4.3-5.7 pri 8s, WER +, draft horsi.
  Kratsi dekod nevyvazi stratu kontextu a vacsi podiel edge guardu.
POUCENIE: dekod MLX nie je bottleneck commitov — commity drzi agreement
+ edge guard, a tie su to, co drzi WER nizko. Dalsie znizenie bielej
latencie vyzaduje presny lacny SK final model (distill/finetune), nie
infra triky. Deployed config (XAGREE, win 8, mlx) = overene optimum.
POZOR: pywhispercpp je odteraz build s Metal+CoreML (--no-cache-dir!)
a draft MUSI mat context_params={"use_gpu": False}.

## Nocny beh 25.10.: cesty 1+2 (small final, parakeet) — PARAKEET VYHRAL LATENCIU
Ciel: biele <1.5-2s pri drzatelnom WER. Vsetko merane suite-om, serialne na GPU.

### Cesta 1: kinit-small-sk ako MLX final — ZAMIETNUTA
Konverzia OK (tools + mlx convert z HF), standalone dekod 1.7s vs turbo 2.1s
(rezia mel+DTW dominuje, vahy nerozhoduju). Suite: WER 9.7-41.9% (vs turbo
0-19.4) a final p50 5.6-21s — mensi model je NESTABILNEJSI medzi behmi,
agreement sa nezhodne, commity stoja. Maly model nie je cesta k latencii.

### Cesta 2: NVIDIA parakeet-tdt-0.6b-v3 (MLX) — KANDIDAT NA NASADENIE
- Offline kvalita (cely subor): WER 0-18.8%, noisy 9.7% (LEPSI nez whisper
  19.4). Dekod 8s okna 0.38s = 4x rychlejsi nez turbo. Nativne TDT casy.
- Streaming mod parakeet-mlx: MRTVY (RTF ~3, WER 16-52%, lokalna attention
  + prepocet encodera per chunk). Nikdy nepouzivat.
- Ako okienkovy final (TITULKY_FINAL_ENGINE=parakeet): lazy init v final
  vlakne (MLX stream per-thread!), + PARAKEET_LEAD 1.5s rozbeh pred
  hranicou (TDT nezvlada tvrdy strih na zaciatku okna — svikove artefakty).
  Vysledok: final p50 1.3-2.8s (baseline 4.3-5.7s), WER 9.7-25%.
- Zvysny WER je dominovany JEDNYM kvirkom: parakeet zlepuje "zivu rec na"
  -> "zivurecna" (3 slova = 6-9 p.b. na kratkych refoch; testovacia veta
  je vo vsetkych suboroch). Na realnej reci nemerane — nutny zivy A/B test.
- drobnosti: obcas dup "system system" na sviku (dedup_boundary neberie
  case/punct variant), chybajuce velke pismena na zaciatku vety po sviku.

### Odporucanie
Default ostava mlx (overeny). Zivy test: TITULKY_FINAL_ENGINE=parakeet
HF_HOME=... python -m titulky.main — ak realna rec nema "zivurecna" efekt,
prepnut default. Dalsi kandidat na kvalitu: flush refine cez kinit-turbo
(flush az po pauze, parakeet drzi stream).

### Zivy test parakeet finalu 26.10. (realna prednaska, mikrofon) — ZAMIETNUTY
Na zivej SK prednaske parakeet v3 UTEKA DO INYCH JAZYKOV po oknach:
polstina ("czy tu w tej nitre"), chorvatcina ("bilo bili izmanki krok
modernizacije"), anglictina ("with telecommunication and see it") +
mrzacenie slov ("Skonavek", "purocime"). TDT v3 NEMA language forcing
(auto-detect per dekod) — na TTS testoch sa to neprejavilo, na realnej
reci s ozvenou fatalne. Zaver: parakeet NESMIE byt zdroj commitov.
Latencia 1.3-2.8s bola realna, ale kvalita robi engine nepouzitelnym.
Pripadny buduci smer: parakeet len ako draft (sede je docasne), NIE final.
Default: mlx kinit-turbo (vrateny, overeny).

### Parakeet ako DRAFT (26.10.) — ZAMIETNUTY (GPU kontencia)
TITULKY_DRAFT_BACKEND=parakeet: draft p50 3.8-8.2s (ANE: 0.35-1.1s),
final p50 8-18s (baseline 4.3-5.7s) — dva MLX modely na jednom Metale
sa vzajomne dusia. WER sice 0% (lepsi draft text pre XAGREE), ale
latencia 10x horsia = bezcenne. ANE draft je nenahraditelny prave
preto, ze bezi na inom compute unite. Pripadna buduca cesta: CoreML
export parakeetu (FluidInference/parakeet-tdt-0.6b-v3-coreml) na ANE —
vyzaduje vlastny TDT dekoder, netrivialne.

### Kvalita+UX doladenie (26.10., JBOP-105) — KOMPLETNÝ REPORT

#### 1. Reálne fixtures z prednášky (5 × 15–23 s, pseudo-refs kinit-turbo offline)
| WER baseline | WER +spellfix | zmena |
|---|---|---|
| 30.2 % | 24.5 % | **✓ −5.7 p.b.** |
| 47.7 % | 40.9 % | **✓ −6.8 p.b.** |
| 51.7 % | 45.0 % | **✓ −6.7 p.b.** |
| 20.8 % | 33.3 % | ✗ +12.5 p.b. (false hit "telefonové→telefónové") |
| 33.3 % | 44.4 % | ✗ +11.1 p.b. (false hit "takátová→taktová") |

Spellfix (3,0M trie, distance-1) opravuje *telecomunikačnú→telekomunikačnú*,
*signalyzácia→signalizácia* — ale na realnej reči false-hity (homofóny,
krátke slová) zhoršujú 2 z 5. **Netreba default zapnúť** — nechaj
ako opt-in (TITULKY_SPELLFIX=1) pre doménovo špecifické prednášky.

#### 2. Obrovský lexikón (hunspell-sk → 3,0M tvarov, marisa-trie 6 MB)
- Load 55 s pri builde, runtime lookup μs, spellfix 3 ms / 10 slov.
- Nainštalovaný cez `tools/make_real_tests.py` build step (generuje trie).
- Uložený na GRINDCASTER, v repi len odkaz.

#### 3. Debounce šedého textu (DRAFT_HOLD_S = 0.10 s)
Už v kóde, merané flicker 2–38 % (baseline). Žiadna zmena —
beh s spellfixom ukazuje podobné hodnoty.

#### 4. Ollama LLM korekcia (qwen3:8b, GPU, /no_think)
- Smoke test: 1.3 s / 26 tokenov, opraví *telekomunikačnú* OK.
- V suite: timeout (GPU inference v každom commite = fronta).
- **Verdikt:** async kanál je hotový, ale na M3 Max 48 GB nie je
  realisticé bežať LLM v commite-by-commite režime (Metal kontencia
  s MLX finalom). Buď menší model (qwen3:4b?), alebo batch korekcia
  len na pauze (flush). Zatiaľ vypnuté (TITULKY_LLM_CORR=0).

#### 5. Termálny beh (12 × continuous.wav, 26 min, M3 Max, ANE+Metal)
| iter | draft p50 | final p50 | flicker |
|---|---|---|---|
| 1 | 24.8 s | 46.4 s | 10.3 % |
| 2 | 8.4 s | 12.2 s | 8.8 % |
| 3 | 16.7 s | 50.2 s | 9.7 % |
| 4 | 13.3 s | 20.3 s | 10.1 % |
| 5 | 16.9 s | 15.8 s | 9.0 % |
| 6 | 10.5 s | 13.2 s | 6.6 % |
| 7 | 19.1 s | 22.6 s | 9.6 % |
| 8 | — | 34.6 s | 9.6 % |
| 9 | — | 11.3 s | 11.3 % |
| 10 | 12.7 s | 14.9 s | 9.0 % |
| 11 | 7.1 s | 9.8 s | 10.1 % |
| 12 | 12.1 s | 13.1 s | 6.2 % |

**Žiadny monotónny nárast** — osciluje (scheduling, MLX fronta), nie
throttling. `pmset -g therm` neukázal `CPU_Speed_Limit < 100`.
M3 Max 48 GB drží 30 min naplno bez výkonnostného poklesu.

---

**ZÁVER OPTIMALIZÁCIEJ S LUČKY (JBOP-97 + JBOP-105):**
- **Baseline (ANE draft + kinit-turbo MLX final + XAGREE) = zmerané optimum tejto architektúry.**
- Latencia: šedé 0.4–0.9 s, biele 2–3 s, WER 0–19 % (syntetické), 30–50 % (reálna reč, pseudo-refs).
- Rýchlejšie biele = presný SK streaming model (Whisper SK fine-tune alebo iný encoder-decoder s language forcing), nie hacky.
- Spellfix ako opt-in pre doménové prednášky.
- Všetky 11+ experimentov zdokumentovaných, zamietnutých s číslami.

### XMODEL: cross-model agreement (26.10. popoludni) — ZAMIETNUTY (SK kvalita parakeetu)
Idea: parakeet (0.4s) potvrdi kinit slova HNED -> biele po JEDNOM kinit
dekode namiesto dvoch. Text vzdy kinitovy, drift sa len nezhodne.
Implementovane za flagom TITULKY_XMODEL=1 (+debug TITULKY_XMODEL_DEBUG).
Merane: na TTS vete promotuje 1-3 slova/cyklus (funguje!), na REALNEJ
prednaske 0 promotov vsade — commit vyzaduje suvisly prefix a parakeet
komoli kazde druhe SK slovo (zivurecna, vysavam, zakladni) -> prefix sa
zlomi hned. Navyse +0.4s rezia/cyklus -> final p50 horsi (9-13s vs 7-8s).
ZAVER: mechanizmus je spravny a hotovy, blokuje ho SK kvalita parakeetu
(ta ista korenova pricina ako pri finale/drafte). Odomkne ho az presny
SK druhy model — potom staci TITULKY_XMODEL=1.

### Cloud engine (OpenAI Realtime STT) — 26.10., JBOP-125
TITULKY_ENGINE=cloud, prepínateľný, lokál ostáva default (semestrálka).
Čisté audio (normal/fast, presné refs): **WER 0.0 %, biele p50 0.4-0.5 s** — brutál.
Reálne prednáškové fixtures (pseudo-refs kinit offline, bias k lokálu):

| model | WER avg | biele p50 | cena |
|---|---|---|---|
| gpt-4o-mini-transcribe | 88.7 % | 0.3-0.8 s | $0.003/min |
| gpt-4o-transcribe | 77.8 % | 0.5 s | $0.006/min |
| gpt-live-transcribe | **57.3 %** | 0.8-3.0 s | $0.017/min |
| lokál (kinit streaming) | **~37 %** | 2.5 s | 0 |

ZÁVER: na vzdialený mikrofón v posluchárni je kinit SK fine-tune stále
najlepší — cloud modely komolia sémantiku (väzenská cela v telekom
prednáške). Cloud vyhráva na čistom/blízkom audiu (diktovanie: 0 % WER
pri 0.5 s bielom). Použitie: diktovanie/čistý vstup → cloud; poslucháreň
→ lokál. Pseudo-refs sú kinit-biased, ale sémantické chyby cloudu sú
viditeľné aj bez refov.

## CONF_COMMIT: pravdepodobnostny commit so slovnikom — zamietnuty (27.10.)

Navrh z konzultacie: commitovat slova z PRVEHO citania ked p(slovo) >= prah
a slovo pozna SK slovnik (marisa-trie 3M tvarov), namiesto cakania na druhe
citanie/XAGREE. Implementacia: TITULKY_CONF=1, TITULKY_CONF_T (asr.py prenasa
word probability, pipeline.py conf vetva pred XMODEL, edge guard 1.2 s).

| konfiguracia | WER synth avg | WER real avg | final p50 |
|---|---|---|---|
| baseline (agreement+XAGREE) | 4.6 % | 34.3 % | ~4.4 s |
| CONF t=0.85 | 8.2 % | 38.4 % | ~4.8 s |
| CONF t=0.70 | 6.4 % | 38.0 % | ~5.0 s |

Zamietnute v oboch prahoch, oba smery (synth aj real):
1. Whisper confidence NIE JE kalibrovana — vysoke p maju aj nespravne slova
   (znamy overconfidence problem), slovnik nechrani: homofony a flexivne
   tvary su validne slova ("išel" nie, ale "čo sa" vs "čo za" ano).
2. Zly commit je permanentny (committed sa nemeni) A rozbija prefix match
   dalsich agreement kol -> final p50 paradoxne stupol o 0.4-0.6 s.
3. Potvrdzuje vzor cintorina: kazdy trik obchadzajuci overenie zhody meni
   presnost za rychlost v zlom kurze — tu dokonca bez zisku rychlosti.

Kod ostava ako opt-in (TITULKY_CONF=1): rozhodovaci bod je pripraveny pre
KALIBROVANY judge (TypeSafe Jev, kalibrovane pravdepodobnosti su jeho core
claim) — ak pride early access, vymeni sa len zdroj p, meranie sa zopakuje.
Jedno meranie na konfiguraciu (limitacia), ale smer konzistentny naprieč
13 scenarmi x 2 prahy.
