# Živé slovenské automatické titulky — Benchmark a evaluačná metodika

*Verzia: 1.0 | Jazyk: slovenský | Čakajúce namerané hodnoty: `[DOPLNIT]`*

---

## Obsah

1. [Metodika](#1-metodika)
   1.1 Hardvér a softvér
   1.2 Testovacie dáta
   1.3 Protokol merania
   1.4 Normalizácia textu
2. [Definície metrík](#2-definície-metrík)
3. [Porovnávacia matica systémov](#3-porovnávacia-matica-systémov)
4. [Návrh vizualizácií](#4-návrh-vizualizácií)
5. [Výsledky](#5-výsledky)
6. [Diskusia](#6-diskusia)
7. [Limitácie](#7-limitácie)
8. [Reprodukcia](#8-reprodukcia)
9. [Obhajoba originality](#9-obhajoba-originality)
10. [Literatúra](#10-literatúra)

---

## 1. Metodika

### 1.1 Hardvér a softvér

| Komponent | Hodnota |
|---|---|
| Zariadenie | [DOPLNIT] (napr. MacBook Pro M3 Pro, 18 GB RAM) |
| Čip | Apple Silicon [DOPLNIT] |
| OS | macOS [DOPLNIT] |
| mlx-whisper | verzia [DOPLNIT] |
| Draft model | `whisper-small` MLX (FP16) |
| Final model | `whisper-large-v3-turbo` MLX, beam=5 |
| Azure STT SDK | `azure-cognitiveservices-speech` v[DOPLNIT] |
| Python | [DOPLNIT] |
| Zvukový vstup | simulovaný cez BlackHole / virtuálne audio zariadenie |

Všetky merania lokálnych modelov sú vykonané **offline** (bez sieťovej závislosti). Azure merania zahŕňajú sieťovú odozvu; nameraný RTT k Azure regiónu `[DOPLNIT]` je `[DOPLNIT] ms` (medián cez 100 pingov počas merania).

---

### 1.2 Testovacie dáta

#### 1.2.1 Sedem WAV súborov (interná sada)

| ID | Popis | Dĺžka [s] | Hovorených slov | SNR [dB] | Rečníkov |
|---|---|---|---|---|---|
| `normal` | Štandardný prejav, tiché prostredie | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | 1 |
| `fast` | Rýchla reč (~180-220 sl/min) | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | 1 |
| `slow` | Pomalá reč (~80-100 sl/min) | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | 1 |
| `quiet` | Tichý prejav (nízka hlasitosť, -20 dBFS) | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | 1 |
| `noisy` | Hluk kancelárie v pozadí (~10 dB SNR) | [DOPLNIT] | [DOPLNIT] | ~10 | 1 |
| `pauses` | Dlhé pauzy medzi vetami (2-5 s) | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | 1 |
| `silence` | Čisté ticho / šum bez reči | 30 | 0 | - | 0 |

**Referenčné prepisy:** manuálne anotované, dvomi nezávislými anotátormi.
Inter-annotator agreement: [DOPLNIT] % zhoda na úrovni slov (alebo Cohen kappa = [DOPLNIT]).
Konflikty riešené konsenzom (tretí posudzovateľ).

#### 1.2.2 Externý referenčný korpus

Common Voice SK, verzia [DOPLNIT]: [DOPLNIT] viet, [DOPLNIT] hodín. Slúži ako **offline referenčný bod** — priame porovnanie s live pipeline nie je validné bez prepočtu latenčnej penalizácie.

---

### 1.3 Protokol merania

1. **Simulácia reálneho vstupu:** Každý WAV je prehrávaný systémom po 250 ms blokoch cez virtuálne audio zariadenie — nie batch spracovanie. Systém dostáva audio rovnakým spôsobom ako pri živom mikrofóne.
2. **Opakovanie:** Každá kombinácia (WAV x systém) sa opakuje **N = 5** krát. Reportujeme **medián +/- IQR** (nie priemer +/- sigma, pretože distribúcie latencie sú vychýlené doprava).
3. **Časové pečiatky:** Získavané z monotónnych hodín (`time.monotonic_ns()`); nevyžadujú NTP, keďže meranie prebieha na jednom zariadení.
4. **Referenčné časy slov:** Odvodené z force-alignment na referenčnom prepise pomocou [DOPLNIT metóda, napr. `whisper-timestamped` alebo `MFA`]. Tieto časy definujú `t_spoken_end` pre každé slovo.
5. **Poradie:** Poradie systémov je pri každom behu náhodné (zabraňuje thermal throttling na jednom systéme).
6. **Izolácia:** Ostatné procesy sú zastavené; meranie prebieha v `nice -20` priorite.

---

### 1.4 Normalizácia textu (pre WER a CER)

Nasledujúce kroky sú aplikované **identicky** na hypotézu aj referenčný prepis:

```
1.  Unicode NFKC normalizácia
2.  Prevod na malé písmená (lower())
3.  Odstránenie interpunkcie: [.,!?;:"''(){}]
4.  Normalizácia číslic na slová
        "3" -> "tri"
        "2024" -> "dvetisícdvadsaťštyri"
        "10:30" -> "desať tridsať"
    (knižnica: num2words, locale='sk')
5.  Komprimácia viacnásobných medzier -> jedna medzera
6.  Orezanie okrajových bielych znakov
```

**Kľúčové odôvodnenie — prečo ZACHOVÁVAME diakritiku:**
Na rozdiel od niektorých anglických benchmarkov je v slovenčine diakritika fonologicky distinktívna. "mas" vs. "máš" sú odlišné slová. Zámenou by sme podhodnotili skutočnú chybovosť systému. Interpunkcia je naopak odstraňovaná, pretože Whisper ju generuje nedeterministicky.

---

## 2. Definície metrík

### 2.1 WER — Word Error Rate

**Definícia:** Podiel chybných operácií na celkovom počte referenčných slov, na normalizovanom texte.

```
WER = (S + D + I) / N * 100  [%]
```

kde:
- `S` = substitúcie (nesprávne slovo)
- `D` = vypustenia (chýbajúce slovo)
- `I` = vloženia (nadbytočné slovo)
- `N` = počet slov v referenčnom prepise

Implementácia: Python balík `jiwer`, funkcia `wer()` s normalizáciou podľa §1.4.

> **Akademická konvencia:** Macháček et al. (2023) [1] reportujú WER ako primárnu metriku kvality pre Whisper-Streaming. Naša definícia je priamo kompatibilná — umožňuje priame porovnanie s ich hodnotami pre češtinu a iné jazyky.

---

### 2.2 CER — Character Error Rate

**Definícia:** Rovnaký vzorec ako WER, na úrovni znakov.

```
CER = (S_c + D_c + I_c) / N_c * 100  [%]
```

CER je doplnkovou metrikou vhodnou pre morfologicky bohatú slovenčinu: čiastočne správne slovo (napr. chybná pádová prípona "-om" vs. "-u") má v CER menšiu váhu ako v WER, čo lepšie odráža percepčnú chybovosť.

---

### 2.3 Latencia prvého slova (FWL — First Word Latency)

**Definícia:** Čas od začiatku rečovej aktivity vo vstupe po moment prvého emitovaného slova (vrátane nestabilného draft výstupu).

```
FWL_i = t_emit_first_i - t_speech_start_i   [ms]
```

kde `t_speech_start_i` je detekovaný začiatok rečovej aktivity (VAD onset).

Reportujeme: **FWL p50 a p95** cez všetky vety (okrem `silence`).

> **Poznámka:** Pre dvojúrovňový pipeline sa FWL vzťahuje na **draft vrstvu** (prvý zobrazený text). Final vrstva má vlastnú CCL (§2.4).

---

### 2.4 Latencia dokončenia titulky (CCL — Caption-Complete Latency)

**Definícia:** Čas od akustického konca hovoreného slova po moment jeho **stabilizácie** v zobrazení (posledná zmena, po ktorej sa slovo nemení).

```
CCL_i = t_stable_emit_i - t_spoken_end_i   [ms]
```

kde `t_stable_emit_i` je čas poslednej modifikácie slova `i` v bufferi; `t_spoken_end_i` je čas konca slova z force-alignment.

Reportujeme: **p50 a p95** cez všetky stabilizované slová.

> **Akademická konvencia:** Macháček et al. (2023) [1] merajú "time difference between when the ASR emitted a word and when the corresponding gold word was spoken." Naša CCL je explicitnejšia — meria **stabilizáciu** (finalizáciu), nie prvý výskyt. Polák et al. (2025) [4] ukazujú, že reportovanie p50 aj p95 dáva robustnejší obraz ako samotný priemer (priemer je citlivý na outliere v dlhých pauzách).

---

### 2.5 RTF — Real-Time Factor

**Definícia:** Pomer výpočtového času k dĺžke spracovaného audia.

```
RTF = t_processing / t_audio_duration
```

Systém pracuje v reálnom čase ak `RTF < 1.0`. Pre pipeline reportujeme RTF každej komponenty (`RTF_draft`, `RTF_final`) aj celkový `RTF_pipeline`.

---

### 2.6 Miera blikania (Flicker Rate)

**Definícia:** Podiel emitovaných slov, ktoré boli po zobrazení zmenené alebo vymazané.

```
Flicker = |{w : w bolo emitované a neskôr zmenené alebo vymazané}|
         ---------------------------------------------------------------- * 100  [%]
                    |{w : všetky emitované slová}|
```

Slovo je "zmenené" ak sa jeho textová reprezentácia v ďalšej iterácii zmení alebo je nahradené iným slovom na tej istej pozícii. Slovo je "vymazané" ak zmizne bez náhrady.

> **Interpretácia:** Flicker = 0 % = ideálna stabilita. Draft vrstva má prirodzene vyšší Flicker. Finálna vrstva by mala mať Flicker blízky 0 % vďaka LocalAgreement-2.

---

### 2.7 Miera halucinácie na tichu (Hallucination Rate)

**Definícia:** Počet emitovaných slov za minútu pri vstupe obsahujúcom výlučne ticho/šum (WAV `silence`).

```
HallucRate = |{slová emitované počas silence WAV}|
             ----------------------------------------   [slov/min]
             dĺžka silence WAV [minúty]
```

Ideálna hodnota: **0 slov/min**. VAD blokácia a dynamický noise gate by mali eliminovať halucinácie na tichu.

---

### 2.8 Čistý prínos korekčnej vrstvy (Correction Net Gain)

**Definícia:** Normalizovaný rozdiel medzi slovami opravenými a slovami poškodenými Azure STT korekciou.

```
NetGain = (W_fixed - W_broken) / N_total * 100  [%]
```

kde:
- `W_fixed` = počet slov, kde editačná vzdialenosť od referencie po korekcii **klesla**
- `W_broken` = počet slov, kde editačná vzdialenosť po korekcii **vzrástla**
- `N_total` = celkový počet slov vo vzorke

`NetGain > 0` = korekcia celkovo pomáha; `NetGain < 0` = korekcia škodí.

---

## 3. Porovnávacia matica systémov

### 3.1 Prehľad systémov

| ID | Systém | Popis | Dôvod zahrnutia |
|---|---|---|---|
| **A** | Whisper turbo offline | `whisper-large-v3-turbo`, batch, bez streamingu | **Primárny baseline** — horná hranica turbo kvality bez latenčnej penalizácie |
| **B** | Whisper large-v3 offline | `whisper-large-v3`, batch, bez streamingu | Referencia pre plnú veľkosť; dokumentuje trade-off turbo vs. large |
| **C** | Draft-only streaming | `whisper-small` MLX, chunk 3 s/krok 0.25 s, LocalAgreement-2 | Izoluje príspevok draft vrstvy; baseline pre two-tier |
| **D** | Two-tier (bez Azure) | Draft (C) + final large-v3-turbo MLX, beam=5 | Jadro nášho príspevku bez externej závislosti |
| **E** | Two-tier + Azure korekcia | D + asynchrónna Azure STT korekcia, hunspell-gated, <=10 s | Kompletný systém |
| **F** | Azure STT samotné | Microsoft Azure Cognitive Services STT, streaming | Komerčná alternatíva — externý referenčný bod |
| **G** *(voliteľné)* | faster-whisper CPU | `faster-whisper` large-v3-turbo, CPU, int8 | Kontext pre zariadenia bez Apple Silicon |

**MUST riadky pre akademickú dôveryhodnosť:** A, B, C, D, E, F

**Zdôvodnenie:** Bez offline baselinov (A, B) nie je možné kvantifikovať latenčnú penalizáciu streamingu. Bez komerčnej alternatívy (F) chýba externý referenčný bod. Bez izolovaného draft (C) nie je preukázateľný príspevok final vrstvy.

---

### 3.2 Hlavná porovnávacia tabuľka

Všetky hodnoty sú **medián cez N=5 behov** na 6 WAV súboroch (okrem `silence`).
Dolná šípka = nižšia je lepšia; Horná šípka = vyššia je lepšia; Pomlčka = metrika nie je relevantná.

| Systém | WER [%] | CER [%] | FWL-draft p50 [ms] | CCL p50 [ms] | CCL p95 [ms] | RTF | Flicker [%] | Halluc/min | NetGain [%] |
|---|---|---|---|---|---|---|---|---|---|
| A: turbo offline | [DOPLNIT] | [DOPLNIT] | - | - | - | [DOPLNIT] | - | [DOPLNIT] | - |
| B: large-v3 offline | [DOPLNIT] | [DOPLNIT] | - | - | - | [DOPLNIT] | - | [DOPLNIT] | - |
| C: draft-only | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | - |
| D: two-tier | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | - |
| E: two-tier + Azure | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| F: Azure STT | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | - |
| G: faster-whisper CPU | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | - |

> Systémy A a B nemajú FWL/CCL (batch offline — latencia je nedefinovaná). Flicker pre offline systémy je taktiež nedefinovaný. NetGain je merateľný len pre E (korekčná vrstva).

---

### 3.3 WER podľa testovacích podmienok — systémy A, D, E, F

Všetky hodnoty v %, medián cez N=5 behov.

| Podmienka | A: turbo offline | D: two-tier | E: two-tier + Azure | F: Azure STT |
|---|---|---|---|---|
| normal | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| fast | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| slow | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| quiet | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| noisy | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| pauses | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |

---

### 3.4 Latencia: draft vs. final vrstva (systém D)

| Meranie | Draft [ms] | Final [ms] | Delta Final-Draft [ms] |
|---|---|---|---|
| FWL p50 | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| FWL p95 | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| CCL p50 | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |
| CCL p95 | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] |

Delta vyjadruje latenčnú cenu za pridanie final vrstvy.

---

### 3.5 Korekčná vrstva — detailná analýza (systém E)

| Podmienka | W_fixed | W_broken | NetGain [%] | Komentár |
|---|---|---|---|---|
| normal | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| fast | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| slow | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| quiet | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| noisy | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| pauses | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |
| **spolu** | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | |

---

### 3.6 Halucinácie na tichu (30 s silence WAV)

| Systém | Emitované slová | HallucRate [slov/min] |
|---|---|---|
| C: draft-only (bez VAD blokovania) | [DOPLNIT] | [DOPLNIT] |
| D: two-tier (s VAD + noise gate) | [DOPLNIT] | [DOPLNIT] |
| E: two-tier + Azure | [DOPLNIT] | [DOPLNIT] |
| F: Azure STT | [DOPLNIT] | [DOPLNIT] |

---

## 4. Návrh vizualizácií

### 4.1 Graf 1 — Latency-Accuracy Trade-off (scatter + Pareto)

**Čo vizualizuje:** Pareto frontier medzi presnosťou (WER) a latenciou (CCL p50). Každý systém je bod; Pareto-dominantné systémy sú prepojené červenou čiarou.

**Dáta z harness (pre každý systém A-G):**

```python
{
  "system": "D",
  "CCL_p50_ms": ...,    # z run-trace JSONL, pole word_ccl_ms -> p50
  "CCL_IQR_ms": ...,    # IQR cez N=5 behov
  "WER_pct": ...,       # z eval/generate_table.py
  "WER_std": ...,       # std cez N=5
}
```

**Špecifikácia grafu:**
- Os X: CCL p50 [ms] (lineárna alebo log), Os Y: WER [%]
- Chybové úsečky: +/-IQR (X), +/-std (Y)
- Pareto frontier: červená čiara spájajúca Pareto-dominantné body
- Šedá zóna: CCL > 2000 ms (nad hranicou akceptovateľnosti pre live titulky)
- Referenčná vodorovná čiara: WER = WER_offline_turbo (offline ceiling)
- Body pomenované ID systémov (A-G)

---

### 4.2 Graf 2 — CDF latencie (per-word CCL)

**Čo vizualizuje:** Empirická CDF per-word CCL pre systémy C, D, E, F — P(slovo stabilizované do X ms).

**Dáta z harness:**

```python
# Pre každý systém: zoznam všetkých CCL hodnôt per-word
# z run-trace JSONL, pole word_ccl_ms
# Minimálne ~500 slov na systém pre spoľahlivú distribúciu
ccl_values = {
  "C_draft":          [...],
  "D_two_tier":       [...],
  "E_two_tier_azure": [...],
  "F_azure":          [...],
}
```

**Špecifikácia grafu:**
- Os X: CCL [ms], logaritmická škála odporúčaná (rozsah 50-5000 ms)
- Os Y: P(CCL <= x), rozsah [0, 1]
- 4 krivky, rôznymi farbami (modrá=C, zelená=D, tmavozelená=E, oranžová=F)
- Vertikálne referenčné čiary: 500 ms (real-time target), 1000 ms, 2000 ms (broadcast limit)
- Legenda s hodnotami p50 a p95 pre každý systém

---

### 4.3 Graf 3 — WER podľa testovacích podmienok (skupinový bar chart)

**Čo vizualizuje:** Odolnosť systémov voči rôznym akustickým podmienkam — kde pipeline degraduje a kde prekonáva baseline.

**Dáta z harness (z tabuľky §3.3):**

```python
data = {
  "conditions": ["normal", "fast", "slow", "quiet", "noisy", "pauses"],
  "systems": {
    "A_turbo_offline":  [...],   # sivá
    "D_two_tier":       [...],   # modrá
    "E_two_tier_azure": [...],   # zelená
    "F_azure":          [...],   # oranžová
  },
  "std": { ... }   # +/-1 std cez N=5 behov pre chybové úsečky
}
```

**Špecifikácia grafu:**
- Skupinový bar chart, 6 skupín (podmienky), 4 bary v každej
- Chybové úsečky: +/-1 std
- Horizontálna čiara: WER = WER_offline_turbo_A (referenčná horná hranica kvality)
- Podmienka `silence` vynechaná (WER nedefinovaný bez reči)

---

## 5. Výsledky

### 5.1 Výsledky na Common Voice SK (offline referencia)

| Model | WER [%] | CER [%] | Počet viet | Poznámka |
|---|---|---|---|---|
| `whisper-small` | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | Naše meranie |
| `whisper-large-v3-turbo` | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | Naše meranie |
| `whisper-large-v3` | [DOPLNIT] | [DOPLNIT] | [DOPLNIT] | Naše meranie |

Tieto čísla sú offline referenciou. Pre porovnanie s live pipeline je potrebné zohľadniť, že streaming zaraďuje latenčnú penalizáciu a segmentácia ovplyvňuje WER.

### 5.2 Hlavné výsledky — streaming

Pozri §3.2 — Hlavná porovnávacia tabuľka.

---

## 6. Diskusia

### 6.1 Prečo turbo prekonáva large-v3 na spontánnej slovenčine?

**Výzva:** Naše merania ukazujú WER(turbo-offline) = [DOPLNIT] % vs. WER(large-v3-offline) = [DOPLNIT] % na spontánnom prejave — napriek tomu, že large-v3 je väčší model.

Diskutujte:
- Tréningová distribúcia: turbo bol distilovaný z large-v3, čo mohlo zlepšiť generalizáciu na spontánnu reč
- Vplyv MLX kvantizácie na oba modely (FP16 vs. INT4)
- Segmentácia pri dlhých spontánnych vetách — large-v3 môže byť citlivejší na dĺžku kontextu
- Porovnanie s číslami Macháček et al. (2023) [1] pre češtinu (typologicky blízky jazyk)

### 6.2 Trade-off draft vs. final vrstva

**Výzva:** Porovnajte systémy C (draft-only) a D (two-tier):
- Aká je latenčná cena Delta CCL p50 za pridanie final vrstvy? (pozri §3.4)
- Pri akej hodnote Flicker Rate je two-tier jednoznačne lepšou voľbou?
- Ako sa mení trade-off medzi podmienkami (napr. `noisy` vs. `normal`)?

### 6.3 Analýza korekčnej vrstvy Azure

**Výzva:** Za akých podmienok má Azure korekcia pozitívny NetGain a kedy je záporný?

Hypotéza: hunspell-gated filtrácia by mala blokovať nesprávne opravy a udržať NetGain > 0.
- Verifikujte hypotézu voči tabuľke §3.5
- Analyzujte: v akom % prípadov hunspell zablokoval navrhnutú opravu?
- Aký je vplyv latency Azure na CCL p95 systému E oproti D (Delta §3.4)?

### 6.4 Porovnanie s komerčnou alternatívou (Azure STT)

**Výzva:** Azure STT (systém F) je komerčné riešenie s SLA. Porovnajte:
- Kde náš pipeline prekonáva Azure: offline prevádzka, GDPR súlad, predvídateľná latencia, nulový prevádzkový náklad po nasadení
- Kde Azure prekonáva náš pipeline: WER na hlučnom vstupe, custom vocabulary, SLA
- Finančný náklad Azure STT pri 8-hodinovom streame vs. lokálny inference

---

## 7. Limitácie

1. **Rozsah testovacích dát:** 7 WAV súborov (~[DOPLNIT] minút reči) je dostatočných pre proof-of-concept, ale nedostatočných pre generalizáciu. Seriózny benchmark vyžaduje aspoň [DOPLNIT] hodín slovenčiny s diverzifikovanými rečníkmi, akcentmi, a doménami.

2. **Referenčné prepisy:** Manuálna anotácia je nákladná. Inter-annotator agreement [DOPLNIT] odráža mieru subjektivity najmä pri spontánnych výplnkových slovách ("ehm", "no teda") — normalizácia tieto nejednoznačnosti čiastočne rieši.

3. **Single-machine meranie:** RTF a latencia sú špecifické pre Apple Silicon [DOPLNIT]. MLX inference je natívne optimalizovaná pre Apple Silicon — na iných platformách by bol potrebný iný inference engine.

4. **Sieťová závislosť Azure:** Latencia zahŕňa RTT k serveru [DOPLNIT] ms. Počas demo prezentácie môže sieťová kvalita fluktuovať a ovplyvniť CCL p95 systému E.

5. **Absencia percepčnej evaluácie:** Benchmark meria automatické metriky. Flicker Rate je proxy pre percepčnú stabilitu, ale nie ekvivalent MOS alebo hodnotenia čitateľnosti ľudskými posudzovateľmi.

6. **VAD kalibrácia:** Prahové hodnoty dynamického noise gate sú nastavené na základe trénovacej sady — môžu byť suboptimálne pre prostredia s iným charakterom šumu (hudba, reverb).

7. **Jazykové obmedzenie:** Systém a benchmark sú optimalizované pre slovenčinu. Výsledky pre iné jazyky nie sú súčasťou tejto práce.

---

## 8. Reprodukcia

### Požiadavky

```bash
# Python 3.11+, Apple Silicon (MLX vyžaduje macOS 13.3+ na M1+)
pip install mlx-whisper jiwer num2words azure-cognitiveservices-speech \
            webrtcvad scipy numpy matplotlib seaborn

# Premenné prostredia pre Azure (len pre systém E a F)
export AZURE_SPEECH_KEY="[DOPLNIT]"
export AZURE_SPEECH_REGION="[DOPLNIT]"
```

### Štruktúra repozitára

```
.
├── test_suite.py
├── test_audio/
│   ├── normal.wav  fast.wav  slow.wav  quiet.wav  noisy.wav  pauses.wav  silence.wav
│   └── references/          # referenčné prepisy (.txt)
├── pipeline/
│   ├── two_tier.py          # dvojúrovňový plánovač
│   ├── agreement.py         # LocalAgreement-2
│   └── vad_gate.py          # dynamický noise gate
├── correction/
│   └── azure_layer.py       # hunspell-gated Azure korekcia
├── eval/
│   ├── normalize.py         # normalizácia textu (§1.4)
│   ├── metrics.py           # WER, CER, CCL, Flicker, Halluc
│   ├── generate_table.py    # generuje §3.2 tabuľku
│   ├── plot_tradeoff.py     # Graf 1 (§4.1)
│   ├── plot_cdf.py          # Graf 2 (§4.2)
│   ├── plot_wer_conditions.py  # Graf 3 (§4.3)
│   └── verify_trace.py      # integrita JSONL trace
└── results/
    └── trace.jsonl           # run-trace log
```

### Spustenie

```bash
# Všetky systémy, všetky WAV, N=5 opakovaní
python test_suite.py \
  --all-systems \
  --wav-dir ./test_audio/ \
  --n-runs 5 \
  --output results/

# Jeden systém (napr. E)
python test_suite.py \
  --system two-tier-azure \
  --wav-dir ./test_audio/ \
  --n-runs 5 \
  --output results/

# Tabuľky a grafy
python eval/generate_table.py --results-dir results/ --output BENCHMARK_results.md
python eval/plot_tradeoff.py  --results-dir results/ --output figures/fig1_tradeoff.pdf
python eval/plot_cdf.py       --results-dir results/ --output figures/fig2_cdf.pdf
python eval/plot_wer_conditions.py --results-dir results/ --output figures/fig3_wer_conditions.pdf

# Verifikácia reprodukovateľnosti
python eval/verify_trace.py --trace results/trace.jsonl
```

### Run-trace formát (JSONL)

Každý beh pridá jeden JSON záznam:

```json
{
  "timestamp": "2026-10-22T10:00:00Z",
  "git_commit": "abc1234",
  "system": "two-tier-azure",
  "wav": "normal",
  "run_index": 1,
  "wav_sha256": "...",
  "WER_pct": 9.2,
  "CER_pct": 3.1,
  "FWL_draft_p50_ms": 320,
  "CCL_final_p50_ms": 1250,
  "CCL_final_p95_ms": 2800,
  "RTF_pipeline": 0.42,
  "flicker_pct": 2.1,
  "halluc_per_min": 0.0,
  "net_gain_pct": 1.8,
  "word_ccl_ms": [480, 1120, 950, ...]
}
```

---

## 9. Obhajoba originality

### 9.1 Off-the-shelf komponenty (nie naša práca)

| Komponent | Zdroj | Licencia |
|---|---|---|
| Whisper modely (small, large-v3, large-v3-turbo) | OpenAI [Radford et al., 2022] [5] | MIT |
| MLX Whisper inference engine | Apple ml-explore/mlx-whisper | MIT |
| LocalAgreement-2 algoritmus | Liu et al. (2020) [6]; impl. UFAL [Macháček et al., 2023] [1] | MIT |
| Azure Cognitive Services Speech-to-Text | Microsoft | Komerčná |
| Hunspell slovenský slovník | Mozilla / Slovak NLP community | LGPL |
| VAD (silero-vad / webrtcvad) | Silero / WebRTC | MIT / BSD |
| jiwer (WER výpočet) | jitsi | Apache 2.0 |
| num2words | community | LGPL |

### 9.2 Naše vlastné príspevky

| Príspevok | Popis a originalita | Kde v kóde |
|---|---|---|
| **Dvojúrovňové plánovanie (two-tier scheduling)** | Asynchrónna architektúra kombinujúca rýchly draft model (small, 3 s/0.25 s) a presný final model (large-v3-turbo, beam=5) s koordinovaným prepínacím mechanizmom. Draft a final bežia paralelne; final prepisuje draft len pri výraznom zlepšení konfidencii. | `pipeline/two_tier.py` |
| **Tuning LocalAgreement-2 pre slovenčinu** | Experimentálne ladenie parametra k=2 a chunk size pre optimálny trade-off latencia/stabilita špecificky pre spontánnu slovenčinu. Zahŕňa meranie dopadu rôznych k hodnôt na Flicker Rate a CCL. | `pipeline/agreement.py`, `experiments/la_tuning/` |
| **Dynamický noise gate s meranými prahmi** | Adaptívne prahové hodnoty SNR odvodené z distribúcie vstupného audia — nie pevné konštanty. Prah sa aktualizuje každých N sekúnd podľa historického pozadia. Validované na `quiet` a `noisy` podmienkach. | `pipeline/vad_gate.py` |
| **Hunspell-gated cross-ASR korekcia** | Originálny filter: Azure STT navrhne alternatívu -> prijatá len ak (a) hunspell akceptuje slovo ako platné slovenské slovo a (b) ide o 1:1 substitúciu <=10 s staré. Zabraňuje regresii na správnych slovách. | `correction/azure_layer.py` |
| **Eval suite pre live streaming** | 7-WAV testovacia sada s referenčnými prepismi a harness `test_suite.py` merajúci WER, CCL, Flicker, HallucRate v reálnom čase (nie batch). Run-trace JSONL so per-commit artefaktmi umožňuje longitudinálne sledovanie kvality. | `eval/`, `test_suite.py` |
| **Normalizačný protokol WER/CER pre slovenčinu** | Formalizovaná sada normalizačných krokov s odôvodnením zachovania diakritiky (§1.4) — aplikovateľná ako štandard pre budúce slovenské ASR benchmarky. | `eval/normalize.py`, §1.4 |

---

## 10. Literatúra

[1] **Macháček, D., Dabre, R., Bojar, O. (2023).** Turning Whisper into Real-Time Transcription System. *Proceedings of the 13th IJCNLP, System Demonstrations*, pp. 17-24. Association for Computational Linguistics.
URL: https://aclanthology.org/2023.ijcnlp-demo.3 | arXiv: https://arxiv.org/abs/2307.14743
*(Definícia latencie pre streaming ASR; LocalAgreement-2 policy; WER reporting konvencia.)*

[2] **Ma, X., et al. (2020).** SimulEval: An Evaluation Toolkit for Simultaneous Translation. *Proceedings of EMNLP 2020: System Demonstrations*, pp. 144-150. ACL.
URL: https://aclanthology.org/2020.emnlp-demos.19
*(Štandardný toolkit pre latenčné metriky: Average Lagging, CL, DAL — naša CCL je kompatibilná s ich word-level latency definíciou.)*

[3] **Ma, M., et al. (2019).** STACL: Simultaneous Translation with Implicit Anticipation and Controllable Latency. *Proceedings of ACL 2019*, pp. 3025-3036.
*(Zavedenie Average Lagging metriky — základ, na ktorom SimulEval stojí.)*

[4] **Polák, P., Papi, S., Bentivogli, L., Bojar, O. (2025).** Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text Translation. *arXiv:2509.17349*.
URL: https://arxiv.org/abs/2509.17349
*(Kritická analýza latenčných metrík; YAAL; zdôvodnenie prečo reportujeme p50 aj p95 namiesto samotného priemeru.)*

[5] **Radford, A., et al. (2022).** Robust Speech Recognition via Large-Scale Weak Supervision. *arXiv:2212.04356*.
*(Whisper — základný model.)*

[6] **Liu, Y., et al. (2020).** Low-Latency Sequence-to-Sequence Speech Recognition and Translation by Partial Hypothesis Selection. *Proceedings of Interspeech 2020*, pp. 3620-3624.
*(LocalAgreement algoritmus — základ stability final vrstvy.)*

---

*Dokument vygenerovaný: [DOPLNIT dátum] | Git commit: [DOPLNIT hash] | Zodpovedná osoba: [DOPLNIT]*
*Repozitár: [DOPLNIT URL] | Testovacia sada SHA-256: [DOPLNIT]*
