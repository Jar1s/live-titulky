# Trade-off: krok final vrstvy × latencia × presnosť

Vstup: demo_sk.wav (47.6 s, syntetická SK reč), MacBook Apple Silicon.

| final krok | draft krok | draft lat p50 | draft lat min | final lat p50 | WER | presnosť | flicker |
|---|---|---|---|---|---|---|---|
| 1.0 s | 0.25 s | 3.11 s | 0.33 s | 4.07 s | 3.6% | 96.4% | 19.7% |
| 1.5 s | 0.25 s | 2.77 s | 0.29 s | 3.90 s | 4.8% | 95.2% | 20.7% |
| 2.0 s | 0.5 s | 2.08 s | 0.35 s | 4.24 s | 2.4% | 97.6% | 6.7% |
| 3.0 s | 0.5 s | 2.00 s | 0.22 s | 5.89 s | 2.4% | 97.6% | 5.9% |
