# Plan: Quasnir-10B, dicht, mit Denkzeit, RLVR, RSI und Subagenten

Stand 20.10.2026. Dieser Plan ersetzt `docs/PLAN_100B_MOE.md` als Hauptziel. Das 108B-MoE bleibt eine Option für
später, wenn breites Faktenwissen gebraucht wird.

## 1. Warum 10B statt 108B-A26B

| | 10B dicht | 108B-A26B MoE |
|---|---|---|
| Vortraining (Annahmen wie im MoE-Plan) | 200–300 Mrd. Tokens, ≈ 18.500–27.800 $ | ≈ 56.500 $ |
| Inferenz | eine GPU (FP8, 10,1 GB Gewichte) | mehrere GPUs (≈ 108 GB in FP8) |
| Subagenten parallel auf einer H100 (je 8k Kontext) | ≈ 42 | – |
| RL-Rollouts je Dollar | ≈ 2,6× mehr (10 statt 26 Mrd. aktiv) | 1× |
| Iterationszeit | kurz; jeder Schritt passt auf 2 GPUs | lang |

Wo Denkzeit Parameter ersetzt:
- **Prüfbare Aufgaben** (Code mit Tests, Mathe, Logik):
  - Snell et al. 2024: Bei gleichen FLOPs schlägt optimal eingesetzte Testzeit-Rechenleistung ein 14-mal größeres
    Modell, solange das kleine Modell die Aufgabe schon gelegentlich löst.
  - rStar-Math: Ein 7B-Modell liegt mit Suche und Selbstverbesserung auf MATH bei 90,0 %, über o1-preview.
- **Breites Faktenwissen** (MMLU-Pro, GPQA, SimpleQA): Hier hilft Denkzeit wenig. Wissen braucht Parameter. Dort
  bleibt ein 10B hinter Frontier-Modellen.
- **Kleine Modelle, die große schlagen:** Solche Ergebnisse gibt es fast immer auf ausgewählten Reasoning-Benchmarks.
  Das kleine Modell bekommt dabei oft viel mehr Testzeit-Rechenleistung oder hat von einem großen Lehrer gelernt
  (zum Beispiel die R1-Distill-Reihe vom 671B-Modell DeepSeek-R1).

## 2. Architektur

| Teil | Wert |
|---|---|
| Schichten, Breite | 44 Schichten, d_model 4096 |
| Attention | 32 Query-Köpfe, 8 KV-Köpfe (GQA), head_dim 128, RoPE, RMSNorm |
| FFN | SwiGLU, hidden 14.336 |
| Vokabular | 65.536 (Tokenizer aus Phase M0), Ein- und Ausgabe getrennt |
| Parameter | 9,60 Mrd. ohne Embeddings + 0,54 Mrd. Embeddings = **10,13 Mrd.** |
| KV-Cache | 180.224 Byte je Token (BF16) |

## 3. Rechnung (FP8; B300 4,5 PFLOP/s, MFU 30 %, 7,40 $/h, wie im MoE-Plan)

| Tokens | FLOP | GPU-Stunden | Kosten | 2 B300 | 8 B300 |
|---|---|---|---|---|---|
| 100 Mrd. | 6,08e21 | 1.251 | ≈ 9.300 $ | 26 Tage | 6,5 Tage |
| 200 Mrd. | 1,22e22 | 2.502 | ≈ 18.500 $ | 52 Tage | 13 Tage |
| 300 Mrd. | 1,82e22 | 3.753 | ≈ 27.800 $ | 78 Tage | 19,5 Tage |

- **Ziel:** 300 Mrd. Tokens aus Corpus v6. Hochwertige Teile dürfen bis zu 4 Epochen wiederholt werden.
- **Reserve:** 15 % für RLVR und RSI. Die Rollouts sind überwiegend Inferenz und werden vorher gemessen.
- **Vor dem Funding** ersetzt ein kurzer Messlauf (Abschnitt 5) die Annahme 4,5 PFLOP/s × 30 % durch einen
  gemessenen Wert.

## 4. Inferenz und Subagenten (FP8-Gewichte 10,1 GB, KV in BF16)

| GPU | Frei für KV (≈) | Tokens im Cache | Agenten × 8k |
|---|---|---|---|
| H100 80 GB | 63 GB | ≈ 349.000 | 42 |
| A100 80 GB (BF16, kein FP8) | 53 GB | ≈ 295.000 | 36 |
| L40S 48 GB | 33 GB | ≈ 182.000 | 22 |
| RTX 5090 32 GB | 18 GB | ≈ 99.000 | 12 |

Rollen, Workflows und Kontrollen wie in `docs/PLAN_SUBAGENTS.md`. Dort gilt jetzt: Hauptmodell 10B statt 13–15B,
Untermodell ≈ 1B.

## 5. Reihenfolge

1. **1B messen** (Auftrag 1: greedy, Denkzeit, Oracle) und **1B-RSI**, wenn `oracle_any_pass` ≥ 5 %. Das belegt die
   Lernkurve billig.
2. **10B-Messlauf**, wenige GPU-Stunden: echte Tokens/s und MFU auf B300 oder H200, Speicher, Stabilität.
3. **Corpus v6 voll, Tokenizer 64k, Wert-Tor** für jede Quelle.
4. **Vortraining Quasnir-10B**, 200–300 Mrd. Tokens, mit Funding.
5. **RLVR und Denkzeit**, dann **kontrolliertes RSI** mit Gates.
6. **Subagenten** auf einer GPU. Danach bauen Rouge und Darus auf dem vortrainierten Quasnir auf.

Tor je Version: Odds Ratio ≥ 2 am harten Test gegen den bisherigen Champion, val höchstens +2 %, Owner-Ja.

Quellen:
- Snell et al., „Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters“,
  arXiv 2408.03314.
- Guan et al., rStar-Math, ICML 2025, arXiv 2501.04519.
- Muennighoff et al., „Scaling Data-Constrained Language Models“, 2023: bis zu 4 Epochen fast so gut wie neue Daten.
