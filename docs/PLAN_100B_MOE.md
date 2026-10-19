# Plan: über 100 Mrd. Parameter, 25–30 Mrd. aktiv, mit 1–2 B300

Stand 19.10.2026. Alle Kosten sind Rechnungen aus Annahmen. Gemessen ist davon noch nichts. Ein Messtag auf einer B300
ersetzt die Annahmen, bevor Geld in Phase M1 fließt.

## 1. Kernaussagen

1. **Speicher ist nicht das Problem, Rechenzeit ist es.** Ein Modell mit 108 Mrd. Parametern
   - passt für Inferenz auf **eine** B300: BF16 216 GB, FP8 108 GB, NVFP4 etwa 61 GB, bei 288 GB HBM;
   - braucht zum Training etwa 1,1–1,7 TB (Gewichte, Master-Kopie, Gradienten, Optimizer), also 4–8 B300.
2. **Kosten folgen den GPU-Stunden, nicht der GPU-Zahl.** 2 GPUs über 5 Monate kosten so viel wie 8 GPUs über
   5 Wochen. Kurze 8×B300-Blöcke mieten ist gleich teuer und 4× schneller.
3. **Mit 1–2 B300 geht es über den Weg Branch-Train-MiX (BTX).** Dichte Äste werden einzeln trainiert; jeder passt auf
   1–2 GPUs. Danach werden sie zu einem MoE gemischt. Nur der letzte Schritt braucht kurz 8 GPUs, und auch der ist
   optional.
4. **Die Größenordnung liegt bei 17.000 bis 57.000 $, nicht bei 1 Mio.** NVFP4 ergibt das untere Ende, FP8 das obere
   (Abschnitt 4). Das liegt weit unter 1 Mio. €, aber weit über den aktuellen 300 €. Abschnitt 6 beschreibt die
   Finanzierung.

## 2. Zielarchitektur „Quasnir-MoE 108B-A26B"

| Teil | Wert |
|---|---|
| Schichten, Breite | 48 Schichten, d_model 5120 |
| Attention | 40 Query-Köpfe, 8 KV-Köpfe (GQA), head_dim 128, RoPE, RMSNorm |
| Experten je Schicht | 44 geroutete (top-8) + 2 gemeinsame, SwiGLU mit hidden 3072 (47,2 Mio. Parameter je Experte) |
| Vokabular | 65.536 (neuer Tokenizer, Phase M0), Ein- und Ausgabe getrennt |
| Parameter gesamt | Attention 3,02 Mrd. + Embeddings 0,67 Mrd. + Experten 104,2 Mrd. = **≈ 108 Mrd.** |
| Parameter aktiv je Token | 3,02 + 0,67 + 10 × 47,2 Mio. × 48 = **≈ 26,3 Mrd.** |

Feinkörnige Experten mit gemeinsamen Experten folgen DeepSeekMoE ([2401.06066](https://arxiv.org/abs/2401.06066)).
Die 44 gerouteten Experten entstehen aus 11 Ästen, deren FFN (hidden 12.288) in je 4 Segmente zerlegt wird. SwiGLU
zerfällt additiv über die Hidden-Einheiten. Deshalb reproduziert ein Router, der die 4 Segmente eines Astes wählt, diesen
Ast exakt. Top-8 entspricht damit „zwei Äste pro Token".

## 3. Weg mit 1–2 B300: BTX statt Training von Grund auf

| Phase | Was | Tokens | Speicher | Passt auf |
|---|---|---|---|---|
| M0 | Corpus v6 (`data/build_corpus_v6.py`), Tokenizer 64k | – | CPU | CPU-Pods |
| M1 | Dichter Stamm „Quasnir-13B": 12,75 Mrd. Parameter (gleiche Attention, FFN hidden 12.288) | 100 Mrd. | ≈ 204 GB (AdamW gemischt) | 1 B300 knapp, 2 B300 sicher |
| M2 | 11 Äste vom Stamm, je eine Domäne: Python, JS/TS, C/C++/Rust/Go, JVM, übrige Sprachen, Mathe, Reasoning-Spuren, Fachartikel, Bücher/Langtexte, Recht/Behörden, Allgemein | 11 × 20 Mrd. | je ≈ 204 GB | je 1–2 B300; unterbrechbar, auch Spot |
| M3 | Mischen: Attention und Embeddings gemittelt, 44 Segmente als Experten, 2 gemeinsame als Mittelwert; zuerst nur Router, Attention und gemeinsame Experten trainieren (Experten eingefroren) | 60 Mrd. | ≈ 208 GB eingefroren + ≈ 132 GB trainierbar (8,2 Mrd. Parameter) + Aktivierungen | 2 B300 |
| M3b | Optional: alle Parameter kurz nachtrainieren | 20 Mrd. | ≈ 1,7 TB | 8 B300, ca. 1–3 Tage |

BTX stammt von Meta (Sukhbaatar et al., [2403.07816](https://arxiv.org/abs/2403.07816)). Dort war es bei gleicher
Rechenzeit besser als Sparse Upcycling ([2212.05055](https://arxiv.org/abs/2212.05055)). Für uns zählen drei
Eigenschaften:
- Jeder Ast ist ein normales dichtes Modell. Kein Expert-Parallel nötig, kein Interconnect, Abbruch kostet nur den
  laufenden Ast.
- Äste laufen nacheinander auf denselben 1–2 GPUs oder gleichzeitig auf billigeren Karten.
- Ein schlechter Ast fällt am Tor G2 heraus, bevor er das MoE verdirbt.

Das heutige 1B liefert dafür keine Gewichte. Breite 1920 statt 5120 lässt sich nicht funktionserhaltend übertragen.
Es bleibt der Prüfstand für Daten, Mix und Evals, und es läuft nach Owner-Entscheidung zuerst weiter.

## 4. Rechnung

Annahmen (vor M1 messen):
- **Hardware:** B300 dicht FP8 ≥ 4,5 PFLOP/s und NVFP4 ≈ 15 PFLOP/s. Quellen nennen für FP8 4,5–7 PFLOP/s; gerechnet
  wird konservativ mit 4,5.
- **Auslastung:** MFU 30 %.
- **FLOP pro Token:** 6 × N_aktiv. Bei 4k Kontext kommen etwa 8 % Attention dazu.
- **Preis:** RunPod B300 6,94–7,89 $/h (Stand Sept. 2026); gerechnet mit 7,40 $/h.

| Phase | FLOP | FP8 GPU-h | NVFP4 GPU-h |
|---|---|---|---|
| M1 Stamm 12,75 Mrd. × 100 Mrd. Tokens | 7,65e21 | 1.574 | 472 |
| M2 11 Äste × 20 Mrd. Tokens | 1,68e22 | 3.463 | 1.039 |
| M3 Mischen 26,3 Mrd. aktiv × 60 Mrd. Tokens | 9,48e21 | 1.951 | 585 |
| M3b optional, 20 Mrd. Tokens | 3,16e21 | 650 | 195 |
| **Summe (400 Mrd. Trainings-Tokens)** | | **7.639 ≈ 56.500 $** | **2.292 ≈ 17.000 $** |
| Wandzeit mit 2 B300 / mit 8 B300 | | 159 / 40 Tage | 48 / 12 Tage |

Zum Vergleich: dasselbe MoE direkt von Grund auf mit 300 Mrd. Tokens braucht in FP8 9.756 GPU-h (≈ 72.000 $), und für
den Speicher ab Tag 1 8 GPUs.

## 5. Hebel, nach Wirkung

1. **NVFP4 auf B300.** Die Rechenleistung ist etwa 3× FP8. NVIDIA hat ein 12-Mrd.-Modell über 10 Billionen Tokens in
   NVFP4 vortrainiert, nahe an FP8 ([2509.25149](https://arxiv.org/abs/2509.25149)). Einschränkung: Unser Backend
   `accel` ist JAX. NVFP4 läuft heute praktisch über PyTorch mit Transformer Engine. Der Messtag klärt, ob FP8 in JAX
   auf B300 schnell ist und ob sich die Portierung des MoE-Blocks lohnt.
2. **Daten statt FLOPs.**
   - AAA-Filter.
   - Bis zu 4 Epochen auf den besten Teilen kosten kaum Qualität (Muennighoff et al.,
     [2305.16264](https://arxiv.org/abs/2305.16264)). 125–150 Mrd. einzigartige Top-Tokens tragen also 400–500 Mrd.
     Trainings-Tokens.
   - Verifizierte Reasoning-Spuren (`data/build_reasoning.py`) gezielt in die Äste Mathe und Reasoning.
3. **Multi-Token-Prediction** (DeepSeek-V3, [2412.19437](https://arxiv.org/abs/2412.19437)): mehr Signal pro Token;
   bei der Inferenz als spekulatives Dekodieren nutzbar.
4. **Eingefrorene Experten in M3.** Nur etwa 8,2 Mrd. Parameter (Attention, Embeddings, gemeinsame Experten) tragen
   Optimizer-Zustände. So passt das MoE-Training
   auf 2 B300.
5. **Tore pro Ast.** Ein Ast geht nur ins MoE, wenn er seine Domäne gegen den Stamm messbar verbessert. Schlechte
   Rechenzeit wird nicht weitergetragen.
6. **Langtexte ganz behalten.** v6 speichert Bücher und Artikel bis 3 Mio. Zeichen am Stück. Das ist das Material für
   die spätere Kontextverlängerung.
7. **Inferenz auf einer GPU.** In NVFP4 braucht das Modell etwa 61 GB, die Bereitstellung kostet also eine Karte.

## 6. Geld und Reihenfolge

| Schritt | Kosten | Deckel |
|---|---|---|
| Grok Auftrag 1: 1B auf dem Pod messen (`docs/PROMPT_GROK_RUNPOD.md`) | CPU-Pod, wenige $ | 5 $ |
| Grok Auftrag 2: Corpus v6, Stufe A (≈ 300 Mrd. Token-Budget) | CPU-Pods; Speicher ≈ 1 TB ≈ 50 $/Monat | 60 $ CPU |
| Messtag 1× B300, ca. 3 h: FP8/NVFP4-Durchsatz des MoE-Blocks | ≈ 25 $ | 30 $ |
| M1–M3 | 17.000 $ (NVFP4) bis 57.000 $ (FP8) | eigene Freigabe je Phase |

Die 300 € decken die ersten drei Zeilen. Für M1–M3 kommen drei Wege in Frage:
- **EuroHPC AI Factories:** kostenlose GPU-Stunden; den Antrag hat Grok vorbereitet.
- **Startup- und Cloud-Programme:** NVIDIA Inception, Credits großer Clouds.
- **Ein zahlender VIP-Kunde.**

Jede Phase startet erst nach dem Tor der vorigen und mit eigener Budgetfreigabe.

## 7. Tore

| Tor | Bedingung |
|---|---|
| G0 | 1B gemessen (HumanEval, MBPP, hart, ARC-AGI-2). Dichtes 1B weiter bis etwa 19 Mrd. Tokens (Owner-Entscheidung). |
| G1 | Stamm 13B gegen das beste 1B: Odds Ratio ≥ 2 am harten Test (unteres 95-%-Band), val höchstens +2 %. Das ist die Regel „jede Version mindestens 2× besser". |
| G2 | Jeder Ast verbessert die val seiner Domäne gegen den Stamm und verliert höchstens 2 % auf der allgemeinen val. Sonst fällt er heraus, und seine Segmente kommen aus dem Stamm. |
| G3 | MoE gegen den Stamm: Odds Ratio ≥ 2 am harten Test, kein Domänenverlust gegen den besten Ast. |
| Nach jedem Tor | Einstufung champion / frontier / dangerous. Ab frontier wird der Owner informiert. Dangerous wird behalten, RSI angehalten, Weitergabe nur an VIP-Firmen. |

## 8. Risiken

- **Verfügbarkeit:** B300 sind knapp. Ausweichen auf B200 kostet etwa 1,5–2× GPU-Stunden.
- **Auslastung:** Liegt die MFU unter 30 %, steigen die Kosten proportional. Der Messtag zeigt es.
- **BTX-Qualität:** Bei unserer Datenlage ist sie unbewiesen; die Tore G2 und G3 fangen das ab.
- **Datenmenge:** Die Ausbeute der Quellen ist erst nach dem Filterlauf bekannt. `status` meldet sie.
- **Tokenizer:** 64k-Vokabular bedeutet neue Tokenisierung. v6 speichert deshalb Text, nicht Tokens.
