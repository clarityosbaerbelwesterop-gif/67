# Plan fürs Funding: über 100 Mrd. Parameter, 25–30 Mrd. aktiv

Stand 19.10.2026, nach Groks Prüfung überarbeitet.

**Rolle dieses Plans.** Das ist der Plan für Funding und EuroHPC. Aus den 300 € läuft heute der 23B-Schritt (Weg B).
Dieser Plan beginnt erst mit Funding.

**Status der Zahlen.** Alle Kosten sind Rechnungen aus Annahmen; gemessen ist noch nichts. Ein Messtag auf einer B300
ersetzt die Annahmen, bevor Geld in Phase M1 fließt.

**Präzision.** 4-Bit-Training (NVFP4) ist nach Owner-Entscheidung ausgeschlossen und im JAX-Trainer auch nicht
verfügbar. Gerechnet wird nur mit FP8.

## 1. Kernaussagen

1. **Speicher ist nicht das Problem, Rechenzeit ist es.**
   - Inferenz: Ein Modell mit 108 Mrd. Parametern passt auf **eine** B300 (BF16 216 GB, FP8 108 GB, bei 288 GB HBM).
   - Training: Es braucht etwa 1,1–1,7 TB, also 4–8 B300.
2. **Kosten folgen den GPU-Stunden, nicht der GPU-Zahl.** 2 GPUs über 5 Monate kosten so viel wie 8 GPUs über
   5 Wochen.
3. **Mit 1–2 B300 geht es über Branch-Train-MiX (BTX).** Dichte Äste werden einzeln trainiert; jeder passt auf 1–2 GPUs.
   Danach werden sie zum MoE gemischt.
4. **Größenordnung ≈ 56.500 $ (FP8).** Das ist weit unter 1 Mio., aber nur mit Funding oder EuroHPC machbar.

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
Die 44 Experten entstehen aus 11 Ästen, deren FFN (hidden 12.288) in je 4 Segmente zerlegt wird. SwiGLU zerfällt
additiv über die Hidden-Einheiten. Deshalb reproduziert ein Router, der die 4 Segmente eines Astes wählt, diesen Ast
exakt.

## 3. Weg mit 1–2 B300: BTX statt Training von Grund auf

| Phase | Was | Tokens | Speicher | Passt auf |
|---|---|---|---|---|
| M0 | Corpus v6 (`data/build_corpus_v6.py`), Wert-Tor je Quelle (Abschnitt 6), Tokenizer 64k | – | CPU, dazu kleine GPU-Proxyläufe | CPU-Pods |
| M1 | Dichter Stamm „Quasnir-13B": 12,75 Mrd. Parameter | 100 Mrd. | ≈ 204 GB (AdamW gemischt) | 2 B300 |
| M2 | 11 Äste vom Stamm, je eine Domäne: Python, JS/TS, C/C++/Rust/Go, JVM, übrige Sprachen, Mathe, Reasoning-Spuren, Fachartikel, Bücher/Langtexte, Recht/Behörden, Allgemein | 11 × 20 Mrd. | je ≈ 204 GB | je 1–2 B300; unterbrechbar |
| M3 | Mischen; zuerst nur Router, Attention und gemeinsame Experten trainieren (Experten eingefroren) | 60 Mrd. | ≈ 208 GB eingefroren + ≈ 132 GB trainierbar (8,2 Mrd. Parameter) + Aktivierungen | 2 B300 |
| M3b | Optional: alle Parameter kurz nachtrainieren | 20 Mrd. | ≈ 1,7 TB | 8 B300, ca. 2–3 Tage |

BTX stammt von Meta (Sukhbaatar et al., [2403.07816](https://arxiv.org/abs/2403.07816)). Dort war es bei gleicher
Rechenzeit besser als Sparse Upcycling ([2212.05055](https://arxiv.org/abs/2212.05055)). Für uns zählen drei
Eigenschaften:
- Jeder Ast ist ein normales dichtes Modell.
- Ein Abbruch kostet nur den laufenden Ast.
- Schlechte Äste fallen am Tor G2 heraus.

Weg B (23B aus dem 1B) bleibt der heutige Schritt und der Prüfstand für Daten, Mix und Evals.

## 4. Rechnung (FP8)

Annahmen (vor M1 messen):
- **Hardware:** B300 dicht FP8 ≥ 4,5 PFLOP/s. Quellen nennen 4,5–7; gerechnet wird mit 4,5.
- **Auslastung:** MFU 30 %.
- **FLOP pro Token:** 6 × N_aktiv. Bei 4k Kontext kommen etwa 8 % Attention dazu.
- **Preis:** RunPod B300 6,94–7,89 $/h (Stand Sept. 2026); gerechnet mit 7,40 $/h.

| Phase | FLOP | GPU-Stunden | Kosten |
|---|---|---|---|
| M1 Stamm 12,75 Mrd. × 100 Mrd. Tokens | 7,65e21 | 1.574 | ≈ 11.600 $ |
| M2 11 Äste × 20 Mrd. Tokens | 1,68e22 | 3.463 | ≈ 25.600 $ |
| M3 Mischen 26,3 Mrd. aktiv × 60 Mrd. Tokens | 9,48e21 | 1.951 | ≈ 14.400 $ |
| M3b optional, 20 Mrd. Tokens | 3,16e21 | 650 | ≈ 4.800 $ |
| **Summe (400 Mrd. Trainings-Tokens)** | | **7.639** | **≈ 56.500 $** |
| Wandzeit | | 2 B300: 159 Tage; 8 B300: 40 Tage | |

Liegt FP8 auf B300 bei 7 statt 4,5 PFLOP/s, sinken Zeit und Kosten um etwa 35 %. Zum Vergleich: dasselbe MoE direkt
von Grund auf mit 300 Mrd. Tokens braucht 9.756 GPU-h (≈ 72.000 $) und ab Tag 1 8 GPUs.

## 5. Hebel, nach Wirkung

1. **Daten mit nachgewiesenem Wert statt mehr FLOPs.** Jede Quelle besteht zuerst das Wert-Tor (Abschnitt 6).
   Bis zu 4 Epochen auf den besten Teilen kosten kaum Qualität (Muennighoff et al.,
   [2305.16264](https://arxiv.org/abs/2305.16264)).
2. **FP8 auf B300.** Der Messtag klärt, wie schnell JAX/XLA FP8 dort rechnet.
3. **Multi-Token-Prediction** (DeepSeek-V3, [2412.19437](https://arxiv.org/abs/2412.19437)): mehr Signal pro Token.
4. **Eingefrorene Experten in M3.** Nur etwa 8,2 Mrd. Parameter tragen Optimizer-Zustände. So passt das MoE-Training
   auf 2 B300.
5. **Tore pro Ast.** Schlechte Rechenzeit wird nicht weitergetragen.
6. **Langtexte ganz behalten.** v6 speichert Bücher und Artikel bis 3 Mio. Zeichen am Stück, als Material für die
   spätere Kontextverlängerung.

## 6. Wert-Tor für Daten (Owner-Regel: Code-Daten nur bei hoher Qualität und nur, wenn sie dem Modell so viel bringen)

Eine Quelle kommt erst in den Mix, wenn ein Token von ihr mindestens so viel bringt wie ein Token der Daten, die sie
ersetzt. Das gilt für Code und genauso für Web-Daten.

1. **Pilot.** `build_corpus_v6.py plan --pilot-tokens 3e9 --sources ...` filtert höchstens 3 Mrd. Tokens je Quelle,
   nach den AAA-Regeln.
2. **Arm B.** `data/build_ablation.py` baut eine Mischung mit 1 Mrd. Tokens: 30 % ganze Dokumente der Kandidatenquelle,
   70 % ganze Dokumente des bisherigen Basis-Stores.
3. **Läufe.** Arm A ist der Basis-Store, er läuft mit zwei Seeds, um das Rauschen zu messen. Arm B läuft mit einem
   Seed. Alle Läufe:
   - starten vom gleichen 100M-Checkpoint,
   - nutzen den gleichen Plan: `accel/run_s2.py --init ... --cap 1000 --batch 128 --accum 4 --seq 512`, also etwa
     262 Mio. Tokens.
4. **Messung** auf der Box: Held-out-Code L0–L2 und val general. Die Held-out-Sets verlassen die Box nicht.
5. **Entscheidung.** Die Quelle besteht, wenn beide Bedingungen gelten:
   - Arm B ist auf Held-out-Code um mehr als das Doppelte des Seed-Rauschens besser.
   - Arm B ist auf val general nicht schlechter als das Rauschen.

   Ihr Anteil im Mix richtet sich nach dem gemessenen Gewinn, nicht nach ihrer Größe.
6. **Kosten.** Je Lauf etwa 1,6e17 FLOP, also Minuten auf einer GPU. 5 Läufe (2 Basis, 3 Kandidaten) liegen unter 10 $.

## 7. Geld und Reihenfolge

| Schritt | Kosten | Deckel | Wann |
|---|---|---|---|
| Weg B (23B-Schritt) | ≈ 190–215 $ | laut Groks Plan | jetzt, hat Vorrang |
| Grok Auftrag 1: öffentliche Benchmarks des 1B auf dem Pod | CPU, wenige $ | 5 $ | nur nach ausdrücklichem Owner-Ja |
| Grok Auftrag 2: Pilot Corpus v6, 3 Code-Quellen à ≤ 3 Mrd. Tokens | CPU; ≤ 100 GB Speicher ≈ 7 $/Monat | 10 $ CPU | jetzt |
| Grok Auftrag 3: Wert-Tor, 5 Proxyläufe 100M | GPU, Minuten | 10 $ | nach Weg B oder mit Owner-Ja |
| Voller Corpus v6 (≈ 300 Mrd., mit Web ≈ 460 Mrd.) | CPU ≈ 60 $; ≈ 1 TB ≈ 50 $/Monat | – | nur Quellen, die das Wert-Tor bestehen; mit Funding |
| Messtag 1× B300 | ≈ 25 $ | 30 $ | mit Funding |
| M1–M3 | ≈ 56.500 $ (FP8) | eigene Freigabe je Phase | mit Funding / EuroHPC |

## 8. Tore

| Tor | Bedingung |
|---|---|
| G0 | 1B gemessen; Weg B mit eigenem Tor laut Groks Plan. |
| G1 | Stamm 13B gegen das beste bisherige Modell: Odds Ratio ≥ 2 am harten Test (unteres 95-%-Band), val höchstens +2 %. Das ist die Regel „jede Version mindestens 2× besser". |
| G2 | Jeder Ast verbessert die val seiner Domäne gegen den Stamm und verliert höchstens 2 % auf der allgemeinen val. Sonst fällt er heraus. |
| G3 | MoE gegen den Stamm: Odds Ratio ≥ 2 am harten Test, kein Domänenverlust gegen den besten Ast. |
| Nach jedem Tor | Einstufung champion / frontier / dangerous. Ab frontier wird der Owner informiert. Dangerous wird behalten, RSI angehalten, Weitergabe nur an VIP-Firmen. |

## 9. Risiken

- **Verfügbarkeit:** B300 sind knapp. Ausweichen auf B200 kostet etwa 1,5–2× GPU-Stunden.
- **Auslastung:** Liegt die MFU unter 30 %, steigen die Kosten proportional.
- **BTX-Qualität:** Bei unserer Datenlage ist sie unbewiesen; die Tore G2 und G3 fangen das ab.
- **Quellennamen:** Teils aus dem Gedächtnis eingetragen. Vor jeder Nutzung prüft `inspect` sie.
- **Datenmenge:** Die Ausbeute ist erst nach dem Filterlauf bekannt.
- **Tokenizer:** 64k-Vokabular bedeutet neue Tokenisierung. v6 speichert deshalb Text, nicht Tokens.
