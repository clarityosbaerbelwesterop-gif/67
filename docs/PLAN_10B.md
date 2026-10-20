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

## 6. Alternative: 23B dicht

Möglicher Aufbau: 56 Schichten, d_model 6144, 48/8 Köpfe × 128, SwiGLU 16.384, insgesamt 22,65 Mrd. Parameter.
Gerechnet mit denselben Annahmen wie oben.

| | 10B | 23B |
|---|---|---|
| 200 Mrd. Tokens | ≈ 18.500 $ | ≈ 41.400 $ |
| 300 Mrd. Tokens | ≈ 27.800 $ | ≈ 62.100 $ (mehr als das 108B-MoE) |
| Wandzeit, 300 Mrd. Tokens auf 8 B300 | 19,5 Tage | 44 Tage |
| Gewichte in FP8 | 10,1 GB | 22,6 GB |
| Agenten × 8k auf H100 / A100 80 GB / L40S / RTX 5090 | 42 / 36 / 22 / 12 | 26 / 15 / 10 / 2 |
| RL-Rollouts je Dollar | 2,25× | 1× |

- **Stärke des 23B:** Bei gleicher Tokenzahl ist es besser beim Faktenwissen und bei den schwersten Aufgaben.
  Gerade dort hilft Denkzeit am wenigsten (Snell et al.).
- **Rechenoptimal** ist bei ≈ 28.000 $ ein Modell von ≈ 12 Mrd. Parametern (Chinchilla, D ≈ 20 N).
- **Inferenz entscheidet:** Unser Ansatz besteht vor allem aus Inferenz: Denkzeit, viele Agenten, RSI-Rollouts.
  Dann lohnt ein kleineres Modell, das länger trainiert wird.
- **Entscheidung:** zuerst das 10B. Das 23B ist Stufe 2, wenn das 10B die Kurve belegt hat und Funding da ist. Es wird
  aus dem 10B gewachsen (Depth Up-Scaling wie bei SOLAR 10.7B aus Mistral 7B) statt von null trainiert; die
  Ersparnis wird vorher gemessen.

## 7. Weg des Owners (20.10.2026): 1B fertig, dann 10B durch Wachsen, ohne MoE

Weg B bleibt der nächste Trainingsschritt. Sein Ziel ist jetzt ein dichtes 10B, das aus dem 1B wächst; MoE-Upcycling
entfällt.

1. **1B fertigstellen.** Bestandsaufnahme, Auftrag 1 (greedy, Denkzeit, Oracle), FineMath-Scan.
2. **RSI und Denkzeit am 1B stark machen** und die Lernkurve belegen: mehrere RSI-Runden, jede mit Gate.
3. **Premium-Tokens.** Ein paar GPU-Stunden am 1B mit den besten Daten.
4. **Wachsen auf ≈ 10B** mit `jobs/grow-10b.sh` (`scripts/grow.py`). Das ist ein CPU-Job für wenige Cent. Das
   10B startet funktionsgleich mit dem 1B; dieser Checkpoint ist der Startpunkt für alle späteren Modelle
   (Quasnir, Rouge 1).
5. **Weitertrainieren am 10B**, so weit das Budget reicht, dann die Tests und das Fine-Tuning.

**Rechnung für die 200–400 €.**
- Auf H200 (FP8, MFU 35 %) oder B300 schafft das 10B ≈ 40–85 Mio. Tokens je GPU-Stunde, also
  ≈ 2,5 Mrd. Tokens je 230 $.
- Das 1B schafft das Zehnfache.
- Damit reichen 200 € für einen echten 10B-Punkt mit einigen Mrd. Tokens. Gemessen wird, ob er das 1B schlägt.
- Ein voll trainiertes 10B (100–300 Mrd. Tokens) bleibt bei ≈ 9.000–28.000 $. Wachsen spart einen Teil davon; wie
  viel, messen wir.

**Bestandsaufnahme Volume `3qlkuqnr1u`** (20.10.2026, Osirus-Lauf 39467175129): 200 GB in US-NC-2, belegt 97,8 GB,
frei ≈ 102 GB, 45 Dateien.
- **`quasnir/run2`** (82,5 GB): Checkpoints step 2000–7000 und 7500 mit je 11,5 GB (Modell 3,8 GB, Optimizer 7,7 GB)
  und der Export `weights-bf16-step007500` (1,9 GB, sha256 `0d5eecc2…`). Das ist der Champion.
- **`quasnir/run1`** (13,4 GB): step 1000 und dessen Export.
- **`quasnir/base-v4`** (1,8 GB): nur 4 von 510 Rohdateien von opc-fineweb-code.
- **Klein:** `quasnir/cot`, `quasnir/handoff`.
- **1B-Aufbau:** dim 1920, 24 Schichten, 30 Köpfe / 10 KV-Köpfe (head_dim 64), FFN 5120, Vokabular 8192, Kontext 512.
  Das sind 0,96 Mrd. Parameter, trainiert mit ≈ 1,97 Mrd. Tokens (7500 × 128 × 512 × 4).
- **Nicht auf dem Volume:** die Trainingsdaten von Lauf 2 (`data/mix`) und Kais FineMath- und StackExchange-Dateien
  (`/workspace/pool-p1`). Sie lagen auf Groks Pod und müssen neu gebaut bzw. neu geladen werden.
- **Wachstumsziel für dieses 1B** (`jobs/grow-10b.sh`): dim 4032, 63/21 Köpfe, FFN 14.080, 44 Schichten, also
  9,43 Mrd. Parameter (37,7 GB in F32).

**Daten-Stand.**
- Corpus v6 ist gebaut und getestet: Pipeline, 47 auf dem Hub geprüfte Quellen für 250–500 Mrd. Tokens, Scan,
  Wert-Tor.
- Heruntergeladen und gefiltert ist er noch nicht. Das braucht CPU-Pods und das 1-TB-Volume.
- Auf dem Volume liegen die Stores des 1B, FineMath-4+ (200,6 Mio. Tokens) und weitere; die Bestandsaufnahme zeigt
  die Zahlen.

## 8. Recherche, langes Denken und Prüfen

**Recherche-Werkzeug** für Planer und Forscher:
- Suche und Abruf laufen nur lesend über einen Proxy mit Allowlist. Keine Formulare, keine Logins, kein Posten.
- Abgerufene Seiten sind Daten, keine Anweisungen.
- Jede Aussage nennt URL und Abrufzeit. Fakten brauchen zwei unabhängige Quellen oder eine Prüfung durch Ausführung.
- Was sich nicht belegen lässt, wird als unbelegt markiert und nicht erfunden.

**Benchmarks:**
- Ohne Internet, sonst sind sie verseucht.
- „Mit Recherche“ wird getrennt gemessen und getrennt berichtet.
- Die Halluzinationsrate wird mit und ohne Recherche gemessen.

**Trainingsdaten aus dem Netz** nur über Scan, Lizenz und Wert-Tor.

**Prüfkette:**
- Code: Tests in der Sandbox.
- Mathe: Ausführung oder symbolische Prüfung.
- Fakten: Quellen.
- Pläne: Teilschritte prüft ein Subagent, der nicht der Autor ist.

## 9. Großer Coding-Test ab dem 10B-Punkt

Aufgabe des Owners: eine Infrastruktur in der Art von Oracle nachbauen, gezielt nach Verbesserungen suchen und alles
mit einem Backend bauen, an das nur noch Supabase und Vercel angeschlossen werden müssen; dazu berichten, worauf
optimiert wurde.

Damit der Test messbar ist:
- **Feste Spezifikation:** mandantenfähiges Backend mit Postgres-Schema und Row Level Security (Supabase), Auth und
  Rollen, API, Hintergrundjobs, Audit-Log, Admin-UI (Next.js für Vercel), Last- und Latenzziele.
- **Verdeckte Abnahmetests**, End-to-End, in der Sandbox. Gezählt wird die Quote bestandener Tests.
- **Optimierungsbericht:** Jede behauptete Verbesserung wird durch Messungen belegt.
- **Funktion nachbauen, nichts kopieren:** kein fremder Code, keine fremde Marke.
- **Zugangsdaten:** Das Modell bekommt keine. Supabase und Vercel schließt der Owner am Ende selbst an.
- Dazu kommen die übrigen Benchmarks. Danach wird gesichert und fein-getunt.
