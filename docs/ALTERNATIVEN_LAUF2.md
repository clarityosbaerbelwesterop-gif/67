# Alternativen zu Lauf 2 mit höchstens 50 €

Stand 21.10.2026. Quasnir ist eingefroren; dieses Dokument bereitet nur vor und startet nichts. Jeder Lauf braucht
die Freigabe des Owners im Environment `RUNPOD` (oder sein Ja zu einem kostenlosen Konto).

Die Quellen stammen aus einer Web-Recherche vom 21.10.2026. Kaggle, die TPU Research Cloud (TRC) und Modal waren
aus der Sitzung nicht direkt abrufbar; die Angaben dazu kommen aus Suchauszügen und Drittseiten und sind als
„ungeprüft“ markiert. Zahlen ohne Quelle sind eigene Rechnungen.

## Kurzantwort

- **Lauf 2 wie geplant passt nicht in 50 €.** 4–5 Mrd. Tokens Lehrer-Daten von Qwen3.8-27B kosten per API etwa
  7.500–15.000 $ (1,875–3,00 $ je Mio. Ausgabe-Tokens,
  [DeepInfra](https://deepinfra.com/blog/qwen3-8-27b-pricing-deepinfra-vs-alibaba)). Selbst erzeugt auf einer
  kostenlosen Kaggle-TPU wären es etwa 39 Mio. Tokens pro Woche
  ([kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab), ungeprüft).
- **Der Engpass sind Pretraining-Tokens, nicht Lehrer-Daten oder Kontextlänge.** Unser 1B hat 1,55 Mrd. Tokens
  gesehen. Zum Vergleich:
  - phi-1-base: 51 Mrd. Tokens, 29 % HumanEval ([phi-1](https://arxiv.org/abs/2306.11644)).
  - nanochat d20: 561 Mio. Parameter, 11,2 Mrd. Tokens, 7,9 % HumanEval
    ([Bericht](https://huggingface.co/richardprobe/nanochat)).
- **Die beste Alternative kostet 0 €:**
  - kostenlose TPU-Zeit;
  - bereits veröffentlichte Daten mit freier Lizenz, statt sie selbst zu erzeugen;
  - ein kleineres Modell (≈ 500 Mio. Parameter), das deutlich länger trainiert wird.
- **Astra und Opus 5.5 schlagen wir mit 50 € nicht.** Das ehrliche Ziel steht unten.

## Warum Nachtraining auf dem jetzigen 1B das Gate nicht erreicht

- **RL mit Tests (RLVR) gewichtet vor allem Antworten neu, die das Basismodell schon erzeugen kann.** Bei großem k
  holen Basismodelle auf oder liegen vorn ([Yue et al. 2025](https://arxiv.org/abs/2504.13837)). Unsere Obergrenze
  von 1,6 % (mindestens ein Versuch richtig) ist damit auch die Grenze für RSI und Best-of-N.
- **Wie weit Fine-Tuning trägt, entscheidet das Basismodell.** Dasselbe Fine-Tuning auf OpenCodeInstruct bringt
  Llama-3.2-1B (Basis mit 9 Billionen Tokens) auf HumanEval 51,8 und MBPP 53,4
  ([OpenCodeInstruct, Tabelle 3](https://arxiv.org/abs/2504.04030)).
- **Das 10B ist mit diesem Budget nicht trainierbar.** Bei gemessenen 137 € je Mrd. Tokens reichen 50 € für
  0,36 Mrd. Tokens. Es bleibt eingefroren.

## Optionen nach Rang

| # | Option | Kosten | Erwarteter Gewinn | Risiko | Zeit |
|---|---|---|---|---|---|
| 1 | Pretraining auf Kaggle TPU v5e-8 | 0 € | hoch: etwa 2,4·10¹⁹ FLOP je Woche, so viel wie ≈ 117 € auf 2× B200 | Kontingent und 9-h-Sitzungen (ungeprüft); accel auf echter TPU noch nicht getestet | 2–4 Wochen |
| 2 | TPU Research Cloud | 0 € für TPUs, wenige € für VM und Speicher | sehr hoch, falls bewilligt: genug für ≈ 50 Mrd. Tokens | Bewilligung offen; verlangt Veröffentlichung der Ergebnisse; Repo seit April 2026 archiviert, Bedingungen ungeprüft | 3–4 Tage Antwort, 30 Tage Zugang |
| 3 | Veröffentlichte Daten statt eigener Erzeugung | 0 € | hoch: OpenCodeInstruct allein ist etwa so groß wie die Lehrer-Daten von Lauf 2 | Dekontamination neu prüfen | 1 Woche Aufbau |
| 4 | Kleineres Modell (≈ 500M), länger trainiert | 0 € extra | hoch: phi-1-small (350M) erreichte 45 % HumanEval, allerdings mit GPT-3.5-Daten, also nur als Obergrenze | Stamm wechselt von 1B auf 500M | – |
| 5 | Kurzer A/B-Test: jetziges 1B auf OpenCodeInstruct nachtrainieren | ≈ 7 € je Epoche (0,38 Mrd. Tokens, Schätzung) | klein, denn das Basismodell ist schwach | gering | 1 Tag |
| 6 | Rejection Sampling oder RL mit Tests | 0 € im RSI-Loop | erst sinnvoll ab pass@10 ≥ 15 %; Beispiel: AceCoder +25 % relativ auf HumanEval+ in 80 RL-Schritten bei 7B ([AceCoder](https://arxiv.org/abs/2502.01718)) | ohne gutes Basismodell wirkungslos | nach 1–4 |
| 7 | Denkzeit bei der Messung | 0 € | CodeT +18,8 Punkte pass@1 ([CodeT](https://arxiv.org/abs/2207.10397)) | durch die Obergrenze begrenzt; getrennt von greedy berichten | sofort |
| 8 | Startup-Credits (Tabelle unten) | 0 €, Papierkram | 1.000–2.000 $ ohne Investoren | Firma nötig | 2–7 Tage |

**Rechnung zu Option 1.**
- v5e-8: 8 Chips × 197 TFLOP/s bf16 = 1,58 PFLOP/s ([Google](https://docs.cloud.google.com/tpu/docs/v5e)).
- Bei 21 % MFU (wie nanochat) und 20 TPU-Stunden je Woche (Drittquellen, ungeprüft) ergibt das ≈ 2,4·10¹⁹ FLOP je
  Woche.
- Ein 500M-Modell braucht 6 × 5·10⁸ = 3·10⁹ FLOP je Token. Drei Wochen reichen für ≈ 7·10¹⁹ FLOP, also
  ≈ 20–23 Mrd. Tokens.

## Daten: Lizenzprüfung

Unsere Regeln: Lehrer nur offene Modelle mit freier Lizenz, keine geschlossenen Modelle, kein CC BY-NC.

**Erlaubt**

| Datensatz | Lizenz | Herkunft | Einsatz |
|---|---|---|---|
| OpenCodeInstruct (5 Mio. Beispiele, je 10 Unit-Tests) | CC BY 4.0 | Aufgaben von Qwen2.5-32B-Instruct, Lösungen, Tests und Bewertung von Qwen2.5-Coder-32B-Instruct, beide Apache-2.0 ([Karte](https://huggingface.co/datasets/nvidia/OpenCodeInstruct), [Paper](https://arxiv.org/abs/2504.04030)) | Annealing und SFT; Filter auf Bewertung 5 und bestandene Tests (laut Paper besser als eine Zufallsauswahl von 500.000) |
| OpenCodeReasoning 1 und 2 | CC BY 4.0 | Antworten von DeepSeek-R1 (MIT); Lizenz der Ausgangsfrage je Zeile vermerkt ([OCR](https://huggingface.co/datasets/nvidia/OpenCodeReasoning), [OCR-2](https://huggingface.co/datasets/nvidia/OpenCodeReasoning-2)) | nur das Feld `solution` oder kurze Zeilen; die Denkspuren sind sehr lang |
| Stack-Edu | Lizenzfelder je Datei, Bedingungen von The Stack v2 | 21,8 Mrd. Tokens allein Python, wir nutzen 553 Mio. ([Karte](https://huggingface.co/datasets/HuggingFaceTB/stack-edu)) | Pretraining, nur Dateien mit freier Lizenz; Inhalte kommen aus dem S3-Bucket von Software Heritage (laut Karte ≈ 6 h auf einer VM mit 16 Kernen) |
| Cosmopedia (Code und Mathe) | Apache-2.0 | erzeugt von Mixtral-8x7B-Instruct, Apache-2.0 ([Karte](https://hf.co/datasets/HuggingFaceTB/cosmopedia)) | Pretraining |
| FineMath, smollm-corpus | ODC-By | ([1](https://hf.co/datasets/HuggingFaceTB/finemath), [2](https://hf.co/datasets/HuggingFaceTB/smollm-corpus)) | schon im Einsatz (FineMath) |
| opc-annealing-corpus, Teil `algorithmic_corpus` | ODC-By | reiner Code aus Stack v2 ([Karte](https://hf.co/datasets/OpenCoder-LLM/opc-annealing-corpus)) | Annealing |
| eigene geprüfte Rechenwege | eigen | 163 Mio. Tokens | Annealing |

**Ausgeschlossen**
- Magicoder-OSS-Instruct-75K: MIT, aber erzeugt von gpt-3.5-turbo, also einem geschlossenen Modell
  ([Karte](https://hf.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K)).
- opc-sft-stage1: enthält GPT-Gespräche aus ShareGPT und WildChat.
- opc-sft-stage2, Teil `evol_instruct` (= Magicoder-Evol-Instruct).
- Glaive code assistant v3: das erzeugende Modell ist nicht offengelegt.
- KodCode: CC BY-NC.

**Entscheidung des Owners nötig (E8)**
- Synthetische Teile von opc-annealing-corpus: erzeugt von Qwen2-72B-Instruct, Lizenz „other“
  ([HF](https://hf.co/Qwen/Qwen2-72B-Instruct)).
- self-oss-instruct-sc2: erzeugt von StarCoder2-15B, Lizenz BigCode OpenRAIL-M, also weder MIT noch Apache
  ([Karte](https://hf.co/datasets/bigcode/self-oss-instruct-sc2-exec-filter-50k)).

**Lehrer für eigene Daten, falls nötig** (alle Apache-2.0):
- Qwen3.8-27B. Laut Modellkarte SWE-bench Pro 61,7 ([Karte](https://hf.co/Qwen/Qwen3.8-27B)).
- Qwen3-Coder-30B-A3B ([HF](https://hf.co/Qwen/Qwen3-Coder-30B-A3B-Instruct)).
- gpt-oss-20b ([HF](https://hf.co/openai/gpt-oss-20b)).

## Rechenzeit: kostenlose und günstige Quellen

| Programm | Umfang und Bedingungen | Passt zu accel (JAX)? |
|---|---|---|
| Kaggle TPU v5e-8 | in Notebooks verfügbar ([Kaggle](https://www.kaggle.com/discussions/product-announcements/607202)); ≈ 20 TPU-Stunden je Woche, Sitzungen knapp 9 h, Telefon- oder Ausweisprüfung (ungeprüft, [kaggle-tpu-lab](https://github.com/ARahim3/kaggle-tpu-lab)) | ja; Anleitung in `docs/ACCEL.md` „Kaggle / Colab TPU“ |
| TPU Research Cloud | über 1.000 Cloud-TPUs, JAX unterstützt, Ergebnisse müssen veröffentlicht werden ([TRC](https://sites.research.google/trc/about/)) | ja |
| Google for Startups, Stufe Start | bis 2.000 $ ohne Investoren, Firma jünger als 5 Jahre ([Folie](https://services.google.com/fh/files/misc/googleforstartups_cloudprogram_overviewslide.pdf)) | ja, TPU |
| Google for Startups, Stufe AI | bis 350.000 $, nur mit VC-Finanzierung von Seed bis Series A ([Seite](https://cloud.google.com/startup/ai)) | ja |
| AWS Activate Founders | 1.000 $, ohne Investoren, Entscheidung in 2–7 Tagen ([AWS](https://aws.amazon.com/aws-startups/learn/applying-for-aws-activate-credits-a-step-by-step-guide)) | GPU über `jax[cuda12]` |
| Microsoft for Startups | 1.000 $ für 90 Tage, 5.000 $ nach Prüfung (Drittquelle) | GPU |
| Modal | 30 $ je Monat kostenlos ([Preise](https://modal.com/pricing), ungeprüft) | GPU |
| RunPod | Forschung bis 25.000 $ ([Research](https://www.runpod.io/research-credits)); Startups 1:1 auf Käufe bis 25.000 $ ([Startup](https://runpod.io/startup-program)) | GPU, unser Workflow |
| Bezahlte TPU (Rückfall) | v5e Flex-start 0,60 $ je Chip-Stunde ([Preise](https://cloud.google.com/tpu/pricing)); 50 € ≈ 12 h auf v5e-8 | ja |

NVIDIA Inception gibt keine Credits. Colab free (T4 ohne Garantie) und HF ZeroGPU (3,5 min am Tag) reichen nur für
Tests.

## Was im Code fehlt (vor dem ersten Lauf)

1. **Muon in accel.** forge hat Muon (opt-in) und lag beim A/B-Test von Grund auf klar vorn: val 4,806 statt 5,051
   nach 300 Schritten (`docs/PROMPT_SCALING.md`). accel kann nur AdamW und lehnt Muon-Konfigurationen ab
   (`docs/ACCEL.md`, Punkt 6). Portieren, mit Paritätstest gegen forge.
2. **accel auf echter TPU.** Bisher nur auf CPU und als simulierter Slice getestet. Erster Schritt: 30 Minuten
   Rauchtest auf Kaggle (Geräte, Durchsatz, Checkpoint).
   - Ein 500M-Modell passt repliziert auf einen Chip mit 16 GB: f32-Gewichte und AdamW ≈ 6 GB, Gradienten ≈ 2 GB.
   - Ein 1B braucht `accel/mgpu.py`, das Gewichte und AdamW per FSDP verteilt.
3. **Sitzungen von 9 Stunden.**
   - Checkpoint jede Stunde.
   - Vor Sitzungsende den Checkpoint nach außen kopieren. `scripts/resume_pack.py` verschlüsselt ihn.
   - Ein 500M-Checkpoint mit AdamW ist ≈ 6 GB.
   - Wie viel Platz Kaggle für Daten und Ausgaben lässt, ist ungeprüft. In Woche 0 klären.
4. **Datenmenge.**
   - 20 Mrd. Tokens mit Vokabular 8.192 sind als uint16 ≈ 40 GB Speicher.
   - Prüfen, ob das als Kaggle-Datensatz geht oder ob die Daten gestreamt werden müssen.
5. **Tokenizer.**
   - Messen, wie viele Bytes je Token unser Vokabular von 8.192 auf Code erreicht. nanochat erreicht mit 65.000
     Tokens 2,56.
   - Ein größeres Vokabular wäre ein neuer Stamm und damit eine Owner-Entscheidung.
6. **Dekontamination.** `data/build_corpus_v6.py scan --evals` über alle neuen Quellen gegen HumanEval, MBPP und
   GSM8K.

## Empfohlener Plan (≤ 50 €)

| Woche | Schritt | Kosten |
|---|---|---|
| 0 | Owner: Kaggle-Konto prüfen lassen, TRC beantragen, mit Firma auch AWS Founders oder Google Start. Claude: Muon in accel, Rauchtest-Notebook, Datenwerkzeuge | 0 € |
| 0–1 | **Pretraining-Daten:** Stack-Edu nur mit freier Lizenz (Python und etwa 3 weitere Sprachen, 10–15 Mrd. Tokens), FineMath, Code und Mathe aus Cosmopedia. **Annealing-Mischung:** OpenCodeInstruct (Bewertung 5, Tests bestanden), `algorithmic_corpus`, eigene Rechenwege, kurze OpenCodeReasoning-Lösungen. Danach Dekontamination | ≤ 5 € (VM für den Download, Schätzung) |
| 1–3 | 500M, Kontext 2.048, Muon; ≈ 20 Mrd. Tokens auf Kaggle; die letzten 10–15 % als Annealing | 0 € |
| 4 | SFT auf 0,5–1 Mio. OpenCodeInstruct-Beispielen, 2–3 Epochen; danach RSI oder Rejection Sampling mit den Tests, sobald pass@10 ≥ 15 % | 0 € |
| Reserve | Liegt der gemessene Kaggle-Durchsatz unter der Hälfte der Schätzung: Annealing und SFT auf 2× B200 (gemessen 18 € je Mrd. Tokens bei 1B; 40 € ≈ 2,2 Mrd. Tokens bei 1B, etwa doppelt so viel bei 500M) oder v5e Flex-start | ≤ 45 € |

**Gate (unverändert):** HumanEval > 5 %, MBPP > 10 % greedy, dazu Validierungs-Loss und A/B gegen das jetzige 1B.
Denkzeit wird getrennt berichtet.

**Erwartung:** 5–15 % HumanEval halte ich für plausibel, weil nanochat d20 ohne Code-Fokus 7,9 % erreicht. Das ist
eine Schätzung, keine Messung.

## Ehrlicher Vergleich mit Astra und Opus 5.5

- **Astra** ist laut `docs/FRONTIER.md` „GPT-6 Astra“ von OpenAI:
  - Humanity's Last Exam mit Tools 57,2 % (offiziell), ARC-AGI-3 62,7 %, FrontierMath Tier 4 ≈ 98 % (Hersteller),
    GPQA Diamond ≈ 96 %.
  - Laut Web erschienen am 03.09.2026, mit Terminal-Bench 4.0 57,7 % und DeepSWE v1.1 74,1 %
    ([Vellum](https://vellum.ai/blog/gpt-6-astra-benchmarks-explained)).
- **Opus 5.5** erreicht laut `docs/FRONTIER.md` 67,7 % auf Humanity's Last Exam mit Tools (offiziell). Für
  SWE-bench Pro widersprechen sich die Aggregatoren, es gibt dort keinen verlässlichen Wert. Opus 5 erreicht 96 %
  auf SWE-bench Verified ([BenchLM](https://benchlm.ai/benchmarks/swe-bench-verified)).
- **HumanEval ist bei Frontier-Modellen gesättigt (91–98 %).**
- **`docs/FRONTIER_GAP.md`** beziffert die Rechenlücke auf etwa 10¹⁰.
- **Beides mit unter 50 € zu schlagen, ist nicht möglich.** Das stärkste Modell, das wir für 0 € nutzen können, ist
  ein offener Lehrer, und dessen Gewichte sind nicht unsere.

**Was wir ehrlich erreichen können:**
1. **Bestes Ergebnis je Rechenaufwand:** ein eigenes Modell von Grund auf, das nanochat d20 (7,9 % HumanEval bei
   3,9·10¹⁹ FLOP) bei gleichem Aufwand schlägt, mit Kurs auf phi-1-small.
2. **Ein enges, geprüftes System:** Quasnir mit Sandbox-Tests und Wiederholungen in einem engen Gebiet. Vorbild
   TinyGSM: ein 1,3B-Generator mit einem 1,3B-Prüfer erreichte 81,5 % GSM8K, mehr als sein Lehrer GPT-3.5 mit
   77,4 % ([TinyGSM](https://arxiv.org/abs/2312.09241)).
3. **Produktqualität in Osirus sofort:** Starke offene Modelle wie Qwen3.8-27B (Apache-2.0) beantworten Anfragen,
   bis Quasnir das Gate besteht. Sie werden als das gekennzeichnet, was sie sind; die Vorbereitung steht in
   `osirus/docs/PROMPT_OSIRUS_STUDIO.md`, Abschnitt 4.8.

## Entscheidungen des Owners

| # | Frage | Empfehlung |
|---|---|---|
| E5 | Kaggle-TPU und TPU Research Cloud nutzen? TRC verlangt, Ergebnisse zu veröffentlichen | Ja zu Kaggle sofort; TRC beantragen und veröffentlicht werden nur Messergebnisse und Methode, keine Gewichte (vor Antrag in den TRC-Bedingungen prüfen) |
| E6 | Neuer Stamm mit ≈ 500M statt Weiterbau am 1B? Das 10B bleibt eingefroren | Ja; das jetzige 1B bleibt Champion, bis das Gate entscheidet |
| E7 | OpenCodeInstruct, OpenCodeReasoning (CC BY 4.0), Cosmopedia (Apache-2.0), Stack-Edu (nur freie Lizenzen) als Trainingsdaten? | Ja, mit Attribution im Manifest und in der Model Card |
| E8 | Grenzfälle: synthetische Teile von opc-annealing-corpus (Qwen2-72B, Lizenz „other“), self-oss-instruct-sc2 (OpenRAIL-M) | Nein, bis die Lizenzen geprüft sind; die erlaubten Quellen reichen |
| E9 | Firma für Startup-Credits (AWS 1.000 $, Google bis 2.000 $)? | Ja, falls eine Firma besteht; das ist kostenlos und kann die Reserve ersetzen |
