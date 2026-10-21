# Plan: Quasnir-MoE 250B-A35B mit Untermodell 1–10B, finanziert über Förderung

Stand 21.10.2026. Dieser Plan ersetzt `docs/PLAN_100B_MOE.md` als Funding-Plan.

**Owner-Entscheidung vom 21.10.2026:**
- Zielmodell: 250 Mrd. Parameter gesamt, 35 Mrd. aktiv, dazu ein Untermodell mit 1–10 Mrd. Parametern.
- Alle Verfahren bleiben: Denkzeit, RSI, Agents, Sandbox, Research.
- Daten: so viel Fertiges wie möglich, den Rest selbst erzeugen.
- Bis die Förderung da ist, gibt es keinen bezahlten Lauf. Bis dahin wird nur kostenlos vorbereitet.

**Status der Zahlen**
- Rechnungen folgen der Regel FLOP = 6 × aktive Parameter × Tokens. Gemessen ist bisher nur unser 1B und 10B auf
  2× B200.
- Die Quellen stammen aus einer Web-Recherche vom 21.10.2026. Huggingface.co, arxiv.org und die EuroHPC-Seiten
  waren aus der Sitzung nicht direkt abrufbar; gelesen wurde über Suchauszüge, Spiegel und den
  Hugging-Face-Connector. **[U]** heißt ungeprüft oder nur aus Drittquellen.
- Vor jedem Antrag die Termine auf den offiziellen Seiten prüfen.

## 1. Kurzfassung

| Frage | Antwort |
|---|---|
| Rechenbedarf 250B-A35B | 6 Bill. Tokens = 1,26·10²⁴ FLOP ≈ **0,7–1,2 Mio. H100-Stunden**; 10 Bill. Tokens ≈ 1,2–2,0 Mio. |
| Marktwert | ≈ 1,4–2,4 Mio. $ für 6 Bill. Tokens (2 $ je GPU-Stunde wie bei DeepSeek), ohne Ablationen und synthetische Daten |
| „200× besser und effizienter“ | Beim Pretraining nicht belegbar. Realistisch **≈ 15×** gegenüber einem naiven dichten 250B (Spanne 7–80×). Ehrlich 200× gibt es nur beim Untermodell: Destillation ≈ 60× günstiger als Training von Grund auf, Betrieb je Token 50–170× günstiger |
| Größter ungenutzter Hebel | Weniger aktive Parameter. 2026 aktivieren Spitzenmodelle 3–5 % ihrer Parameter, wir planen 14 %. Mit 13B statt 35B aktiv kauft dieselbe Rechenzeit 16 statt 6 Bill. Tokens (Entscheidung E10) |
| Läuft das 250B auf einer RTX 4090/5090? | Nein. FP8 sind 250 GB, 4-Bit ≈ 125–150 GB. Mit Experten im Arbeitsspeicher (KTransformers) und ≥ 192 GB RAM etwa 3–5 Tokens/s. Auf die Karte passt das Untermodell: 10B in 8-Bit ≈ 10 GB, ≈ 100 Tokens/s (4090) bis ≈ 180 Tokens/s (5090), geschätzt |
| Rechtlicher Rahmen | 1,3·10²⁴ FLOP liegt unter der Schwelle für systemisches Risiko im AI Act (10²⁵). Die GPAI-Pflichten gelten trotzdem: Urheberrechts-Policy und öffentliche Zusammenfassung der Trainingsdaten (Art. 53 Abs. 1 c, d), auch bei Open Source |
| Beste Quelle für Rechenzeit | EuroHPC AI Factories, Zugang „Large Scale“ (für KI-Startups und KMU kostenlos, über 50.000 GPU-Stunden, 3–12 Monate) |
| Daten | Echt frei lizenziert und ohne Webtext gibt es nur ≈ 2,5–2,9 Bill. Tokens vor Filterung. Für 6 Bill. braucht es Webtext aus Common Crawl (Entscheidung E11) oder ≈ 2,4 Bill. synthetische Tokens |

## 2. Rechenzeit und Förderung

### 2.1 Rangliste

| Rang | Programm | Was es bringt | Bedingungen | Termine | Quelle |
|---|---|---|---|---|---|
| 1 | **EuroHPC AI Factory, Large Scale** (Track „AI for Industrial Innovation“) | > 50.000 GPU-Stunden, Zuteilung 3, 6 oder 12 Monate | Kostenlos für KI-KMU und Startups. Peer Review; zeigen, dass ein kleineres System nicht reicht. Verlangt laut Drittquelle USt-ID, EU-PIC und angestellten Lead Engineer **[U]**. Pflichten: Danksagung, Abschlussbericht; bei Geschäftsgeheimnissen reicht ein eingeschränkter öffentlicher Bericht | Zwei Stichtage je Monat (z. B. 04.01.2027); Zugang 10 Arbeitstage nach Stichtag | [EuroHPC](https://eurohpc-ju.europa.eu/large-scale-access-ai-factories_en), [Access Policy](https://www.eurohpc-ju.europa.eu/document/download/a473e939-1131-4a75-bf0b-4d8659d6bce0_en?filename=EuroHPC+JU+-+Access+Policy_April2025.pdf) |
| 1a | Welche Factory | JUPITER AI Factory (Jülich), HammerHAI (Stuttgart: 860 B200, Betrieb ab H2 2026 erwartet), Leonardo | LUMI hat für 2026 keine Large-Scale-Kapazität mehr. MareNostrum 5 und LUMI-G sind laut Drittquelle für neue Large-Scale-Zuteilungen zu **[U]** | jede Factory direkt nach Kapazität 2027 fragen | [LUMI](https://lumi-ai-factory.eu/large-scale-access-to-lumi-ai-factory/), [HLRS](https://www.hlrs.de/news/detail/hammerhai-a-new-supercomputer-for-artificial-intelligence), [WestAI](https://westai.de/en/press/jupiter-ai-factory-europes-new-ai-turbo-for-smes-industry-and-research/) |
| 2 | **AI Factory Playground und Fast Lane** | Playground ohne Auswahl, in 2 Arbeitstagen, 1–3 Monate (laut Folien ≈ 5.000 GPU-Stunden **[U]**). Fast Lane bis 50.000 GPU-Stunden für 3 Monate, Zusage in 4 Arbeitstagen | Kostenlos für KI-Startups und KMU. Die Seiten zeigten zuletzt „vorübergehend geschlossen, Umzug auf neue Plattform“ **[U]** | laufend | [Playground](https://www.eurohpc-ju.europa.eu/playground-access-ai-factories_en), [Fast Lane](https://eurohpc-ju.europa.eu/fast-lane-access-ai-factories_en) |
| 3 | **EuroHPC Benchmark und Development** | kleine Kontingente für Skalierungstests, als Nachweis für Rang 1 | offene Forschung | Development: Stichtag am 1. jedes Monats (01.11., 01.12.2026) | [Development](https://eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-development-access_en), [Benchmark](https://www.eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-benchmark-access_en) |
| 4 | **Gauss Centre (GCS) Call 37** | Large-Scale-Projekte ab 2 % der Jahreskapazität eines Systems (z. B. ≥ 140.000 GPU-Stunden auf SuperMUC-NG Phase 2); JUPITER, Hunter, SuperMUC-NG | Nur Forschende an deutschen Hochschulen oder öffentlichen Instituten, also mit einer Hochschule als Partner (Projektleitung dort). OpenGPT-X wurde so trainiert | 11.01.–15.02.2027, Zuteilung ab 01.05.2027 | [GCS](https://www.gauss-centre.eu/for-users/hpc-access), [LRZ](https://doku.lrz.de/gcs-large-scale-project-on-supermuc-ng-10746588.html) |
| 5 | **EuroHPC Extreme Scale, Industrie-Track** | Größenordnung Millionen GPU-Stunden; OpenEuroLLM erhielt über 10 Mio. GPU-Stunden über einen strategischen Zugang | Veröffentlichung der Ergebnisse; Entscheidung ≈ 5–6 Monate nach dem Stichtag | Stichtage 2026 vorbei (04.05., 19.10.); Termine 2027 nicht gefunden **[U]** | [EuroHPC](https://eurohpc-ju.europa.eu/eurohpc-ju-call-proposals-extreme-scale-access-mode_en), [OpenEuroLLM](https://openeurollm.eu/blog/strategic-access-EuroHPC-OpenEuroLLM) |
| 6 | **hessian.AI** | 632 A100 und 280 H100; externe Antragsteller möglich, 3 Monate, verlängerbar | – | laufend | [hessian.AI](https://hessian.ai/computeservices/) |

**Geld statt Rechenzeit**

| Programm | Umfang | Bedingungen | Termine | Quelle |
|---|---|---|---|---|
| EIC Accelerator | Zuschuss bis 2,5 Mio. €, dazu optional Beteiligung des EIC Fund | Kurzantrag jederzeit; Erfolgsquote ≈ 6 % **[U]** | letzter Stichtag 2026: 04.11.; Entwurf 2027: 03.02., 01.04., 01.09., 03.11. **[U]** | [EIC](https://eic.ec.europa.eu/eic-funding-opportunities/eic-accelerator_en) |
| Horizon Europe „EU Frontier AI Initiative“ (HORIZON-CL4-2027-04-DIGITAL-EMERGING-11) | 44 Mio. €, etwa ein Zuschlag | Konsortium | öffnet 17.11.2026, schließt 18.03.2027 **[U]** | [horizon-europe.gouv.fr](https://www.horizon-europe.gouv.fr/eu-frontier-ai-initiative-developing-frontier-ai-solutions-are-safe-and-computationally-efficient) |
| Forschungszulage | 35 % für KMU (sonst 25 %) auf bis 12 Mio. € Kosten je Jahr, also bis 4,2 Mio. € | Ob Cloud-GPU-Kosten zählen, ist offen **[U]**; Steuerberater fragen | laufend | [Forvis Mazars](https://www.forvismazars.com/de/en/services/consulting/management-consulting/foerdermittelberatung/update-2026-ausweitung-der-forschungszulage) |
| AWS Generative AI Accelerator | bis 1 Mio. $ Credits (Kohorte 2024) | Auswahl | je Kohorte | [Startups Magazine](https://startupsmagazine.co.uk/article-aws-commits-230-million-generative-ai-startups) |
| Google for Startups, AI-Stufe | bis 350.000 $ | nur mit VC-Finanzierung | laufend | [Google](https://cloud.google.com/startup/ai) |
| Kleinere Programme | Together bis 50.000 $, RunPod bis 25.000 $, Lambda 5.000 $ | – | laufend | [Together](https://together.ai/startup-accelerator), [RunPod](https://www.runpod.io/startup-program) |

**Partner statt allein.** Drei Projekte haben schon Rechenzeit oder Geld und könnten Partner sein:
- das EUROPA-Konsortium (Domyn, Gewinner der Frontier AI Grand Challenge, Modell über 400B);
- OpenEuroLLM;
- SOOFI (vom BMWE mit ≈ 20 Mio. € gefördert, TU Darmstadt).

Quellen: [heise](https://www.heise.de/en/news/400-Billion-Parameter-Model-Consortium-Europa-Wins-AI-Competition-11339046.html),
[BMWE](https://www.bundeswirtschaftsministerium.de/Redaktion/DE/Pressemitteilungen/2025/11/20251118-bundeswirtschaftsministerium-foerdert-aufbau-eines-europaeischen-ki-sprachmodells.html).

**Geschlossen oder pausiert:**
- SPRIND „Next Frontier AI“ (Finalisten am 17.07.2026 gewählt).
- Large AI Grand Challenge 2024; eine neue Runde wurde nicht gefunden.
- BMFTR „KI-Methoden und ML-Modelle“ (Frist war der 30.04.2026).
- ZIM (Neuanträge gestoppt, Wiederöffnung für Anfang 2027 geplant).

### 2.2 Was eine Zuteilung kauft

Annahme: ≈ 400 TFLOP/s effektiv je H100 (zwischen BF16 bei 30 % MFU und FP8 bei 25 % MFU), also
1,44·10¹⁸ FLOP je GPU-Stunde.

| Zuteilung | FLOP | 250B-A35B | 250B-A13B | Sinnvoll stattdessen |
|---|---|---|---|---|
| 50.000 h (Fast Lane) | 7,2·10²² | 0,34 Bill. Tokens (zu wenig) | 0,9 Bill. | Skalierungsleiter und Untermodell-Pilot; z. B. 40B-A6B auf 2 Bill. Tokens |
| 200.000 h | 2,9·10²³ | 1,4 Bill. | 3,7 Bill. | 100B-A13B auf 3,7 Bill. |
| 500.000 h | 7,2·10²³ | 3,4 Bill. | 9,2 Bill. | 250B-A13B auf 9 Bill. |
| 1.000.000 h | 1,44·10²⁴ | 6,9 Bill. | 18 Bill. | Zielmodell |

Die Systeme haben unterschiedliche GPUs: JUPITER GH200, HammerHAI B200, Leonardo A100, LUMI AMD. JAX läuft auf
NVIDIA über CUDA; auf AMD über ROCm ist es weniger ausgereift. GPU-Stunden auf A100 sind etwa 2–3× weniger wert.

### 2.3 Was wir vorbereiten

| # | Unterlage | Wer | Stand |
|---|---|---|---|
| 1 | Rechtsform mit KMU-Status, USt-ID, EU-PIC (Participant Identification Code), benannter Lead Engineer | Owner | offen (E12) |
| 2 | Begründung der Rechenzeit (Abschnitt 6) mit Token-Budget und Formel | Claude | Entwurf hier |
| 3 | **Gemessene Skalierung** auf dem Zielsystem: Durchsatz und MFU auf 8 → 64 → 256 GPUs | Claude baut, Lauf über Playground oder Benchmark | fehlt; braucht MoE und Expert Parallelism in accel (Abschnitt 5) |
| 4 | Container-Image, Checkpoint und Neustart über viele Knoten, Datenpipeline mit Lizenz-Manifest | Claude | teilweise (Checkpoints, Manifest, `resume_pack`) |
| 5 | Meilensteine und Tore (Abschnitt 7) | Claude | Entwurf hier |
| 6 | Urheberrechts-Policy und Trainingsdaten-Zusammenfassung nach AI Act (AI-Office-Vorlage vom 24.07.2025) | Claude entwirft, Owner prüft mit Anwalt | offen |
| 7 | Hochschulpartner für GCS | Owner | offen (E14) |

**Zeitplan, realistisch:**
- Playground oder Fast Lane: Tage.
- Benchmark und Development: nächster Monatsstichtag.
- Large Scale: Antrag Dezember 2026 bis Januar 2027, Rechenzeit ab etwa Februar 2027, falls Kapazität frei ist.
- GCS: Rechenzeit ab 01.05.2027.
- EIC: 6–12 Monate bis zum Geld.

## 3. Daten

### 3.1 Was es unter welchen Bedingungen gibt

| Kategorie | Frei lizenziert je Dokument, ohne Common-Crawl-Web | Aus Common Crawl (Owner-Entscheidung E11, Stufe C) |
|---|---|---|
| Web Englisch | ≈ 0,1 Bill. (Common Corpus OpenWeb, StackExchange) | Nemotron-CC v1 6,3 Bill. (≈ 4,4 Bill. echte Tokens, Lizenz Common Crawl ToU **[U]**), TxT360 ≈ 5 Bill. (ODC-By), FineWeb-Edu 1,3 Bill. (ODC-By) |
| Code | ≈ 0,35 Bill. (Common Corpus Code 0,28, Common Pile ≈ 0,08) | RefineCode 0,96 Bill., Stack v2 0,9 Bill.+ (enthalten Dateien ohne oder mit unklarer Lizenz) |
| Mathe | ≈ 0,01–0,05 Bill. | MegaMath-Web 0,28 Bill., FineMath 0,05 Bill. |
| Deutsch | ≈ 0,17 Bill. (German Commons 0,155 Bill., Common Corpus) | FineWeb-2 Deutsch ≈ 0,4 Bill. (Schätzung), RedPajama-V2 Deutsch 3,0 Bill. |
| Bücher, Wissenschaft, Behörden | ≈ 2,0 Bill. (Common Corpus 1,83 Bill., Common Pile ≈ 0,35 Bill.) | – |
| **Summe** | **≈ 2,5–2,9 Bill. vor Dedup und Qualitätsfilter** | |

Quellen:
[Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus),
[Common Pile](https://arxiv.org/abs/2506.05209),
[German Commons](https://huggingface.co/datasets/coral-nlp/german-commons),
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb),
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[TxT360](https://huggingface.co/datasets/IFM/TxT360),
[Nemotron-CC](https://arxiv.org/abs/2412.02595),
[FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
[Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu),
[MegaMath](https://huggingface.co/datasets/IFM/MegaMath).

**Nicht nutzbar:**

| Datensatz | Grund | Quelle |
|---|---|---|
| Nemotron-CC v2, v2.1, Math-v1, Code-v1 | NVIDIA Data Agreement: nur internes Training, keine Weitergabe, kündbar mit 30 Tagen, danach löschen | [Lizenz](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample/blob/main/LICENSE.md) |
| DCLM-baseline | laut Karte „research use only“ | [Karte](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) |
| Aleph-Alpha-GermanWeb | nicht kommerziell | [Karte](https://huggingface.co/datasets/Aleph-Alpha/Aleph-Alpha-GermanWeb) |
| synthetischer Teil von MegaMath | erzeugt von Qwen2.5-72B (Lizenz „other“) und Llama-3.3 | – |
| NuminaMath | GPT-4-Beteiligung **[U]**, enthält GSM8K-Aufgaben | – |

**Nachtraining, erlaubt** (Erzeuger mit MIT oder Apache-2.0):
- OpenMathReasoning (CC BY 4.0, DeepSeek-R1, QwQ-32B).
- OpenCodeReasoning (CC BY 4.0, DeepSeek-R1).
- OpenCodeInstruct (CC BY 4.0, Qwen2.5-Coder-32B).
- Nemotron-Post-Training-v1 (CC BY 4.0, DeepSeek-R1-0528, Qwen3-235B-A22B).
- Llama-Nemotron-Post-Training nur nach Filter auf den Erzeuger.

**Dekontamination:** Nemotron-CC, FineWeb-Edu, DCLM, NuminaMath und OpenMathReasoning wurden ohne
Dekontamination veröffentlicht. `build_corpus_v6.py scan --evals` läuft über alles.

### 3.2 Selbst erzeugen

Messung von Hugging Face (FinePhrase, vLLM auf H100, Umschreiben von FineWeb-Edu,
[Quelle](https://huggingface.co/spaces/HuggingFaceFW/finephrase)):

| Erzeuger (Lizenz) | Ausgabe-Tokens/s je GPU | H100-Stunden je Mrd. Tokens |
|---|---|---|
| Qwen3-30B-A3B (Apache-2.0), optimiert | 5.310 | 52 |
| gpt-oss-120b (Apache-2.0), optimiert | 6.117 | 45 |
| Qwen3-8B (Apache-2.0) | 6.443 | 43 |

Das ergibt ≈ 50.000 H100-Stunden je Billion synthetischer Tokens, also ≈ 100.000–180.000 € zu Marktpreisen.
Deshalb gehört die Erzeugung in den Förderantrag (Rechenzeit), nicht in unser Budget.

Kimi K2 hat mit 10 Umschreibungen in einer Epoche mehr erreicht als mit den Rohdaten in 10 Epochen (SimpleQA
28,94 statt 23,76, [K2](https://arxiv.org/abs/2507.20534)). Bis zu 4 Epochen auf denselben Daten wirken etwa wie
neue Daten ([Muennighoff](https://arxiv.org/abs/2305.16264)).

### 3.3 Mischung für 6 Bill. Tokens (Vorschlag)

| Anteil | A: nur freie Lizenzen | B: mit Common-Crawl-Web |
|---|---|---|
| Web, allgemein Englisch | 1,6 Bill. offener Text (2–4 Epochen) + 1,5 Bill. Umschreibungen | 2,6 Bill. (Nemotron-CC v1, TxT360, FineWeb-Edu) |
| Code | 1,5 Bill. (≈ 1,0 Bill. Code mit freier Lizenz über Epochen + 0,5 Bill. synthetisch) | 1,5 Bill. |
| Mathe | 0,6 Bill. (überwiegend synthetisch) | 0,6 Bill. (MegaMath-Web, FineMath, synthetisch) |
| Deutsch | 0,6 Bill. (German Commons ×2, Common Corpus, 0,25 Bill. synthetisch) | 0,6 Bill. (FineWeb-2, German Commons) |
| Bücher, Wissenschaft, Recht | in „offener Text“ enthalten | 0,5 Bill. |
| Reasoning im Mid-Training | 0,2 Bill. | 0,2 Bill. |
| **Synthetisch zu erzeugen** | **≈ 2,4 Bill. (≈ 120.000 H100-h)** | **≈ 0,3–0,5 Bill.** |

### 3.4 Rechtlicher Rahmen (Fakten, keine Rechtsberatung)

- **§ 44b UrhG** erlaubt Text und Data Mining, außer der Rechteinhaber widerspricht in maschinenlesbarer Form.
  Kopien sind zu löschen, wenn sie nicht mehr gebraucht werden
  ([Gesetz](https://www.gesetze-im-internet.de/urhg/__44b.html)).
- **OLG Hamburg, 10.12.2025, 5 U 104/24 (LAION):** Ein Vorbehalt in natürlicher Sprache ist nicht maschinenlesbar.
  Die Revision zum BGH ist anhängig **[U]**; beim EuGH ist C-250/25 anhängig.
- **AI Act:**
  - Art. 53 Abs. 1 c und d gelten seit 02.08.2025, Bußgelder ab 02.08.2026.
  - Die Open-Source-Ausnahme befreit nur von a und b, nicht von c und d.
  - Systemisches Risiko gilt ab 10²⁵ FLOP vermutet; unser Ziel liegt bei ≈ 1,3·10²⁴.
- **GPAI Code of Practice** (freiwillig): Crawler beachten robots.txt und maschinenlesbare Opt-outs. Common Crawl
  beachtet robots.txt beim Crawlen und führt eine Opt-out-Liste
  ([CCBot](https://commoncrawl.org/ccbot)).
- **§ 5 UrhG:** Gesetze, Verordnungen und Urteile sind gemeinfrei.
- Vor der Weitergabe von Gewichten an Unternehmen: anwaltliche Prüfung (wie schon in E3 festgehalten).

## 4. „200× besser“: was wirklich wirkt

Basis des Vergleichs: dichtes 250B, ungefiltertes Web, BF16, AdamW, von Grund auf. Multipliziert werden nur
Faktoren, die weitgehend unabhängig sind.

| Hebel | Belegter Gewinn | Ansatz im Plan | Quelle |
|---|---|---|---|
| MoE statt dicht | DeepSeekMoE 16B = LLaMA2-7B mit 39,6 % der Rechenzeit (2,5×); bei 1/32 Aktivierung über 7× | 2,5–4× bei 14 % Aktivierung, mehr bei weniger aktiven Parametern | [DeepSeekMoE](https://github.com/deepseek-ai/DeepSeek-MoE), [2507.17702](https://arxiv.org/abs/2507.17702) |
| FP8 | DeepSeek-V3: Verlust höchstens 0,25 % schlechter als BF16; 1,25–1,5× Durchsatz | 1,3× | [V3](https://arxiv.org/abs/2412.19437), [torchao](https://docs.pytorch.org/ao/stable/workflows/training.html) |
| Muon | Moonlight ≈ 2× laut Autoren; unabhängig getestet 1,4× bei 0,1B, nur 1,1× bei 1,2B. Bewiesener Nutzen in großem Maßstab: Stabilität (K2 ohne Loss-Spitzen) | 1,1–1,3×. Unser 300-Schritte-Test (4,806 statt 5,051) sagt über das Ende des Laufs noch nichts | [Moonlight](https://arxiv.org/abs/2502.16982), [2509.02046](https://arxiv.org/abs/2509.02046) |
| Datenqualität und Umschreiben | FineWeb-Edu: 10× weniger Tokens für dasselbe MMLU (kleines Modell); Umschreiben 2,7–7,7× | zusammen 2–4×, nicht doppelt zählen | [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), [BeyondWeb](https://arxiv.org/abs/2508.10975) |
| Multi-Token-Prediction | +12 % HumanEval, +17 % MBPP bei 13B; 1,8× schnellere Ausgabe | Qualität und Inferenz, beim Training ≈ 1,0–1,1× | [2404.19737](https://arxiv.org/abs/2404.19737) |
| Wachsen und Upcycling | Stacking 1,55×; Upcycling mit ≈ 50 % der Kosten des dichten Modells | 1,0–1,3×, überlappt mit MoE | [2405.15319](https://arxiv.org/abs/2405.15319) |
| Kontextverlängerung (YaRN) | bei DeepSeek-V3 nur 4,3 % der GPU-Stunden | kleiner Posten | [YaRN](https://arxiv.org/abs/2309.00071) |
| **Untermodell destillieren** | Minitron bis 40× weniger Tokens je abgeleitetem Modell; ≈ 100 Mrd. Tokens je Modell | 8B für ≈ 10.000 H100-h statt ≈ 580.000 von Grund auf | [Minitron](https://arxiv.org/abs/2407.14679) |

**Gesamt:**
- **Mitte:** MoE 3 × FP8 1,3 × Muon 1,2 × Daten 3 × MTP und Annealing 1,1 ≈ **15×**.
- **Pessimistisch:** ≈ 7×.
- **Obergrenze:** ≈ 80×, mit 250B-A13B, besten Daten und idealem FP8. Alle Faktoren gleichzeitig an ihrem
  veröffentlichten Maximum gab es noch nie.

**Wo 200× ehrlich stimmt:**
- **Untermodell:** Ein destilliertes 8B ist ≈ 60× günstiger als eines von Grund auf. Im Betrieb kostet ein 3–10B in
  8-Bit je Token ≈ 50–170× weniger als ein dichtes 250B in BF16.
- **Qualität je Euro im Produkt:** Osirus nutzt das Untermodell für schnelle Agents und den großen Mega Bot nur für
  schwere Aufgaben.

Logit-Destillation braucht denselben Tokenizer bei Lehrer und Schüler. Weil wir beide selbst bauen, ist das bei uns
möglich, bei fremden Lehrern nicht.

## 5. Was wir bauen müssen, bevor ein Antrag glaubwürdig ist

Unser Trainer `accel` ist dicht, AdamW, BF16, mit FSDP (`accel/mgpu.py`). Für 250B-A35B fehlt:

| # | Baustein | Warum | Prüfung |
|---|---|---|---|
| B1 | MoE-Schicht: feinkörnige Experten, gemeinsame Experten, Router mit Lastausgleich ohne Hilfsverlust (DeepSeek-V3) | Zielarchitektur | Paritätstest gegen eine Referenz auf CPU; Lastverteilung messen |
| B2 | Expert Parallelism (all-to-all) neben FSDP, mit Überlappung von Kommunikation und Rechnung | 250B passt sonst nicht und wäre langsam | MFU auf 8 → 64 → 256 GPUs (Unterlage 3) |
| B3 | FP8-Matmuls mit feiner Skalierung | 1,3× Durchsatz | Verlust gegen BF16 auf einem Proxy, höchstens 0,25 % Abstand |
| B4 | Muon in accel (gibt es in forge) | Stabilität, 1,1–1,3× | A/B bis zum Ende der Lernrate, gegen ein abgestimmtes AdamW, ab 1B |
| B5 | Multi-Token-Prediction-Kopf | Qualität beim Coden, schnellere Ausgabe | HumanEval und MBPP auf einem Proxy |
| B6 | Tokenizer mit 64k Vokabular (wie in `PLAN_100B_MOE.md`) | heute 8.192; messen, wie viele Bytes je Token auf Code | Bytes je Token, Durchsatz |
| B7 | Checkpoint und Neustart über viele Knoten, Container-Image | Pflicht auf Supercomputern | Abbruch und Neustart ohne Abweichung |
| B8 | Destillation (Logits) und Pruning für das Untermodell | Untermodell 1–10B | Untermodell gegen dasselbe Modell von Grund auf |

Alles bis B7 lässt sich auf CPU und kostenlosen Kontingenten (Kaggle-TPU, Playground) vorbereiten. Die
Skalierungsmessung braucht das Zielsystem.

## 6. Begründung der Rechenzeit (Baustein für den Antrag)

> **Ziel.** Ein europäisches Sprachmodell aus eigener Hand für Code, Mathematik und logisches Denken, auf Deutsch
> und Englisch: Mixture-of-Experts mit 250 Mrd. Parametern, davon 35 Mrd. aktiv je Token, dazu ein destilliertes
> Untermodell mit 1–10 Mrd. Parametern für den Betrieb auf einer einzelnen GPU.
>
> **Rechenbedarf.** Training ≈ 6 × N_aktiv × D FLOP. Mit N_aktiv = 35·10⁹ und D = 6·10¹² Tokens sind das
> 1,26·10²⁴ FLOP. Bei gemessener Leistung von ≈ 400 TFLOP/s je H100-Äquivalent (FP8) sind das ≈ 875.000
> GPU-Stunden.
>
> Dazu kommen:
> - Erzeugung synthetischer Daten: ≈ 50.000 GPU-Stunden je Billion Tokens.
> - Ablationen: ≈ 5 %.
> - Kontextverlängerung: ≈ 4 %.
> - Destillation des Untermodells: ≈ 10.000 GPU-Stunden.
>
> **Warum kein kleineres System reicht.** Gewichte und Optimizer-Zustände eines 250B-Modells brauchen mehrere
> Terabyte Speicher und damit Hunderte GPUs mit schneller Verbindung.
>
> **Vorleistung.** Eigener Trainings-Stack (Rust-CPU-Engine und JAX-Trainer mit FSDP). Gemessen auf 2× B200:
> 222.000 Tokens/s beim 1B und 30.000 Tokens/s beim 10B, mit funktionserhaltendem Wachsen vom 1B zum 10B.
> Außerdem:
> - Benchmark-Harness mit Sandbox-Ausführung;
> - Datenpipeline mit Lizenz-Manifest und Dekontamination;
> - gemessene Skalierung auf [Zielsystem, Zahl GPUs, MFU — nach Unterlage 3 eintragen].
>
> **Offenheit.** [Nach E13 eintragen: Bericht, Paper, Gewichte des Untermodells.]

## 7. Phasen und Tore

| Phase | Was | Rechenzeit | Tor |
|---|---|---|---|
| P0 (jetzt, 0 €) | Anträge, Rechtsform, B1–B7 auf CPU und kostenlosen Kontingenten, Datenpipeline und Lizenz-Manifest | kostenlos | B1–B7 grün |
| P1 | Skalierungsleiter aus 4 MoE-Proxys (z. B. 0,5B-A0,1B, 2B-A0,4B, 8B-A1,5B, 30B-A4B) → Skalierungsgesetz → vorhergesagter Loss des Zielmodells. Dabei auch E10 (aktive Parameter) messen | Playground, Fast Lane, Benchmark, hessian.AI (≤ 50.000 h) | Vorhersage mit Fehlerband im Antrag |
| P2 | Pretraining Quasnir-MoE, Annealing auf den besten Daten in den letzten 10–15 % | Large Scale oder Extreme Scale (0,7–1,2 Mio. h) | Validierungs-Loss auf der Vorhersage; Champion-Gate (Odds Ratio ≥ 2, val höchstens +2 %) |
| P3 | Kontextverlängerung, Nachtraining (SFT, RL mit Tests), RSI, Denkzeit, Agents | ≈ 5–10 % von P2 | Benchmarks greedy und mit Denkzeit, getrennt berichtet |
| P4 | Untermodell 1–10B per Pruning und Destillation | ≈ 10.000 h je Modell | gegen dasselbe Modell von Grund auf; läuft in 8-Bit auf RTX 4090/5090 |
| P5 | Pre-Trained-Version einfrieren: Gewichte, Manifest, Model Card, AI-Act-Zusammenfassung | – | Owner-Freigabe |
| P6 | Fine-Tuning, danach Benchmark-Tests (Reihenfolge des Owners) | – | – |

Bis P2 bleibt der heutige Champion (1B) bestehen. Das 10B bleibt eingefroren.

## 8. Entscheidungen des Owners

| # | Frage | Empfehlung |
|---|---|---|
| E10 | Aktive Parameter: 35B wie vorgegeben, oder 13–15B für 2,5× mehr Tokens bei gleicher Rechenzeit? | In P1 auf Proxys messen und dann entscheiden. Der Antrag rechnet zunächst mit 35B |
| E11 | Webtext aus Common Crawl (Stufe C): Nemotron-CC v1, TxT360, FineWeb-Edu, FineWeb-2? | Ja, mit Opt-out-Filter und Urheberrechts-Policy; sonst ≈ 2,4 Bill. synthetische Tokens nötig |
| E12 | Rechtsform mit KMU-Status, USt-ID, EU-PIC, Lead Engineer | Ja, Voraussetzung für die AI Factories und den EIC |
| E13 | Offenheit: was wird veröffentlicht (Bericht, Paper, Gewichte des Untermodells)? | Bericht und Paper ja; Gewichte des Untermodells unter freier Lizenz erhöhen die Chancen; das große Modell nach Entscheidung |
| E14 | Hochschulpartner für GCS (z. B. Jülich, TU Darmstadt, Stuttgart) | Ja, ansprechen |
| E15 | Allein oder als Partner (EUROPA, OpenEuroLLM, SOOFI)? | Beides prüfen; ein Partner verkürzt den Weg zur Rechenzeit |
| E16 | Mit B1–B7 jetzt beginnen (0 €)? | Ja; ohne gemessene Skalierung ist ein Large-Scale-Antrag schwach |
