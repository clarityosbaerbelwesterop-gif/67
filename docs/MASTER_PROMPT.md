# Master-Prompt forge: vom Start bis zum Ende

Stand: 2026-10-18, Commit `83521dc`, Branch `claude/gitops-multi-model-finetuning-ort4m9`.
Dieses Dokument erklärt forge von Grund auf, nennt den gemessenen Produktstand und den
vollständigen Fahrplan bis zum Ziel. Es ersetzt keine Detaildokumente, sondern ordnet sie:
`docs/HANDOVER.md` (Betrieb und Regeln), `docs/PROMPT_PHASE_A.md` (100 M),
`docs/PROMPT_SCALING.md` (Bausteine und Speicherbudget).

Kopiere alles ab „Prompt“ in den ausführenden Agenten.

---

## Prompt

Du übernimmst **forge**, Repo `clarityosbaerbelwesterop-gif/67`, Branch
`claude/gitops-multi-model-finetuning-ort4m9`. Du brauchst ein Terminal auf der
Trainingsmaschine und Schreibrecht auf diesen Branch. Ohne Terminal kannst du nur planen
und prüfen. Arbeite in der Reihenfolge dieses Dokuments und berichte nur gemessene Zahlen.

### 1. Was forge ist

forge ist ein eigener Trainings-Stack, komplett als Code, ohne fremde Basisgewichte. Jedes
Modell wird von Grund auf trainiert.

| Teil | Ort | Aufgabe |
|---|---|---|
| Kern-Engine (Rust) | `core_engine/`, Binary `forge` (`cargo build --release -p forge`) | Training, Eval, Generierung, Logits; CPU-Kernels AVX2, AVX-512, AMX |
| Orchestrierung (Python) | `scripts/longrun.py` | Zyklen aus Training, RSI, Champion-Wahl, Releases; Platzhygiene |
| Champion-Regeln | `scripts/champions.py`, `scripts/releases.py`, `scripts/tiers.py` | Wer Champion wird, wann es ein Release gibt, welche Stufe |
| RSI | `training/rsi/loop.py` | Selbstverbesserung: Aufgaben lösen, in Sandbox prüfen, nur Bestandenes trainieren |
| Beschleuniger (JAX) | `accel/` | Exakter Port von forge für GPU/TPU (25 Paritätstests); `accel/mgpu.py` FSDP über viele Geräte; `accel/run_s2.py` Lauf mit Zeitbudget |
| Cluster | `cluster/`, `.github/workflows/forge-cluster.yml` | DiLoCo über mehrere Rechner; gehostete Runner nur für Tests |
| Sicherung | `scripts/resume_pack.py` | Verschlüsselte Übergabepakete (age, öffentlicher Schlüssel) |
| Tokenizer | `data/tokenizer.json` | Byte-BPE, vocab 8192, sha256 `fe1ee90a…8012` |

### 2. Die Modelle

| Familie | Zweck | Zugang |
|---|---|---|
| **base** | Fundament, auf dem die anderen aufbauen | öffentlich |
| **Quasnir** | Coding-Modell, **aktueller Fokus** | öffentlich |
| **Rouge** | eingeschränkt, nur ausgewählte Unternehmen | Gewichte nur verschlüsselt, Modus 0700 |
| **Darus** | Flaggschiff, Merge aus den anderen | ruht, nur im verschlüsselten Paket |

Rouge und Darus sind auf Anweisung des Owners ausgeschaltet (`disabled_families`). Es läuft
ein einziger maßgeblicher Lauf: Quasnir.

**Architektur der bisherigen Champions** (alle Familien gleich, 12.582.912 Parameter):
dim 384, 6 Layer, 6 Heads, 2 KV-Heads, head_dim 64, FFN 1024, vocab 8192, Kontext 512,
RMSNorm eps 1e-5, RoPE θ 10.000, Embedding an den Ausgabekopf gebunden.

**Größenleiter:** 12,58 M (Champion) → 100,66 M (Phase A: dim 768, 15 Layer) →
959,5 M (s2, gewachsen: dim 1920, 24 Layer, 30 Heads, 10 KV-Heads, FFN 5120;
`accel/mgpu.py`).

### 3. Gemessener Produktstand

**Champions** (`checkpoints/champions.json`, Zyklus 5, 2026-10-14):

| Familie | Lauf | sha256 | Messwerte |
|---|---|---|---|
| base | base-g4 FINAL | `104654de…e52b` | val 1,992 |
| Quasnir | rsi-quasnir-g3-r8 step 120 | `4da6575f…588a` | Held-out L0 41/128, L1 58/128, L2 2/128 |

Quasnir-Baseline auf den aktuellen v2-Stores (`data/manifests/evals.sha256.json`):
val code 1,830 / general 2,735 (seed 1234), 1,862 / 2,753 (seed 4242), batch 8, seq 256,
32 Batches. Die Werte 2,219 / 2,873 in `champions.json` stammen von den alten v3-Stores und
sind damit nicht vergleichbar.

**Releases** (`runs/releases.json`): base 2, Quasnir 2, Rouge 2, Darus 1.

**Öffentliche Benchmarks** (`runs/benchmarks.jsonl`, greedy, 192 Tokens): HumanEval 0/164
und MBPP 0/500 für alle 12,6-M-Modelle. Frontier-Schwellen (`scripts/tiers.py`):
HumanEval 0,99, MBPP 0,95, ARC-AGI-2 0,942. **Kein Modell ist frontier oder dangerous.**

**Abgeschlossen:**
- Phase 0: alle 13 Einträge von `checkpoints/MANIFEST.json` auf Commit `031b008` geprüft.
- JAX-Port bitgenau (Logits ≤ 2,2e-5, Eval-Loss identisch).
- Muon-Optimierer: von Grund auf val 4,806 statt 5,051 (AdamW); beim Weitertraining kein Unterschied.
- DiLoCo-Cluster: 4 Runner, von Grund auf, val 9,01 → 5,49 nach einer Runde.

**Laut Operator-Bericht vom 19.10. (im Repo noch nicht belegt):**
- 100 M `quasnir-1015e-s5744`: H200, 5.744 Schritte, ~1,5 Mrd. Tokens, ~300k Tokens/s; schlug
  g3-r8 auf dekontaminiertem Code-val 1,754 gegen 2,288.
- 1B Lauf 1 (1 × B200) knapp neuer Champion; Lauf 2 bis step-007500, beim Code-val in 122 bzw.
  128 von 128 gepaarten Paketen besser. Allgemeinwissen-Test offen. Tokens gesamt unbekannt.
- MoE-Upcycling 1B → 23B gebaut und getestet, nicht gestartet.

**Beim Operator, nicht im Repo:**
- `quasnir-phase-a-100m` step-005744. Seine Konfiguration ist in `accel/mgpu.py` als Preset
  `champion` hinterlegt; Messwerte, sha256 und die Gate-A-Entscheidung fehlen im Repo.
- s2-Startgewichte (959.541.120 Parameter) und der Stand von `s2_run1` (1.000 Schritte).

**Container-Langlauf:** absichtlich gestoppt (`runs/longrun/STOP`), letzter Stand base-g4
step 16324, val 1,966. Training läuft beim Operator.

### 4. Was wo liegt

| Inhalt | Ort | Zugang |
|---|---|---|
| Gesamter Code und alle Dokumente | Branch, Commit `83521dc` | öffentlich |
| base-g5 step 500 (`8b36f7fd…`, val 1,999), Quasnir-Champion (`4da6575f…`), quasnir-g4 step 4567 (`d7595a91…`, val 1,772) | `checkpoints/` mit `MANIFEST.json` | öffentlich; Teile per `cat` zusammensetzen, sha256 prüfen |
| Alle vier Familien-Champions inkl. Rouge/Darus, base-g4 step 16324 mit Optimizer | `resume/pack/` (8 Teile, 871 MB, 232 Dateien) | verschlüsselt an `age17f8gymjphpzklavte56njerdfgjxtl5hf66dgdymyqx4p39lt55qmzntqg` |
| Tokenizer | `data/tokenizer.json` | öffentlich |
| Eval-Sets, Held-out-Hashes, Quasnir-Baseline | `data/manifests/evals.sha256.json` | öffentlich |
| Korpora (7,4 GB) | nicht im Git; aus `data/manifests/` mit sha256-Pins neu baubar | – |
| Konfigurationen | `checkpoints/configs/` (`longrun.json`, `longrun-quasnir-focus.json`) | öffentlich |
| Korpus v5 (19.10.): `code-v5add`, `general-v5add` (nur neue Pakete), AAA-Stufe `code-v5aaa`, `general-v5aaa` | nicht im Git; Manifeste `data/manifests/v5/` (`stores.json`, `aaa.json`) | neu baubar: `data/build_corpus_v5.py`, dann `--aaa` |
| Reasoning-Store `reasoning-v5` (Problem/Thinking/Answer; GSM8K-Train, geprüfte AQuA-Teilmenge, 20 verifizierte synthetische Familien) | nicht im Git; Manifest `data/manifests/v5/reasoning.json` | neu baubar: `data/build_reasoning.py` (deterministisch, Seed 5) |

Wiederherstellen des Pakets:
`python3 scripts/resume_pack.py restore --age-identity forge.key`.
Die Datei `forge.key` liegt nur beim Operator.

### 5. Schlüssel und Geheimnisse

Schlüssel gehören **nie** in einen Chat, ein Issue, einen Commit oder eine Log-Datei. Wer
einen Schlüssel braucht, bekommt ihn direkt vom Owner über den jeweiligen Secret-Speicher.

| Schlüssel | Liegt in | Hinweis |
|---|---|---|
| age-Privatschlüssel `forge.key` | nur auf der Maschine des Operators | Der öffentliche Schlüssel oben ist unbedenklich. |
| UNOROUTER-Keys | GitHub-Actions-Secrets im Repo Osirus | Setzen über Settings → Secrets and variables → Actions. |
| `FORGE_PACK_KEY` | nicht gesetzt | Altes AES-Verfahren, durch age ersetzt. |
| Hugging-Face-Token | nicht nötig für das öffentliche Qwen | Für private Uploads als `HF_TOKEN` nur auf der Trainingsmaschine. |
| Git-Push | eigene Zugangsdaten des Operators | – |

Taucht ein Schlüssel versehentlich in einem Chat oder Commit auf, gilt er als verbrannt:
sofort beim Anbieter widerrufen und neu ausstellen.

### 6. Fahrplan bis zum Ende

Jede Stufe hat ein messbares Tor. Ohne bestandenes Tor beginnt die nächste Stufe nicht.

**Stufe A – 100 M abschließen** (`docs/PROMPT_PHASE_A.md`)
1. `quasnir-phase-a-100m` step-005744 in `checkpoints/MANIFEST.json` eintragen: sha256,
   Speicherort, Teile unter 100 MB nur nach Freigabe des Owners.
2. Messen, mit derselben Methode wie die Baseline:
   - Held-out L0 bis L2 (je 128 Aufgaben);
   - val code und general, seeds 1234 und 4242;
   - HumanEval, MBPP, ARC-AGI-2 und der harte Test HumanEval-hard / MBPP-hard
     (`scripts/humaneval.py CKPT --suite humaneval-hard`; ~37 Zusatztests je Aufgabe,
     `data/build_hardbench.py`). bf16-Exporte lädt forge direkt
     (`scripts/import_export.py EXPORT OUT`).
3. Gate A: Arbiter gegen `4da6575f` (tol 0,02, margin 0,03).
4. Release **Quasnir 3** nur bei Odds Ratio ≥ 2 (unteres 95-%-Band), ohne McNemar-Verlust
   und mit val höchstens +2 %. Diese Regel setzt die Owner-Vorgabe „jede Version mindestens
   2× besser“ um.

**Stufe S2 – ~1 Mrd. Parameter** (`accel/mgpu.py`, `accel/run_s2.py`)
1. `s2_run1` mit Deadline auf einem GPU-Pod: 1.000 Schritte, warmup 300, WSD-Abklingen über
   die letzten 200 Schritte, lr 2e-4 → 3e-5.
2. Daten nur gefiltert über die Kept-Docs-TSV. Fehlt der Filter, bricht der Trainer ab
   (`DataSafetyError`). Das bleibt so.
3. **Dichtes 1B zuerst** (Owner-Entscheidung 19.10.): weitertrainieren bis etwa 19 Mrd. Tokens,
   dann benchmarken; MoE-Upcycling erst danach. Ein 23B-Checkpoint mit AdamW hat 276–368 GB und
   passt nicht auf ein 200-GB-Volume.
   Datenmix dafür: base-v4 plus die AAA-Stufe v5 und `reasoning-v5`. Der Reasoning-Anteil ist
   ein Experiment, kein fester Wert. Er wird an val und am harten Test gemessen, bevor er steigt.
   Synthetische Spuren nicht öfter als etwa 2× wiederholen.
   Abruf des 1B-Exports: Diese Cloud-Box hat kein ausgehendes SSH, `runpod-fetch` braucht es aber.
   Einen Runner-seitigen Abruf (nur Chiffretext, Entwurfs-Release) legt der Owner selbst an oder
   gibt ihn ausdrücklich frei.
   Heute: Weg B (23B-Schritt) aus dem 300-€-Budget. Plan fürs Funding: `docs/PLAN_100B_MOE.md`
   (BTX, 108B gesamt, ≈ 26B aktiv, ≈ 56.500 $ in FP8). Daten: `data/build_corpus_v6.py`, zuerst als Pilot.
   Neue Quellen nur über das Wert-Tor (`data/build_ablation.py`). Auftrag an Grok: `docs/PROMPT_GROK_RUNPOD.md`.
   Subagenten ab 13–15B (Rollen, Workflows, GPU-Rechnung): `docs/PLAN_SUBAGENTS.md`.
   Grok Bot ist seit 20.10.2026 ausgesetzt. Plan ohne Grok, offene Owner-Entscheidungen E1–E4: `docs/PLAN_10_STEPS.md`.
4. Danach der volle s2-Lauf. Chinchilla-Ziel: etwa 19 Mrd. Tokens, also
   6 · 9,6e8 · 1,9e10 ≈ 1,1e20 FLOP. Das sind etwa 3 Tage auf einer H100 bei 40 % MFU oder
   etwa 10 Stunden auf 8 H100.
5. Gate S2 wie Gate A, gegen den Stufe-A-Champion.
6. Größen: Ein voller f32-Checkpoint hat 11,5 GB und gehört nie ins Git. Der bf16-Export hat
   1,92 GB und wird mit sha256 dort abgelegt, wo der Owner es freigibt.

**Stufe B – Bausteine** (`docs/PROMPT_SCALING.md`)
- Aufbau: gemeinsamer Stamm plus kleine Delta-Experten mit Router.
- Zuerst 8 Experten, zum Beispiel Python, JS/TS, Rust/C++, SQL, Mathe, Algorithmen,
  Refactoring, Tests.
- Jeder Experte muss den Stamm auf seinem Gebiet messbar schlagen, ohne anderswo zu
  verlieren; sonst wird er verworfen.

**Stufe C – Skalieren im Speicherbudget 20–50 GB**
- Eindeutige Parameter = Stamm + E × Delta.
- Beispiel 100-M-Stamm mit Rang-1-Deltas: 203.520 Parameter je Experte. 200.000 Experten
  ergeben 40,7 Mrd. eindeutige Parameter, das sind 20 GB in int4.
- 200.000 unabhängige 100-M-Kopien bräuchten 40 TB. Das ist nicht machbar.
- Fähigkeit hängt an den aktiven Parametern pro Token und an der Trainingsmenge, nicht an der
  Nennzahl. Berichte deshalb immer beide Zahlen.

**Stufe D – Metis-Rat und Dauerbetrieb**
- Metis: Mehrere Champions beraten zur Inferenzzeit (Vorschlag, Kritik, Abstimmung). Ein
  Gewichtsmerge mit Fremdmodellen ist wegen anderer Architektur und anderem Tokenizer nicht
  möglich.
- Qwen3.8-27B (Apache-2.0) dient nur als Lehrer. Seine Ausgaben gehen erst in die Daten,
  wenn sie in der Sandbox bestanden haben und ein eigener A/B-Test sie bestätigt.
- RSI läuft auf jedem neuen Champion mit allen Gates:
  - held-out +0,03;
  - val tol 0,01;
  - Anker tol 0,02;
  - McNemar α 0,05;
  - Dekontamination;
  - Lineage.

**Ende** ist erreicht, wenn ein forge-Modell auf denselben dekontaminierten Benchmarks unter
gleichem Protokoll die veröffentlichten Werte von Astra, Opus 5.5 und GPT-7 erreicht oder
übertrifft. Benchmarks: HumanEval, MBPP, ARC-AGI-2, ergänzt um LiveCodeBench und
SWE-bench Verified. Vorher gibt es keine Vergleichsbehauptung.

**Ehrliche Rechengröße:** Öffentliche Schätzungen für Frontier-Modelle liegen bei 1e25 bis
1e26 FLOP. Der volle s2-Lauf hat 1,1e20 FLOP, also 100.000- bis 1.000.000-mal weniger. Code
kann Rechenleistung nicht ersetzen. Er kann sie nur besser ausnutzen: Muon, Wachstum statt
Neustart, Datenqualität, Experten, RSI. Plane deshalb jede Stufe mit gemessenen Tokens/s
und nenne die ETA.

### 7. Stufen und Pflichten

| Stufe | Bedingung | Pflicht |
|---|---|---|
| champion | gewinnt den Arbiter | normal weiter |
| frontier | erreicht die Schwellen aus `scripts/tiers.py` | Owner sofort informieren |
| dangerous | schlägt alle Tests, löst jede Coding-Aufgabe, entwickelt ein Untermodell und verselbstständigt sich | Owner sofort informieren; nie verwerfen; behalten und weiterentwickeln; RSI anhalten; nur für ausgewählte VIP-Unternehmen, niemals für Einzelpersonen; Release erst nach Freigabe des Owners |

### 8. Feste Regeln

- **Löschen:** Nichts ohne ausdrückliche Freigabe des Owners, außer der eingebauten
  Checkpoint-Rotation.
- **Rouge:** Gewichte nie unverschlüsselt an öffentliche Orte.
- **GitHub:** Gehostete Runner nur für Tests, nie als Trainingsfarm. Eine Sperre träfe alle
  Repos.
- **Push:** nur auf diesen Branch. Ausnahme ist der Cluster-Workflow auf `main`.
- **Commits:** keine Modellkennungen in Commits, Code oder Dokumenten.
- **Zahlen:** keine erfundenen. Vergleiche nur über gemessene Benchmarks.
- **Code:** Änderungen nur mit Tests (`python3 -m pytest -q scripts/tests`,
  `python3 -m pytest -q accel/tests/test_parity.py`) und zuerst in einer separaten Arbeitskopie.
- **Sicherheitsprüfungen:** Werden sie abgelehnt, nicht umgehen, sondern den Owner fragen.
- **Präzision:** kein 4-Bit-Training (NVFP4); FP8 und BF16 sind erlaubt.
- **Evals:** Held-out-Sets verlassen die Box nie. Öffentliche Benchmarks auf Pods nur nach Owner-Ja
  (für das 1B, s2-run2 step 7500, erteilt am 20.10.2026).
- **Denkzeit:** `scripts/humaneval.py` bietet Stichproben, weitere Runden für ungelöste Aufgaben, eigene
  Testzeilen des Modells und eine Reparaturrunde (`--samples --rounds --self-tests --repair`).
  - Ausgewählt wird nur mit den Tests im Prompt und mit der Übereinstimmung der Lösungen untereinander.
  - Die verdeckten Tests laufen einmal auf der gewählten Lösung.
  - Berichtet werden immer `ablation_pass@1` (greedy, Best-of-N, voll) und die Zahl der dekodierten
    Lösungen.
  - `oracle_any_pass` ist eine Obergrenze und nie ein pass@1.
  - Eine Stufe bleibt nur, wenn sie gemessen etwas bringt.
- **Daten auf einer Maschine prüfen:** `python3 data/build_corpus_v6.py scan --files GLOB` (13-Gramm-Dekontamination und Inhaltsregeln, ändert nichts).
- **Daten:** Code- und neue Daten nur bei hoher Qualität und nur mit bestandenem Wert-Tor
  (Ersatz-Ablation am 100M-Proxy, `docs/PLAN_100B_MOE.md` Abschnitt 6).

### 9. Berichte

Alle 2 Stunden, 3–6 Zeilen auf Deutsch, nur Messwerte:
1. Stufe, Schritt / Ziel, Tokens/s, ETA.
2. Trainings-Loss, val code/general mit Trend.
3. Gate-Entscheidungen und Releases.
4. Ausfälle, freier Platz, Auffälligkeiten.

Sofortmeldung bei: neuem Champion, Release, frontier oder dangerous, Abbruch, NaN.
