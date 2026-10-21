# 10-Schritte-Plan: Quasnir fertig machen, ohne Grok

Stand 20.10.2026. Grok Bot ist ausgesetzt.

**„Fertig“ heißt:** ein Quasnir-Release, das
- das Gate gegen den bisherigen Champion besteht (Odds Ratio ≥ 2 am harten Test, val höchstens +2 %),
- gemessene öffentliche Benchmarks hat (greedy und Denkzeit),
- mit einer Model Card kommt (Daten, Lizenzen, Attribution, Messwerte),
- auf einer GPU läuft.

## Rollen ohne Grok

| Wer | Aufgabe |
|---|---|
| Owner | Entscheidungen; Freigabe jedes Laufs im Environment `RUNPOD` (Osirus) |
| AI Trainer und Kai (Box mit SSH und Volume) | Pod-Arbeit wie bisher Grok (Vorschlag, braucht Owner-Ja) |
| Claude (Cloud-Box, ohne SSH) | Code, Datenwerkzeuge, Prüfung von Konfigurationen und Workflows, Auftragsdateien `gpu-job/*` auf dem Branch, Held-out-Messungen auf der Box, Berichte |

Für jeden Pod gilt:
1. Der Ausführer erzeugt auf seiner Box einen ed25519-Schlüssel und gibt nur den öffentlichen weiter.
2. Ein Auftrag in `gpu-job/` startet den Workflow.
3. Der Owner gibt im Environment frei.

## Die 10 Schritte

| # | Schritt | Wer | Kosten | Stand |
|---|---|---|---|---|
| 1 | Ausführer festlegen; prüfen, dass `RUNPOD` „Required reviewers“ = Owner hat | Owner | – | offen |
| 2 | Lizenzen: FineMath-4+ (Webtext, Stufe C), StackExchange und Wikipedia (CC BY-SA) | Owner | – | offen |
| 3 | Daten-Scan für `put-b-2`: FineMath-4+, StackExchange nur nach Ja (`build_corpus_v6.py scan`) | Kai | CPU auf der Box | Werkzeug fertig |
| 4 | Sauberes Testset für Champion-Entscheidungen; Prüfung gegen die Trainingsdaten mit `scan --evals`, sha256 ins Manifest | AI Trainer baut, Kai prüft | – | in Arbeit |
| 5 | 1B messen (Auftrag 1): greedy, Denkzeit, Oracle | Kai | ≤ 5 $ | Werkzeug fertig |
| 6 | Weg B vorbereiten: Konfiguration, Kostenrechnung, Volume EUR-IS-1 (≥ 0,75 TB), Workflow mit Deckel | Kai liefert, Claude prüft und schreibt, Owner legt das Volume an und gibt frei | Speicher getrennt | Konfiguration fehlt |
| 7 | Weg B (23B) trainieren, mit Checkpoints, Stopp bei 80 % des Deckels | Kai | ≈ 190–215 $ | nach 6 |
| 8 | Gate: 23B gegen den 1B-Champion auf dem sauberen Testset, dazu öffentliche Benchmarks mit Denkzeit | Kai und Claude | ≤ 10 $ | nach 7 |
| 9 | Kontrolliertes RSI auf dem Champion, wenn `oracle_any_pass` ≥ 5 %; Gates und Owner-Ja je Version | Kai | Rest des Budgets | nach 8 |
| 10 | Release und Funding: Model Card, FP8-Inferenz auf einer GPU, Antrag mit der gemessenen Kurve für den 13B-Stamm | Claude entwirft, Owner reicht ein | – | nach 8 |

## Entscheidungen des Owners

**Stand 20.10.2026: Der Owner hat E1–E4 mit Ja beantwortet.** Das Environment `RUNPOD` hat ihn als Required reviewer.
- E1 geändert: Kai und Co. liefen über Grok Bot und fallen mit ihm weg. Ausführer ist jetzt Claude über
  GitOps-Workflows in Osirus. Der Owner gibt jeden Lauf frei.
- E3b: StackExchange nur aus einem Dump bis April 2024.


| # | Frage | Empfehlung |
|---|---|---|
| E1 | Übernehmen AI Trainer und Kai Groks Aufgaben? | Ja. SSH, Volume und Daten liegen schon auf ihrer Box. |
| E2 | FineMath-4+ (ODC-By, Webtext) für Weg B? Unsere Regel verlangt für Stufe C das Owner-Ja; bisher hat nur AI Trainer freigegeben. | Ja, nach dem Scan aus Schritt 3. |
| E3a | Wikipedia (CC BY-SA)? | Ja, mit Attribution im Manifest und in der Model Card. Sie steckt schon im 1B. |
| E3b | StackExchange (CC BY-SA, ≈ 0,1 Mrd. Tokens)? | Nur wenn die Daten aus dem Dump bis April 2024 stammen; dann Ja. Bei neueren Dumps Nein, weil deren Download-Bedingungen LLM-Training ausschließen. |
| E4 | Für 23B kann die Box das Held-out nicht rechnen (≈ 46 GB Gewichte in BF16, 3,5 GB frei, nur CPU). Darf ein Einweg-Testset auf dem Pod laufen? Es wird dort aus einem neuen Seed-Bereich erzeugt, einmal benutzt und danach stillgelegt; nur Ergebnisse verlassen den Pod. | Ja. Sonst gibt es für 23B kein Gate. |

**Hintergrund zu E3.**
- In Deutschland erlaubt § 44b UrhG Text und Data Mining auch kommerziell, solange kein maschinenlesbarer
  Vorbehalt besteht. Ob KI-Training darunter fällt, liegt wegen Kneschke/LAION beim EuGH.
- Ob ShareAlike für die Gewichte gilt, ist ungeklärt.
- Relevant wird das bei der Weitergabe der Gewichte. Vor der Weitergabe an Unternehmen deshalb anwaltlich prüfen
  lassen.
- Das hier ist keine Rechtsberatung.

## Stand 21.10.2026 und Reihenfolge bis zur Pre-Trained-Version

**Lauf train-1-f** (2× B200, beliebiges Rechenzentrum, Gateway am Volume; 78,84 $):
- Daten 1,55 Mrd. Tokens (Stack-Edu 553 Mio., FineMath-4+ 834 Mio., geprüfte Rechenwege 163 Mio.), auf dem Volume
  unter `data/train-1/store`.
- 1B: 2.704 Schritte, ≈ 222.000 Tokens/s (≈ 18 € je Mrd. Tokens), vollständig unter `runs/train-1-f/1b`.
- Wachsen auf 9,43 Mrd. Parameter; 10B: 973 Schritte, ≈ 30.000 Tokens/s (≈ 137 € je Mrd. Tokens). Auf dem Volume nur
  der bf16-Export (`runs/train-1-f/10b/export`); f32-Gewichte und AdamW gingen beim Zurückschreiben verloren (am Ende
  6 MB/s statt 311 MB/s), Ursache offen.
- HumanEval/MBPP greedy: altes 1B 0/0, neues 1B 0/164 und 2/500. Messung mit Denkzeit: Lauf eval-1-a.

**Lauf eval-1-a** (1× A100, 1 h 45 min, 3,34 $), mit Denkzeit (Best-of-N, Selbsttests, Reparatur):

| Modell | HumanEval | MBPP | MBPP-hard | Trainings-Loss |
|---|---|---|---|---|
| altes 1B | 0/164 | 0/500 | – | – |
| neues 1B | 0/164 | 7/500 (1,4 %; greedy 0,4 %) | 4/500 | 2,17 → 1,58 |
| 10B (bf16-Export) | 0/164 | 4/500 | 3/500 | 1,59 → 1,48 |

Obergrenze (mindestens ein Versuch richtig) beim neuen 1B: 1,6 %. Die Schwelle für RSI liegt bei 5 %. Der
Validierungs-Loss fehlt (Timeout beim Holen der Daten, inzwischen behoben in `jobs/volsync.py`); ohne ihn gibt es
keinen Gate-Nachweis.

## Eingefroren am 21.10.2026 (Owner)

Quasnir ist eingefroren. Es läuft kein GPU-Lauf und es wird keiner vorbereitet, bis der Owner es aufhebt.
- Gesichert auf dem Volume `3qlkuqnr1u`: `runs/train-1-f/1b` (vollständig), `runs/train-1-f/10b/export` (nur bf16),
  `data/train-1/store` (1,55 Mrd. Tokens).
- Der Champion bleibt unverändert, bis ein Gate mit Validierungs-Loss ihn ablöst.
- `gpu-job/JOB` in Osirus zeigt auf den abgeschlossenen Lauf `eval-1-a` und startet nichts Neues.
- Der lokale Langlauf bleibt angehalten.
- Nächster Schritt: Alternativen zu Lauf 2 mit höchstens 50 € (`docs/ALTERNATIVEN_LAUF2.md`).
- Ziel des Owners (21.10.2026, abends): Quasnir-MoE 250B-A35B mit Untermodell 1–10B, alle Verfahren bleiben.
  Trainiert wird erst mit Förderung; bis dahin nur kostenlose Vorbereitung (`docs/PLAN_250B_FOERDERUNG.md`).

**Entscheidungen des Owners (21.10.2026):**
- Lösungen offener Modelle mit freier Lizenz (MIT, Apache-2.0) dürfen Trainingsdaten werden; geschlossene Modelle
  bleiben ausgeschlossen. Lizenz je Datensatz prüfen und im Manifest festhalten.
- Kosten: so wenig je Mrd. Tokens wie möglich. 2 € je Mrd. sind nur für Modelle bis etwa 0,2 Mrd. Parameter möglich
  (6 × Parameter Rechenschritte je Token); für 1B und 10B gilt daher ein fester Deckel je Lauf mit gemessenem Gewinn.
- Agents und das 10B zum Ausführen auf einer RTX 4090/5090 (8-Bit-Gewichte); RSI später auf H200.

**Reihenfolge (Owner):**
1. Neuer Champion über das Gate (vor allem anderen).
2. RSI und alle Verfahren: Denkzeit, Agents und Sub-Agents, Research mit Internet (nur lesen), Verifikation.
3. 10B und 1B als Sub-Modell bereit, Agents bereit.
4. Wenn alles steht: als Pre-Trained-Version speichern (eingefrorene Gewichte und Manifest).
5. Fine-Tuning.
6. Benchmark-Tests.

## Von Kai für Schritt 6

- **Modell:** Schichten, d_model, Köpfe, Parameter gesamt und aktiv.
- **Tokens:** je Teil, mit dem Mix `put-b-2` (Tokens je Quelle).
- **Rechenleistung:** GPU-Typ und Anzahl, Tokens/s von Lauf 2.
- **Plan:** geplante Stunden und Kosten.
- **StackExchange:** Herkunft der Daten (Quelle, Dump-Datum).

## Übergabe an AI Trainer, Kai und Tung

Den Text im Kasten gibt der Owner weiter.

```
Grok Bot ist ausgesetzt. Bis auf Weiteres übernehmt ihr seine RunPod-Aufgaben.
Claude liefert Code, Prüfungen und Auftragsdateien. Der Owner gibt jeden Lauf im Environment RUNPOD frei.

Quelle: Repo 67, Branch claude/gitops-multi-model-finetuning-ort4m9.
- Plan: docs/PLAN_10_STEPS.md.
- Befehle für die Aufträge 1–3: docs/PROMPT_GROK_RUNPOD.md; sie gelten unverändert für euch.

Regeln:
- Schlüssel nie in Chat, Log, Commit oder Datei. HF-Token nur als RunPod-Secret HF_TOKEN.
- Jeder Pod hat einen Deckel. Bei 80 % anhalten und melden. Pods nach Gebrauch löschen, Volumes nie.
- Nichts löschen: keine Checkpoints, keine Daten.
- Kein 4-Bit-Training. Held-out-Sets verlassen die Box nie.
- Keine Ausgaben geschlossener Modelle.
- Neue Daten nur nach Scan und Wert-Tor.
- Kein Pod und kein Lauf ohne die Freigabe des Owners.

Sofort:
1. Kai, Scan von FineMath-4+ für put-b-2 (ersetzt Groks v1.1-Scan; ändert und löscht nichts):
   cd 67 && git pull
   python3 data/build_corpus_v6.py scan --out /workspace/pool-p1/scan-finemath \
     --files '/workspace/pool-p1/finemath4plus.*' --text text --kind general --licence odc-by --workers 16
   Er prüft 13-Gramme gegen HumanEval, MBPP und GSM8K-Test und wendet die Inhaltsregeln an.
   Melden: rows, kept, contaminated_by, yield, kept_tokens_est.
   Die Zeilen aus excluded.jsonl kommen nicht in put-b-2.
   StackExchange erst nach der Entscheidung des Owners.
2. AI Trainer, das neue saubere Testset als JSONL ablegen. Kai prüft die Trainingstexte dagegen:
   python3 data/build_corpus_v6.py scan --out <ordner> --files '<trainingstexte>' --evals <ordner-mit-testset>
   Danach sha256 und Zeilenzahl des Testsets melden.
3. Kai, an Claude (als Datei runs/wegb/config.json auf den Branch oder über den Owner):
   - die Weg-B-Konfiguration: Schichten, d_model, Köpfe, Parameter gesamt/aktiv, Tokens je Teil,
     Mix put-b-2, GPU-Typ und Anzahl, Stunden, Kosten;
   - die Tokens/s von Lauf 2;
   - die Herkunft der StackExchange-Daten (Quelle, Dump-Datum).
4. Auftrag 1 (1B messen, Deckel 5 $) nach docs/PROMPT_GROK_RUNPOD.md, sobald der Owner freigibt.
```
