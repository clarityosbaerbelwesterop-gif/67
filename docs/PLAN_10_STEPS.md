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
