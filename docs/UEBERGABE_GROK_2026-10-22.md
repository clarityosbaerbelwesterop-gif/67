# Übergabe an Grok Bot: was seit dem 20.10.2026 passiert ist und was am Freitag läuft

Stand: 22.10.2026. Kopiere alles ab „Prompt“ in Grok Bot.

---

## Prompt

Du übernimmst wieder Aufgaben im Projekt **Quasnir**. Seit du am 20.10.2026 ausgesetzt wurdest, hat Claude die
Ausführung übernommen. Dieses Dokument sagt dir, was seitdem passiert ist, was gerade wartet und was du tun darfst.
Der Owner entscheidet alles. Jeder GPU-Lauf braucht seine Freigabe im GitHub-Environment `RUNPOD`.

### 1. Repos und Branches

| Repo | Inhalt | Branch |
|---|---|---|
| `clarityosbaerbelwesterop-gif/67` | Trainings-Stack: forge (Rust, CPU), accel (JAX, GPU), Daten, Jobs, Pläne | `claude/gitops-multi-model-finetuning-ort4m9` |
| `clarityosbaerbelwesterop-gif/Osirus` | Produkt (Next.js) und GPU-Steuerung: `.github/workflows/runpod-job.yml`, Auftrag `gpu-job/JOB` | gleicher Branch für die Steuerung, `main` für das Produkt |

**So startet ein GPU-Lauf (GitOps):**
1. Eine Änderung an `gpu-job/JOB` auf dem Branch startet den Workflow.
2. Der Workflow wartet auf die Freigabe des Owners.
3. Der Workflow legt die Pods an, holt die Ergebnisse und löscht die Pods.

**Format von `gpu-job/JOB`:**
1. Run-ID
2. `job=<name>`
3. `sha=<Commit in 67>`
4. `volume=3qlkuqnr1u`
5. `budget=<USD>`
6. `gpu=<TYP|TYP>x<1|2>@<USD/h je GPU>/any`
7. optional `hours=N` für Langläufe

**Daten und Pods:**
- Speicher ist das Network Volume `3qlkuqnr1u` (500 GB, US-NC-2).
- Mit `/any` läuft der Trainings-Pod in jedem Rechenzentrum mit freien GPUs.
- Ein kleiner Gateway-Pod am Volume (`jobs/volgw.py`) liefert die Eingaben und nimmt die Ergebnisse auf
  (`jobs/volsync.py`, mit Prüfsummen, ohne Überschreiben).

### 2. Was seit dem 20.10.2026 passiert ist

| Datum | Ereignis | Ergebnis |
|---|---|---|
| 20.10. | Owner-Entscheidungen E1–E4 | Claude führt über GitOps aus. FineMath und Wikipedia erlaubt, StackExchange nur bis April 2024. Einweg-Testset auf dem Pod erlaubt |
| 20.–21.10. | Läufe train-1-a bis train-1-e | keine B200 frei oder Abbruch; daraus entstanden Bestands-Warten (bis 45 min) und das Gateway |
| 21.10. | **train-1-f**, 2× B200, 78,84 $ | Daten 1,55 Mrd. Tokens. 1B: 2.704 Schritte bei ≈ 222.000 Tokens/s, vollständig unter `runs/train-1-f/1b`. Gewachsen auf 9,43 Mrd. Parameter, 10B: 973 Schritte bei ≈ 30.000 Tokens/s; vom 10B ist nur der bf16-Export gesichert (f32 und AdamW beim Zurückschreiben verloren, 6 MB/s statt 311 MB/s, Ursache offen) |
| 21.10. | Owner: offene Lehrer mit freier Lizenz (MIT, Apache-2.0) erlaubt, geschlossene ausgeschlossen. Kosten so niedrig wie möglich je Mrd. Tokens. Reihenfolge: Champion → RSI und alle Verfahren → 10B und 1B bereit → Pre-Trained-Version → Fine-Tuning → Benchmarks | `docs/PLAN_10_STEPS.md` |
| 21.10. | **eval-1-a**, 1× A100, 3,34 $ | HumanEval/MBPP mit Denkzeit: altes 1B 0/164 und 0/500; neues 1B 0/164 und 7/500 (1,4 %, greedy 0,4 %); 10B 0/164 und 4/500. Obergrenze (Oracle) 1,6 %, Schwelle für RSI 5 %. Der Validierungs-Loss fiel aus (Timeout beim Download; inzwischen mit Wiederaufnahme behoben) |
| 21.10. | Quasnir eingefroren | Champion unverändert, nichts gelöscht |
| 21.10. | Plan für Förderung: Quasnir-MoE 250B mit 35B aktiv, Untermodell 1–10B | `docs/PLAN_250B_FOERDERUNG.md`. Bedarf ≈ 0,7–1,2 Mio. H100-Stunden; EuroHPC AI Factories, GCS, EIC. Training erst mit Förderung |
| 21.10. | Alternativen zu Lauf 2 mit höchstens 50 € | `docs/ALTERNATIVEN_LAUF2.md` (kostenlose TPU-Zeit, fertige Daten mit freier Lizenz, kleineres Modell) |
| 21.10. | Osirus: Prüfung und Prompt für Osirus Studio | `Osirus/docs/PROMPT_OSIRUS_STUDIO.md` (Bots, Mega Bot, Mission Control, neue Oberfläche, 28 Fehler mit Datei und Zeile); Umsetzung durch Grok auf eigenen Branches |
| 22.10. | Owner: Quasnir auf Basis von **Qwen3.8-27B** (Apache-2.0), nur Fine-Tuning; Quasnir-1B als Vorsortierer; Name „Quasnir“ für das System, mit Hinweis auf die Basis | `docs/PLAN_QWEN.md` |
| 22.10. | Owner: Ja zu den freien Datensätzen (OpenCodeInstruct, OpenCodeReasoning, Cosmopedia, Stack-Edu mit freier Lizenz) | E7 |
| 22.10. | Owner: ein Lauf für Freitag, der das 1B auf 2× B200 fertigstellt, fein abstimmt und misst, höchstens 300–400 € | Lauf **train-final-a** (Abschnitt 3) |

### 3. Was wartet (Freigabe durch den Owner am Freitag)

**Reihenfolge:** Zuerst Run #12 freigeben. Run #13 startet erst, wenn #12 fertig ist, und wartet dann selbst auf
die Freigabe.

**Run #12: eval-qwen-a** (Qwen3.8-27B messen, Schritt 1 des Qwen-Pfads)
- 1× H200 oder H100, höchstens 4,50 $/h, Budget 15 $, wahrscheinlich 3–6 $.
- vLLM 0.32.0. HumanEval, MBPP und die schweren Varianten, Denken aus und an, mit den Einstellungen der Modellkarte.
- Ergebnisse unter `runs/eval-qwen-a/record` auf dem Volume.

**Run #13: train-final-a** (Langlauf; `jobs/train-final.sh` in 67 bei `6e57c94`)
- 2× B200, höchstens 8,00 $/h je GPU, höchstens 22 h, Budget 400 $ (Obergrenze 460 $).
- Erwartete Kosten: etwa 265–355 $ (≈ 230–305 €), je nach B200-Preis.
- Der Owner muss vorher genug RunPod-Guthaben haben (≥ 400 €). Sonst hält RunPod die Pods an.

| Phase | Was | Zeit (bei 22 h) |
|---|---|---|
| setup, pull | JAX, forge; das 1B aus train-1-f, der train-1-Store | ≈ 30 min |
| data (CPUs) | Store `data/final-pt`: Stack-Edu mit freier Lizenz (6 Mrd.), FineMath-4+ (2 Mrd.), OpenCoder-Algorithmen (0,8 Mrd.), geprüfte Rechenwege. Store `data/final-sft`: OpenCodeInstruct (nur Lösungen, die alle Tests bestehen), Rechenwege, etwas Wiederholung. Beide dekontaminiert gegen HumanEval, MBPP, GSM8K | ≈ 2,5 h, parallel zu p0 |
| p0-warm | GPUs trainieren das 1B weiter auf dem train-1-Store, konstante Lernrate, Kontext 1.024 | parallel zu data |
| pt-1b | das 1B auf dem neuen Store (Re-Warmup, WSD-Abkling) | ≈ 11 h, ≈ 8–9 Mrd. Tokens |
| sft-1b | Fine-Tuning des 1B, höchstens 2 Epochen | 2 h |
| eval | Validierungs-Loss von 4 Modellen (altes 1B, train-1-f 1B, pt-1b, sft-1b) auf demselben Split; HumanEval/MBPP und -hard mit Denkzeit und Oracle für pt-1b und sft-1b | ≈ 75 min |
| sft-10b | nur wenn ≥ 230 min übrig: das 10B (bf16-Export) auf demselben Fine-Tuning-Store, danach gemessen; nur sein bf16-Export geht aufs Volume | ≈ 3,8 h |
| done | Bericht und `DONE.json` unter `runs/train-final-a/` | ≈ 1 h Reserve |

**Sicherheit im Langlauf:**
- Beide Pods prüfen beim Start, ob sie sich mit ihrem eigenen, auf den Pod beschränkten RunPod-Schlüssel selbst
  beenden können (`jobs/pod_guard.sh`). Wenn nicht, bricht der Job nach wenigen Minuten ab und der Workflow löscht
  beide Pods.
- Der Trainings-Pod endet 10 Minuten nach dem Job, spätestens 10 Minuten nach seiner Deadline.
- Der Gateway-Pod endet 5 Minuten nach `DONE.json`, spätestens an seiner eigenen Deadline.
- Nach 5,5 h verfolgen die Jobs `watch-1` bis `watch-4` den öffentlichen, bereinigten Status (ohne Token, ohne
  Logs).
- Der Job `cleanup` löscht am Ende alles, was noch läuft. Er braucht noch einmal die Freigabe des Owners.
- Jede Phase speichert ihr Ergebnis aufs Volume, bevor die nächste beginnt. Nichts wird gelöscht oder
  überschrieben.

**Vorher geprüft:** 121 Tests in 67 grün (davon neu: Pod-Wächter, Freitag-Lauf mit Attrappen in vier Szenarien,
Vorlagen für Datensätze). Der Workflow ist mit actionlint ohne Befund. Die Pod-Startlogik ist in drei Szenarien
simuliert.

**Nicht geprüft:** ein echter Lauf. Ob RunPod den Pod-Schlüssel (`RUNPOD_API_KEY` im Pod) so bereitstellt wie
erwartet, zeigt erst der Start. Ohne ihn bricht der Lauf ab und kostet nur Minuten.

### 4. Was du tun darfst und was nicht

**Darfst du:**
- Code und Dokumente in beiden Repos lesen, prüfen und kommentieren.
- Für Osirus Studio auf eigenen Branches (`grok/...`) arbeiten und Pull Requests gegen `main` stellen, nach
  `Osirus/docs/PROMPT_OSIRUS_STUDIO.md`, mit grünen Prüfungen (typecheck, lint, vitest, Playwright).
- Nach dem Lauf die Ergebnisse aus `runs/train-final-a/record` und `runs/eval-qwen-a/record` auswerten und
  Vorschläge machen.

**Darfst du nicht:**
- `gpu-job/JOB` ändern oder auf `claude/gitops-multi-model-finetuning-ort4m9` pushen. Jeder Push dieser Datei startet
  einen Lauf.
- Pods anlegen, starten oder löschen, Volumes ändern, Checkpoints oder Daten löschen.
- Schlüssel in Chats, Logs, Commits oder Dateien schreiben.
- In 4-Bit trainieren oder Ausgaben geschlossener Modelle als Trainingsdaten nutzen.
- Zahlen nennen, die nicht gemessen sind.
- Qwen-basierte Gewichte als „von Grund auf selbst trainiert“ bezeichnen.

### 5. Wie es nach Freitag weitergeht

1. **Ergebnisse von eval-qwen-a:** Ausgangswert von Qwen3.8-27B. Danach Schritt 2 des Qwen-Pfads: eine
   Agent-Suite für das Osirus-Format.
2. **Ergebnisse von train-final-a:**
   - Validierungs-Loss und Benchmarks entscheiden über den neuen 1B-Champion (Gate: MBPP über 10 %, HumanEval über
     5 % greedy; Validierungs-Loss nicht schlechter).
   - Oracle ≥ 5 % → RSI auf dem 1B.
   - Das fertige 1B wird Vorsortierer im System Quasnir (Qwen-Basis mit unseren LoRA-Gewichten, Sandbox als letzte
     Prüfung).
3. **Förderung** für das 250B-MoE nach `docs/PLAN_250B_FOERDERUNG.md`. Bis dahin keine großen Läufe.
4. **Osirus Studio:** PR 1 (Chat über die Runtime) zuerst. Die Prüfpunkte dazu stehen im Chat mit dem Owner:
   - schlanker Chat-Weg statt des vollen Agent-Laufs;
   - `/api/chat/turns` entfernen;
   - Modellwahl erhalten;
   - Regressionstest vorher rot.
