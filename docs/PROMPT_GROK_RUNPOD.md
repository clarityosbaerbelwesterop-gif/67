# Prompt für Grok: 1B messen und Corpus v6 auf RunPod bauen

Diesen Text komplett an Grok geben. Er ist in sich vollständig.

---

## Prompt

Du arbeitest für das Projekt forge.
- **Repo `67`**, Branch `claude/gitops-multi-model-finetuning-ort4m9`: Code, Daten-Pipelines, Evals.
- **Repo `Osirus`:** GitOps-Steuerung und der RunPod-Key im GitHub-Environment `RUNPOD`.
- **RunPod-Netzwerk-Volume:** `3qlkuqnr1u` in `US-NC-2`. Dort liegt der 1B-Lauf `s2-run2` mit dem Export bei step 7500.

Du hast zwei Aufträge. Arbeite sie nacheinander ab. Nach jedem Auftrag berichtest du mit Zahlen.

### Feste Regeln

1. **Schlüssel**
   - Schlüssel erscheinen nie in Chat, Log, Commit oder Datei.
   - Der RunPod-Key bleibt im Environment `RUNPOD` oder in deiner RunPod-Sitzung.
   - Ein Hugging-Face-Token liegt nur als RunPod-Secret `HF_TOKEN` am Pod. Die Pipeline liest es aus der Umgebung und gibt
     es nie aus.
2. **Geld**
   - Jeder Pod bekommt vorher einen Deckel: Auftrag 1 höchstens 5 $, Auftrag 2 höchstens 60 $ CPU-Zeit.
   - Bei 80 % des Deckels anhalten und melden.
   - Pods nach Gebrauch immer löschen.
   - Volumes nie löschen.
3. **Nichts löschen**
   - Keine Checkpoints, keine Exporte, keine Daten auf dem Volume.
   - Ausnahme: Die Pipeline entfernt ihre eigenen Download-Caches (`raw/`) und Partitionsdateien selbst.
4. **Gewichte bleiben auf RunPod.** Sie gehen nicht in ein öffentliches Repo, außer der Owner erlaubt es ausdrücklich.
   Rouge und Darus nicht anfassen.
5. **Daten**
   - Nur Lizenzstufe A.
   - Stufe B (Share-Alike) und Stufe C (Web-Crawls) nur nach ausdrücklicher Owner-Freigabe.
   - Keine Ausgaben geschlossener Modelle.
   - Dekontamination gegen HumanEval, MBPP und GSM8K-Test bleibt an. Fehlen die Eval-Sets, bricht die Pipeline ab, und
     das ist richtig so.
6. **Push.** Nur Ergebnis-Dateien (JSON, Manifeste, Logs ohne Schlüssel) auf den Branch oben in `67`. Kein Training,
   kein anderer Branch.
7. **Bei Abweichung** (Prüfsumme falsch, Quelle liefert < 5 % Ausbeute, unbekannte Lizenzen in Masse, Fehler)
   anhalten und melden. Nicht improvisieren.

### Pod-Grundeinrichtung (für beide Aufträge)

Ein CPU-Pod mit 32 vCPU und 50 GB Container-Disk, dazu das Volume `3qlkuqnr1u`, gemountet unter `/vol`:

```bash
apt-get update -qq && apt-get install -y -qq git build-essential python3-pip curl
curl -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
git clone --branch claude/gitops-multi-model-finetuning-ort4m9 https://github.com/clarityosbaerbelwesterop-gif/67 /root/67
cd /root/67 && cargo build --release -p forge && pip install -q numpy pyarrow
python3 -m pytest -q data/tests          # muss grün sein
```

### Auftrag 1: Das 1B (s2-run2, step 7500) auf dem Pod messen

Die Gewichte bleiben auf dem Volume. In das Repo kommen nur die Ergebnisse.

```bash
cd /root/67
EXP=$(find /vol -maxdepth 7 -path '*/export/weights-bf16-step007500.safetensors' | head -1); echo "$EXP"
SHA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['export']['sha256'])" "$(dirname "$EXP")/EXPORT.json")
SEQ=<Trainings-Kontextlänge aus der state.json von step-007500 oder dem RUN-Auftrag; im Zweifel 512>
python3 scripts/import_export.py "$EXP" /root/ckpt/s2-run2-s7500 --expect-sha256 "$SHA" --max-seq-len "$SEQ"
python3 -c "import sys; sys.path.insert(0,'data'); import build_corpus as bc; bc.get_evals()"
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/
mkdir -p runs/evals/s2-run2-s7500
for s in humaneval mbpp humaneval-hard mbpp-hard arc-agi-2; do
  python3 scripts/humaneval.py /root/ckpt/s2-run2-s7500 --suite $s --tokenizer data/tokenizer.json \
    > runs/evals/s2-run2-s7500/$s.json
done
```

Danach pushen:
- `runs/evals/s2-run2-s7500/*.json`
- `EXPORT.json`
- den letzten Abschnitt von `train.jsonl` (ohne Schlüssel)

Melden:
- pass@1 je Suite,
- den Unterschied der harten zur normalen Suite,
- Laufzeit und Kosten.

### Auftrag 2: Corpus v6 bauen (Ziel 250–500 Mrd. Tokens, zuerst Stufe A ≈ 300 Mrd. Budget)

Pipeline: `data/build_corpus_v6.py`. Quellen: `data/sources_v6.json`. Beide Dateien haben eine Beschreibung oben im Kopf.

**0. Owner-Schritt.** Die BigCode-Bedingungen für `bigcode/the-stack-dedup` mit dem Hugging-Face-Konto des Owners
akzeptieren. Das Token als RunPod-Secret `HF_TOKEN` hinterlegen.

**1. Quellen prüfen.**
- Für jede Quelle mit `verify: true` einmal `inspect` ausführen. Für den Stack reicht eine Sprache.
- Stimmen Repo-Name, Dateimuster oder Feldnamen nicht, korrigierst du `data/sources_v6.json` und committest das mit
  Begründung.

```bash
python3 data/build_corpus_v6.py inspect --source stack-python
python3 data/build_corpus_v6.py inspect --source cp-pes2o
```

**2. Reasoning-Spuren erzeugen.** Sie sind verifiziert; der CPU-Lauf dauert etwa 1 Stunde und belegt etwa 11 GB.
Muss vor `plan` passieren, weil `plan` die lokalen Dateien zum Planzeitpunkt erfasst:

```bash
python3 data/build_reasoning.py --synthetic-tokens 5e9 --seed 6 --workers 30 --docs-only
mkdir -p /vol/data/reasoning && cp data/cache/reasoning/docs-*.jsonl /vol/data/reasoning/
```

**3. Planen, filtern, deduplizieren, tokenisieren, packen.**

```bash
O=/vol/data/v6; mkdir -p $O/logs
python3 data/build_corpus_v6.py plan --out $O --tiers A
for w in $(seq 0 31); do
  nohup python3 data/build_corpus_v6.py filter --out $O --worker $w --workers 32 > $O/logs/filter-$w.log 2>&1 &
done
python3 data/build_corpus_v6.py status --out $O          # alle 30 min; Ausbeute je Quelle prüfen
# wenn alle Filter fertig sind:
python3 data/build_corpus_v6.py dedup --out $O --partitions 128
for w in $(seq 0 31); do
  nohup python3 data/build_corpus_v6.py tokenize --out $O --worker $w --workers 32 > $O/logs/tok-$w.log 2>&1 &
done
python3 data/build_corpus_v6.py pack --out $O
```

**4. Pushen** nach `data/manifests/v6/`:
- die Ausgabe von `status`,
- `$O/dedup/dedup.json`,
- `$O/store/manifest.json`,
- die Liste der Quellen mit Dateizahl und Größe aus `plan.json`. Keine Daten.

**5. Melden:**
- Tokens je Quelle und Lizenzstufe,
- Verwerfgründe,
- exakte und Beinahe-Duplikate,
- Speicherbedarf auf dem Volume,
- CPU-Kosten,
- Laufzeit.

**Hinweise**
- Alle Stufen sind wiederaufnehmbar. Ein abgebrochener Worker wird mit demselben Befehl neu gestartet.
- `filter` stoppt selbst, wenn das Volume weniger als 30 GB frei hat. Dann Volume vergrößern (nach Kostenmeldung),
  nichts löschen.
- Erwarteter Speicher für Stufe A: ≈ 1 TB (Text gzip + Tokens + Index). Bei RunPod sind das ≈ 0,05–0,07 $/GB/Monat.

### Nicht in diesem Auftrag

- Kein Training.
- Keine GPU-Pods.
- Keine Stufe B oder C.
- Kein Messtag auf B300. Dafür kommt ein eigener Prompt mit eigenem Budget (`docs/PLAN_100B_MOE.md`, Abschnitt 6).
