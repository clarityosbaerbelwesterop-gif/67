# Prompt für Grok: 1B messen, Daten-Pilot, Wert-Tor

Diesen Text komplett an Grok geben. Er ist in sich vollständig. Stand 19.10.2026, nach Groks Prüfung überarbeitet.

---

## Prompt

Du arbeitest für das Projekt forge.
- **Repo `67`**, Branch `claude/gitops-multi-model-finetuning-ort4m9`: Code, Daten-Pipelines, Evals.
- **Repo `Osirus`:** GitOps-Steuerung und der RunPod-Key im GitHub-Environment `RUNPOD`.
- **RunPod-Netzwerk-Volume:** `3qlkuqnr1u` in `US-NC-2`.

Weg B, der 23B-Schritt, hat beim 300-€-Budget Vorrang. Die drei Aufträge hier sind klein und gedeckelt, zusammen
höchstens 25 $; Speicher läuft getrennt. Nach jedem Auftrag berichtest du mit Zahlen.

### Feste Regeln

1. **Schlüssel**
   - Schlüssel erscheinen nie in Chat, Log, Commit oder Datei.
   - Der RunPod-Key bleibt im Environment `RUNPOD` oder in deiner RunPod-Sitzung.
   - Das Hugging-Face-Token liegt nur als RunPod-Secret `HF_TOKEN` am Pod. Die Pipeline liest es aus der Umgebung,
     gibt es nie aus und sendet es nur an Hugging-Face-Hosts.
2. **Geld**
   - Jeder Pod bekommt vorher einen Deckel.
   - Bei 80 % des Deckels anhalten und melden.
   - Pods nach Gebrauch löschen, Volumes nie.
3. **Nichts löschen**
   - Keine Checkpoints, keine Exporte, keine Daten.
   - Ausnahme: Die Pipeline entfernt ihre eigenen Download-Caches und Partitionsdateien selbst.
4. **Gewichte** bleiben auf RunPod oder der Box. Kein öffentliches Repo ohne ausdrückliches Owner-Ja. Rouge und Darus
   nicht anfassen. Kein 4-Bit-Training.
5. **Evals**
   - Held-out-Sets (L0–L2) verlassen die Box nie.
   - Auf einem Pod laufen nur öffentliche Benchmarks. Für Auftrag 1 hat der Owner am 20.10.2026 Ja gesagt.
6. **Daten**
   - Lizenzstufe A.
   - Stufe C (FineWeb-Edu, FineMath) erst nach ausdrücklichem Owner-Ja, und dann ebenfalls durch das Wert-Tor.
   - Code- und alle neuen Daten kommen nur in den Mix, wenn sie das Wert-Tor bestehen (Auftrag 3).
   - Keine Ausgaben geschlossener Modelle.
7. **Push.** Nur Ergebnis-Dateien (JSON, Manifeste) auf den Branch oben in `67`. Kein anderer Branch.
8. **Bei Abweichung** (Prüfsumme falsch, Ausbeute < 5 %, unbekannte Lizenzen in Masse, Fehler) anhalten und melden.

### Pod-Grundeinrichtung

CPU-Pod mit 32 vCPU und 50 GB Container-Disk, dazu das Volume `3qlkuqnr1u` unter `/vol`:

```bash
apt-get update -qq && apt-get install -y -qq git build-essential python3-pip curl
curl -sSf https://sh.rustup.rs | sh -s -- -y && . "$HOME/.cargo/env"
git clone --branch claude/gitops-multi-model-finetuning-ort4m9 https://github.com/clarityosbaerbelwesterop-gif/67 /root/67
cd /root/67 && cargo build --release -p forge && pip install -q numpy pyarrow
python3 -m pytest -q data/tests          # muss grün sein
```

### Auftrag 1: Öffentliche Benchmarks des 1B (s2-run2, step 7500). Owner-Ja liegt vor (20.10.2026). Deckel 5 $.

Gemessen werden HumanEval, MBPP und die harten Varianten. Die harten Varianten bestehen aus öffentlichen Aufgaben plus
generierten Zusatztests, die bereits im Repo liegen. ARC-AGI-2 entfällt, weil es für 1B wenig aussagt. Die Gewichte
bleiben auf dem Volume.

```bash
cd /root/67
EXP=$(find /vol -maxdepth 7 -path '*/export/weights-bf16-step007500.safetensors' | head -1); echo "$EXP"
SHA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['export']['sha256'])" "$(dirname "$EXP")/EXPORT.json")
SEQ=<Trainings-Kontextlänge aus der state.json von step-007500 oder dem RUN-Auftrag; im Zweifel 512>
python3 scripts/import_export.py "$EXP" /root/ckpt/s2-run2-s7500 --expect-sha256 "$SHA" --max-seq-len "$SEQ"
python3 -c "import sys; sys.path.insert(0,'data'); import build_corpus as bc; bc.get_evals()"
mkdir -p data/stores/evals && cp data/out/evals/*.jsonl data/stores/evals/
mkdir -p runs/evals/s2-run2-s7500
for s in humaneval mbpp humaneval-hard mbpp-hard; do
  python3 scripts/humaneval.py /root/ckpt/s2-run2-s7500 --suite $s --tokenizer data/tokenizer.json \
    --cache-dir /root/gens > runs/evals/s2-run2-s7500/$s.json
done
```

**Denkzeit.** Diesen Teil nur ausführen, wenn die Hochrechnung unter 80 % des Deckels bleibt. Er dekodiert je Aufgabe
bis zu 24 Lösungen und 16 kurze Testzeilen, also grob das 20- bis 25-Fache des greedy-Laufs. Darum zuerst
HumanEval allein laufen lassen, die Zeit messen und erst dann MBPP hochrechnen.

Ablauf:
1. Runde 1: greedy plus 7 Stichproben.
2. Runde 2: nur für Aufgaben, deren Prompt-Tests noch scheitern.
3. Das Modell schreibt als Tester eigene Testzeilen.
4. Eine Reparaturrunde: Das Modell sieht seinen besten Fehlversuch und den Test, an dem er scheitert.

Ausgewählt wird nur über die Tests aus dem Prompt und über die Übereinstimmung mit den eigenen Tests. Die harten
Varianten verwenden alle gespeicherten Lösungen wieder.

```bash
for s in humaneval mbpp humaneval-hard mbpp-hard; do
  python3 scripts/humaneval.py /root/ckpt/s2-run2-s7500 --suite $s --tokenizer data/tokenizer.json \
    --samples 8 --rounds 2 --self-tests 8 --repair --oracle --cache-dir /root/gens \
    > runs/evals/s2-run2-s7500/$s-think.json
done
```

Ergebnis pushen: `runs/evals/s2-run2-s7500/*.json` und `EXPORT.json`. `/root/gens` nicht pushen.

Melden:
- pass@1 je Suite;
- `ablation_pass@1`: greedy, Best-of-8 und volle Denkzeit, aus denselben Lösungen;
- `oracle_any_pass` als Obergrenze;
- `candidates_decoded`;
- den Abstand der harten zur normalen Suite;
- Laufzeit und Kosten.

### Auftrag 2: Pilot Corpus v6, nur Code mit hoher Qualität. Deckel 10 $ CPU, höchstens 100 GB Speicher.

Das Volume hat heute 200 GB, der Pilot passt hinein. Das Upgrade auf 1 TB ist geplant. Speicher läuft nicht über das
300-€-Budget.

Pipeline: `data/build_corpus_v6.py`. Quellen: `data/sources_v6.json`.

**0. Reihenfolge.**
- **Sofort:** `cp-stackv2-edu` (Common Pile Stack-Edu). Nicht gesperrt, Lizenz pro Datei, Qualitätsnote
  `int_score` ≥ 3. Am 20.10. auf dem Hub geprüft.
- **Danach:** `stack-python` und `stack-typescript`, sobald der Owner die BigCode-Bedingungen für
  `bigcode/the-stack-dedup` selbst akzeptiert hat. Das Token kommt nur als RunPod-Secret `HF_TOKEN` an den Pod.
  Dieser zweite Plan bekommt ein eigenes Verzeichnis: `/vol/data/v6-pilot-stack`.

**1. Quellen prüfen.** Stimmt Repo, Dateimuster oder Feld nicht, korrigierst du `data/sources_v6.json` und committest
das mit Begründung.

```bash
python3 data/build_corpus_v6.py inspect --source cp-stackv2-edu
# nach der BigCode-Freigabe:
python3 data/build_corpus_v6.py inspect --source stack-python
```

**2. Pilot**, höchstens 3 Mrd. Tokens je Quelle:

```bash
O=/vol/data/v6-pilot; mkdir -p $O/logs
python3 data/build_corpus_v6.py plan --out $O --tiers A --sources cp-stackv2-edu --pilot-tokens 3e9
for w in $(seq 0 31); do
  nohup python3 data/build_corpus_v6.py filter --out $O --worker $w --workers 32 > $O/logs/filter-$w.log 2>&1 &
done
python3 data/build_corpus_v6.py status --out $O          # alle 30 min
python3 data/build_corpus_v6.py dedup --out $O --partitions 64
for w in $(seq 0 31); do
  nohup python3 data/build_corpus_v6.py tokenize --out $O --worker $w --workers 32 > $O/logs/tok-$w.log 2>&1 &
done
python3 data/build_corpus_v6.py pack --out $O
```

**3. Pushen** nach `data/manifests/v6-pilot/`:
- die Ausgabe von `status`,
- `dedup.json`,
- `store/manifest.json`.

**4. Melden:**
- Tokens je Quelle,
- Verwerfgründe und Ausbeute,
- Duplikate,
- Speicher, CPU-Kosten, Laufzeit.

### Auftrag 3: Wert-Tor (nach Weg B oder mit Owner-Ja). Deckel 10 $ GPU.

Frage: Bringt ein Token der neuen Code-Quelle dem Modell mindestens so viel wie ein Token der bisherigen Daten?

1. **Je Kandidat Arm B bauen.** 30 % Kandidat, 70 % bisheriger Basis-Store; das ist der Store mit der Kept-TSV, mit
   der Lauf 2 trainiert wurde.

   ```bash
   python3 data/build_ablation.py --base-meta <BASIS>/train.meta.json --base-index <BASIS-KEPT.tsv> \
     --cand-meta /vol/data/v6-pilot/store/train.meta.json --cand-index /vol/data/v6-pilot/store/kept.npy \
     --frac 0.3 --tokens 1e9 --seed 1 --out /vol/data/ablation/code-pilot
   ```

   Ein Arm je Quelle: zuerst Stack-Edu, nach der BigCode-Freigabe der Arm aus `/vol/data/v6-pilot-stack/store`.
2. **Läufe auf 1 GPU**, alle vom 100M-Champion (`quasnir-1015e-s5744`), mit gleichem Plan:
   - Arm A, Seed 1234
   - Arm A, Seed 4242 (Rauschen)
   - Arm B, Seed 1234

   ```bash
   python3 -m accel.run_s2 --init <100M-CKPT> --data <ARM>/train.meta.json --clean-tsv <ARM-INDEX> \
     --out /vol/runs/ablation/<ARM>-<SEED> --deadline $(( $(date +%s) + 3600 )) --cap 1000 \
     --batch 128 --accum 4 --seq 512 --seed <SEED>
   ```

   Für Arm A gelten der Basis-`train.meta.json` und die Basis-Kept-TSV, für Arm B `<ARM>/kept.npy`. Vorher einmal
   `python3 -m pytest -q accel/tests/test_parity.py` auf dem GPU-Pod: Der Trainer liest den `.npy`-Index neu.
3. **Messen auf der Box**, wie bisher: Held-out-Code L0–L2 und val general, mit denselben Seeds. Die Exporte der Arme
   (≈ 200 MB je Arm) kommen dafür zur Box; die Held-out-Sets nicht zum Pod.
4. **Entscheidung.** Die Quelle besteht, wenn beide Bedingungen gelten:
   - Arm B ist auf Held-out-Code um mehr als das Doppelte des Seed-Rauschens besser als Arm A.
   - Arm B ist auf val general nicht schlechter als das Rauschen.

   Sonst bleibt sie draußen. Melde die Tabelle: Arm, Seed, Held-out-Code, val general, Kosten.

### Nicht in diesem Auftrag

- Kein voller Corpus v6.
- Keine Stufe C ohne Owner-Ja.
- Kein Training außer den drei Proxyläufen.
- Kein B300-Messtag. Das kommt erst mit Funding (`docs/PLAN_100B_MOE.md`).
