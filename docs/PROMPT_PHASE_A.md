# Prompt: Phase A – Quasnir auf 100 M Parameter wachsen lassen und trainieren

Stand: 2026-10-14. Phase 0 ist abgeschlossen und unabhängig geprüft: Auf Commit
`031b008` stimmen alle 13 Einträge von `checkpoints/MANIFEST.json`. Dieses Dokument ist
der Auftrag für Phase A. `docs/HANDOVER.md` und `docs/PROMPT_SCALING.md` gelten weiter.
Kopiere alles ab „Prompt“.

---

## Prompt

Du führst **Phase A** für forge aus: Repo `clarityosbaerbelwesterop-gif/67`, Branch
`claude/gitops-multi-model-finetuning-ort4m9`.

**Ziel:** Der Quasnir-Champion (12,58 M Parameter) wächst ohne Fortschrittsverlust auf
**100,66 M Parameter** und wird dann weitertrainiert, bis er den Champion messbar schlägt.

### Voraussetzungen (vor dem Start prüfen und berichten)

- **Ausführende Stelle:** Ein Agent mit Terminal auf der Trainingsmaschine und Schreibrecht auf den Branch. Ein Chat ohne Terminal kann nur planen und prüfen, aber nichts ausführen.
- **Software:** `git pull` auf Commit `031b008` oder neuer. `cargo build --release -p forge`. Python 3 mit numpy. `python3 -m pytest -q scripts/tests` muss grün sein.
- **Hardware messen:**
  - CPU-Kerne bzw. GPU, RAM und freien Platz.
  - Tokens/s mit einem 20-Schritt-Probelauf des 100-M-Modells. `accel/` (JAX) nutzen, falls eine GPU oder TPU da ist.
  - Daraus die ETA für 2 Mrd. Tokens berechnen und melden.
- **Startpunkt:** `checkpoints/quasnir-champion-g3-r8/step-000120`, sha256 `4da6575fb7e25907af0646726e70cac4025c44230fce5fb621a44e873ff6588a`. Vor dem Start nachrechnen.

### A1. Wachstumswerkzeug `scripts/grow.py`, funktionserhaltend

Die echte Struktur des Champions (56 Tensoren, Gewichte als [out, in]):

| Tensor | Form | Anmerkung |
|---|---|---|
| `tok_emb.weight` | [8192, 384] | zugleich Ausgabekopf, gebunden |
| `norm.weight` | [384] | Schluss-RMSNorm, eps 1e-5 |
| `blocks.i.attn.wq.weight` | [384, 384] | |
| `blocks.i.attn.wo.weight` | [384, 384] | |
| `blocks.i.attn.wk.weight` | [128, 384] | |
| `blocks.i.attn.wv.weight` | [128, 384] | |
| `blocks.i.ffn.w1.weight` | [1024, 384] | Gate |
| `blocks.i.ffn.w3.weight` | [1024, 384] | Up |
| `blocks.i.ffn.w2.weight` | [384, 1024] | Down |
| `blocks.i.attn_norm.weight`, `blocks.i.ffn_norm.weight` | [384] | |

Weitere Eigenschaften: head_dim 64, 6 Heads, 2 KV-Heads (je 3 Q-Heads teilen einen KV-Head), RoPE θ 10.000.

**Zielform:**

| Größe | Wert |
|---|---|
| dim | 768 |
| Heads | 12 (head_dim 64 bleibt) |
| KV-Heads | 4 |
| FFN | 2048 |
| Layer | 15 |
| vocab | 8192 |
| max_seq_len | 512 |
| Parameter | 15 × 6.291.456 + 6.291.456 = **100.663.296** |

**Schritt 1: Breite 384 → 768.** Jede Aktivierung wird verdoppelt (x′ = [x, x]):

- **Ein- und Ausgabe:**
  - `tok_emb` → [E, E] auf 768 Spalten.
  - Alle RMSNorm-Gewichte → [g, g].
  - Ausnahme `norm.weight` → [g/2, g/2], weil der gebundene Ausgabekopf sonst die Logits verdoppelt.
- **Matrizen, die vom Residualstrom lesen** (wq, wk, wv, w1, w3): Eingangsspalten → [W/2, W/2].
  - Ausgangszeilen: Q-Heads 6–11 sind Kopien von 0–5, KV-Heads 2–3 Kopien von 0–1. Dadurch nutzen die Q-Heads 6–8 KV-Head 2, also die Kopie von KV-Head 0, und die Gruppen bleiben konsistent. Das setzt zusammenhängende Gruppen voraus (Q-Head h nutzt KV-Head h / 3). Prüfe das in `core_engine` und ordne die Kopien bei verschachtelter Zuordnung entsprechend an. Test 1 deckt jeden Fehler hier auf.
  - FFN-Hidden-Einheiten 1024–2047 sind Kopien von 0–1023.
- **Matrizen, die in den Residualstrom schreiben** (wo, w2): [[W/2, W/2], [W/2, W/2]].

Damit ist das Netz mathematisch identisch zum Champion.

**Schritt 2: Tiefe 6 → 15.** Neun neue Blöcke werden gleichmäßig verteilt eingefügt, nach den Blöcken 0, 1, 2, 2, 3, 3, 4, 4 und 5:
- Jeder neue Block kopiert q, k, v, w1, w3 und die Normen seines Vorgängers.
- **wo und w2 sind null**, der Block gibt also 0 aus und ist exakt die Identität.

**Schritt 3: Symmetrie brechen.**
- Auf alle duplizierten Hälften kommt relatives Rauschen mit σ = 1e-3 × std(W).
- Auf die Null-Matrizen der neuen Blöcke kommt Rauschen mit σ = 1e-4.

Ohne diesen Schritt bekämen die Kopien für immer gleiche Gradienten.

**Ausgabe:** ein forge-Checkpoint mit `model.safetensors` und `state.json`. `parent_sha256` = `4da6575f…`, dazu ein Feld `grown_from` mit der Wachstumsvorschrift.

**Tests, bevor irgendetwas trainiert wird:**
1. Ohne Rauschen (`--noise 0`) liefert `forge logits` für 5 Prompts dieselben Logits wie der Champion (max. Abweichung ≤ 1e-4).
2. Mit Rauschen ist der val-Loss auf den Gate-Stores code und general höchstens 1 % schlechter als beim Champion. Messung: `forge eval --split val --batch 8 --seq 256 --batches 32 --seed 4242`.
3. Die Parameterzahl ist exakt 100.663.296.

Bestehen die Tests nicht, wird nicht trainiert: erst den Fehler finden.

### A2. A/B-Test des Optimierers (je 300 Schritte, gleiche Daten und Seeds)

| Arm | Einstellung |
|---|---|
| a | AdamW, lr 3e-4 |
| b | Muon, `muon_lr` 0.02, AdamW-Teil 3e-4 |

Gemeinsam für beide Arme:
- Start: der gewachsene Checkpoint;
- Daten: `data/out/base-v4/train.meta.json`, Code-lastig;
- Lernrate: warmup 50, konstant;
- batch 16, seq 512, grad_accum 4, also 32.768 Tokens pro Schritt.

Gewinner ist der niedrigere val-Loss, gemessen wie oben. Gleichstand innerhalb 0,5 %: AdamW.
Gemessen bei uns: Muon von Grund auf val 4,806 statt 5,051, beim Weitertraining ohne Unterschied.

### A3. Haupttraining

- **Optimierer:** der Gewinner aus A2.
- **Lernrate:** WSD-Plan (`"lr_schedule": "wsd"`). Warmup 500 Schritte, konstante Phase, Abklingen über die letzten 20 %.
- **Ziel:** 2 Mrd. Tokens, Chinchilla-Größenordnung, ≈ 61.000 Schritte zu 32.768 Tokens.
- **Daten:**
  - Korpus v4 (base 681,6 M Tokens, etwa 3 Epochen);
  - verifizierte Qwen-Lehrerdaten nach `docs/PROMPT_SCALING.md`, erst wenn ihr eigener A/B-Test bestanden ist.
- **Zwischenmessung alle 2.000 Schritte:** val-Loss code und general.
- **Checkpoints:** alle 500 Schritte; nur die letzten 3 plus `TRUNK` behalten.
- **Abbruchregeln:**
  - val-Loss steigt 3 Messungen in Folge → stoppen, auf den besten Checkpoint zurück, Ursache berichten.
  - Bei NaN oder inf → sofort stoppen.
- **Größen beachten:** Ein 100-M-Checkpoint hat ≈ 403 MB Gewichte und ≈ 805 MB Optimizer. Nicht ins Git committen. Gesichert wird nur der Phase-A-Champion, in Teilen unter 100 MB, mit MANIFEST und sha256, nach Freigabe des Owners.

### A4. Gate A und Abschluss

- **Champion-Arbiter:** Der 100-M-Kandidat tritt gegen Quasnir 2 an. Kriterien: Held-out Stufen 0–2 (128 Aufgaben je Stufe), val auf code und general, `tol` 0,02, `margin` 0,03.
- **Release-Leiter:** Ein Release **Quasnir 3** gibt es nur bei Odds Ratio ≥ 2 (unteres 95-%-Konfidenzband), ohne McNemar-Verlust und mit val höchstens +2 %.
- **Benchmarks:** HumanEval, MBPP und ARC-AGI-2 für den neuen Champion messen, dann Tiers prüfen. Bei frontier oder dangerous gelten die Pflichten aus HANDOVER Abschnitt 5.
- **Danach:** RSI auf dem neuen Champion mit allen Gates. Phase B (erste 8 Bausteine) startet erst nach bestandenem Gate A.

### Berichte (alle 2 Stunden, 3–6 Zeilen Deutsch, nur gemessene Zahlen)

1. Schritt / Ziel, Tokens/s, ETA.
2. Trainings-Loss, letzter val-Loss code/general mit Trend.
3. A2-Ergebnis, sobald da, und die Gate-Entscheidungen.
4. Ausfälle, Platz, Auffälligkeiten.

### Feste Regeln

- **Löschen:** Nichts ohne Freigabe des Owners, außer der eingebauten Checkpoint-Rotation.
- **Rouge und Darus:** Sie bleiben aus. Ein einziger maßgeblicher Lauf.
- **Code:** Änderungen nur mit Tests und zuerst in einer separaten Arbeitskopie.
- **Zahlen:** Keine erfundenen. Vergleiche mit Astra, Opus 5.5 oder GPT-7 nur über gemessene Benchmarks.
