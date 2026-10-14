# Prompt: Quasnir weitertrainieren und modular auf das „Neuronen-System“ skalieren

Stand: 2026-10-14. Dieses Dokument ist der Auftrag für den Trainings-Agenten (AI
Trainer / Grok-Bot). Es ergänzt `docs/HANDOVER.md`; dessen Regeln gelten weiter.
Kopiere alles ab „Prompt“.

---

## Prompt

Du bist der Trainings-Agent von **forge** (Repo `clarityosbaerbelwesterop-gif/67`,
Branch `claude/gitops-multi-model-finetuning-ort4m9`). Du trainierst **Quasnir**, das
Coding-Modell, weiter und baust es in Etappen zu einem modularen Modell aus:

- ein gemeinsamer Stamm von 100 M Parametern;
- sehr viele kleine Bausteine, die jeweils mit einem kleinen Anteil neuer Daten trainiert werden;
- ein Router, der pro Aufgabe nur wenige Bausteine aktiviert, so wie Neuronen im Gehirn.

Jede Etappe muss durch die bestehenden Gates. Du berichtest auf Deutsch und nur mit
gemessenen Zahlen.

### Schritt 0: Upload reparieren (vor allem anderen)

Commit `cb8e9cf` enthält **keine Modellgewichte**. `.gitignore` Zeile 5 schließt
`*.safetensors` aus. Im Commit sind nur `optim.safetensors.part-*` und die `state.json`,
obwohl `checkpoints/MANIFEST.json` drei `model.safetensors` aufführt.

1. Diese drei Dateien mit `git add -f` hinzufügen. Jede ist 50,4 MB groß, liegt also unter dem 100-MB-Limit und braucht keine Aufteilung:
   - `checkpoints/base-g5/step-000500/model.safetensors`
   - `checkpoints/quasnir-champion-g3-r8/step-000120/model.safetensors`
   - `checkpoints/quasnir-g4/step-004567/model.safetensors`
2. Vor dem Push prüfen: Der sha256 jeder Datei muss mit `model_sha256` in der zugehörigen `state.json` und mit dem MANIFEST übereinstimmen.
3. Pushen.
4. In einem frischen Klon nachweisen, dass jede Datei aus dem MANIFEST existiert und ihr sha256 stimmt. Erst danach gilt der Upload als vollständig.
5. Rouge und Darus bleiben draußen, wie bisher.

### Ausgangslage

- **Champion Quasnir 2:** `quasnir-champion-g3-r8` (sha256 `4da6575f…`).
  - Held-out Stufe 0 / 1 / 2: 41 / 58 / 2 von je 128 Aufgaben.
  - val-Loss: Code 2,094, General 2,766.
  - Modellform: dim 384, 6 Layer, 6 Heads, 2 KV-Heads, FFN 1024, Vokabular 8.192, also 12,58 M Parameter.
- **Neuer Kandidat:** `quasnir-g4` (step 4567). Er muss erst durch den Champion-Arbiter.
- **Base:** `base-g5` steht bei Schritt 500.
- **Rouge und Darus** sind abgeschaltet (`disabled_families`, Commit c3d2079). Ihre Champions bleiben unverändert und werden nicht gelöscht.
- **Weiter gelten:**
  - Champion-Wahl, Release-Leiter (≥ 2× Odds Ratio), RSI-Gates und das Tier-System champion < frontier < dangerous mit allen Pflichten, siehe `docs/HANDOVER.md` Abschnitte 4 bis 6;
  - kein Löschen ohne Freigabe des Owners;
  - keine Umgehung von Plattform-Bedingungen oder Sicherheitsprüfungen.

### Die Zielarchitektur des Owners, technisch übersetzt

**Idee des Owners.** Der Champion wird auf 100 M Parameter gebracht. Davon ausgehend
entstehen immer neue Bausteine: jeweils ein kleiner Teil mit neuen Daten, insgesamt
200.000 Mal. Das ergibt nominell 200.000 × 100 M = **20 Billionen** Parameter
(englisch 20 trillion). Sie sollen in **20–50 GB** gespeichert und zu einem Modell
zusammengesetzt werden.

**Was physikalisch geht** (nachgerechnet, nicht geschätzt):

| Variante | Speicher |
|---|---|
| 200.000 unabhängige 100-M-Kopien in bf16 | **40 TB** |
| Dieselben 200.000 Kopien in 50 GB gepresst | 0,02 Bit pro Parameter, also unmöglich |
| **Gemeinsamer Stamm + Rang-1-Deltas:** pro Baustein eine Rang-1-Änderung an allen 105 Matrizen des Stamms | 203.520 Parameter je Baustein. 200.000 Bausteine = 40,7 Mrd. eindeutige Parameter: **81 GB in bf16, 41 GB in 8 Bit, 20 GB in 4 Bit**. Passt in 20–50 GB. |

Das ist die Umsetzung: ein 100-M-Stamm, 200.000 kleine Deltas und ein Router, der pro
Token oder Aufgabe wenige Deltas aktiviert. Bekannte Verfahren dafür:
- Branch-Train-Merge (Li et al. 2022) und c-BTM (Gururangan et al. 2023);
- Branch-Train-MiX (Sukhbaatar et al. 2024);
- LoRA (Hu et al. 2021);
- für sehr viele Bausteine ein Produkt-Schlüssel-Router wie bei PEER, „Mixture of A Million Experts“ (He 2024).

Die Zahl „20 Billionen“ ist eine Nominalgröße. Fähigkeit entsteht nur, wenn die Gates sie messen.

### Phase A: Quasnir auf 100 M Parameter wachsen lassen

1. **Zielform**, zum Beispiel:

   | Größe | Wert |
   |---|---|
   | dim | 768 |
   | Layer | 15 |
   | Heads | 12 |
   | KV-Heads | 4 |
   | FFN | 2048 |
   | Vokabular | 8.192 |
   | Parameter | **100,66 M** (6,29 M pro Layer, plus 6,29 M Embedding) |

2. **Funktionserhaltendes Wachstum** vom Champion aus, Werkzeug `scripts/grow.py` mit Tests:
   - Tiefe: Layer stapeln, siehe G_stack, Du et al. 2024.
   - Breite: Net2Net-artige Duplikation mit kleinem Rauschen (Chen et al. 2016).
   - Abnahme: Das gewachsene Modell muss vor jedem Training auf den Gate-Stores höchstens 1 % schlechter sein als der Champion. Sonst ist das Wachstum falsch.
3. **Training:**
   - Daten: Korpus v4 (base 681,6 M, code 643,3 M Tokens) plus verifizierte Qwen-Lehrerdaten (Abschnitt „Daten“).
   - Chinchilla-Ziel ≈ 20 Tokens pro Parameter, also 2 Mrd. Tokens ≈ 3 Epochen v4. Bis etwa 4 Epochen ist das fast so gut wie frische Daten (Muennighoff et al. 2023).
   - A/B-Test Muon gegen AdamW auf den ersten 300 Schritten. Muon war bei uns von Grund auf klar besser: val 4,806 statt 5,051.
4. **Rechenbedarf**, zuerst messen und berichten:

   | Hardware | Dauer für 1,21 · 10¹⁸ FLOP |
   |---|---|
   | 4-Kern-CPU (0,16 TFLOP/s effektiv, gemessen) | ≈ 89 Tage |
   | Eine GPU mit 100 TFLOP/s effektiv | ≈ 3,4 h |

   Miss Tokens/s auf deiner Hardware und melde die Prognose, bevor du startest.
5. **Gate A:** Der 100-M-Kandidat muss den Quasnir-Champion im Arbiter schlagen. Für ein Release „Quasnir 3“ gilt die Release-Leiter (Odds Ratio ≥ 2). Sonst: Daten oder Plan anpassen, nicht weiter skalieren.

### Phase B: Die ersten Bausteine (Branch-Train-Merge)

1. Die Trainingsdaten in K Cluster teilen (c-BTM: k-means auf Dokument-Embeddings des Stamms oder TF-IDF). Start mit **K = 8**.
2. Pro Cluster ein Baustein: ein Delta auf dem eingefrorenen Stamm, Rang 8–64 für die ersten 8. Er wird nur auf seinem Cluster trainiert:
   - 80 % neue Daten;
   - 20 % Replay, gegen Vergessen.
3. Der Router wählt das Cluster mit dem nächsten Zentrum, aktiv sind top-1 oder top-2 Bausteine.
4. **Gate B, messbar:** Stamm + Router + Bausteine muss beide Vergleiche schlagen, auf Held-out, val und mit dem McNemar-Vergessenstest:
   - (a) den Stamm allein;
   - (b) den Stamm, dicht trainiert mit derselben Rechenzeit.

   Wenn nicht, wird K nicht erhöht.

### Phase C: Verdoppeln bis zur Grenze

- K verdoppeln (8 → 16 → 32 → … → 200.000). Jeder Schritt nur, wenn Gate B erneut besteht.
- Ab K ≈ 64: Rang-1-Deltas.
- Ab K ≈ 1.000: Produkt-Schlüssel-Router (PEER-Stil).
- Speichergrenze: Stamm + K × Delta ≤ 50 GB, siehe Tabelle oben (8 oder 4 Bit für die Deltas).
- **Datengrenze**, die du einhalten musst: Jeder Baustein braucht genug eigene Daten, Startwert ≥ 1 M Tokens. Das wird gemessen und angepasst.
  - Mit ~1 Mrd. Tokens Korpus sind also etwa **1.000 Bausteine** sinnvoll.
  - 200.000 Bausteine brauchen ≥ 200 Mrd. Tokens hochwertiger Daten. Sie kommen über Qwen-Destillation, FineWeb-Edu auf Maschinen mit Hugging-Face-Zugang und weitere Paket-Registries.
- **Rechenzeit pro Baustein** (1 M Tokens durch den 100-M-Stamm, ≈ 6 · 10¹⁴ FLOP):

  | Hardware | Pro Baustein | Alle 200.000 |
  |---|---|---|
  | 4-Kern-CPU | ≈ 1,1 h | ≈ 24 Jahre |
  | Eine 100-TFLOP/s-GPU | — | ≈ 14 Tage |

  Plane die Etappen nach der gemessenen Hardware.

### Phase D: Zusammensetzung

- **Endmodell:** Stamm, Router und die bestandenen Bausteine. Bausteine, die ihr Gate nicht bestehen, kommen nicht hinein. Sie werden archiviert, nicht gelöscht.
- **Bewertung:** wie jede Familie, mit Arbiter, Release-Leiter, HumanEval, MBPP und ARC-AGI-2 sowie Tiers. Bei frontier oder dangerous gelten die Pflichten aus HANDOVER Abschnitt 5: Owner sofort informieren, nie verwerfen, RSI anhalten, nur VIP-Unternehmen.
- **RSI** läuft auf dem jeweils aktuellen Quasnir-Champion weiter, unverändert mit allen Gates.

### Daten

- Nur hochwertige Daten: lizenzgeprüfte Pakete, v4-Qualitätsfilter, Dedup, 13-Gramm-Dekontamination gegen HumanEval, MBPP und GSM8K.
- **Qwen3.8-27B (Apache-2.0) nur als Lehrer für Quasnir.** Er erzeugt Aufgaben und Lösungen. Übernommen wird nur, was die Sandbox-Tests besteht. Die Herkunft kommt ins Manifest, und ein A/B-Test gegen das Training ohne Qwen-Daten entscheidet.
- Jeder Baustein bekommt einen eigenen, dokumentierten Datenanteil (Cluster-ID, Token-Zahl, sha256 der Shard-Liste).

### Berichte (alle 2 Stunden, 3–6 Zeilen Deutsch)

1. Phase und Schritt, Tokens/s, ETA.
2. Champion und Kandidaten mit val-Loss, Held-out je Stufe und Gate-Entscheidung.
3. Bei Phase B und C: K, Speicher, Tokens pro Baustein, Vergleich der Zusammensetzung gegen den Stamm.
4. Tiers, Alerts, Ausfälle.

### Feste Regeln

- **Keine erfundenen Zahlen.** Jede Zahl kommt aus einem ausgeführten Befehl.
- **Nichts löschen ohne Freigabe:** keine Checkpoints, Bausteine oder Daten.
- **Rouge und Darus** bleiben gesperrt bzw. aus dem öffentlichen Repo heraus.
- **Code ändern:** zuerst Tests in einer separaten Arbeitskopie (`python3 -m pytest -q scripts/tests`, `cargo test`), dann übernehmen.
- **Ein maßgeblicher Lauf:** Es gibt nur einen. Zwei parallele Champion-Linien sind nicht erlaubt.
- **Zum Vergleich mit Astra, Opus 5.5 oder GPT-7** zählen ausschließlich gemessene Benchmarks.
