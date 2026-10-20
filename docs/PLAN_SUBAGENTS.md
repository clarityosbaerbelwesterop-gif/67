# Plan: Denkende Subagenten ab 13–15 Mrd. Parametern

Stand 20.10.2026.

Vorgabe des Owners:
- Subagenten kontrollieren Lösungen und arbeiten mit.
- Das kommt erst, wenn das Modell die richtige Größe hat: 13–15 Mrd. Parameter mit einem Untermodell.
- Denken und Forschen laufen dauerhaft auf einer GPU der A100/H100-Klasse oder kleiner.
- Kontrolliertes RSI läuft auf den großen GPUs.

Alle Zahlen hier sind Rechnungen und werden vor dem Einsatz gemessen.

## 1. Was heute schon steht

**Denkzeit** (`scripts/humaneval.py`). Die Rollen sind heute noch dasselbe Modell und dienen der Messung.
- Mehrere Runden: Weitere Lösungen gibt es nur für Aufgaben, die noch scheitern.
- Tester-Rolle: Das Modell schreibt eigene Testzeilen.
- Reparatur-Rolle: Das Modell sieht seinen Fehlversuch und den Test, an dem er scheitert.
- Auswahl: zuerst über die sichtbaren Tests, danach über die Übereinstimmung der Lösungen bei der Ausführung.

**RSI-Schleife** (`training/rsi/loop.py`):
1. Aufgaben.
2. K Kandidaten.
3. Sandbox.
4. Training nur auf geprüften Erfolgen.
5. Gates, dann Freigabe.

## 2. Rollen

| Rolle | Aufgabe | Wer |
|---|---|---|
| Orchestrator | Reihenfolge, Budgets, Gates, STOP | deterministischer Code, kein Modell |
| Planer | zerlegt eine Aufgabe in Teilschritte | Hauptmodell |
| Löser (× N) | schreibt Lösungen | Hauptmodell |
| Tester | schreibt Tests und Grenzfall-Eingaben | Untermodell |
| Prüfer | führt aus, vergleicht, misst | Code in der Sandbox, kein Modell |
| Reparierer | korrigiert anhand der Fehlermeldung | Hauptmodell |
| Forscher | schlägt Experimente vor (Daten, Mix, Hyperparameter) und wertet Proxy-Läufe aus | Hauptmodell |
| Kurator | geprüfte Lösungswege in den Reasoning-Store, nur über das Wert-Tor | Code und Untermodell |

**Grundsatz: Die Kontrolle liegt im Code, nie beim Modell.** Budgets, Gates, Freigaben, Prüfregeln und die
STOP-Datei kann kein Agent ändern.

**Rollen als Adapter.** Rollen sind dieselben Gewichte mit einem Rollen-Prompt oder einem LoRA-Adapter. Ein Adapter
mit Rang 32 auf allen Linearschichten des 13B-Stamms hat ≈ 131 Mio. Parameter, also ≈ 0,26 GB in BF16. Rechnung je
Schicht:

| Schichten | Rechnung | Parameter |
|---|---|---|
| Q, O | 2 × 32 × (5120 + 5120) | 655.360 |
| K, V | 2 × 32 × (5120 + 1024) | 393.216 |
| Gate, Up, Down | 3 × 32 × (5120 + 12.288) | 1.671.168 |
| je Schicht | | 2.719.744 |
| × 48 Schichten | | ≈ 131 Mio. |

**Untermodell.** Etwa 1 Mrd. Parameter, mit demselben 64k-Tokenizer, aus dem eigenen 13B destilliert (nie aus
geschlossenen Modellen). Zwei Aufgaben:
1. Die Tester-Rolle, billig und parallel.
2. Entwurfsmodell für spekulatives Dekodieren. Die Ausgabeverteilung des Hauptmodells bleibt dabei unverändert;
   den Gewinn an Geschwindigkeit messen wir.

## 3. Hardware (Rechnung)

Angenommen ist der 13B-Stamm aus `docs/PLAN_100B_MOE.md`: 12,75 Mrd. Parameter, 48 Schichten, 8 KV-Köpfe × 128.
- **Gewichte:** 25,5 GB in BF16 oder 12,75 GB in FP8.
- **KV-Cache je Token:** 2 × 48 × 8 × 128 × 2 Byte = 196.608 Byte ≈ 0,19 MB (BF16). Mit FP8-KV ist es die Hälfte.

| GPU | Gewichte | Frei für KV (≈) | Tokens im Cache (BF16-KV) | Beispiel |
|---|---|---|---|---|
| H100 80 GB | FP8, 12,75 GB | 60 GB | ≈ 305.000 | 37 Agenten × 8k oder 9 × 32k |
| A100 80 GB | BF16, 25,5 GB (kein FP8) | 48 GB | ≈ 244.000 | 29 × 8k |
| L40S 48 GB | FP8, 12,75 GB | 30 GB | ≈ 152.000 | 18 × 8k |
| RTX 5090 32 GB | FP8, 12,75 GB | 15 GB | ≈ 76.000 | 9 × 8k oder 2 × 32k |

- **15B:** Gewichte und KV-Cache wachsen um etwa 18 %.
- **Inferenz in FP8.** 4-Bit-Inferenz nur nach eigener Qualitätsmessung und mit Owner-Ja. Training in 4 Bit bleibt
  ausgeschlossen.

## 4. Workflows

| Workflow | Was | Wo | Läuft |
|---|---|---|---|
| W1 Denken | Aufgaben, Planer, Löser × N, Tester, Prüfer, Reparierer; Ergebnis: geprüfte Lösungswege | 1 kleine GPU | dauerhaft |
| W2 Prüfen | Ausführung in der Sandbox, ohne Netz, mit Limits | CPU | dauerhaft |
| W3 Kuratieren | Dedup, Dekontamination, Wert-Tor, Reasoning-Store | CPU | dauerhaft |
| W4 Forschen | Hypothese, dann 100M-Proxy-Lauf mit Deckel, dann Bericht | kleine GPU und CPU | dauerhaft, nur Proxy-Läufe |
| W5 RSI-Training | Training auf geprüften Lösungswegen | B300 / H200 | je Lauf Owner-Ja und Budgetdeckel |
| W6 Gate | Held-out L0–L2, McNemar, Odds Ratio ≥ 2, dann Champion | Box | nach jedem W5, Champion nur mit Owner-Ja |

## 5. Kontrollen (gelten für alle Agenten)

- Sandbox ohne Netz, Schreibrechte nur im Arbeitsverzeichnis, CPU- und Speicherlimits.
- Held-out-Sets nie in W1–W5, nur in W6 auf der Box.
- Jede neue Version braucht die Gates und das Owner-Ja. Die STOP-Datei beendet alle Workflows.
- Agenten starten keine großen Läufe, ändern keine Gates und Budgets und nicht ihre eigenen Prüfregeln.
- Stufe dangerous: RSI anhalten, Zugang nur für VIP-Unternehmen.
- Keine Ausgaben geschlossener Modelle als Trainingsdaten.

## 6. Reihenfolge und Tore

1. **Jetzt (1B):** Grok misst in Auftrag 1 die Denkzeit (`ablation_pass@1`). Jede Stufe (Runden, Selbsttests,
   Reparatur) bleibt nur, wenn sie gemessen etwas bringt.
2. **1B-RSI-Lauf**, wenn `oracle_any_pass` ≥ 5 %. Das belegt die Lernkurve und ist der Nachweis für das Funding.
3. **M1:** 13B-Stamm, mit Funding (`docs/PLAN_100B_MOE.md`).
4. **Untermodell:** 1B mit 64k-Tokenizer, aus dem 13B destilliert.
5. **Betrieb:** Rollen-Adapter; W1–W4 dauerhaft auf einer GPU, W5 auf B300.

**Tor für das Agentensystem.** Bei gleicher Zahl dekodierter Tokens muss es messbar besser sein als Best-of-N.
Sonst bleibt es beim einfacheren Verfahren.
