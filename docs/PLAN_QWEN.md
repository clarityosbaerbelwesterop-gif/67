# Plan: Quasnir auf Basis von Qwen3.8-27B (Budget 300 €)

Stand 22.10.2026. Owner-Entscheidungen vom 22.10.2026: Ja zu allem (PyTorch und LoRA, Name, Schritt 1), Training
auf 2× B200.

## Was „Quasnir“ jetzt ist

Quasnir ist der Name des Systems, das Nutzer in Osirus auswählen. Es besteht aus drei Teilen:

| Teil | Modell | Herkunft |
|---|---|---|
| Hauptmodell | Qwen3.8-27B mit unseren LoRA-Gewichten | Basis von Qwen (Apache-2.0), nachtrainiert von uns |
| Vorsortierer und Prüfer | Quasnir-1B | von Grund auf selbst trainiert (train-1-f) |
| Letzte Prüfung | Sandbox-Tests, Build, Typprüfung | deterministisch, in Osirus |

Regeln für den Namen:
- In der Modellauswahl steht „Quasnir“.
- Ein Info-Hinweis und die Model Card nennen die Basis „Qwen3.8-27B (Apache-2.0)“ und unsere Änderungen. Lizenz und
  Hinweisdatei von Qwen liegen jeder Weitergabe bei.
- Nie „selbst von Grund auf trainiert“ für das Hauptmodell; in Osirus bleibt `selfTrained: false` für diese
  Gewichte.
- „Qwen“ wird nicht unser Markenname.

Quasnir-Daten, Checkpoints und Pläne (1B, 10B, `PLAN_250B_FOERDERUNG.md`) bleiben vollständig erhalten.

## Fakten zu Qwen3.8-27B (Hugging Face, 22.10.2026)

- Apache-2.0, 27,8 Mrd. Parameter, dicht, mit Bild-Encoder.
- 64 Schichten, davon 48 mit linearer Attention und 16 mit voller Attention (GQA, 4 KV-Köpfe).
- Multi-Token-Prediction eingebaut, Kontext 262.144 Tokens, Vokabular 248.320.
- Denken ist standardmäßig an und pro Anfrage abschaltbar; `reasoning_effort` ist einstellbar.
- Empfohlene Sampling-Einstellungen:
  - mit Denken: Temperatur 1,0, top_p 0,95, top_k 20;
  - ohne Denken: Temperatur 0,7, top_p 0,8, top_k 20, presence_penalty 1,5.
- Der JAX-Trainer `accel` kennt diese Architektur nicht. Nachtrainiert wird mit PyTorch, Transformers, PEFT und
  TRL (Owner-Ja).

## Schritte

| # | Schritt | Hardware | Kosten (Schätzung) | Tor |
|---|---|---|---|---|
| 1 | Qwen messen, wie es ist: HumanEval, MBPP und -hard, Denken aus und an (`jobs/eval-qwen.sh`, Lauf eval-qwen-a) | 1× H200 oder H100 | ≤ 15 $ | – |
| 2 | Agent-Suite für Osirus bauen (Entscheidungs-JSON der Agent-Schleife, Tool-Aufrufe, Delegation an Sub-Agents, Deutsch) und Qwen darauf messen | 1 GPU | ≈ 10 € | Basiswert steht |
| 3 | Daten: Qwen erzeugt Agent-Abläufe im Osirus-Format; nur was Sandbox oder Schema-Prüfung besteht, bleibt. Keine Benchmark-Aufgaben | 2× B200 | ≈ 40 € | dekontaminiert |
| 4 | LoRA-Feintuning (BF16, kein 4-Bit-Training) | 2× B200, parallel | ≈ 50 € | Agent-Suite besser; HumanEval und MBPP höchstens 1 Punkt schlechter |
| 5 | RL mit Tests (GRPO) | 2× B200 | ≈ 110 € | gleiches Tor wie 4 |
| 6 | Quasnir-1B als Vorsortierer auf den Pass/Fail-Beispielen aus 3–5 | 1 GPU | ≈ 10 € | Best-of-N mit Quasnir-1B besser als zufällig und als Mehrheit |
| – | Reserve | | ≈ 65 € | |

Mit zwei GPUs ist das Training doppelt so schnell. Die Kosten je Token bleiben gleich.

**Nicht im Budget:** der Betrieb in Osirus. Optionen:
- eigene GPU rund um die Uhr (≈ 1.400 $ im Monat für eine H100);
- serverlos;
- ein Anbieter, der LoRA-Adapter auf Qwen bereitstellt.

Auf einer RTX 5090 läuft das Modell in FP8 (≈ 28 GB) mit moderatem Kontext.

## AI Act (Kenntnisstand, keine Rechtsberatung)

Nach den GPAI-Leitlinien der Kommission (Juli 2025) wird man durch Nachtrainieren erst dann Anbieter eines neuen
Modells, wenn dafür mehr als ein Drittel der ursprünglichen Rechenleistung eingesetzt wird. 300 € liegen weit
darunter.
