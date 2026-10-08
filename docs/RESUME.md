# Den 10-Tage-Lauf in einer neuen Sitzung fortsetzen

Checkpoints und Daten liegen nur auf der Festplatte des Containers. Damit eine neue
Sitzung dort weitermacht, wo die alte stand, liegt unter `resume/pack/` ein
AES-256-verschlüsseltes Paket: die Gewichte der aktuellen Champions, die Basen, von
denen sie abstammen, alles aus dem laufenden Zyklus (inkl. RSI-Champion und Anker)
und der komplette Long-Run-/RSI-Zustand. Das Repo ist öffentlich; ohne den
Schlüssel ist das Paket unlesbar (Rouge bleibt vertraulich). Den Schlüssel hat nur
der Owner. Neu sichern: `FORGE_PACK_KEY=… python3 scripts/resume_pack.py save`,
dann committen und pushen (jede Sicherung kostet ~310 MB Git-Historie).

## Prompt für die neue Sitzung

Den Schlüssel an der markierten Stelle einsetzen:

```
Setze den 10-Tage-Lauf in Repo 67 fort, Branch claude/gitops-multi-model-finetuning-ort4m9.
NICHT von vorn anfangen. Ablauf, siehe docs/RESUME.md und docs/LONGRUN.md:
1. export FORGE_PACK_KEY=<SCHLÜSSEL>; python3 scripts/resume_pack.py restore
   (stellt runs/ mit Champions, Zustand und RSI-Stand wieder her).
2. cargo build --release (forge-Binary), dann die Korpora neu bauen, nacheinander und mit nice:
   python3 data/build_corpus_v2.py, danach python3 data/build_corpus_v3.py --target-tokens 400e6 --max-disk-gb 6.
3. bash scripts/resume_all.sh; prüfen mit python3 scripts/longrun.py --status und tail runs/longrun/longrun.out,
   dass der Lauf im gespeicherten Zyklus/Schritt weitermacht. Weicht ein neu gebauter Korpus vom
   gespeicherten ab, gilt er wie jede neue Korpusversion erst ab dem nächsten Zyklus.
4. Keep-alive: Hintergrundaufgabe (run_in_background, timeout 7200000; Schleife sleep 300 + resume_all.sh
   für 7080 s), nach Ablauf neu starten. Eine Routine alle 2 h anlegen, die dasselbe prüft.
5. RSI läuft automatisch mit Gates (≤ +1 % Verlust je Runde, ≤ 2 % ab Startmodell, ≥ +3 Punkte
   Testaufgaben), der Arbiter wählt pro Zyklus die Champions. Nichts abschwächen.
6. Regeln: keine Checkpoints löschen, kein Geld ausgeben, keine Agents/Workflows, nur auf diesen
   Branch pushen, keine erfundenen Benchmark-Zahlen, den Schlüssel nie in Dateien oder Commits schreiben.
7. Antworte nur mit einer kurzen Zeile; ausführlich nur bei einem nicht behebbaren Fehler oder am
   Ende des Budgets (runs/champions.json und letzten Zyklusbericht zusammenfassen).
```
