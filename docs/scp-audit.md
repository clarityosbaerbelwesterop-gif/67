# Audit: `swarm-compute-protocol-` (SCP), read-only, 2026-10-05

Commit `ab2db73` (2026-09-02, "Phase 2f: bind authenticated governance identity"); 384 files. Nothing was changed in SCP.

## Compute: no real GPU pipeline

- **Swarm.** `config/swarm_config.json` describes a **simulation**: "Mainnet Simulation", `swarm_size` 10,000 simulated nodes, simulated credits and simulated GPU tiers. No real worker nodes are connected.
- **Own model (`model/`).** SCP-1 is a Llama-style decoder (RMSNorm, RoPE, GQA, SwiGLU) with presets up to 5.8B, tests and a trainer.
  - GPU training is a **manual runbook** (`docs/training-runbook.md`): SSH from an iPad to an IONOS H200 instance with 144 TB storage.
  - No workflow starts GPU work.
  - Whether that IONOS H200 exists is not visible in the repo.
- **Workflows (9).** None runs on a GPU:
  - deploy;
  - `distill`;
  - keep-alive (schedule disabled);
  - LLM handshake;
  - n8n publish;
  - Python CI;
  - secret sync to Render and to Vercel;
  - verify-live (weekly cron).

## Findings for the owner

1. **Distillation data** (`distill.yml`) is generated through NVIDIA's hosted API (Kimi K2 / Nemotron). Under the Rouge rules, API outputs are not training data unless the provider's terms allow it and the owner decides so. The Rouge teacher stays self-hosted (DeepSeek-V4-Pro, MIT).
2. **Several numbered keys per provider:** OpenRouter × 3, Groq × 3, NVIDIA × 5.
   - If these come from separate accounts to multiply free quotas, that can break the providers' terms.
   - Keys from one account are fine.
   - Owner to check; no action taken here.
3. **Reusable for 67:** the SCP-1 transformer and runbook. The fabric uses the Osirus native Rouge model, which is further along (tournament, ternary R1.29b, Muon). Its run presets (`model/configs/runs.json`) can become fabric jobs once the H200 is confirmed.
