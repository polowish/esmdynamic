# Investigation: ESMDynamic's DynamicModule (DCM) blocks are untrained

Branch `investigate/dynamicmodule-untrained` of the `polowish/esmdynamic` fork. Not for
merging upstream. Upstream is untouched; everything for this investigation lives here.

**Finding.** In both released weight files (Illinois Data Bank V1, 2025-06, and V2, 2026-04)
every parameter inside every DCM block is at its initial value, so the blocks are identity
maps. Cause: `torch.autocast`'s weight cache combined with the DCM's `no_grad` recycling
passes stops all gradient reaching the blocks during training. Full evidence and history:
`dyn_models/docs/esmdynamic_dynamicmodule_handoff.md`, and the scripts in
`dyn_models/scripts/diagnostics/esmdyn_*.py`.

## Layout

| folder | holds |
|---|---|
| `notes/` | numbered write-ups, one per check (`01_preprint_review.md`, ...) |
| `scripts/` | scripts for the checks on this branch (run from the repo root) |
| `results/` | small machine-readable outputs (json/csv); large data stays out of git |
| `figures/weights_audit/` | what the released weight files contain |
| `figures/gradient_flow/` | where gradient does and does not reach |
| `figures/reproduction/` | reproducing published numbers with the released weights |
| `figures/training_check/` | short training runs with and without the fix |

## Checks

| # | check | status |
|---|---|---|
| 3 | preprint: how the DCM, its training and its ablation are described | done — `notes/01_preprint_review.md` |
| 1 | minimal pure-PyTorch reproduction of the autocast-cache mechanism | done — reproduced; 3 fixes verified — `notes/04_minimal_reproduction.md` |
| 2 | did the June-2025 `train.py` (V1 era) also train under autocast? | done — yes; DCM code unchanged since — `notes/02_training_code_history.md` |
| 4 | which weights the Colab notebook and the Dockerfile load | done — all load V2 `7odsk` — `notes/03_public_weight_sources.md` |
| 5 | deleting the DCM blocks leaves outputs bitwise identical | done — 11/11 outputs identical — `notes/05_dcm_identity.md` |
| 6 | reproduce the published ATLAS-test numbers with the released weights | done — 0.880 ± 0.010 vs 87% — `notes/06_atlas_reproduction.md` |
| 7 | short training with and without the fix (if 1–6 justify it) | conditional |
| 8 | released weights vs the published human-proteome predictions (if justified) | conditional |
