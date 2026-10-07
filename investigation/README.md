# Investigation: ESMDynamic's DynamicModule (DCM) blocks are untrained

Branch `investigate/dynamicmodule-untrained` of the `polowish/esmdynamic` fork. Not for
merging upstream. Upstream is untouched; everything for this investigation lives here.

**Finding.** In both released weight files (Illinois Data Bank V1, 2025-06, and V2, 2026-04)
every parameter inside every DCM block is at its initial value, so the blocks are identity
maps. Cause: `torch.autocast`'s weight cache combined with the DCM's `no_grad` recycling
passes stops all gradient reaching the blocks during training. Full evidence and history:
`dyn_models/docs/esmdynamic_dynamicmodule_handoff.md`, and the scripts in
`dyn_models/scripts/diagnostics/esmdyn_*.py`.

**Start with `EVIDENCE.md`**: the whole case, one figure per claim.

## Layout

| folder | holds |
|---|---|
| `notes/` | numbered write-ups, one per check (`01_preprint_review.md`, ...) |
| `scripts/` | scripts for the checks on this branch (run from the repo root) |
| `results/` | small machine-readable outputs (json/csv); large data stays out of git |
| `figures/evidence/` | the figures in `EVIDENCE.md` (E0–E8), one claim each |
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
| 7 | short training with and without the fix (if 1–6 justify it) | done — blocks train only with the fix; training loss −6%; held-out mixed — `notes/07_training_check.md` |
| — | supplement: parameter table and ablation numbers | done — `notes/09_supplement_ablation.md` |
| — | mdCATH test split vs SI Table 6, and the paper-comparison figure | done — 320 K reproduces (0.806 vs 0.796) — `notes/10_mdcath_test_reproduction.md` (scoring superseded by 12) |
| — | all six metrics of Tables 1–2 under the original protocol (crop, logits, `frequency_pred`) | done — every 320 K number reproduces; RMSE gap was our output choice — `notes/12_eval_protocol.md` |
| 8 | released weights vs the published human-proteome predictions (if justified) | skipped — settled by checks 4 and 6 — `notes/08_proteome_check_skipped.md` |
