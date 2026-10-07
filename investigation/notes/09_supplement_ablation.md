# 09 — The supplement's parameter table and ablation numbers

Source: the preprint's supplementary information (55 pp; obtained manually, not in git). Its
table numbers differ from the main text's: main-text "Table S4" is SI Table 5, the ablation
tables "S12–S19" are SI Tables 13–20.

## SI Table 5 — the DCM was meant to train

- "Dynamic Contact Module (3X): 28,600,448 — Trainable: Yes", including "Evoformer (2X)
  14,293,888", besides the positional embedding (8,448), recycle norms (2,048 + 256) and the
  recycle distogram embedding (1,920).
- Trainable total **95,894,843** — exactly the number of head parameters our gradient check
  reports as `requires_grad` (`dyn_models/scripts/diagnostics/esmdyn_grad_check.py`). The
  authors counted the blocks as trained parameters; in both released checkpoints they are
  at initialisation (`esmdyn_checkpoint_audit.py`), and the mechanism is in 04.

## SI Tables 6–9, 13–20 — the ablation (mdCATH test split, 320 K)

| model | dynamic bal. acc. | AUROC | occupancy RMSE | on-time recall | off-time recall |
|---|---|---|---|---|---|
| full (SI 6–9) | **0.796 ± 0.007** | 0.889 | **0.076 ± 0.002** | 0.339 | 0.432 |
| no auxiliary inputs (SI 13–16) | 0.771 ± 0.006 | 0.860 | 0.095 ± 0.002 | — | — |
| no DCM (SI 17–20) | 0.764 ± 0.006 | 0.839 | **0.183 ± 0.003** | 0.348 | 0.416 |

(no-aux kinetics: SI 15–16, not transcribed here.) Full model, all splits at 320 K: training
0.811, validation 0.812, test 0.796 balanced accuracy — consistent with the main text's 80%.

## What this means for the claim that the DCM "does improve the performance"

- **None of these variants had working DCM blocks.** All were trained with the same code, so
  in every variant that has a DCM (full, no-aux) its blocks received no gradient. The
  comparison is never "trained Evoformer blocks vs none".
- **What "no DCM" removes, per the Methods**: "the pairwise representation from the final
  Evoformer block of ESMFold was used in place of the DCM output." The pair transition is
  added to ESMFold's pair state *before* the DCM (`s_z_0 = s_z + pair_transition(...)`), so
  replacing the DCM's output with ESMFold's pair state most likely also drops the pair
  transition from the pair readout. The no-DCM model then lacks (a) the auxiliary pair input
  and (b) the DCM's non-block terms — positional embedding and recycling — and, by our
  results, nothing else that worked. (The ablation code is not in the repository; this is a
  reading of the Methods, to confirm with the authors.)
- **Classification:** full 0.796 vs no-aux 0.771 vs no-DCM 0.764. The no-aux and no-DCM
  variants differ by less than their standard errors, so the full model's ~3-point lead is
  what the auxiliary inputs and the DCM's non-block terms add — not the Evoformer blocks.
- **Occupancy:** no-DCM's 320 K RMSE (0.183) is more than twice the full model's (0.076) and
  far worse than no-aux (0.095) — but the same no-DCM model scores ~0.10 at every other
  temperature (SI 18: 0.102, 0.111, 0.107, 0.099), so its 320 K value looks like a
  training artefact of that run, not an architectural effect. Reported as a gap from the DCM,
  it is not attributable to the blocks either way.
- **On-time kinetics:** no-DCM is marginally better (0.348 vs 0.339 recall), as the main text
  notes.

So the supplement is consistent with everything else: the published model's accuracy comes
from ESMFold plus the heads' shallow trained parts; the ablation measures the auxiliary
inputs and the DCM's non-block terms, and cannot say anything about Evoformer blocks that
never trained. The ATLAS comparison table (main text "S21") is not in this supplement.
