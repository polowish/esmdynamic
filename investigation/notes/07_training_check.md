# 07 — With the fix, the DCM blocks train and fit better; held-out effect mixed so far (check 7)

Script: `scripts/short_training_check.py` · results: `results/07_training_check_cache_{on,off}.json`
· figure: `figures/training_check/07_training_check.png` · jobs 2157527 (cache on) and
2157528 (cache off), L40S, ~46 min each.

## Setup

Two runs from the released V2 weights (DCM blocks at initialisation), identical in every
respect except `torch.autocast(..., cache_enabled=...)`:

- **as released**: cache on (what `train.py` does) · **fixed**: `cache_enabled=False`
- 500 optimiser steps, each 4 mdCATH training samples (batch 1 × 4 accumulated — batch 4
  with gradient through every block does not fit a 44 GB L40S, and the preprint's batch 64
  on a 24 GB card implies accumulation too); 256-residue crops, length-weighted sampling —
  **pre-drawn from one seed, so both runs see the same samples and crops** (confirmed: same
  first batch and bit-identical step-1 loss, 2267.2194)
- Adam lr 1e-4, bf16 autocast, ESMFold frozen, chunk size 256 (as `train.py`), ESMDynamic's
  own losses on the three main heads (dynamic focal α 0.85 γ 2, kinetics weighted CE,
  frequency MSE), targets from `train.py`'s `build_outputs_and_targets_for_loss`
- Validation before and after: 48 fixed mdCATH validation crops (first 256 residues)
- Two packages the inference env lacks were not installed: torchvision's 10-line
  `sigmoid_focal_loss` is provided verbatim, and tensorboard's `SummaryWriter` by a no-op.

## Results

**The blocks only learn with the fix.** Norm of all DCM blocks' zero-initialised output
layers: as released **0.000 → 0.000** over 500 steps; fixed **0.000 → 26.3**, rising steadily.

**The fixed model fits the training data increasingly better.** Paired training-loss
difference (fixed − as released, same batches):

| steps | mean difference (± se) |
|---|---|
| 1–100 | −7 ± 5 |
| 101–250 | −79 ± 8 |
| 251–400 | −140 ± 10 |
| 401–500 | **−200 ± 17** (≈ 6% lower) |

**Held-out: mixed.** Change on the 48 validation crops after 500 steps:

| | as released | fixed |
|---|---|---|
| dynamic-contact balanced accuracy | −0.0 pts (0.800 → 0.800) | **+2.6 pts (0.800 → 0.826)** |
| frequency loss (MSE) | −1.0% | **−7.2%** |
| dynamic loss (focal) | −0.2% | +5.4% |
| kinetic loss (weighted CE) | −0.1% | +0.9% |

## Reading it

- The mechanism and the fix are confirmed in ESMDynamic's own training loop: with the
  cache on, 500 steps of training leave the blocks exactly at initialisation; with it off
  they learn from step 1.
- Working blocks lower the training loss, by a widening margin: the architecture has
  capacity the released model never used.
- Whether that generalises is **not settled by this run**: balanced accuracy and frequency
  improve on held-out proteins, the focal and kinetic losses get slightly worse. 500 steps
  is ~2 of the preprint's fine-tuning "epochs" (1,000 samples each, early stopping on
  validation loss), from one seed, scored on 48 proteins, starting from heads that were
  tuned to work *without* blocks. A worsening focal loss alongside a better thresholded
  accuracy can come from calibration shifting while the blocks are still settling.
- What a proper answer needs: training the heads from scratch (or fully re-fine-tuning)
  with the fix to convergence under the preprint's recipe, several seeds, and evaluation
  on the full mdCATH and ATLAS test sets — i.e. retraining, which is the authors' call or a
  separate project.
