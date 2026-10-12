# Plan: a cheap trial of "do working Evoformer blocks help?"

Status: **plan, not started** (2026-10-12). A low-cost first answer before committing to the
faithful protocol in `02_retraining_protocol.md` (~1,400 GPU-h). Target: **~25 GPU-h** for the
first answer, ~60 GPU-h with seeds.

## Where the cost of the faithful protocol goes

| driver | share | how the trial avoids it |
|---|---|---|
| ESMFold (3B LM + 48-block trunk) run on every sample, every epoch | most of the ~1.4 s/sample measured in check 7 | **run ESMFold once** per crop and train the heads from cached outputs: ESMFold is frozen, and the heads see only its outputs |
| stage 1 on RCSB clusters, 10,000 samples per epoch | ~90% of the samples | **skip stage 1**; start the dynamic head from released weights |
| 2 arms × 3 seeds × 2 stages | ×12 | 2 arms × 1 seed first; seeds only if there is an effect |

Everything else stays as the authors ran stage 2 (plan 02, section 3): mdCATH train/val splits,
1,000 / 100 samples per epoch, 256-residue length-weighted crops, Adam 1e-4, batch 64 (via
accumulation), focal α = 0.85 γ = 2, MSE and weighted cross-entropy, mean over heads, auxiliary
heads after 10 epochs, bf16, early stopping after 10 epochs without validation improvement. The
two arms differ only in `torch.autocast(cache_enabled=...)`.

## Steps

**0. Time it (~1 GPU-h).** On 50 cached crops, measure seconds per sample for the head
forward + backward, arm A (cache on) vs arm B (cache off, gradients through the blocks). This
sets the real budget; the numbers below assume ~0.3 s/sample and are replaced by the
measurement.

**1. Cache ESMFold's outputs once (~3 GPU-h, ~60 GB on scratch).**
- Train: every mdCATH training domain; a domain ≤ 256 residues gives one crop (the whole
  domain), a longer one gives 3 random 256-residue windows, drawn length-weighted with a fixed
  seed. About 6,000 crops.
- Validation: the 270 validation domains, one fixed crop each.
- Saved per crop (fp16): what the heads read from ESMFold: `s_s`, `s_z`, `lddt_logits`,
  `lm_logits`, `ptm_logits`, `distogram_logits`, plus the crop window.
- Check: a head run on cached features equals the head run on live ESMFold, to fp16 rounding,
  on 20 crops.

**2. Two trial pairs, one seed (~4 runs × ~5 GPU-h ≈ 20 GPU-h).**

| pair | starting point | question it answers |
|---|---|---|
| **fine-tune** | released V2 heads (blocks at initialisation, everything else trained) | does switching the blocks on improve the shipped model? |
| **re-train stage 2** | V1's dynamic head (the nearest public stand-in for a stage-1 checkpoint), fresh kinetics and occupancy heads | the authors' stage 2, with and without working blocks |

Each pair is arm A vs arm B, identical sample sequence and initialisation.

**3. Only if B differs from A:** 2 more seeds for the better-motivated pair (~20 GPU-h); then
decide whether the faithful two-stage protocol (plan 02) is worth its cost.

## What the trial gives up, stated in the results

- **A finite crop pool** (~6,000 crops, re-sampled with replacement) instead of a fresh random
  crop per draw. It could favour memorisation; the validation loss and the test sets guard it.
- **No re-run of stage 1** (RCSB pre-training): both arms inherit the same released starting
  point, so the comparison is fair, but absolute numbers may differ from a full retrain.
- **One seed** until there is something to confirm.

## Evaluations

Each trained checkpoint (best validation loss), plus the released weights for reference, scored
exactly as the reproduction figures (notes 12, 13). In order of priority:

**Core: is B better than A?**
1. All Table 1 / Table 2 / SI Table 6 metrics: balanced accuracy, precision, recall, F1, AUROC,
   occupancy RMSE; mdCATH at all five temperatures (pairs ≥ 3 apart, scikit-learn balanced
   accuracy) and ATLAS (`train.py` protocol). **B − A paired per protein**, protein-level
   bootstrap 95% interval, decision rule fixed in advance (blocks help if the interval excludes 0
   on both test sets; no detectable effect if it lies within ±0.01).
2. **Kinetics** (SI Tables 8–9, on/off-time macro recall and precision) and the **confidence and
   residual heads** (SI Tables 10–12).

**Sanity: did the experiment do what it should?**
3. Arm A's block output layers stay at exactly 0; arm B's grow from step 1.
4. Arm A reproduces the released numbers within noise (the pipeline matches the authors').
5. **Deleting the blocks from trained B changes its outputs** (unlike the released model): the
   trained model actually uses them. The size of that change is a direct measure of how much the
   blocks contribute.
6. Learning curves per head (training and validation loss), and a **Fig. 2A analogue**:
   balanced accuracy and occupancy RMSE on train / validation / test, to see overfitting.

**Where and how: what do working blocks change?**
7. **By sequence separation** (|i − j| = 3–5, 6–11, 12–23, ≥ 24): triangle updates exist to
   propagate pair information, so any gain should be largest for long-range pairs.
8. By temperature, protein length and fraction of dynamic pairs.
9. **Calibration**: reliability diagram, Brier score, expected calibration error. In check 7 the
   fixed model's balanced accuracy rose while its focal loss got worse, which is a calibration
   signature.
10. **Out of distribution**: ATLAS (different force field, 300 K, never trained on) vs the mdCATH
    test split. Does a gain on mdCATH carry over?

**For dyn_models**
11. Re-extract the heads' internal states from trained B and probe them on atlas, allobench and
    passerrank. In the released model these states are ESMFold's to within 1% (note: "dyn_s ≈
    trunk_s"); trained blocks would make them genuinely ESMDynamic representations.

Optional: the case-study maps (Fig. 3 ASCT2, Fig. 5), side by side for A and B.

## Outputs

Note 15 (trial), results `results/15_*`, an A-vs-B figure in the style of the reproduction
figures (published / A / B), and the per-separation and calibration panels.
