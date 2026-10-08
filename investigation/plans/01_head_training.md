# Plan: train ESMDynamic's heads ourselves and test what working blocks change

Status: **plan, not started** (2026-10-08). Needs the cluster rebuilt first
(`dyn_models/docs/REBUILD.md` phases A–B, plus the mdCATH/ATLAS data from phase G).

## The questions

1. **Do working Evoformer blocks improve the predictions?** Train the same head with the
   autocast cache on (the released code's bug: blocks stay at init) and off (the fix), and
   compare on the held-out test sets. This is the question the paper's ablation cannot answer.
2. **How much do the heads add over ESMFold at all?** Compare against a linear readout of
   ESMFold's own pair state, trained the same way.
3. **Is our training pipeline faithful?** Our "cache on" model, trained from scratch, should
   land near the released model's numbers (note 12). If it does not, questions 1–2 are about
   our pipeline, not theirs.

Scope: the **dynamic-contact head only** (the paper's headline metric) until question 1 has an
answer. Kinetics and occupancy heads are phase 6, optional.

## Variants

| id | what trains | blocks | purpose |
|---|---|---|---|
| R | nothing: the released V2 weights | dead | reference numbers (already measured, note 12) |
| A | full dynamic head, `autocast(cache_enabled=True)` | dead (the bug) | reproduces the authors' setup in our pipeline |
| B | full dynamic head, `autocast(cache_enabled=False)` | **trained** | the fix |
| C | full dynamic head with `blocks = ModuleList()` | none | should equal A (A's blocks are identity); a pipeline check, and cheaper |
| D | `Linear(128 → 5)` on ESMFold's `z` only (no transitions, no DCM) | none | floor: what ESMFold gives with one linear layer |
| E | rule: predicted dynamic if ESMFold's predicted Cα distance lies in a band around 8 Å (band fit on train) | none | floor without any learned representation |

Two starting points for A and B:

- **from scratch** (heads re-initialised): the clean comparison; what the authors should have run.
- **from the released weights** (fine-tune, as check 7 did): cheaper, answers "does turning the
  blocks on improve the shipped model?".

Run from-scratch first if compute allows; fine-tune is the fallback.

## Recipe (the paper's fine-tuning stage, Methods)

- mdCATH train split (4,858 domains), all 5 temperatures; validation 270, test 270 (`train.csv`
  etc. from the Data Bank `mdcath.zip`)
- 256-residue random crops, sampling weighted by protein length
- Adam, lr 1e-4; focal loss α = 0.85, γ = 2 (dynamic head); bf16 autocast
- **effective batch 64** (paper) via gradient accumulation; 1,000 training samples and 100
  validation samples per epoch; early stopping when validation loss has not improved for 10
  epochs; cap at 150 epochs
- **Deviation, stated up front:** no pretraining on the RCSB experimental clusters (the paper
  pretrains the dynamic head there first). All variants skip it equally, so the comparison
  among A–D is fair; R is not directly comparable to A–D because of it. If A from scratch falls
  far short of R, add the pretraining stage (the RCSB data is on the Data Bank) before
  drawing conclusions.

## Phase 1 — training harness (laptop, no GPU)

`investigation/scripts/train_heads.py`, grown from `short_training_check.py` (which already has
the dataset, the losses, the focal-loss and tensorboard shims, the matched-batch sampler, and
the block-norm logging):

- `--variant {A,B,C,D}`, `--init {scratch,released}`, `--seed`, `--features <cache dir>`
- the paper's recipe above, early stopping, a checkpoint per improvement, a JSON log per epoch
  (train loss, val loss, val balanced accuracy, block output-layer norm)
- the same pre-drawn sample/crop sequence for every variant with the same seed (paired runs)

CPU tests before any GPU time (`investigation/tests/test_train_heads.py`):

1. variant B gives every block tensor a gradient; variant A gives the blocks none (the
   notebook's §10 mechanism, on the real `DynamicModule` with a tiny config, CPU autocast)
2. variant C's head output equals variant A's at initialisation, bit for bit
3. the loss on one fake batch equals `training/loss.py`'s `esmdynamic_loss`
4. one optimiser step changes exactly the parameters it should

## Phase 2 — cache ESMFold's outputs once (GPU, ~2–3 h)

The heads only see ESMFold's outputs, and ESMFold is frozen, so run it **once** per crop and
train every variant from disk. That removes the 3B-parameter forward pass from every training
step (check 7 took ~1.4 s per sample with it).

`investigation/scripts/cache_esmfold_features.py`: for each training protein, a fixed pool of
crops (all of a protein ≤ 256 residues; 4 random 256-windows of a longer one, drawn
length-weighted), ESMFold run on the crop's sequence exactly as `forward_from_seq` does, saving
what the heads read: `s_s`, `s_z`, `lddt_logits`, `lm_logits`, `ptm_logits`, `distogram_logits`
(fp16) plus the crop window. Validation and test proteins: the crops `eval_protocol.py` uses.

- Size: ~11 MB per crop at typical mdCATH lengths, ~60 GB per crop of every training protein;
  keep it on scratch, not in git. Budget ~150 GB.
- Deviation: a finite crop pool instead of fresh crops every draw. Fine for comparing variants;
  stated in the write-up.
- **Check:** the dynamic head run on cached features equals the head run on live ESMFold
  outputs (max difference at fp16 rounding) for 20 proteins.

## Phase 3 — pilot (GPU, ~4 h)

One seed each of A, B, C, D for ~10 epochs:

- measure seconds per sample and memory (B is the expensive one: gradients through
  2 blocks × 4 recycles on 256² pairs) and fix walltimes
- confirm curves move: B's block norm rises from 0, A's stays 0; C tracks A
- confirm validation balanced accuracy is heading toward the released model's ~0.81

Stop here and review if A is not heading toward R's numbers.

## Phase 4 — main runs (GPU; estimate after the pilot)

| variant | init | seeds |
|---|---|---|
| A, B | scratch | 3 each |
| A, B | released (fine-tune) | 2 each |
| C, D | scratch | 1 each (C must match A; D is a floor) |

Rough budget before measuring: 4–8 h per from-scratch run on cached features → ~50–80 GPU-h
in total on L40S. E needs no training (one fit of the distance band on train).

## Phase 5 — evaluation

`eval_protocol.py` extended with `--checkpoint`, scoring each best-validation checkpoint
exactly as note 12 did (both crop protocols; all six metrics; mdCATH at all five
temperatures and ATLAS).

- **Primary comparison:** B − A, test balanced accuracy and AUROC on mdCATH 320 K and ATLAS,
  **paired per protein** (same proteins, same crops), mean over seeds, with a protein-level
  bootstrap 95% interval
- **Decision rule, set before looking:** working blocks "help" if B − A is positive and its
  interval excludes 0 on both test sets; "no detectable effect" if the interval sits inside
  ±0.01; anything else reported as mixed
- Also: training-fit gap (train loss at the stopping epoch); A vs C agreement (must be ~0);
  A, B vs D and E (what the heads add over ESMFold); A vs R (pipeline fidelity)

## Phase 6 — optional follow-ons

- the same comparison for the kinetics and occupancy heads (all three losses, as in check 7)
- a version with a working recycling distance embedding (it currently only ever reads row 0;
  see the notebook §8)
- the RCSB pretraining stage, if A from scratch falls short of R

## Outputs

- note `14_head_training.md`; results under `results/14_*` (JSON summaries; checkpoints stay on
  scratch and are backed up off-cluster, per REBUILD.md §10)
- figure E9: validation curves per variant, and test metrics per variant with intervals
- an EVIDENCE.md section stating the answer to question 1, whatever it is

## What could go wrong

- **A from scratch never reaches R:** likely the missing pretraining stage, or a recipe detail
  (class weighting, sampler weights in `splits_20_id_cutoff/*_weights.pt`, which are not in
  `mdcath.zip`). Compare with fine-tune results; add pretraining if needed.
- **B is unstable** (blocks waking up mid-training at lr 1e-4): add warm-up or gradient
  clipping, to *both* A and B, so the comparison stays matched.
- **Small effect, noisy test sets** (270 / 82 proteins): this is why the comparison is paired
  and uses several seeds; report the interval, not just the mean.
