# Plan 05: fine-tune the released V2 heads with working DCM blocks

Decided with the user on 2026-10-13. Supersedes plan 04 for what runs first. Plan 04's
two-stage retraining from scratch becomes **Phase 2**, decided after Phase 1's results.

## Question

What do working Evoformer blocks add to the published model?

The published V2 weights have every DCM block tensor at its initial value (the autocast-cache
bug), and the block output layers are zero. So V2 with the fix is *exactly* the published
model at step 0, and anything that changes comes from training with blocks that compute.

## Phase 1, option 1 (first): train only what never trained

- **Trainable:** the 488 tensors still at initialisation, 88,737,698 parameters
  (`retrain_lib.at_init_names`, checked by `at_init_set.py`):
  - all 480 DCM block tensors (3 heads × 2 blocks);
  - the occupancy head's single-state path: `frequency.seq_transition.*` and
    `frequency.dynamic_module.recycle_s_norm.*`. This path reaches its loss only through the
    blocks, so it never got gradient. Its LayerNorms are exactly 1 and 0.
- **Frozen:** everything the authors actually trained, about 7.2 M parameters. It is checked
  bit-identical after every epoch.
- **Audit false positives:** the checkpoint audit's two other flags (one tensor each in
  `kinetic.confidence_head.3` and `frequency.residual_head.3`) are biases inside the init bound
  whose weights are clearly trained. They stay frozen.
- **Arm:** `cache_enabled=False` (the fix). No control is needed: with the bug on, none of
  these tensors can receive gradient.

## Phase 1, option 2 (second): every head parameter, plus a control

- `--trainable all`, as `train.py` does, both `--arm fixed` and `--arm bug`.
- Same seed, the same draws and the same start (V2).
- The control separates "the blocks work" from "the trained layers trained longer".

## Recipe (both options)

**As the authors** (Methods, and `training/{train,loss,data_reader}.py`):
- mdCATH train / val splits;
- 1,000 draws per epoch, domain ∝ length, with replacement;
- a fresh 256-residue crop per draw, start uniform on 0 … L−257, the code's off-by-one kept;
- Adam, lr 1e-4;
- batch 64, as 1 × 64 accumulation, with a final partial step per epoch;
- bf16 autocast;
- their loss: `esmdynamic_loss` over all six heads, focal α 0.85 γ 2, weighted CE with
  `kinetic_weights.pt`, MSEs; summed over pairs, mean over heads;
- best-validation-loss checkpoint; early stopping with patience 10; our cap of 200 epochs.

**Ours (stated departures):**
1. **All six losses from step 0.** The epoch-10 delay protected untrained confidence heads;
   V2's are trained.
2. **A fixed validation set:** every val domain, one window each, drawn once, every epoch.
   It replaces 100 fresh random draws, which make early stopping noisy.
3. **ESMFold cached, not run live, in eval mode.** Every possible training window is cached
   (35,012), so draws are exactly the original's fresh crops.
   - **Why the mode matters:** `train.py`'s `model.train()` also puts the frozen ESMFold in
     train mode, so its structure-module dropout (0.1) is on. The noise spreads through
     recycling, and measured on 12 mdCATH windows (job 2229389) it changes:
     - `s_z` by about 5% (largest relative difference);
     - the pLDDT features by about 25%;
     - the released model's probabilities by up to 0.05–0.2;
     - about 0.4–0.5% of pairs per window across the 0.5 threshold.

     Eval mode is deterministic (repeat difference 0).
   - **Decided with the user: eval mode.** Everything else runs ESMFold in eval mode
     (prediction, validation, every evaluation), and nothing indicates the train-mode
     dropout was intended. It was Meta's regulariser for training ESMFold, not a designed
     augmentation for the heads.
   - **Consequences:** training inputs now match test inputs, all arms see identical
     features, and one cache serves every variant. What's given up is the incidental
     augmentation the authors' heads had; deliberate augmentation can be added later if
     option 2 overfits.
4. **Optional warm-up and gradient clipping** (`--warmup-steps`, `--clip`), off by default.
   Turned on only if the smoke or timing runs show instability.
5. **Per-epoch monitoring of the published protocol's metrics** (pairs |i−j| ≥ 3,
   symmetrised probability, scikit-learn-style balanced accuracy, AUROC, occupancy RMSE).
   Monitoring only.

## Cache (`cache_esmfold_features.py`)

- **What:** mdCATH train (`--windows all`, 35,012 windows) and val (`--windows fixed`, 270).
- **Storage:** stored in ESMFold's own output dtypes (lossless). Labels are stored once per
  domain and cropped on read.
- **Size:** about 1.2–2.4 TB, an estimated 3–7 GPU-h.
- **Reuse:** options 1, 2, the control and any later variant reuse it; it is re-run only for
  a new crop length, dataset or ESMFold setting.
- **Test-set caching:** deferred. Live evaluation is under 1 GPU-h per checkpoint, and keeps
  the verified scripts (notes 12–13) unchanged.

## Evaluation

- **Models:** released V2, option 1, option 2 fixed and option 2 bug.
- **Protocols:**
  - ATLAS: `eval_protocol.py` (256-residue crops, raw logits);
  - mdCATH test: `eval_pairset.py` (|i−j| ≥ 3, symmetrised, scikit-learn-style balanced accuracy).
  - Both take `--heads-checkpoint` and `--tag`.
- **Comparison:** per-protein paired differences with bootstrap 95% intervals. The decision
  rule from plan 02 applies: working blocks help if the interval excludes 0 on both test sets,
  and there is no detectable effect if it lies within ±0.01.
- **Leakage:** every result is reported on all test proteins and on the subset absent from
  RCSB pre-training (`results/15_pretrain_overlap.json`). 138 of 270 mdCATH test domains and at
  least 32 of 82 ATLAS chains are in RCSB train clusters.
- **Also:**
  - kinetics and the confidence and residual heads (SI Tables 8–12);
  - breakdown by sequence separation;
  - calibration;
  - learning curves;
  - deleting the trained blocks must now change the outputs.

## Order

1. ✅ Code: `retrain_lib.py`, `cache_esmfold_features.py`, `train_from_v2.py`,
   `retrain_timing.py`, `tests/test_retrain.py`.
2. ✅ Timing and smoke job (`slurm/retrain_timing.sbatch`, job 2229389). It measured:
   - ESMFold seconds per window, the dtypes and the dropout effect;
   - head seconds per sample for each option;
   - disk per window;
   - and runs a smoke run of all three configurations.

   Results:
   - ESMFold: 1.81 s per 256-residue window, 0.85 s per short window;
   - disk: 52.5 MB per 256-residue window;
   - heads: 0.32–0.34 s per training sample, peak memory 12 GB;
   - all three smoke configurations ran;
   - the blocks move only in the fixed arms.

   Reported to the user.
3. Build the mdCATH cache, eval mode (`slurm/cache_mdcath.sbatch`): about 16.5 GPU-h and
   1.7 TB.
4. **With the user's go-ahead:** option 1, then evaluate and write it up; then option 2 and
   its control.

## Checks built in

- **Step 0 equals the published model:** `test_v2_step0_is_released_model`, plus the epoch-0
  validation entry.
- **Frozen tensors stay frozen:** bit-identical after every epoch, asserted.
- **The blocks learn only with the fix:** the block output-layer norm per head, per step.
  The fixed arm must rise from step 1; the bug arm must stay at 0.000.
- **The cache is faithful:** cached windows equal live ESMFold (`test_cached_equals_live`).
- **The arms see the same data:** identical draws across runs at one seed (`plans/epoch_*.npy`).
