# 16 — How pre-training was done: the paper, the supplement and the code

Sources: the published paper (Nature Communications 17:9623, 2026, doi 10.1038/s41467-026-76361-2;
Methods p.14) and its Supplementary Information (Supp. Note 1 §2.1, Table 5, Figs. 18–19); the
fork's code at HEAD and at `b51b35f` (2025-06-22, the V1 release); the V1 and V2 weights.

## What the paper specifies

- **Data:** RCSB PDB structures *and AlphaFold DB models*, grouped by MMseqs2 into 128,660 clusters
  at > 95% identity; the 88,294 with at least one dynamic contact are used (946,397 chains,
  598,144 structures, median four conformers per protein). Each cluster's members are aligned to
  a consensus sequence, which is the model input (Supp. Fig. 18).
- **Label:** a pair is dynamic if its Cα–Cα distance is < 7.5 Å in one structure and > 8.5 Å in
  another (a 0.5 Å margin around 8 Å). One condition, no temperature.
- **Sparsity:** ~0.3% of pairs are dynamic in these clusters, against ~15% in mdCATH at 320 K
  (Supp. Note 1 §2.1, 3,310 shared proteins). Hence the higher focal α.
- **Splits:** 90/5/5, MMseqs2 clusters, ≤ 20% identity between splits, 80% coverage.
- **Recipe:** dynamic-contact classification only ("other heads were fitted on MD data only");
  10,000 training and 1,000 validation samples per epoch; 256-residue random crops, sampling
  weighted by length (93% of dynamic contacts lie within 256 residues, Supp. Fig. 19C); Adam
  lr 1e-4; focal loss **α 0.99**, γ 2; batch 64; stop after 10 epochs without validation
  improvement; one RTX 4090; ESMFold frozen.
- **Not specified:** the pre-training prediction layer, the number of epochs run, and how the
  stage-1 weights were carried into fine-tuning. Supp. Table 5 lists only the final (V2)
  architecture (classification prediction layer 645 parameters = 128 → 5).

## What the code shows

- **The released code cannot pre-train.** `training/train.py` and `data_reader.py` at HEAD
  (2025-11/12) read the mdCATH layout only: `dynamic_contacts.pt` as (5, L, L) plus
  `kinetics.pt` and `frequency.pt`, which RCSB clusters do not have. `convert_csv_to_torch.py`
  writes the RCSB labels as a flat 1-D tensor that nothing at HEAD reshapes. The README's
  training section says it covers mdCATH only.
- **Pre-training is the V1-era code** (`b51b35f`): a single dynamic head, prediction layer
  LayerNorm + Linear(128 → 1); the reader turns the flat labels into L × L with
  `scipy.spatial.distance.squareform` (row-major strict upper triangle, as inferred in plan 04);
  `sigmoid_focal_loss` summed over pairs; α passed as `--weight_positive`; bf16 autocast and the
  `no_grad` recycles, so the same autocast-cache bug (note 02).
- **No early stopping in either version.** Both scripts run `--epochs` and keep the
  best-validation checkpoint; the paper's patience-10 rule was applied outside the code.
- **A V1-era masking quirk.** The V1 loss zeroes only `loss[i, :, L:, L:]`, so pairs between a
  real residue and a padding position stay in the loss as negatives; and the V1 reader passes
  the full, uncropped length. Cropped samples are always 256 long, the batch maximum, so only
  uncropped short chains batched with longer ones are affected; with α 0.99 negatives carry
  weight 0.01. HEAD masks `[:L, :L]` correctly.

## What the weights show (`scripts/compare_v1_v2_heads.py`)

- **V2's dynamic head was initialised from V1.** Its transitions and positional embedding are
  close to V1's (cosine 0.86–0.999, relative difference 0.04–0.54) — further trained, not
  re-initialised. Two unrelated V2 heads (dynamic vs kinetic) sit at cosine ≈ 0.
- **All 160 of V2's dynamic DCM block tensors are bit-identical to V1's**, including the 54
  with random initial values. The kinetic and frequency heads have different random
  initialisations. So the blocks were initialised once, for V1, and survived both training
  rounds untouched.
- Consequence: V2's dynamic head carries V1's training (which, per preprint v1, was the same
  RCSB-then-mdCATH recipe) forward. Whether the RCSB stage was re-run for V2 or V2 simply
  started from the V1 checkpoint cannot be told from the weights; the bit-identical blocks
  are consistent with either, since the blocks never train.

## For replicating stage 1 (plan 04)

Use the paper's numbers (10,000 / 1,000 samples per epoch, α 0.99, patience 10), the V1-era
reader's `squareform` layout and 1-output head, and the corrected `[:L, :L]` mask. The
published data covers 78,326 / 4,352 / 4,351 clusters (87,029 of the paper's 88,294; 181 are
truncated in the archive and excluded).
