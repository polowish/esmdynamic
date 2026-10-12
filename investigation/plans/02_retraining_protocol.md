# Protocol: retrain ESMDynamic's heads the way the authors did

Status: **protocol, not started** (2026-10-12). Supersedes the training sections of
`01_head_training.md` where they differ; the evaluation half of that plan still stands.

**Goal**: train the heads with the authors' own recipe, twice per seed, differing only in the
autocast weight cache, to answer *what do working Evoformer blocks add?* on the paper's own
benchmarks. Everything below is taken from the paper's Methods (Nat Commun 2026, 17:9623) and
the released code (`esm/esmdynamic/training/{train,loss,data_reader}.py` at `b3c7f04`); every
place we cannot match them exactly is listed in section 5 with the reason.

## 1. Arms

| arm | what differs | why |
|---|---|---|
| **A, as released** | `torch.autocast(bf16)`, cache on (`train.py` as shipped) | replicates the authors' run, bug included; must land near the released weights' numbers, or our pipeline differs from theirs |
| **B, fixed** | `torch.autocast(bf16, cache_enabled=False)` | the one-line fix; the only arm in which the Evoformer blocks train |
| C, no Evoformer (optional) | blocks removed from the architecture, cache irrelevant | a cleaner "no blocks" baseline; should equal A within seed noise, since A's blocks never move |

Everything else is identical across arms and paired by seed: same data order, same crops, same
initialisation. 3 seeds per arm (A, B); 1 for C.

## 2. Stage 1: pre-training the dynamic-contact head on experimental structures

From Methods ("Pretraining") and `train.py`:

| setting | value | source |
|---|---|---|
| data | RCSB + AlphaFold DB structure clusters | Data Bank `rcsb.zip` (datafile `97ijo`), unpacked with `decompress_rcsb_dataset.py` (`n5f4t`) |
| heads | **dynamic-contact head only** (`--loss_heads dynamic_logits`) | "this pretraining is only relevant for the dynamic contact classification task" |
| trainable | the dynamic head's parameters; ESMFold frozen (`requires_grad_(False)`) | Methods; `init_optimizer_for_heads` |
| optimiser | Adam, lr 1e-4, no scheduler, no weight decay, no clipping | Methods; `train.py` |
| loss | sigmoid focal loss, **α = 0.99**, γ = 2, mean over valid pairs | Methods (`train.py`'s default α = 0.25 is overridden) |
| crops | 256 residues, random start, all L × L pairs of the crop | `DynContactDataset`, Methods |
| sampling | with replacement, probability ∝ protein length | Methods; `DynContactDataset._compute_sampling_weights` |
| per epoch | 10,000 training samples, 1,000 validation samples | Methods; `train.py` defaults |
| batch | **64** | Methods (`train.py` default 4 × 8 = 32 is overridden) |
| precision | bf16 autocast over the whole forward; cache per arm (section 1) | `train.py` |
| recycling | the DCM's default 4 passes, `no_grad` on the first 3 | `dynamic_module.py` |
| chunk size | 256 | `train.py` |
| stopping | **early stopping: stop when validation loss has not improved for 10 epochs**; keep the best-validation-loss checkpoint | Methods (`train.py` itself runs fixed epochs and keeps the best checkpoint) |

## 3. Stage 2: fine-tuning all heads on mdCATH

From Methods ("Finetuning"): "the same procedure as pretraining, except":

| setting | value |
|---|---|
| data | mdCATH, all 5 temperatures; splits `train.csv` (4,858) / `val.csv` (270) from Data Bank `mdcath.zip`; `test.csv` (270) held out entirely |
| initialisation | dynamic head from stage 1 (same arm, same seed); kinetics and occupancy heads fresh |
| per epoch | **1,000 training, 100 validation samples** |
| dynamic loss | focal, **α = 0.85**, γ = 2 |
| occupancy loss | MSE on `frequency_pred` |
| kinetics loss | weighted cross-entropy, class weights `kinetic_weights.pt` (in `mdcath.zip`) |
| combining | `esmdynamic_loss`: **mean** over the active heads |
| auxiliary heads | **confidence (dynamic, kinetics) and occupancy-residual heads enabled after 10 epochs** ("to avoid fitting the confidence and residual heads to random noise"): epochs 1–10 with `dynamic_logits kinetic_logits frequency_pred`, then add `dynamic_confidence kinetic_confidence frequency_residual_pred` |
| everything else | as stage 1: Adam 1e-4, batch 64, 256-residue length-weighted crops, bf16, early stopping (patience 10) |

## 4. Evaluation

Each arm's best-validation checkpoint (and the released weights, for reference), scored exactly
as the reproduction figures:

- **ATLAS test** (82 chains): `train.py`'s protocol, 256-residue crops, raw logits > 0.5, all six
  Table 2 metrics (note 12).
- **mdCATH test** (270 domains), all 5 temperatures: pairs |i − j| ≥ 3, symmetrised probability
  > 0.5, scikit-learn per-protein balanced accuracy; plus AUROC, precision, recall, F1,
  occupancy RMSE (note 13).
- **Fig. 2A analogue**: balanced accuracy and occupancy RMSE on the train, validation and test
  splits, per arm.
- **Kinetics** (SI Tables 8–9): macro recall / precision for on- and off-times, per arm.

**Comparison**: B − A per protein (paired), averaged over seeds, with a protein-level bootstrap
95% interval. Decision rule, fixed before looking: working blocks help if the interval excludes 0
on both test sets; no detectable effect if it lies within ±0.01; otherwise mixed.

**Pipeline checks** that must pass before reading B − A:
- arm A's blocks stay at exactly 0 (output-layer norm 0.000 every step); arm B's grow from step 1
- arm A lands within seed noise of the released weights on every metric above
- arm C matches arm A within seed noise

## 5. Where we cannot match them exactly

| item | theirs | ours | effect |
|---|---|---|---|
| `train.py` cannot run as shipped | n/a (it predates the native-contact step) | run `model.forward` up to the end of the heads, as `short_training_check.py` does; `forward_from_seq`'s final native-contact step calls `.numpy()` on outputs that require grad | none: that step feeds no loss |
| sampler weight files | `splits_20_id_cutoff/{train,val}_weights.pt` | not distributed; length-proportional weights, the dataset class's own fallback and what Methods describes | small; unknown if theirs also down-weighted large clusters |
| batch 64 on one GPU | RTX 4090 (24 GB); presumably accumulated | batch 1 × 64 accumulation for arm B (gradients through the blocks need ~44 GB at batch 4); batch 4 × 16 where memory allows; loss divided by the accumulation count as in `train_one_epoch` | numerically equivalent apart from summation order |
| hardware / precision | RTX 4090, bf16 | L40S / H100 / RTX PRO 6000, bf16 | rounding-level |
| number of epochs | early stopping, count not reported | same rule; cap 300 (stage 1) and 200 (stage 2) | none unless the cap binds (logged) |
| seeds | not reported (1 run) | 3 per arm | adds error bars they did not have |
| pre-training split | RCSB/AFDB train/val split | whatever split `rcsb.zip` ships (to verify on unpacking); `mdcath_to_rcsb_mapping.csv` used to check no mdCATH test domain's structures are in pre-training | none if their split ships; checked either way |

## 6. Implementation

`investigation/scripts/train_faithful.py`: `train.py`'s loop with only the changes in section 5,
plus `--autocast-cache {on,off}`, `--no-evoformer`, `--stage {pretrain,finetune}`,
`--aux-heads-after 10`, `--patience 10`, `--seed`, and per-step logging of training loss and the
blocks' output-layer norm per head (missing from check 7). Same data order across arms for a
given seed (pre-drawn sample/crop sequence, logged to disk).

CPU tests first: (1) arm A gives the block Linears no gradient and arm B does; (2) the loss on a
fake batch equals `esmdynamic_loss`; (3) a no-Evoformer head's output equals arm A's at
initialisation; (4) the sample/crop sequence is identical across arms.

## 7. Cost and order

Measured in check 7: ~1.4 s per sample with gradients through the blocks (L40S, ESMFold run on
every sample, as the authors did).

| stage | samples per epoch | per epoch | at ~50 epochs |
|---|---|---|---|
| pre-training | 11,000 | ~4.3 h | ~210 GPU-h per run |
| fine-tuning | 1,100 | ~0.4 h | ~20 GPU-h per run |

So roughly **230 GPU-h per arm per seed**, ~1,400 GPU-h for A + B × 3 seeds: too much to run
blind. Order:

1. **Pilot**: arms A and B, one seed, stage 2 only, with the dynamic head initialised from the
   V1 release. V2 was trained onward from V1 (all 160 V1 block tensors are bit-identical to
   V2's dynamic head; none of V1's 22 trained non-block tensors are), so V1 is the nearest
   public stand-in for a stage-1 checkpoint. It is not known to be exactly the authors'
   pre-training output (V1 may also have seen mdCATH), so the pilot is labelled as such.
   ~20 GPU-h per arm. It already answers whether turning the blocks on changes the shipped
   model's stage 2.
2. If B differs from A: the full two-stage run, 3 seeds.
3. Optional speed-up for step 2 (a stated deviation): cache ESMFold's outputs for a fixed pool of
   crops, which removes the 3B-parameter forward pass from every step (several-fold faster); report
   that the crop pool was finite.

Outputs: note 15, results `results/15_*`, a Fig. 2A-style figure and the reproduction figures
with a "trained blocks" series.
