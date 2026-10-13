# Handoff: run the cached two-stage retraining (one seed, arms A and B)

Written 2026-10-13 for a fresh session. **The user approved this run** ("continue with the 80 hour
GPU plan (pre trained, cache, with one seed)"); still **check with the user before submitting the
main GPU training jobs** (step 6), and report the timing measurement (step 4) first.

Read first: `plans/02_retraining_protocol.md` (the authors' recipe, cited line by line) and
`plans/03_cheap_trial.md` (the evaluations). This file is the operational plan that combines them:
**both stages, ESMFold outputs cached once, one seed, two arms**, ~85 GPU-h.

## 0. State of things right now

**Repos** (laptop `/Users/smadejski/disco_lab/github/`, cluster `~/scratch/github/`):
- esmdynamic fork, branch `investigate/dynamicmodule-untrained` (never open a PR); the cluster's
  investigation worktree is `~/scratch/github/esmdynamic-investigation`.
- dyn_models `main`. `CLAUDE.md` + `docs/ENVIRONMENT.md` + `docs/REBUILD.md` describe the cluster.

**Cluster access**: `ssh ucsf-cluster` (ControlMaster). If it fails, other live sockets may exist
(`ls ~/.ssh/control-*`; test with `ssh -S <socket> -O check x`); otherwise ask the user to run
`! ssh ucsf-cluster`. Use `-o BatchMode=yes`.

**Data on the cluster** (`~/scratch/github/esmdynamic-investigation/investigation/data/`):

| dataset | path | contents |
|---|---|---|
| RCSB + AFDB clusters (stage 1) | `rcsb/rcsb/` | `train.csv` 78,326 ids, `val.csv` 4,352, `test.csv` 4,351; `train_weights.pt`, `val_weights.pt`, `test_weights.pt` (the sampler weights `train.py` loads); `rcsb/<id>/consensus.fasta` + `dynamic_contacts.csv` (125,082 folders). **Not yet converted** to `.pt`: the labels are a flattened 0/1 matrix, one integer per line, a single condition (no temperatures). Converter: `rcsb/decompress_rcsb_dataset.py <root>` (writes `dynamic_contacts.pt` next to each csv). Also `rcsb/mdcath_to_rcsb_mapping.csv` |
| mdCATH (stage 2) | `mdcath/` | `train.csv` 4,858 / `val.csv` 270 / `test.csv` 270; `kinetic_weights.pt`; `dataset/<id>/` 5,398 domains with `consensus.fasta`, `dynamic_contacts.pt` (5, L, L), `kinetics.pt`, `frequency.pt` |
| ATLAS test | `atlas_test/` | 82 chains, `*_dyn_cont.npy`, `*_freq.npy`, `atlas_test.csv` |

**Weights**: `~/scratch/github/esmdynamic/model_weights/hub/checkpoints/` (`esmdynamic.pt` V2,
sha256 `b41a1a22…`, plus ESMFold and ESM-2 3B). V1 weights: laptop
`~/disco_lab/rescued_from_cluster_2026-10-07/esmdynamic_V1.pt` (scp it up when needed).

**Environment for all of this**: the ESMDynamic env. In a job:
```bash
cd $HOME/scratch/github/dyn_models
MODEL_SPEC=esmdynamic . slurm/config.sh; eval "$ENV_SETUP_EXTRACT"     # exports TORCH_HOME + tool caches
cd $HOME/scratch/github/esmdynamic-investigation
```
GPU jobs: `sbatch -p gpu --gres=gpu:nvidia_l40s:1 --mem=48G -c 8 -t HH:MM:SS -o <abs path>`.
Compute nodes have **no internet** (do all downloads on the login node). **Nothing in `$HOME`**
(caches go to `~/scratch/tool_caches`, `config.sh` 2b). Avoid `pkill -f`/`pgrep -f` patterns that
also match your own ssh command line (it kills the session; happened 2026-10-12). After copying
results back and committing them, the cluster worktree holds untracked copies that block
`git pull`: move them aside, pull, `cmp`, delete.

**Other work in flight**: the dyn_models ESM sweep (jobs 2200251–2200307) has **finished**: the queue
is empty and `results/` holds 28 files. Not yet done: check every cell succeeded
(`sacct -j 2200251-2200307`), `sbatch slurm/aggregate.sbatch`, compare with `docs/REBUILD.md`
section 8, back up `results/ figures/ slurm/logs/` to the laptop (REBUILD.md section 10), tick the
phase D boxes. Phases E (ESMDynamic sweep) and F (window/mix variants) follow.

## 1. What the run must match (and the one inferred detail)

From Methods + `esm/esmdynamic/training/{train,loss,data_reader}.py` (details and citations in
plan 02):

| | stage 1: pre-training | stage 2: fine-tuning |
|---|---|---|
| data | RCSB clusters, their train/val split, **their sampler weights** (`train_weights.pt`) | mdCATH train/val, length-proportional sampling (`DynContactDataset` default; mdCATH weight files not distributed) |
| heads trained | dynamic only | dynamic, kinetics, occupancy; then + dynamic/kinetic confidence and occupancy residual after 10 epochs |
| prediction layer | **1 output, LayerNorm + Linear** (as in V1: `prediction_layer.0` LN(128), `prediction_layer.1` Linear 128→1) | **5 outputs, Linear 128→5** (as in V2 `prediction_linear`), fresh; transitions + DCM carried over from stage 1 |
| focal α / γ | 0.99 / 2 | 0.85 / 2 |
| samples per epoch (train / val) | 10,000 / 1,000 | 1,000 / 100 |
| shared | Adam lr 1e-4 over the head parameters; batch 64 (accumulation); 256-residue random crops; bf16 autocast; ESMFold frozen; losses = mean over active heads (`esmdynamic_loss`); early stopping, patience 10 on validation loss, keep best checkpoint | same |

The 1-output → 5-output hand-over is **inferred from the V1/V2 weight shapes** (V2's dynamic-head
blocks are bit-identical to V1's; its other layers differ; V1 has a 1-output LN+Linear prediction
layer, V2 a 5-output Linear). Say so in the write-up. **Check before coding**: how the current
model code builds a 1-condition head (`DynamicHead(n_conditions=...)` in `esmdynamic.py`), and
whether V1 loads into a 1-condition head with an LN+Linear prediction layer; if the current code
cannot express V1's layer, add it in the training script (not in upstream code).

**Arms** (identical sample sequence, crops and initialisation for the same seed):
- **A**: `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` (cache on, as released)
- **B**: same with `cache_enabled=False` (the fix)

**Deviations to state** (plan 02 section 5): ESMFold outputs cached per crop from a fixed crop
pool instead of a fresh random crop per draw; forward stops before the native-contact step
(`.numpy()` on grad tensors); batch 64 via accumulation; L40S not RTX 4090; one seed; epoch cap.

## 2. Step 1: prepare stage-1 labels

On a CPU node (not the login node; 125k small files):
```bash
python investigation/data/rcsb/rcsb/../decompress_rcsb_dataset.py investigation/data/rcsb/rcsb/rcsb
```
(path: the script is `investigation/data/rcsb/decompress_rcsb_dataset.py`; root is
`investigation/data/rcsb/rcsb/rcsb`). Then check one: `dynamic_contacts.pt` length must be L² for
`L = len(consensus sequence)`; reshape to (L, L). If it is not L², stop and inspect.

## 3. Step 2: write the scripts (laptop, commit to the branch)

**`investigation/scripts/cache_esmfold_features.py`**: `--dataset {rcsb,mdcath} --split
{train,val} --crops-per-long N --seed S --out <dir>`.
- Crop pool: a sequence ≤ 256 → one crop (whole); longer → N random 256 windows
  (`torch.randint(0, L-256)`, as `DynContactDataset`). Suggested N = 2 for stage-1 train (≈ the
  number of draws per cluster over training is ~6), 3 for mdCATH train, 1 for val.
- For each crop: `model.esmfold(aatype, mask, residx, None, None)` on the **cropped sequence**
  (as `train.py` does via `forward_from_seq`), under `torch.no_grad()` and the same bf16 autocast
  as training; save fp16: `s_s`, `s_z`, `lddt_head[-1]`, `lm_logits`, `ptm_logits`,
  `distogram_logits`, `residue_index`, `mask`, plus the crop window and the cluster id. Exactly
  the keys `DynamicHead.forward` reads (`esmdynamic.py` ~lines 119–140).
- Shard output (e.g. 1,000 crops per file) to keep file counts sane.
- Storage: ~20–35 MB per 256-crop → **~2–3 TB for stage 1**, ~60 GB for mdCATH. Put it under
  `~/scratch/esmdyn_cache/` (not inside a git worktree). Check free quota first
  (`quota -s`; group 50 TB).

**`investigation/scripts/train_faithful.py`**: the `train.py` loop over cached features.
`--stage {pretrain,finetune} --arm {A,B} --seed S --features <dir> --init <ckpt>
--patience 10 --max-epochs (300 pretrain / 200 finetune) --aux-heads-after 10 --out <dir>`.
- Builds the heads exactly as the model does (`model.heads[...]`), stage-1 head with the 1-output
  LN+Linear prediction layer, stage-2 with the standard 5-output heads; stage 2 loads stage 1's
  transitions + DynamicModule into the dynamic head.
- Samples per epoch as in section 1, drawn with the dataset's weights from the crop pool; the
  draw sequence comes from `--seed` only, so arms A and B see identical data. Log it.
- Targets with `train.py`'s `build_outputs_and_targets_for_loss`; loss with
  `training/loss.py`'s `esmdynamic_loss` (provide torchvision's `sigmoid_focal_loss` with the shim
  in `short_training_check.py`; no tensorboard: no-op writer).
- Logs per step: loss per head; **block output-layer norm per head**; per epoch: train/val loss,
  val balanced accuracy. Saves best-val-loss and last checkpoints of the head state dicts.
- Reuse `short_training_check.py` for the shims, target building and block-norm code.

**CPU tests** (`investigation/tests/test_train_faithful.py`, run on the cluster in the
ESMDynamic env, which has openfold): (1) arm A gives block Linears no gradient, arm B does (the
real `DynamicModule`, tiny inputs, CPU bf16 autocast reproduces the bug: notebook §10);
(2) loss on a fake batch equals `esmdynamic_loss`; (3) a head run on cached features equals the
head run on live ESMFold output for one crop (GPU, in step 4); (4) identical draw sequence for A
and B at the same seed; (5) stage-1 → stage-2 hand-over copies exactly the intended tensors.

## 4. Step 3: measure, then report to the user

One short GPU job: cache 50 mdCATH crops; time ESMFold per crop; time a head forward+backward
per sample for arm A and arm B (dynamic head only, and all heads). Re-estimate: stage-1 caching
(~83k clusters × ~1.3 crops), per-epoch time per stage, total GPU-h and disk. **Report these to
the user before step 6.** The ~85 GPU-h figure assumed ~0.15 s/sample head-only and ~50 epochs.

## 5. Step 4: build the caches

Array jobs on L40S over shards: stage-1 train + val, mdCATH train + val. Verify counts, spot-check
a few crops against live ESMFold (test 3). Expect ~20–25 GPU-h for stage 1, ~3 for mdCATH.

## 6. Step 5: train (after the user's go-ahead)

For arm in A, B (parallel): stage 1 (`--stage pretrain`) → best checkpoint → stage 2
(`--stage finetune --init <stage1 best>`). Long jobs: walltime per job ≤ what the partition allows,
with resume-from-last-checkpoint (`--resume`) and `--requeue`. Watch: arm A's block norm must stay
0.000; arm B's must rise from step 1.

## 7. Step 6: evaluate

Add `--heads-checkpoint <path>` to `eval_protocol.py` and `eval_pairset.py` (load trained head
state dicts into `esmdynamic()` after loading), then run for A, B (and the released weights for
reference), scored exactly as the reproduction figures:
- ATLAS: `eval_protocol.py --dataset atlas` (figure uses `crop256/unsym`)
- mdCATH: `eval_pairset.py --dataset mdcath` (figure uses k = 3, symmetrised, sklearn-style BA)
Then the evaluations in plan 03 (priority order): B − A paired per protein with bootstrap 95%
intervals and the pre-set decision rule; kinetics and confidence heads; sanity checks (A ≈ released,
A blocks 0, deleting blocks from trained B changes outputs); learning curves and a Fig. 2A analogue;
by sequence separation; calibration; ATLAS vs mdCATH.

## 8. Outputs and housekeeping

- Note `notes/15_retraining.md`; results `results/15_*` (json/csv only; checkpoints and caches stay
  on scratch, back up checkpoints to the laptop); figures in `figures/retraining/`, styled like the
  reproduction figures (Open Sans; published `#1D3BE2`, reproduced `#1DE2C4`; add a colour for
  arm B and say which in the legend).
- Update `investigation/README.md` (checks table) and, if the result changes the story,
  `EVIDENCE.md` + the shared page https://claude.ai/artifact/AbJa2BJYpo1Tp9EhassLLp (private;
  read it with the Artifact tool before republishing).
- Commit each step on `investigate/dynamicmodule-untrained`; never open a PR.
