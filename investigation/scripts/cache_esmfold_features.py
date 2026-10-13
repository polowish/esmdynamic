"""Run frozen ESMFold once per crop and cache what the heads read (plan 04, step 2).

    # the crop pool only (cheap, CPU): writes <out>/pool.csv
    python investigation/scripts/cache_esmfold_features.py --dataset rcsb --split train \
        --crops-per-long 2 --seed 0 --out ~/scratch/esmdyn_cache/rcsb_train --plan-only
    # features, one array task per GPU (shards k = task, task + n_tasks, ...)
    python investigation/scripts/cache_esmfold_features.py ... --task $SLURM_ARRAY_TASK_ID --n-tasks 16

The crop pool (the stated deviation from train.py, which draws a fresh crop each time): a chain
of <= 256 residues gives one crop, the whole chain; a longer one `--crops-per-long` windows,
starts drawn as DynContactDataset draws them. Deterministic in (dataset, split, n, seed).

Per crop, ESMFold runs as in train.py's forward_from_seq on the CROPPED sequence: batch of one,
batch_encode_sequences(..., 512, "G" * 25), chunk size 256, inside torch.no_grad() and bf16
autocast. `--esmfold-mode` sets ESMFold's module mode: train.py calls model.train() before its
training epochs (ESMFold included) and model.eval() before validation, so training pools are
cached in `train` mode and validation pools in `eval` mode; in `train` mode the RNG is seeded
per crop (seed + crop id) so a crop's features can be recomputed exactly (test 3).

Saved per crop, fp16: s_s (l, 1024), s_z (l, l, 128), lddt_last = lddt_head[-1] (l, 37, 50),
lm_logits (l, 23), ptm_logits (l, l, 64), distogram_logits (l, l, 64); residue_index, mask;
labels cropped like DynContactDataset: dyn uint8 (C, l, l) [C = 1 RCSB, 5 mdCATH], kin uint8
(5, 2, l, l), freq float32 (5, l, l) [mdCATH only]; plus the cluster id, crop window and length.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from retrain_lib import (CROP, DATASETS, build_crop_pool, load_labels, read_seq, split_table)

FP16_MAX = 65504.0


def write_pool(out, ids, lengths, pool, shard_size):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "pool.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crop_id", "cluster_index", "cluster", "start", "end", "length_full",
                    "shard", "pos"])
        for c, (i, s, e) in enumerate(pool):
            w.writerow([c, i, ids[i], s, e, int(lengths[i]), c // shard_size, c % shard_size])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["rcsb", "mdcath"], required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--crops-per-long", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--esmfold-mode", choices=["train", "eval"], default=None,
                    help="default: train for the train split, eval otherwise")
    ap.add_argument("--max-crops", type=int, default=None, help="first N crops only (timing)")
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))
    mode = args.esmfold_mode or ("train" if args.split == "train" else "eval")
    folders = Path(DATASETS[args.dataset]["folders"])

    ids, lengths, _ = split_table(args.dataset, args.split)
    pool = build_crop_pool(lengths, args.crops_per_long, args.seed)
    if args.max_crops:
        pool = pool[:args.max_crops]
    n_shards = (len(pool) + args.shard_size - 1) // args.shard_size
    meta = dict(dataset=args.dataset, split=args.split, crops_per_long=args.crops_per_long,
                seed=args.seed, n_clusters=len(ids), n_crops=len(pool), shard_size=args.shard_size,
                n_shards=n_shards, esmfold_mode=mode, crop=CROP)
    if args.task == 0 or args.plan_only:
        write_pool(out, ids, lengths, pool, args.shard_size)
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    if args.plan_only:
        return

    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    model = esm.pretrained.esmfold_v1().cuda()
    model.requires_grad_(False)
    model.set_chunk_size(256)
    model.train(mode == "train")

    mine = [k for k in range(n_shards) if k % args.n_tasks == args.task]
    t_start, n_done, t_fold = time.time(), 0, 0.0
    for k in mine:
        path = out / f"shard_{k:05d}.pt"
        if path.exists():
            continue
        recs = []
        for c in range(k * args.shard_size, min(len(pool), (k + 1) * args.shard_size)):
            i, s, e = pool[c]
            folder = folders / ids[i]
            seq = read_seq(folder)
            dyn, kin, freq = load_labels(args.dataset, folder, len(seq))
            crop = seq[s:e]
            aatype, mask, residx, _, _ = batch_encode_sequences([crop], 512, "G" * 25)
            aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
            if mode == "train":
                torch.manual_seed(args.seed * 1_000_003 + c)
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = model(aatype, mask, residx, None, None)
            torch.cuda.synchronize()
            t_fold += time.time() - t0
            feats = {"s_s": st["s_s"][0], "s_z": st["s_z"][0], "lddt_last": st["lddt_head"][-1][0],
                     "lm_logits": st["lm_logits"][0], "ptm_logits": st["ptm_logits"][0],
                     "distogram_logits": st["distogram_logits"][0]}
            rec = {}
            for key, v in feats.items():
                v = v.float()
                assert torch.isfinite(v).all() and v.abs().max() < FP16_MAX, (c, key)
                rec[key] = v.half().cpu()
            rec.update(residue_index=residx[0].cpu(), mask=mask[0].cpu(),
                       dyn=dyn[:, s:e, s:e].clone(), crop_id=c, cluster=ids[i], start=s, end=e,
                       length=e - s)
            if kin is not None:
                rec.update(kin=kin[:, :, s:e, s:e].clone(), freq=freq[:, s:e, s:e].clone())
            recs.append(rec)
            n_done += 1
        tmp = path.with_suffix(".tmp")
        torch.save(recs, tmp)
        tmp.rename(path)
        el = time.time() - t_start
        print(f"shard {k}: {len(recs)} crops; {n_done} done in {el:.0f} s "
              f"({el / n_done:.2f} s/crop overall, {t_fold / n_done:.2f} s/crop ESMFold)", flush=True)
    print(f"task {args.task} done: {n_done} crops, ESMFold {t_fold / max(n_done, 1):.3f} s/crop")


if __name__ == "__main__":
    main()
