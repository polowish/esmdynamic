"""Run frozen ESMFold once per window and cache exactly what the heads read (plan 05).

    # 1. the window table and the labels (CPU): <out>/windows.csv, labels.pt, meta.json
    python investigation/scripts/cache_esmfold_features.py --split train --windows all \
        --esmfold-mode eval --out ~/scratch/esmdyn_cache/mdcath_train --plan-only
    # 2. features (GPU array; task k writes shards k, k + n, k + 2n, ...; resumable)
    python investigation/scripts/cache_esmfold_features.py <same args> --task $SLURM_ARRAY_TASK_ID --n-tasks 8

Windows (DynContactDataset's crop: a chain of <= 256 residues whole, else 256 residues from a
start uniform on 0 .. L - 257):
  --windows all     every window the training sampler can draw, so training draws fresh crops
                    exactly as train.py does (mdCATH train: 35,012 windows);
  --windows fixed   one window per domain drawn once with --seed (the fixed validation set).

ESMFold runs as in train.py's forward_from_seq on the window's sequence: one sequence (no
padding), batch_encode_sequences(..., 512, "G" * 25), chunk size 256, default recycles, under
torch.no_grad() and bf16 autocast. --esmfold-mode sets ESMFold's module mode (train.py trains
with model.train(), i.e. structure-module dropout on; validation in eval mode); in train mode
the RNG is seeded per window (seed * 1_000_003 + window id) so a window can be recomputed.

Saved per window, in the dtypes ESMFold produced (lossless): s_s, s_z, lddt_last
(= lddt_head[-1]), lm_logits, ptm_logits, distogram_logits, residue_index, mask. Labels are
stored once per domain at full length (labels.pt) and cropped when a window is read.
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch

from retrain_lib import (CROP, DATASETS, enumerate_windows, fixed_windows, load_mdcath_labels,
                         read_seq, split_table)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["mdcath"], default="mdcath")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--windows", choices=["all", "fixed"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--esmfold-mode", choices=["train", "eval"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=32, help="windows per shard file")
    ap.add_argument("--task", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--n-tasks", type=int, default=1)
    ap.add_argument("--max-windows", type=int, default=None, help="first N windows only (timing)")
    ap.add_argument("--min-length", type=int, default=0, help="only domains this long (timing)")
    ap.add_argument("--plan-only", action="store_true", help="write windows.csv, labels.pt, meta.json")
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))
    folders = Path(DATASETS[args.dataset]["folders"])

    ids, lengths = split_table(args.dataset, args.split)
    wins = enumerate_windows(lengths) if args.windows == "all" else fixed_windows(lengths, args.seed)
    wins = [w for w in wins if lengths[w[0]] >= args.min_length]
    if args.max_windows:   # whole domains only, so every crop a draw can pick is in the cache
        keep, count = set(), {}
        for i, _, _ in wins:
            count[i] = count.get(i, 0) + 1
        total = 0
        for i in dict.fromkeys(i for i, _, _ in wins):
            if total + count[i] > args.max_windows and keep:
                continue
            keep.add(i); total += count[i]
        wins = [w for w in wins if w[0] in keep]
    n_shards = (len(wins) + args.shard_size - 1) // args.shard_size
    meta = dict(dataset=args.dataset, split=args.split, windows=args.windows, seed=args.seed,
                esmfold_mode=args.esmfold_mode, crop=CROP, n_domains=len(ids),
                n_windows=len(wins), shard_size=args.shard_size, n_shards=n_shards)

    if args.plan_only:
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "windows.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window_id", "domain_index", "domain", "start", "end", "length_full",
                        "shard", "pos"])
            for k, (i, s, e) in enumerate(wins):
                w.writerow([k, i, ids[i], s, e, int(lengths[i]), k // args.shard_size, k % args.shard_size])
        used = sorted({i for i, _, _ in wins})
        labels = {}
        for i in used:
            dyn, kin, freq = load_mdcath_labels(folders / ids[i], int(lengths[i]))
            labels[ids[i]] = dict(dyn=dyn, kin=kin, freq=freq)
        torch.save(labels, out / "labels.tmp")
        (out / "labels.tmp").rename(out / "labels.pt")
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta), flush=True)
        return

    assert json.loads((out / "meta.json").read_text())["n_windows"] == len(wins), "run --plan-only first"
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    model = esm.pretrained.esmfold_v1().cuda()
    model.requires_grad_(False)
    model.set_chunk_size(256)
    model.train(args.esmfold_mode == "train")

    seqs = {}
    dtypes, t_fold, n_done, t0 = {}, 0.0, 0, time.time()
    for k in range(args.task, n_shards, args.n_tasks):
        path = out / f"shard_{k:05d}.pt"
        if path.exists():
            continue
        recs = []
        for wid in range(k * args.shard_size, min(len(wins), (k + 1) * args.shard_size)):
            i, s, e = wins[wid]
            if i not in seqs:
                seqs = {i: read_seq(folders / ids[i])}
            aatype, mask, residx, _, _ = batch_encode_sequences([seqs[i][s:e]], 512, "G" * 25)
            aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
            if args.esmfold_mode == "train":
                torch.manual_seed(args.seed * 1_000_003 + wid)
            torch.cuda.synchronize(); ts = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                st = model(aatype, mask, residx, None, None)
            torch.cuda.synchronize(); t_fold += time.time() - ts
            feats = {"s_s": st["s_s"][0], "s_z": st["s_z"][0], "lddt_last": st["lddt_head"][-1][0],
                     "lm_logits": st["lm_logits"][0], "ptm_logits": st["ptm_logits"][0],
                     "distogram_logits": st["distogram_logits"][0]}
            rec = {"window_id": wid, "residue_index": residx[0].cpu(), "mask": mask[0].cpu()}
            for key, v in feats.items():
                assert torch.isfinite(v).all(), (wid, key)
                rec[key] = v.detach().cpu().clone()
                dtypes[key] = str(v.dtype)
            recs.append(rec)
            n_done += 1
        tmp = path.with_suffix(".tmp")
        torch.save(recs, tmp)
        tmp.rename(path)
        el = time.time() - t0
        print(f"shard {k}: {len(recs)} windows; {n_done} done, {el / n_done:.2f} s/window overall, "
              f"{t_fold / n_done:.2f} s/window ESMFold", flush=True)
    rep = dict(task=args.task, windows=n_done, esmfold_s_per_window=t_fold / max(n_done, 1),
               wall_s=time.time() - t0, dtypes=dtypes, device=torch.cuda.get_device_name(0))
    (out / f"task_{args.task:03d}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
