"""Step 3 of plan 04: measure before committing GPU time.

    python investigation/scripts/retrain_timing.py --mdcath-cache <dir> --rcsb-cache <dir> \
        --out investigation/results/15_timing.json

1. ESMFold train vs eval mode on the same crop (train.py trains with model.train(), which
   reaches ESMFold: does it change what the heads see?).
2. Head forward + backward per sample, arm A (cache on) and B (cache off): stage 1 (dynamic
   head, 1 condition, dynamic loss) and stage 2 (all heads, all six losses); a validation
   forward; peak GPU memory; time to read a crop from the cache.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import retrain_lib as R


def time_heads(model, store, n, cache_enabled, loss_heads, alpha, conditions, kin_w, train=True):
    opt = torch.optim.Adam(model.heads.parameters(), lr=1e-4)
    model.train(train)
    ts, t_read = [], []
    torch.cuda.reset_peak_memory_stats()
    for k in range(n):
        t0 = time.time()
        rec = store.get(k % len(store))
        rec = {key: (v.clone() if torch.is_tensor(v) else v) for key, v in rec.items()}
        t_read.append(time.time() - t0)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.set_grad_enabled(train):
            st = R.run_heads(model, R.to_structure(rec, "cuda"), cache_enabled)
            total, *_ = R.compute_loss(st, R.to_targets(rec, conditions), loss_heads, alpha, kin_w)
            if train:
                (total / 64).backward()
                if (k + 1) % 8 == 0:
                    opt.step(); opt.zero_grad()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    ts = ts[3:]                                  # warm-up
    return dict(s_per_sample=float(np.mean(ts)), s_per_sample_median=float(np.median(ts)),
                read_s_per_crop=float(np.mean(t_read)), peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9,
                mean_crop_len=float(np.mean([int(store.get(k % len(store))["length"]) for k in range(n)])),
                block_norms=R.block_output_norms(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdcath-cache", required=True)
    ap.add_argument("--rcsb-cache", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R.provide_training_imports()
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}

    # 1. ESMFold train vs eval mode
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    fold = esm.pretrained.esmfold_v1().cuda().requires_grad_(False)
    fold.set_chunk_size(256)
    seq = R.read_seq(Path(R.DATASETS["mdcath"]["folders"]) / R.read_ids("investigation/data/mdcath/val.csv")[0])[:256]
    aa, mask, residx, _, _ = (x.cuda() if torch.is_tensor(x) else x for x in batch_encode_sequences([seq], 512, "G" * 25))
    outs = {}
    for name, mode, seed in [("eval", False, 0), ("eval2", False, 1), ("train_s0", True, 0), ("train_s1", True, 1), ("train_s0b", True, 0)]:
        fold.train(mode); torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = fold(aa, mask, residx, None, None)
        outs[name] = {k: st[k].float() for k in ("s_s", "s_z", "ptm_logits", "distogram_logits", "lm_logits")}
        outs[name]["lddt_last"] = st["lddt_head"][-1].float()
    def diff(a, b):
        return {k: float((outs[a][k] - outs[b][k]).abs().max() / outs[a][k].abs().max()) for k in outs[a]}
    res["esmfold_mode"] = {"eval_vs_eval_other_seed": diff("eval", "eval2"),
                           "eval_vs_train": diff("eval", "train_s0"),
                           "train_seed0_vs_seed1": diff("train_s0", "train_s1"),
                           "train_seed0_repeat": diff("train_s0", "train_s0b")}
    print(json.dumps(res["esmfold_mode"], indent=1), flush=True)
    del fold, outs; torch.cuda.empty_cache()

    # 2. heads
    md, rc = R.CropStore(args.mdcath_cache), R.CropStore(args.rcsb_cache)
    kin_w = torch.load("investigation/data/mdcath/kinetic_weights.pt", map_location="cpu").cuda()
    for arm, cache in (("A", True), ("B", False)):
        m1 = R.build_stage1(0).cuda(); m1.set_chunk_size(256)
        res[f"stage1_{arm}_train"] = time_heads(m1, rc, args.n, cache, ["dynamic_logits"], 0.99, 1, None)
        res[f"stage1_{arm}_val"] = time_heads(m1, rc, args.n, cache, ["dynamic_logits"], 0.99, 1, None, train=False)
        del m1
        m2 = R.build_stage2(0).cuda(); m2.set_chunk_size(256)
        res[f"stage2_{arm}_train"] = time_heads(m2, md, args.n, cache, R.STAGE2_FULL, 0.85, 5, kin_w)
        res[f"stage2_{arm}_val"] = time_heads(m2, md, args.n, cache, R.STAGE2_FULL, 0.85, 5, kin_w, train=False)
        del m2; torch.cuda.empty_cache()
        print(arm, json.dumps({k: v for k, v in res.items() if k.endswith(arm + "_train") or k.endswith(arm + "_val")}), flush=True)

    # 3. disk per crop
    for name, store in (("mdcath", md), ("rcsb", rc)):
        size = sum(p.stat().st_size for p in Path(store.dir).glob("shard_*.pt"))
        lens = np.array([int(r["end"]) - int(r["start"]) for r in store.rows])
        res[f"disk_{name}"] = dict(bytes_per_crop=size / len(store), mean_crop_len=float(lens.mean()),
                                   bytes_per_full_256_crop_est=size / float((lens ** 2).sum()) * 256 ** 2)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
