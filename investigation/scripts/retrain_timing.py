"""Plan 05 step 3: measure before committing GPU time.

    python investigation/scripts/retrain_timing.py --train-cache <dir> --val-cache <dir> \
        --out investigation/results/15_timing.json

1. ESMFold on real mdCATH windows: seconds per window (256-residue and short), the dtypes of
   what the heads read, and how much train-mode dropout (train.py's model.train() reaches
   ESMFold) changes (a) the features and (b) the released model's dynamic-contact probabilities
   (largest change, fraction of pairs crossing 0.5), next to an eval-mode repeat (ESMFold cannot
   run without autocast: its language model is fp16).
2. Heads on cached windows: seconds per sample (forward + backward) for option 1 (at-init,
   fixed), option 2 (all, fixed) and its control (all, bug); a validation forward; peak GPU
   memory; seconds to read a window from the cache.
3. Disk per cached window.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import retrain_lib as R


def esmfold_checks(n_windows=12):
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences
    ids, lengths = R.split_table("mdcath", "val")
    folders = Path(R.DATASETS["mdcath"]["folders"])
    fold = esm.pretrained.esmfold_v1().cuda().requires_grad_(False)
    fold.set_chunk_size(256)
    heads = R.load_v2(device="cuda").eval()
    heads.set_chunk_size(256)

    def run(seq, mode, seed, autocast=True):
        aa, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        aa, mask, residx = aa.cuda(), mask.cuda(), residx.cuda()
        fold.train(mode == "train"); torch.manual_seed(seed)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            st = fold(aa, mask, residx, None, None)
        torch.cuda.synchronize(); dt = time.time() - t0
        st = {k: st[k] for k in ("s_s", "s_z", "lm_logits", "ptm_logits", "distogram_logits", "lddt_head")}
        st["residue_index"], st["mask"] = residx, mask
        with torch.no_grad():
            prob = R.run_heads(heads, st, True)["dynamic_prob"].float()
        return st, prob, dt

    long_ = [i for i in range(len(ids)) if lengths[i] >= 256][:n_windows // 2]
    short = [i for i in range(len(ids)) if lengths[i] < 200][:n_windows // 2]
    rows, times = [], {"256": [], "short": []}
    for i in long_ + short:
        seq = R.read_seq(folders / ids[i])[:256]
        ref, p_ref, dt = run(seq, "eval", 0)
        times["256" if len(seq) == 256 else "short"].append(dt)
        variants = {"eval_repeat": run(seq, "eval", 1), "train_seed0": run(seq, "train", 0),
                    "train_seed1": run(seq, "train", 1)}
        row = {"domain": ids[i], "L": len(seq)}
        for name, (st, p, _) in variants.items():
            feat = {k: float((st[k].float() - ref[k].float()).abs().max() / ref[k].float().abs().max())
                    for k in ("s_s", "s_z", "lddt_head", "ptm_logits", "distogram_logits")}
            row[name] = dict(feature_rel_maxdiff=feat,
                             prob_maxdiff=float((p - p_ref).abs().max()),
                             frac_pairs_flipped=float(((p > 0.5) != (p_ref > 0.5)).float().mean()))
        rows.append(row)
    dtypes = {k: str(v.dtype) for k, v in ref.items()}
    del fold, heads; torch.cuda.empty_cache()

    def agg(name, key):
        return float(np.median([r[name][key] for r in rows]))
    summary = {name: dict(prob_maxdiff_median=agg(name, "prob_maxdiff"),
                          frac_pairs_flipped_median=agg(name, "frac_pairs_flipped"),
                          s_z_rel_maxdiff_median=float(np.median([r[name]["feature_rel_maxdiff"]["s_z"] for r in rows])))
               for name in ("eval_repeat", "train_seed0", "train_seed1")}
    return dict(esmfold_s_per_window={k: float(np.mean(v)) for k, v in times.items() if v},
                dtypes=dtypes, dropout_summary=summary, per_window=rows)


def time_heads(store, n, trainable, cache_enabled, kin_w, train=True):
    model = R.load_v2(device="cuda")
    R.set_trainable(model, trainable)
    model.set_chunk_size(256)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4) if params else None
    model.train(train)
    ts, t_read = [], []
    torch.cuda.reset_peak_memory_stats()
    for k in range(n):
        t0 = time.time(); rec = store.get(k % len(store)); t_read.append(time.time() - t0)
        torch.cuda.synchronize(); t0 = time.time()
        with torch.set_grad_enabled(train):
            st = R.run_heads(model, R.to_structure(rec, "cuda"), cache_enabled)
            total, *_ = R.compute_loss(st, R.to_targets(rec), R.ALL_LOSSES, 0.85, kin_w)
            if train:
                (total / 64).backward()
                if opt and (k + 1) % 8 == 0:
                    opt.step(); opt.zero_grad()
        torch.cuda.synchronize(); ts.append(time.time() - t0)
    lens = [int(store.get(k % len(store))["length"]) for k in range(n)]
    res = dict(s_per_sample=float(np.mean(ts[3:])), read_s_per_window=float(np.mean(t_read)),
               peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, mean_window_len=float(np.mean(lens)),
               block_norms=R.block_output_norms(model))
    del model, opt; torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--long-cache", default=None, help="a cache of full 256-residue windows")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    R.provide_training_imports()
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    res["esmfold"] = esmfold_checks()
    print(json.dumps({k: v for k, v in res["esmfold"].items() if k != "per_window"}, indent=1), flush=True)

    tr, va = R.WindowStore(args.train_cache), R.WindowStore(args.val_cache)
    kin_w = torch.load("investigation/data/mdcath/kinetic_weights.pt", map_location="cpu").cuda()
    for name, trainable, cache in (("opt1_at-init_fixed", "at-init", False),
                                   ("opt2_all_fixed", "all", False), ("opt2_all_bug", "all", True)):
        res[f"heads_{name}"] = time_heads(tr, args.n, trainable, cache, kin_w)
        print(name, json.dumps(res[f"heads_{name}"]), flush=True)
    res["heads_val_forward"] = time_heads(va, args.n, "all", False, kin_w, train=False)
    if args.long_cache:
        lg = R.WindowStore(args.long_cache)
        for name, trainable, cache in (("opt1_at-init_fixed", "at-init", False), ("opt2_all_fixed", "all", False)):
            res[f"heads256_{name}"] = time_heads(lg, min(args.n, 3 * len(lg)), trainable, cache, kin_w)
            print("256:", name, json.dumps(res[f"heads256_{name}"]), flush=True)
        res["heads256_val_forward"] = time_heads(lg, min(args.n, 3 * len(lg)), "all", False, kin_w, train=False)

    stores = [("train", tr), ("val", va)] + ([("long", R.WindowStore(args.long_cache))] if args.long_cache else [])
    for name, store in stores:
        size = sum(p.stat().st_size for p in Path(store.dir).glob("shard_*.pt"))
        lens = np.array([int(r["end"]) - int(r["start"]) for r in store.rows])
        res[f"disk_{name}"] = dict(windows=len(store), bytes_per_window=size / len(store),
                                   mean_window_len=float(lens.mean()),
                                   bytes_per_256_window_est=size / float((lens ** 2).sum()) * 256 ** 2)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "esmfold"}, indent=2))


if __name__ == "__main__":
    main()
