"""Verify an ESMFold window cache before training reads it (plan 05).

    # every shard, every window (CPU, workers in parallel)
    python investigation/scripts/verify_cache.py --cache ~/scratch/esmdyn_cache/mdcath_train integrity -j 16
    # a stratified sample recomputed with live ESMFold and compared (GPU)
    python investigation/scripts/verify_cache.py --cache ~/scratch/esmdyn_cache/mdcath_train live -n 48

integrity  meta.json agrees with windows.csv; every shard exists, loads, holds exactly the
           windows windows.csv assigns it, in order; every feature has the window's length on
           its residue axes, one trailing shape and one dtype across the whole cache, and only
           finite values; labels.pt has every domain at full length; no .tmp left behind.
live       recomputes a sample (equal per array task, so every GPU that wrote shards is
           covered) on this GPU, as cache_esmfold_features.py did, and reports per feature
           whether it is bit-identical and the largest difference relative to the feature's
           scale, plus the largest difference in the released V2 heads' dynamic probabilities.
           Run on each GPU type that wrote shards: bf16 kernels differ across architectures,
           so cross-GPU differences are the floor a cached-vs-live difference is judged against.

Writes a json report (--report) and exits non-zero on any failure.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

FEATS = ("s_s", "s_z", "lddt_last", "lm_logits", "ptm_logits", "distogram_logits")
PAIR = {"s_z", "ptm_logits", "distogram_logits"}            # (L, L, ...) features


def _rows(cache):
    return list(csv.DictReader(open(cache / "windows.csv")))


def _check_shard(args):
    path, expected = args                                # expected: [(window_id, L), ...]
    out = {"shard": path.name, "errors": [], "trailing": Counter(), "dtypes": Counter(),
           "n": 0, "bytes": path.stat().st_size if path.exists() else 0}
    if not path.exists():
        out["errors"].append("missing")
        return out
    try:
        recs = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    except Exception as e:                               # noqa: BLE001
        out["errors"].append(f"load failed: {type(e).__name__}: {e}")
        return out
    if len(recs) != len(expected):
        out["errors"].append(f"{len(recs)} windows, expected {len(expected)}")
    for rec, (wid, L) in zip(recs, expected):
        out["n"] += 1
        if rec.get("window_id") != wid:
            out["errors"].append(f"window_id {rec.get('window_id')} at the slot of {wid}")
            continue
        for key in ("residue_index", "mask"):
            if tuple(rec[key].shape) != (L,):
                out["errors"].append(f"{wid} {key} shape {tuple(rec[key].shape)}, L={L}")
        if rec["mask"].numel() and not bool(rec["mask"].all()):
            out["errors"].append(f"{wid} mask has padding")
        for key in FEATS:
            v = rec[key]
            lead = 2 if key in PAIR else 1
            if tuple(v.shape[:lead]) != (L,) * lead:
                out["errors"].append(f"{wid} {key} shape {tuple(v.shape)}, L={L}")
            out["trailing"][f"{key}{tuple(v.shape[lead:])}"] += 1
            out["dtypes"][f"{key}:{v.dtype}"] += 1
            if not bool(torch.isfinite(v).all()):
                out["errors"].append(f"{wid} {key} non-finite")
    return out


def integrity(cache, jobs):
    from multiprocessing import Pool

    meta = json.loads((cache / "meta.json").read_text())
    rows = _rows(cache)
    errors = []
    if len(rows) != meta["n_windows"]:
        errors.append(f"windows.csv has {len(rows)} rows, meta says {meta['n_windows']}")
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[int(r["shard"])].append((int(r["window_id"]), int(r["end"]) - int(r["start"])))
    if sorted(by_shard) != list(range(meta["n_shards"])):
        errors.append("windows.csv shard numbers are not 0..n_shards-1")
    stray = sorted(p.name for p in cache.glob("*.tmp"))
    if stray:
        errors.append(f"leftover temp files: {stray[:5]}")
    extra = sorted(p.name for p in cache.glob("shard_*.pt")
                   if int(p.stem.split("_")[1]) >= meta["n_shards"])
    if extra:
        errors.append(f"shards beyond n_shards: {extra[:5]}")

    t0 = time.time()
    work = [(cache / f"shard_{k:05d}.pt", by_shard[k]) for k in sorted(by_shard)]
    trailing, dtypes, n, nbytes = Counter(), Counter(), 0, 0
    with Pool(jobs) as pool:
        for i, res in enumerate(pool.imap_unordered(_check_shard, work, chunksize=4), 1):
            errors += [f"{res['shard']}: {e}" for e in res["errors"][:20]]
            trailing.update(res["trailing"]); dtypes.update(res["dtypes"])
            n += res["n"]; nbytes += res["bytes"]
            if i % 100 == 0:
                print(f"  {i}/{len(work)} shards, {time.time() - t0:.0f} s", flush=True)
    for key in FEATS:                                     # one shape and one dtype per feature
        shapes = {s for s in trailing if s.split("(")[0] == key}
        kinds = {d for d in dtypes if d.split(":")[0] == key}
        if len(shapes) != 1:
            errors.append(f"{key}: {len(shapes)} trailing shapes {sorted(shapes)[:4]}")
        if len(kinds) != 1:
            errors.append(f"{key}: {len(kinds)} dtypes {sorted(kinds)}")

    labels = torch.load(cache / "labels.pt", mmap=True, weights_only=True)
    full = {r["domain"]: int(r["length_full"]) for r in rows}
    missing = [d for d in full if d not in labels]
    if missing:
        errors.append(f"labels.pt lacks {len(missing)} domains, e.g. {missing[:3]}")
    for d, L in full.items():
        if d in labels:
            lab = labels[d]
            if lab["dyn"].shape[-1] != L or lab["kin"].shape[-1] != L or lab["freq"].shape[-1] != L:
                errors.append(f"labels {d}: shapes {[tuple(lab[k].shape) for k in ('dyn', 'kin', 'freq')]}, L={L}")

    tasks = {p.name: json.loads(p.read_text()) for p in sorted(cache.glob("task_*.json"))}
    return {"mode": "integrity", "cache": str(cache), "meta": meta, "windows_checked": n,
            "shards": len(work), "bytes": nbytes, "seconds": round(time.time() - t0),
            "trailing_shapes": sorted(trailing), "dtypes": sorted(dtypes),
            "domains": len(full), "writers": {k: v.get("device") for k, v in tasks.items()},
            "errors": errors, "ok": not errors}


def live(cache, n, seed, tol):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import esm
    from esm.esmfold.v1.misc import batch_encode_sequences

    import retrain_lib as R

    meta = json.loads((cache / "meta.json").read_text())
    tasks = {p.name: json.loads(p.read_text()) for p in sorted(cache.glob("task_*.json"))}
    n_tasks = max(len(tasks), 1)
    writer = {int(v["task"]): v.get("device", "?") for v in tasks.values()}
    store = R.WindowStore(cache)
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for wid in range(len(store)):
        by_task[int(store.shard[wid]) % n_tasks].append(wid)
    sample = [w for t in sorted(by_task) for w in rng.sample(by_task[t], min(len(by_task[t]), -(-n // n_tasks)))]

    fold = esm.pretrained.esmfold_v1().cuda().requires_grad_(False)
    fold.set_chunk_size(256)
    fold.train(meta["esmfold_mode"] == "train")
    heads = R.load_v2(device="cuda").eval()
    heads.set_chunk_size(256)
    here = torch.cuda.get_device_name(0)
    folders = Path(R.DATASETS[meta["dataset"]]["folders"])

    per = []
    for wid in sample:
        rec = store.get(wid)
        seq = R.read_seq(folders / rec["domain"])[rec["start"]:rec["end"]]
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        if meta["esmfold_mode"] == "train":
            torch.manual_seed(meta["seed"] * 1_000_003 + wid)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = fold(aatype.cuda(), mask.cuda(), residx.cuda(), None, None)
        lv = {"s_s": st["s_s"][0], "s_z": st["s_z"][0], "lddt_last": st["lddt_head"][-1][0],
              "lm_logits": st["lm_logits"][0], "ptm_logits": st["ptm_logits"][0],
              "distogram_logits": st["distogram_logits"][0]}
        row = {"window_id": wid, "L": rec["length"], "task": int(store.shard[wid]) % n_tasks,
               "written_on": writer.get(int(store.shard[wid]) % n_tasks, "?")}
        for k in FEATS:
            a, b = lv[k].float(), rec[k].cuda().float()
            row[f"{k}_identical"] = bool(torch.equal(lv[k], rec[k].cuda()))
            row[f"{k}_rel"] = float((a - b).abs().max() / b.abs().max().clamp_min(1e-12))
        live_struct = {"s_s": st["s_s"], "s_z": st["s_z"], "lm_logits": st["lm_logits"],
                       "ptm_logits": st["ptm_logits"], "distogram_logits": st["distogram_logits"],
                       "lddt_head": st["lddt_head"], "residue_index": residx.cuda(), "mask": mask.cuda()}
        with torch.no_grad():
            pl = R.run_heads(heads, live_struct, True)["dynamic_prob"].float()
            pc = R.run_heads(heads, R.to_structure(rec, "cuda"), True)["dynamic_prob"].float()
        row["prob_maxdiff"] = float((pl - pc).abs().max())
        per.append(row)
        print(f"  w{wid:6d} L={row['L']:3d} task {row['task']} ({row['written_on'][:16]}): "
              f"identical={all(row[f'{k}_identical'] for k in FEATS)}  "
              f"s_s rel {row['s_s_rel']:.1e}  s_z rel {row['s_z_rel']:.1e}  prob {row['prob_maxdiff']:.1e}",
              flush=True)

    groups = defaultdict(list)
    for r in per:
        groups[r["written_on"]].append(r)
    summary = {}
    for dev, rs in groups.items():
        summary[dev] = {"windows": len(rs),
                        "all_features_identical": sum(all(r[f"{k}_identical"] for k in FEATS) for r in rs),
                        "max_prob_diff": max(r["prob_maxdiff"] for r in rs),
                        **{f"max_{k}_rel": max(r[f"{k}_rel"] for r in rs) for k in FEATS}}
    worst = max(r["prob_maxdiff"] for r in per)
    errors = [] if worst < tol else [f"max dynamic-prob difference {worst:.2e} >= {tol}"]
    return {"mode": "live", "cache": str(cache), "checked_on": here, "sample": len(per),
            "seed": seed, "tolerance_prob": tol, "by_writer": summary, "windows": per,
            "errors": errors, "ok": not errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", required=True)
    ap.add_argument("--report", default=None, help="json report path (default: print only)")
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("integrity")
    a.add_argument("-j", "--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    b = sub.add_parser("live")
    b.add_argument("-n", type=int, default=48, help="windows, split evenly across array tasks")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tol", type=float, default=1e-3, help="max |dynamic prob| difference")
    args = ap.parse_args()

    cache = Path(os.path.expanduser(args.cache))
    rep = integrity(cache, args.jobs) if args.mode == "integrity" else live(cache, args.n, args.seed, args.tol)
    brief = {k: v for k, v in rep.items() if k != "windows"}
    print(json.dumps(brief, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
