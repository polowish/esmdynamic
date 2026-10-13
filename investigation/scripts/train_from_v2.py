"""Fine-tune ESMDynamic's released V2 heads on mdCATH with working DCM blocks (plan 05).

    # option 1: train only what never trained (blocks + the occupancy head's single-state path)
    python investigation/scripts/train_from_v2.py --trainable at-init --arm fixed --seed 0 \
        --train-cache ~/scratch/esmdyn_cache/mdcath_train --val-cache ~/scratch/esmdyn_cache/mdcath_val \
        --out ~/scratch/esmdyn_retrain/runs/opt1_fixed_s0 --resume
    # option 2: every head parameter, with the fix and (control) without it
    python investigation/scripts/train_from_v2.py --trainable all --arm fixed ...
    python investigation/scripts/train_from_v2.py --trainable all --arm bug ...

Start: the released V2 heads (strict load). Their DCM blocks are at initialisation with
zero output layers, so at step 0 the model IS the published model. --arm sets the one thing
that decides whether the blocks can learn: torch.autocast(cache_enabled=...) (bug = True, as
train.py; fixed = False).

As the authors' fine-tuning (Methods; training/{train,loss,data_reader}.py): mdCATH train,
1,000 draws per epoch, domains with probability proportional to length, with replacement,
fresh 256-residue crop per draw (all windows are cached); Adam lr 1e-4; batch 64 as 1 x 64
accumulation (a final partial step per epoch, as train_one_epoch); bf16 autocast; their loss
(esmdynamic_loss over all six heads: focal alpha 0.85 gamma 2, weighted CE, MSEs; sums over
pairs, mean over heads); best-validation-loss checkpoint; early stopping, patience 10.
Ours (stated in plan 05): start from V2 with all six losses from step 0 (the epoch-10 delay
protected untrained confidence heads; V2's are trained); a FIXED validation set (every val
domain, one window each, every epoch) instead of 100 fresh random draws; optional linear
warm-up and gradient clipping (--warmup-steps, --clip; off by default); per-epoch monitoring
of the published protocol's metrics (pairs |i-j| >= 3, symmetrised probability).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from retrain_lib import (ALL_LOSSES, V2_WEIGHTS, WindowStore, block_output_norms, check_at_init,
                         compute_loss, draw_epoch, epoch_seed, load_v2, loader, paper_metrics,
                         provide_training_imports, run_heads, seed_all, set_trainable,
                         to_structure, to_targets)
from wandb_tracking import Tracker

BATCH = 64
ALPHA = 0.85


def validate(model, store, window_ids, cache_enabled, kin_w, workers):
    """Loss (all six heads) and published-protocol metrics on the fixed validation windows."""
    model.eval()
    losses, per = [], {h: [] for h in ALL_LOSSES}
    m = {k: [[] for _ in range(5)] for k in ("bal_acc", "auroc", "rmse")}
    with torch.no_grad():
        for rec in loader(store, window_ids, workers):
            st = run_heads(model, to_structure(rec, "cuda"), cache_enabled)
            total, ph, _, _ = compute_loss(st, to_targets(rec), ALL_LOSSES, ALPHA, kin_w)
            losses.append(float(total))
            for h in ALL_LOSSES:
                per[h].append(ph[h])
            for t, row in enumerate(paper_metrics(st, rec)):
                for k in m:
                    m[k][t].append(row[k])
    out = dict(val_loss=float(np.mean(losses)), **{f"val_loss_{h}": float(np.mean(v)) for h, v in per.items()})
    for k, per_t in m.items():
        for t, T in enumerate((320, 348, 379, 413, 450)):
            out[f"val_{k}_{T}K"] = float(np.nanmean(per_t[t]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trainable", choices=["at-init", "all"], required=True)
    ap.add_argument("--arm", choices=["fixed", "bug"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--init", default=V2_WEIGHTS)
    ap.add_argument("--kin-weights", default="investigation/data/mdcath/kinetic_weights.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-train", type=int, default=1000, help="draws per epoch")
    ap.add_argument("--n-val", type=int, default=None, help="first N validation windows (smoke tests)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--warmup-steps", type=int, default=0, help="linear lr warm-up (0 = off, as the authors)")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip (0 = off, as the authors)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-limit-h", type=float, default=None,
                    help="stop cleanly after an epoch when the next one would pass this wall time")
    ap.add_argument("--wandb", choices=["offline", "online", "off"], default="offline")
    ap.add_argument("--wandb-project", default="esmdynamic-retrain")
    ap.add_argument("--artifact-every", type=int, default=10)
    args = ap.parse_args()
    provide_training_imports()
    t_job = time.time()
    cache_enabled = args.arm == "bug"
    out = Path(os.path.expanduser(args.out))
    (out / "plans").mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    model = load_v2(args.init)
    check_at_init(model)                                  # V2's blocks etc. really are at init
    trainable = set_trainable(model, args.trainable)
    if args.arm == "bug" and args.trainable == "at-init":
        print("WARNING: bug arm + at-init: no trainable tensor can receive gradient", flush=True)
    model.cuda().set_chunk_size(256)
    params = [p for p in model.parameters() if p.requires_grad]
    frozen0 = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}
    print(f"trainable: {len(trainable)} tensors, {sum(p.numel() for p in params):,} parameters; "
          f"frozen: {len(frozen0)} tensors", flush=True)

    store_tr, store_va = (WindowStore(os.path.expanduser(args.train_cache)),
                          WindowStore(os.path.expanduser(args.val_cache)))
    val_ids = list(range(len(store_va)))[:args.n_val]
    kin_w = torch.load(args.kin_weights, map_location="cpu").cuda()

    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup_steps) if args.warmup_steps else 1.0)
    state = dict(epoch=0, global_step=0, best=float("inf"), best_epoch=None, bad=0, history=[],
                 done=False, stop_reason=None)
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        state = ck["state"]
        print(f"resumed after epoch {state['epoch']}", flush=True)
    if state["done"]:
        print("already finished:", state["stop_reason"]); return

    meta = dict(vars(args), stage=f"v2-{args.trainable}", cache_enabled=cache_enabled,
                n_trainable_tensors=len(trainable), n_trainable_params=sum(p.numel() for p in params),
                batch=BATCH, alpha=ALPHA, torch=torch.__version__, device=torch.cuda.get_device_name(0),
                n_train_windows=len(store_tr), n_val_windows=len(val_ids))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (out / "trainable.txt").write_text("\n".join(trainable) + "\n")
    tracker = Tracker(args.wandb, args.wandb_project, out, run_name=out.name, group=f"seed{args.seed}",
                      tags=[f"v2-{args.trainable}", f"arm-{args.arm}"], config=meta, model=model,
                      artifact_every=args.artifact_every)
    steplog = open(out / "steps.jsonl", "a")

    def save(path, extra=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"heads": model.heads.state_dict(), **(extra or {})}, tmp)
        tmp.rename(path)

    if state["epoch"] == 0 and not state["history"]:        # the starting point = released V2
        v0 = validate(model, store_va, val_ids, cache_enabled, kin_w, args.workers)
        state["history"].append(dict(epoch=0, **v0, block_norms=block_output_norms(model)))
        print(json.dumps(state["history"][0]), flush=True)
        tracker.epoch(dict(epoch=0, **v0), model, 0)

    while not state["done"]:
        e = state["epoch"]
        draws = draw_epoch(store_tr.lengths, args.n_train, epoch_seed(args.seed, e, "train"))
        ids = [store_tr.window_id(d, s) for d, s in draws]
        np.save(out / "plans" / f"epoch_{e + 1:03d}.npy", np.array(ids))

        model.train()
        opt.zero_grad()
        t0 = time.time()
        acc, acc_h, n_acc, ep, ep_h, ep_n = 0.0, dict.fromkeys(ALL_LOSSES, 0.0), 0, 0.0, dict.fromkeys(ALL_LOSSES, 0.0), 0
        for i, rec in enumerate(loader(store_tr, ids, args.workers)):
            st = run_heads(model, to_structure(rec, "cuda"), cache_enabled)
            total, per, _, _ = compute_loss(st, to_targets(rec), ALL_LOSSES, ALPHA, kin_w)
            (total / BATCH).backward()
            acc += float(total); n_acc += 1
            for h in ALL_LOSSES:
                acc_h[h] += per[h]
            if (i + 1) % BATCH == 0 or i + 1 == len(ids):
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip else float("inf")))
                opt.step(); sched.step(); opt.zero_grad()
                state["global_step"] += 1
                r = dict(epoch=e + 1, step=state["global_step"], samples=n_acc, loss=acc / n_acc,
                         grad_norm=gnorm, lr=sched.get_last_lr()[0],
                         **{f"loss_{h}": v / n_acc for h, v in acc_h.items()},
                         **{f"block_norm_{k}": v for k, v in block_output_norms(model).items()},
                         t=time.time() - t0)
                if not np.isfinite(r["loss"]) or not np.isfinite(gnorm):
                    raise FloatingPointError(f"non-finite loss/gradient at step {state['global_step']}: {r}")
                steplog.write(json.dumps(r) + "\n")
                tracker.step(r, state["global_step"])
                ep += acc; ep_n += n_acc
                for h in ALL_LOSSES:
                    ep_h[h] += acc_h[h]
                acc, acc_h, n_acc = 0.0, dict.fromkeys(ALL_LOSSES, 0.0), 0
        steplog.flush()
        t_train = time.time() - t0

        t1 = time.time()
        v = validate(model, store_va, val_ids, cache_enabled, kin_w, args.workers)
        changed = [n for n, p in model.named_parameters() if not p.requires_grad
                   and not torch.equal(p.detach().cpu(), frozen0[n])]
        assert not changed, f"frozen tensors changed: {changed[:5]}"
        state["epoch"] += 1
        hist = dict(epoch=state["epoch"], train_loss=ep / ep_n,
                    **{f"train_loss_{h}": x / ep_n for h, x in ep_h.items()}, **v,
                    **{f"block_norm_{k}": x for k, x in block_output_norms(model).items()},
                    t_train=t_train, t_val=time.time() - t1)
        if v["val_loss"] < state["best"]:
            state.update(best=v["val_loss"], best_epoch=state["epoch"], bad=0)
            save(out / "best.pt", {"epoch": state["epoch"], "val": v})
        else:
            state["bad"] += 1
        hist["best_epoch"] = state["best_epoch"]
        state["history"].append(hist)
        print(json.dumps(hist), flush=True)
        if state["bad"] >= args.patience:
            state.update(done=True, stop_reason=f"early stop: no improvement for {args.patience} epochs")
        elif state["epoch"] >= args.max_epochs:
            state.update(done=True, stop_reason=f"epoch cap {args.max_epochs} reached (the cap binds)")
        save(last, {"opt": opt.state_dict(), "sched": sched.state_dict(), "state": state})
        (out / "history.json").write_text(json.dumps(state["history"], indent=1))
        tracker.epoch(hist, model, state["global_step"])
        if args.time_limit_h and not state["done"] and \
                time.time() - t_job + 1.5 * (hist["t_train"] + hist["t_val"]) > args.time_limit_h * 3600:
            print("time limit: stopping after this epoch; rerun with --resume", flush=True)
            tracker.finish(final=False)
            return
    summary = dict(stop_reason=state["stop_reason"], best_epoch=state["best_epoch"], best_val=state["best"])
    (out / "DONE").write_text(json.dumps(summary))
    tracker.finish(summary)
    print("finished:", summary)


if __name__ == "__main__":
    main()
