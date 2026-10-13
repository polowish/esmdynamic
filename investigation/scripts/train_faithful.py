"""train.py's recipe over cached ESMFold features, arms A (as released) and B (fixed) (plan 04).

    # stage 1: RCSB pre-training, dynamic head only
    python investigation/scripts/train_faithful.py --stage pretrain --arm A --seed 0 \
        --train-cache ~/scratch/esmdyn_cache/rcsb_train --val-cache ~/scratch/esmdyn_cache/rcsb_val \
        --out ~/scratch/esmdyn_retrain/runs/s0_A_pretrain --resume
    # stage 2: mdCATH fine-tuning, all heads, from stage 1's best checkpoint of the same arm
    python investigation/scripts/train_faithful.py --stage finetune --arm A --seed 0 \
        --train-cache ~/scratch/esmdyn_cache/mdcath_train --val-cache ~/scratch/esmdyn_cache/mdcath_val \
        --init ~/scratch/esmdyn_retrain/runs/s0_A_pretrain/best.pt \
        --out ~/scratch/esmdyn_retrain/runs/s0_A_finetune --resume

The two arms differ ONLY in torch.autocast(cache_enabled=...): A = True (train.py as shipped,
the DCM blocks get no gradient), B = False (the fix). Initialisation, the sample sequence and
the crops are functions of --seed alone (logged per epoch under <out>/plans/).

As the authors (Methods; training/{train,loss,data_reader}.py):
  Adam lr 1e-4 over the head parameters; batch 64 (here 1 x 64 accumulation, loss / 64, a final
  partial step as train_one_epoch makes); bf16 autocast; ESMFold frozen (cached);
  esmdynamic_loss (mean over active heads; per-head sums over pairs) with train.py's targets;
  per epoch 10,000 / 1,000 (pretrain) or 1,000 / 100 (finetune) draws, cluster with probability
  proportional to length, with replacement, fresh draws each epoch for train and validation;
  focal alpha 0.99 (pretrain) / 0.85 (finetune), gamma 2; keep the best-validation-loss heads;
  early stopping after --patience epochs without improvement.
Stage 2: epochs 1..--aux-heads-after with dynamic/kinetic/frequency losses, then the
confidence and residual losses added. We run the switch as train.py would run it, a second
invocation: carry the heads over, fresh Adam, fresh best-loss and patience. (Not stated by
the authors; noted in note 15.)
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from wandb_tracking import Tracker
from retrain_lib import (STAGE2_FULL, STAGE2_WARM, CropStore, block_output_norms, build_stage1,
                         build_stage2, compute_loss, draw_plan, handover, loader, plan_seed,
                         provide_training_imports, run_heads, to_structure, to_targets)

STAGES = {
    "pretrain": dict(alpha=0.99, n_train=10_000, n_val=1_000, max_epochs=300, conditions=1),
    "finetune": dict(alpha=0.85, n_train=1_000, n_val=100, max_epochs=200, conditions=5),
}
BATCH = 64


def cluster_weights(store):
    """Per-cluster sampling weight = length (RCSB train/val_weights.pt are exactly the lengths;
    mdCATH: DynContactDataset's fallback). Taken from the pool so it matches the cache."""
    n = int(store.cluster_index.max()) + 1
    w = np.zeros(n)
    for r in store.rows:
        w[int(r["cluster_index"])] = float(r["length_full"])
    assert (w > 0).all()
    return w


def phase_schedule(stage, aux_after):
    if stage == "pretrain":
        return [("main", ["dynamic_logits"])]
    return [("warm", STAGE2_WARM), ("full", STAGE2_FULL)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-cache", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--init", default=None, help="finetune: stage-1 best.pt of the same arm")
    ap.add_argument("--kin-weights", default="investigation/data/mdcath/kinetic_weights.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--aux-heads-after", type=int, default=10)
    ap.add_argument("--n-train", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--n-val", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wandb", choices=["offline", "online", "off"], default="offline",
                    help="Weights & Biases tracking (compute nodes: offline, then sync)")
    ap.add_argument("--wandb-project", default="esmdynamic-retrain")
    ap.add_argument("--artifact-every", type=int, default=10,
                    help="log best/last checkpoints as W&B artifacts every N epochs (and at the end)")
    ap.add_argument("--time-limit-h", type=float, default=None,
                    help="stop cleanly (after an epoch) when this much wall time is used")
    args = ap.parse_args()
    provide_training_imports()
    from esm.esmdynamic.training.train import metrics_dynamic_batch

    cfg = dict(STAGES[args.stage])
    cfg["n_train"] = args.n_train or cfg["n_train"]
    cfg["n_val"] = args.n_val or cfg["n_val"]
    cfg["max_epochs"] = args.max_epochs or cfg["max_epochs"]
    cache_enabled = args.arm == "A"
    out = Path(os.path.expanduser(args.out))
    (out / "plans").mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    t_job = time.time()

    store_tr, store_va = CropStore(os.path.expanduser(args.train_cache)), CropStore(os.path.expanduser(args.val_cache))
    w_tr, w_va = cluster_weights(store_tr), cluster_weights(store_va)
    kin_w = (torch.load(args.kin_weights, map_location="cpu").to(dev)
             if args.stage == "finetune" else None)

    # ---- model (identical for both arms at a given seed)
    if args.stage == "pretrain":
        model = build_stage1(args.seed)
        copied = []
    else:
        model = build_stage2(args.seed)
        assert args.init, "--init (stage-1 best.pt) is required for finetune"
        sd1 = torch.load(os.path.expanduser(args.init), map_location="cpu")["heads"]
        copied = handover({k[len("dynamic."):]: v for k, v in sd1.items() if k.startswith("dynamic.")},
                          model)
    model.to(dev)
    model.set_chunk_size(256)
    phases = phase_schedule(args.stage, args.aux_heads_after)

    state = dict(epoch=0, global_step=0, phase=0, best=float("inf"), best_epoch=None, bad=0, history=[],
                 done=False, stop_reason=None)
    opt = torch.optim.Adam(model.heads.parameters(), lr=1e-4)
    last = out / "last.pt"
    if args.resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.heads.load_state_dict(ck["heads"])
        opt.load_state_dict(ck["opt"])
        state = ck["state"]
        print(f"resumed at epoch {state['epoch']} (phase {phases[state['phase']][0]})", flush=True)
    if state["done"]:
        print("already finished:", state["stop_reason"])
        return

    meta = dict(vars(args), **cfg, cache_enabled=cache_enabled, handover_keys=len(copied),
                torch=torch.__version__, device=torch.cuda.get_device_name(0), batch=BATCH)
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    steplog = open(out / "steps.jsonl", "a")
    tracker = Tracker(args.wandb, args.wandb_project, out, run_name=out.name, group=f"seed{args.seed}",
                      tags=[args.stage, f"arm{args.arm}"], config=meta, model=model,
                      artifact_every=args.artifact_every)

    def save(path, extra=None):
        tmp = Path(str(path) + ".tmp")
        torch.save({"heads": model.heads.state_dict(), **(extra or {})}, tmp)
        tmp.rename(path)

    while not state["done"]:
        phase_name, loss_heads = phases[state["phase"]]
        e = state["epoch"]                        # 0-based index of the epoch about to run
        plan_tr = draw_plan(store_tr.cluster_index, w_tr, cfg["n_train"], plan_seed(args.seed, args.stage, e, "train"))
        plan_va = draw_plan(store_va.cluster_index, w_va, cfg["n_val"], plan_seed(args.seed, args.stage, e, "val"))
        np.save(out / "plans" / f"epoch_{e + 1:03d}_train.npy", plan_tr)
        np.save(out / "plans" / f"epoch_{e + 1:03d}_val.npy", plan_va)

        # ---- train (train_one_epoch)
        model.train()
        opt.zero_grad()
        t0 = time.time()
        acc_loss, acc_heads, n_acc, step = 0.0, {h: 0.0 for h in loss_heads}, 0, 0
        ep_loss, ep_heads, ep_n = 0.0, {h: 0.0 for h in loss_heads}, 0
        n = len(plan_tr)
        for i, rec in enumerate(loader(store_tr, plan_tr, args.workers)):
            st = run_heads(model, to_structure(rec, dev), cache_enabled)
            total, per, _, _ = compute_loss(st, to_targets(rec, cfg["conditions"]), loss_heads,
                                            cfg["alpha"], kin_w)
            (total / BATCH).backward()
            acc_loss += float(total); n_acc += 1
            for h in loss_heads:
                acc_heads[h] += per[h]
            if (i + 1) % BATCH == 0 or i + 1 == n:
                opt.step()
                opt.zero_grad()
                step += 1
                state["global_step"] += 1
                rec_log = dict(epoch=e + 1, phase=phase_name, step=step, samples=n_acc,
                               loss=acc_loss / n_acc, **{f"loss_{h}": v / n_acc for h, v in acc_heads.items()},
                               **{f"block_norm_{k}": v for k, v in block_output_norms(model).items()},
                               t=time.time() - t0)
                steplog.write(json.dumps(rec_log) + "\n")
                tracker.step(rec_log, state["global_step"])
                ep_loss += acc_loss; ep_n += n_acc
                for h in loss_heads:
                    ep_heads[h] += acc_heads[h]
                acc_loss, acc_heads, n_acc = 0.0, {h: 0.0 for h in loss_heads}, 0
        steplog.flush()
        t_train = time.time() - t0

        # ---- validate (compute_validation)
        model.eval()
        t1 = time.time()
        v_loss, v_heads, ba = [], {h: [] for h in loss_heads}, []
        with torch.no_grad():
            for rec in loader(store_va, plan_va, args.workers):
                st = run_heads(model, to_structure(rec, dev), cache_enabled)
                total, per, outs, tgts = compute_loss(st, to_targets(rec, cfg["conditions"]),
                                                      loss_heads, cfg["alpha"], kin_w)
                v_loss.append(float(total))
                for h in loss_heads:
                    v_heads[h].append(per[h])
                ba.append(metrics_dynamic_batch(outs["dynamic_logits"].float(),
                                                tgts["dynamic_logits"].float(),
                                                torch.tensor([int(rec["length"])]))["bal_acc"])
        val_loss = float(np.mean(v_loss))
        norms = block_output_norms(model)
        state["epoch"] += 1
        hist = dict(epoch=state["epoch"], phase=phase_name, train_loss=ep_loss / ep_n,
                    **{f"train_loss_{h}": v / ep_n for h, v in ep_heads.items()},
                    val_loss=val_loss, **{f"val_loss_{h}": float(np.mean(v)) for h, v in v_heads.items()},
                    val_dynamic_bal_acc=float(np.mean(ba)),
                    **{f"block_norm_{k}": v for k, v in norms.items()},
                    t_train=t_train, t_val=time.time() - t1)
        print(json.dumps(hist), flush=True)

        if val_loss < state["best"]:
            state["best"], state["best_epoch"], state["bad"] = val_loss, state["epoch"], 0
            if phase_name != "warm":
                save(out / "best.pt", {"epoch": state["epoch"], "val_loss": val_loss})
        else:
            state["bad"] += 1
        hist["best_epoch"] = state["best_epoch"]
        state["history"].append(hist)

        # ---- phase change / stopping
        if phase_name == "warm" and state["epoch"] >= args.aux_heads_after:
            save(out / "warm_last.pt", {"epoch": state["epoch"]})
            state.update(phase=state["phase"] + 1, best=float("inf"), best_epoch=None, bad=0)
            opt = torch.optim.Adam(model.heads.parameters(), lr=1e-4)   # a fresh train.py run
        elif phase_name != "warm" and state["bad"] >= args.patience:
            state.update(done=True, stop_reason=f"early stop: no improvement for {args.patience} epochs")
        if state["epoch"] >= cfg["max_epochs"] and not state["done"]:
            state.update(done=True, stop_reason=f"epoch cap {cfg['max_epochs']} reached (cap binds)")
        save(last, {"opt": opt.state_dict(), "state": state})
        (out / "history.json").write_text(json.dumps(state["history"], indent=1))
        tracker.epoch(hist, model, state["global_step"])
        if args.time_limit_h and not state["done"]:
            if time.time() - t_job + 1.5 * (hist["t_train"] + hist["t_val"]) > args.time_limit_h * 3600:
                print("time limit: stopping after this epoch; resume with --resume", flush=True)
                tracker.finish(final=False)
                return
    print("finished:", state["stop_reason"], "best epoch", state["best_epoch"], "val", state["best"])
    summary = dict(stop_reason=state["stop_reason"], best_epoch=state["best_epoch"], best_val=state["best"])
    (out / "DONE").write_text(json.dumps(summary))
    tracker.finish(summary)


if __name__ == "__main__":
    main()
