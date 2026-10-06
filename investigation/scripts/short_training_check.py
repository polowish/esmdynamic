"""Check 7: does the fix let the DCM blocks train, and does that help? A short, matched run.

    python investigation/scripts/short_training_check.py --cache on  [--steps 300]
    python investigation/scripts/short_training_check.py --cache off [--steps 300]

Both runs start from the released V2 weights (whose DCM blocks are at initialisation) and
see EXACTLY the same batches: indices and crop windows are pre-drawn from a fixed seed with
the dataset's own length-proportional weights and 256-residue crops (as in the preprint).
The only difference is `torch.autocast(..., cache_enabled=...)`:

  --cache on   as train.py runs (the bug)
  --cache off  the fix

Losses are ESMDynamic's own (`training/loss.py`) on the three main heads, with targets built
by train.py's `build_outputs_and_targets_for_loss`; Adam, lr 1e-4, ESMFold frozen, bf16
autocast. The forward is the model's up to the end of the heads: its native-contact step
cannot run with gradients (see notes/02).

Logged: training loss; the Frobenius norm of every DCM block's zero-initialised output
layers (0 at the start; it can only move if gradient reaches the blocks); and, before and
after, the loss and dynamic-contact balanced accuracy on a fixed set of validation proteins.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ZERO_INIT = ("seq_attention.o_proj.weight", "mlp_seq.mlp.3.weight", "mlp_pair.mlp.3.weight",
             "sequence_to_pair.o_proj.weight", "pair_to_sequence.linear.weight",
             "tri_mul_in.linear_z.weight", "tri_mul_out.linear_z.weight",
             "tri_att_start.mha.linear_o.weight", "tri_att_end.mha.linear_o.weight")
LOSS_HEADS = ["dynamic_logits", "kinetic_logits", "frequency_pred"]


def block_output_norm(model):
    total = 0.0
    for n, p in model.heads.named_parameters():
        if ".blocks." in n and any(n.endswith(z) for z in ZERO_INIT):
            total += float(p.detach().float().norm()) ** 2
    return total ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache", choices=["on", "off"], required=True)
    ap.add_argument("--data", default="investigation/data/mdcath/dataset")
    ap.add_argument("--splits", default="investigation/data/mdcath")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"investigation/results/07_training_check_cache_{args.cache}.json")

    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences
    from esm.esmdynamic.training.data_reader import DynContactDataset
    from esm.esmdynamic.training import loss as loss_mod
    from esm.esmdynamic.training.train import (build_outputs_and_targets_for_loss,
                                                metrics_dynamic_batch)

    cache = args.cache == "on"
    torch.manual_seed(args.seed)
    model = esmdynamic().cuda()
    model.esmfold.requires_grad_(False)
    model.train()
    opt = torch.optim.Adam(model.heads.parameters(), lr=1e-4)
    kin_w = torch.load(Path(args.splits) / "kinetic_weights.pt", map_location="cpu").cuda()

    train_ids = [x.strip() for x in open(Path(args.splits) / "train.csv") if x.strip()]
    val_ids = [x.strip() for x in open(Path(args.splits) / "val.csv") if x.strip()]
    ds_train = DynContactDataset(args.data, train_ids, crop_length=256)
    ds_val = DynContactDataset(args.data, val_ids, crop_length=256)

    def load(ds, i, start):
        d = Path(ds.data_dir) / ds.identifiers[i]
        seq = ds._load_sequence(str(d / "consensus.fasta"))
        dyn = ds._load_dynamic(str(d / "dynamic_contacts.pt"))
        kin = ds._load_kinetics(str(d / "kinetics.pt"))
        freq = ds._load_frequency(str(d / "frequency.pt"))
        end = min(len(seq), start + 256)
        return (seq[start:end], dyn[:, start:end, start:end], kin[:, :, start:end, start:end],
                freq[:, start:end, start:end], end - start)

    # the same batches for both runs: length-proportional sampling and random crops, drawn
    # once from a dedicated generator before any model computation
    g = torch.Generator().manual_seed(args.seed)
    lengths = np.array([len(ds_train._load_sequence(str(Path(args.data) / i / "consensus.fasta")))
                        for i in train_ids], float)
    w = torch.tensor(lengths / lengths.sum())
    plan = []
    for _ in range(args.steps):
        idx = torch.multinomial(w, args.batch, replacement=True, generator=g).tolist()
        starts = [int(torch.randint(0, max(1, int(lengths[i]) - 256), (), generator=g))
                  if lengths[i] > 256 else 0 for i in idx]
        plan.append(list(zip(idx, starts)))
    val_plan = [(i, 0) for i in range(min(args.n_val, len(val_ids)))]

    def forward(batch, train):
        seqs, dyn, kin, freq, L = ds_train.custom_collate_fn(batch)
        aatype, mask, residx, _, _ = batch_encode_sequences(list(seqs), 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
            with torch.no_grad():
                st = model.esmfold(aatype, mask, residx, None, None)
            st["mask"] = mask
            with torch.set_grad_enabled(train):
                for h in model.heads.values():
                    st = h(st, num_recycles=None)
        outs, tgts = build_outputs_and_targets_for_loss(st, dyn, kin, freq, L, LOSS_HEADS,
                                                        "cuda", kin_w)
        kw = kin_w.to(outs["kinetic_logits"].dtype) if "kinetic_logits" in outs else None
        loss = loss_mod.esmdynamic_loss(outs, tgts, L.cuda(), active_heads=LOSS_HEADS,
                                        kin_class_weights=kw, alpha=0.85, gamma=2)
        return loss, outs, tgts, L

    def evaluate():
        model.eval()
        losses, ba = [], []
        for k in range(0, len(val_plan), args.batch):
            batch = [load(ds_val, i, s) for i, s in val_plan[k:k + args.batch]]
            with torch.no_grad():
                loss, outs, tgts, L = forward(batch, train=False)
            losses.append(float(loss))
            m = metrics_dynamic_batch(outs["dynamic_logits"].float(), tgts["dynamic_logits"].float(), L)
            ba.append(m["bal_acc"])
        model.train()
        return {"val_loss": float(np.mean(losses)), "val_dynamic_bal_acc": float(np.mean(ba))}

    log = {"cache_enabled": cache, "steps": args.steps, "batch": args.batch, "seed": args.seed,
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "first_batch_ids": [train_ids[i] for i, _ in plan[0]], "train": [],
           "before": evaluate()}
    log["before"]["block_output_norm"] = block_output_norm(model)
    print(f"cache={args.cache}  before: {log['before']}  first batch {log['first_batch_ids']}")
    t0 = time.time()
    for step, b in enumerate(plan, 1):
        batch = [load(ds_train, i, s) for i, s in b]
        loss, *_ = forward(batch, train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rec = {"step": step, "loss": float(loss), "block_output_norm": block_output_norm(model)}
        log["train"].append(rec)
        if step % 10 == 0 or step == 1:
            print(f"  step {step:>4}  loss {rec['loss']:.4f}  block-output norm "
                  f"{rec['block_output_norm']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
    log["after"] = evaluate()
    log["after"]["block_output_norm"] = block_output_norm(model)
    print(f"cache={args.cache}  after: {log['after']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
