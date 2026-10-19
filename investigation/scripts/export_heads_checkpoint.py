"""Export fine-tuned heads (train_from_v2.py best.pt) in the released weights' layout.

    python investigation/scripts/export_heads_checkpoint.py \
        ~/scratch/esmdyn_retrain/runs/v2ft-all-heads-fixed-s0/best.pt \
        ~/scratch/esmdyn_retrain/runs/v2ft-all-heads-fixed-s0/v2ft-all-heads-fixed-s0.pt

train_from_v2.py saves {"heads": model.heads.state_dict(), "epoch": ...} with keys like
`dynamic.seq_transition.0.weight`; the released esmdynamic.pt is a flat state dict with keys
`heads.dynamic.seq_transition.0.weight` (what esm.esmdynamic.pretrained._load_model, and so
dyn_models' esmdynamic adapter with `checkpoint: <path>.pt`, expects). This adds the prefix,
and refuses to write unless the key set and every shape and dtype match the released file.
Name the output so it does NOT start with "esmdynamic": dyn_models labels a checkpoint by its
file name, and one starting with the adapter's name would collapse into the released model's.
"""
import argparse
import os
from pathlib import Path

import torch

RELEASED = "~/scratch/github/esmdynamic/model_weights/hub/checkpoints/esmdynamic.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("best")
    ap.add_argument("out")
    ap.add_argument("--released", default=RELEASED)
    args = ap.parse_args()
    out = Path(os.path.expanduser(args.out))
    if out.name.lower().startswith("esmdynamic"):
        raise SystemExit(f"{out.name}: must not start with 'esmdynamic' (it would share the released label)")
    ck = torch.load(os.path.expanduser(args.best), map_location="cpu", weights_only=False)
    sd = {f"heads.{k}": v for k, v in ck["heads"].items()}
    ref = torch.load(os.path.expanduser(args.released), map_location="cpu", weights_only=False)
    ref = {k: v for k, v in ref.items() if k.startswith("heads.")}
    if set(sd) != set(ref):
        raise SystemExit(f"key sets differ: only ours {sorted(set(sd) - set(ref))[:5]}, "
                         f"only released {sorted(set(ref) - set(sd))[:5]}")
    bad = [k for k in sd if sd[k].shape != ref[k].shape or sd[k].dtype != ref[k].dtype]
    if bad:
        raise SystemExit(f"shape/dtype mismatch: {bad[:5]}")
    changed = sum(not torch.equal(sd[k], ref[k]) for k in sd)
    tmp = out.with_suffix(".tmp")
    torch.save(sd, tmp)
    tmp.rename(out)
    print(f"{out}: {len(sd)} tensors (epoch {ck.get('epoch')}); {changed} differ from the released weights")


if __name__ == "__main__":
    main()
