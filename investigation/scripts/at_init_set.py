"""Which released V2 head tensors are still at initialisation, and so trainable in option 1?

    python investigation/scripts/at_init_set.py [--weights PATH]

Option 1 (plan 05) trains exactly the tensors that never moved because the dead DCM blocks cut
off their gradient, and freezes everything the authors actually trained. The set, by name:
  - every tensor of every head's dynamic_module.blocks (480; all at init, audit + EVIDENCE E1);
  - frequency.seq_transition.* and frequency.dynamic_module.recycle_s_norm.* (8): the
    occupancy head's single-state path, which reaches its loss only through the blocks.
This script checks that set against the weights with STRICT tests where they exist (LayerNorm
weight == 1 and bias == 0; zero-initialised output layers == 0) and prints the audit's two
other flags (kinetic.confidence_head.3, frequency.residual_head.3) for inspection.
"""
import argparse
import math
import os

import torch

V2 = os.path.expanduser("~/scratch/github/esmdynamic/model_weights/hub/checkpoints/esmdynamic.pt")


def at_init_names(keys):
    return sorted(k for k in keys if ".dynamic_module.blocks." in k
                  or k.startswith("heads.frequency.seq_transition.")
                  or k.startswith("heads.frequency.dynamic_module.recycle_s_norm."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=V2)
    sd = torch.load(ap.parse_args().weights, map_location="cpu")
    names = at_init_names(sd)
    print(f"at-init set: {len(names)} tensors, {sum(sd[k].numel() for k in names):,} parameters "
          f"(of {sum(v.numel() for k, v in sd.items() if k.startswith('heads.')):,} head parameters)")
    for k in ("heads.frequency.seq_transition.0.weight", "heads.frequency.seq_transition.0.bias",
              "heads.frequency.dynamic_module.recycle_s_norm.weight",
              "heads.frequency.dynamic_module.recycle_s_norm.bias",
              "heads.dynamic.dynamic_module.recycle_s_norm.weight"):
        t = sd[k].float()
        print(f"  {k}: all ones {torch.equal(t, torch.ones_like(t))}, all zeros {not t.any().item()}")
    for k in ("heads.kinetic.confidence_head.3.weight", "heads.kinetic.confidence_head.3.bias",
              "heads.frequency.residual_head.3.weight", "heads.frequency.residual_head.3.bias"):
        t = sd[k].float()
        fan_in = sd[k.replace("bias", "weight")].shape[1]
        print(f"  {k}: shape {tuple(t.shape)} max|v| {t.abs().max():.4f} init bound "
              f"1/sqrt({fan_in}) = {1 / math.sqrt(fan_in):.4f} values {t.flatten()[:5].tolist()}")


if __name__ == "__main__":
    main()
