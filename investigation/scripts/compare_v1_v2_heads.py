"""Is V2's dynamic head a continuation of the V1 weights? (note 16)

    python investigation/scripts/compare_v1_v2_heads.py <esmdynamic_V1.pt> <esmdynamic_model_weights_V2.pt>

V1 (Data Bank, 2025-06) is the single-head model: top-level seq_transition / pair_transition /
dynamic_module / prediction_layer (LayerNorm + Linear 128 -> 1). V2 (2026-04) has three heads
under heads.{dynamic,kinetic,frequency}. For every V1 tensor with a same-shaped counterpart in
heads.dynamic, prints the relative difference and the cosine similarity. The baseline is two
unrelated V2 heads (dynamic vs kinetic) on the same tensors: independently initialised and
trained, they sit at cosine ~0. A high cosine to V1 means V2's dynamic head started from V1.
"""
import sys

import torch


def main(v1_path, v2_path):
    v1 = torch.load(v1_path, map_location="cpu", weights_only=False)
    v2 = torch.load(v2_path, map_location="cpu", weights_only=False)
    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))  # noqa: E731
    print(f"{'V1 tensor':55s} {'rel diff':>9s} {'cosine':>7s}")
    for k, a in v1.items():
        k2 = "heads.dynamic." + k
        if k2 not in v2 or v2[k2].shape != a.shape or a.dim() == 0:
            continue
        a, b = a.float(), v2[k2].float()
        rel = float((a - b).norm() / a.norm()) if a.norm() > 0 else float("inf")
        print(f"{k:55s} {rel:9.3f} {cos(a, b):7.3f}")
    print("baseline, V2 dynamic vs V2 kinetic:")
    for k in ("seq_transition.1.weight", "pair_transition.1.weight", "seq_transition.2.weight"):
        print(f"  {k:53s} {'':9s} {cos(v2['heads.dynamic.' + k].float(), v2['heads.kinetic.' + k].float()):7.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
