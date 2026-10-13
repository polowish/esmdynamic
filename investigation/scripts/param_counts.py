"""Per head of the released V2 weights: parameters the authors trained vs still at initialisation.

    python investigation/scripts/param_counts.py [--out investigation/results/15_param_counts.json]

"Untrained" is the set retrain_lib.at_init_names defines and check_at_init verifies on the
weights: every DCM block tensor (all three heads) plus the occupancy head's single-state path
(seq_transition, recycle_s_norm), which reaches its loss only through the blocks. Everything
else in the heads trained. Totals match SI Table 5 (95,894,843 trainable head parameters).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import retrain_lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="investigation/results/15_param_counts.json")
    args = ap.parse_args()
    m = R.load_v2()
    R.check_at_init(m)
    params = dict(m.named_parameters())
    untrained = set(R.at_init_names(params))
    out = {}
    for head in ("dynamic", "kinetic", "frequency"):
        names = [n for n in params if n.startswith(f"heads.{head}.")]
        u = sum(params[n].numel() for n in names if n in untrained)
        t = sum(params[n].numel() for n in names if n not in untrained)
        out[head] = dict(total=t + u, trained=t, untrained=u, untrained_tensors=sum(n in untrained for n in names))
    out["all_heads"] = {k: sum(out[h][k] for h in ("dynamic", "kinetic", "frequency"))
                        for k in ("total", "trained", "untrained", "untrained_tensors")}
    assert out["all_heads"]["total"] == 95_894_843, out["all_heads"]
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
