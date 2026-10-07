"""Does excluding near-diagonal residue pairs explain SI Table 6's high-temperature numbers?

    python investigation/scripts/eval_pairset.py --dataset mdcath
    python investigation/scripts/eval_pairset.py --dataset atlas

Note 12: at 348-450 K our precision, recall, F1 and RMSE match SI Tables 6-7, but balanced
accuracy and AUROC fall far short of matching (450 K: published 0.542 vs ours 0.753). The
published numbers imply a specificity of 0.09 at 450 K against our 0.51.

Hypothesis: the paper scored only pairs with |i - j| >= k for some k >= 1, while we score all
L x L pairs. Pairs close in sequence (a residue with itself, chain neighbours ~3.8 A apart)
are always in contact, hence never dynamic, and the model gets them right: easy true
negatives. At 450 K ~90% of pairs are dynamic, so these easy negatives are a large share of
all negatives and dropping them collapses specificity (and AUROC) while barely moving
precision and recall, whose pairs are rarely sequence neighbours. At 320 K negatives are
plentiful and mostly distant, so the exclusion barely matters there.

Test: score the same predictions on pair sets |i - j| >= k, k in K (k = 0 = all pairs, our
note-12 scoring), full length, both thresholded scores (symmetrised probability; raw
logits). Per-protein mean +- se, as note 12. If some k reproduces balanced accuracy and
AUROC at every temperature while precision / recall / F1 stay matched, the hypothesis
explains the gap.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import PUBLISHED, TEMPS, classification, load_items, mean_se, predict  # noqa: E402

K = [0, 1, 2, 3, 4, 6, 12]
METRICS = ["bal_acc", "precision", "recall", "f1", "auroc"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["mdcath", "atlas"], default="mdcath")
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default="investigation/data/mdcath")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    temp_index = list(range(5)) if args.dataset == "mdcath" else [0]
    items = load_items(args)
    model = esmdynamic().cuda().eval()
    rows = []
    for n, (pid, seq, dyn, _) in enumerate(items):
        L = len(seq)
        if dyn.shape[-1] != L:
            continue
        out = predict(model, seq, batch_encode_sequences)
        sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
        for k in K:
            sel = sep >= k
            for t, T in enumerate(temp_index):
                y = dyn[t][sel]
                for v in ("sym", "unsym"):
                    rows.append({"id": pid, "length": L, "k": k, "variant": v, "temp": TEMPS[T],
                                 "n_pairs": int(sel.sum()), "n_negative": int((~y).sum()),
                                 **classification(out[v][T][sel], y)})
        if (n + 1) % 25 == 0 or n == 0:
            print(f"  [{n + 1:>3}/{len(items)}] {pid} L={L}", flush=True)

    pub = PUBLISHED[args.dataset]
    summary = {}
    for v in ("sym", "unsym"):
        for k in K:
            for T in pub:
                sub = [r for r in rows if r["variant"] == v and r["k"] == k and r["temp"] == T]
                cell = {m: dict(zip(("mean", "se", "n"), mean_se([r[m] for r in sub]))) for m in METRICS}
                cell["specificity"] = dict(zip(("mean", "se", "n"), mean_se(
                    [2 * r["bal_acc"] - r["recall"] for r in sub])))
                cell["negatives_per_protein"] = float(np.mean([r["n_negative"] for r in sub]))
                summary[f"{v}/k{k}/{T}K"] = cell

    for v in ("sym", "unsym"):
        print(f"\n=== {args.dataset}, thresholded score: {v}   (cell = ours; published in the header row)")
        for m in METRICS + ["specificity"]:
            head = f"{m:<12}{'k':>4}" + "".join(
                f"{T:>9}K" + (f" ({pub[T][m][0]:.3f})" if m in pub[T] else
                              f" ({2 * pub[T]['bal_acc'][0] - pub[T]['recall'][0]:.3f})") for T in pub)
            print(head)
            for k in K:
                print(f"{'':<12}{k:>4}" + "".join(f"{summary[f'{v}/k{k}/{T}K'][m]['mean']:>18.3f}" for T in pub))
        print(f"{'negatives/protein':<12}" + "".join(
            f"   k={k}: " + " ".join(f"{summary[f'{v}/k{k}/{T}K']['negatives_per_protein']:.0f}" for T in pub)
            for k in (0, 1, 3)))

    out = Path(args.out or f"investigation/results/13_eval_pairset_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "k_values": K, "n_proteins": len({r["id"] for r in rows}),
                               "published": {str(T): v for T, v in pub.items()}, "summary": summary}, indent=2))
    with open(out.with_name(out.stem + "_per_protein.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
