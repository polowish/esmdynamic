"""Every metric of the published ATLAS table, from the released V2 weights.

    python investigation/scripts/atlas_full_metrics.py --data investigation/data/atlas_test

Nature Communications (2026) 17:9623, Table 2 ("Performance on the ATLAS test set") reports
ESMDynamic as mean +- standard error over the 82 chains: balanced accuracy 0.872 +- 0.010,
precision 0.284 +- 0.013, recall 0.890 +- 0.011, F1 0.413 +- 0.016, AUROC 0.942 +- 0.007,
contact-occupancy RMSE 0.063 +- 0.003. This computes each of them PER PROTEIN and then
mean +- se over proteins, from the 320 K output (nearest to ATLAS's 300 K), prediction =
probability > 0.5.

The pair set is not stated in the paper, so two variants are reported: all L x L pairs, and
the upper triangle i < j (no diagonal, each pair once). AUROC uses the rank (Mann-Whitney)
formula with average ranks for ties; precision is undefined for a protein with no predicted
positives and F1 then counts as 0 (both rare at a 0.5 threshold, and counted in the output).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

PUBLISHED = {"bal_acc": (0.872, 0.010), "precision": (0.284, 0.013), "recall": (0.890, 0.011),
             "f1": (0.413, 0.016), "auroc": (0.942, 0.007), "rmse": (0.063, 0.003)}


def auroc(scores, labels):
    labels = labels.astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), float)
    i = 0
    while i < len(s):                       # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1
        i = j + 1
    r = np.empty(len(s), float)
    r[order] = ranks
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def metrics(prob, occ_pred, lab, occ, sel):
    p, y, q, o = prob[sel], lab[sel], occ_pred[sel], occ[sel]
    pred = p > 0.5
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and (prec + rec) else 0.0
    return {"bal_acc": 0.5 * (rec + tnr), "precision": prec, "recall": rec, "f1": f1,
            "auroc": auroc(p, y), "rmse": float(np.sqrt(np.mean((q - o) ** 2))),
            "no_predicted_positive": int(tp + fp == 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="investigation/data/atlas_test")
    ap.add_argument("--out", default="investigation/results/12_atlas_full_metrics.json")
    args = ap.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    data = Path(args.data)
    rows = list(csv.DictReader(open(data / "atlas_test.csv")))
    model = esmdynamic().cuda().eval()
    per = {"all": [], "triu": []}
    for k, r in enumerate(rows):
        name, seq = r["name"], r["seqres"]
        lab = np.load(data / f"{name}_dyn_cont.npy") > 0.5
        occ = np.load(data / f"{name}_freq.npy")
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.no_grad():
            st = model.esmfold(aatype, mask, residx, None, None)
            st["mask"] = mask
            for h in model.heads.values():
                st = h(st, num_recycles=None)
        prob = st["dynamic_prob"][0, 0].float().cpu().numpy()                 # 320 K
        q = np.clip(st["frequency_value"][0, 0].float().cpu().numpy(), 0, 1)  # 320 K
        L = len(seq)
        sel_all = np.ones((L, L), bool)
        sel_triu = np.triu(np.ones((L, L), bool), 1)
        for key, sel in (("all", sel_all), ("triu", sel_triu)):
            per[key].append({"name": name, "length": L, **metrics(prob, q, lab, occ, sel)})
        if (k + 1) % 20 == 0:
            print(f"  [{k + 1}/{len(rows)}]", flush=True)

    out = {"n_proteins": len(rows), "published": PUBLISHED, "ours": {}}
    print(f"\n{len(rows)} ATLAS proteins, 320 K output, per-protein mean +- se")
    print(f"{'metric':<10} {'published':>16} {'ours: all pairs':>18} {'ours: i < j':>16}")
    for m, (pv, ps) in PUBLISHED.items():
        line = f"{m:<10} {pv:>9.3f} ± {ps:.3f}"
        for key in ("all", "triu"):
            v = np.array([x[m] for x in per[key]], float)
            v = v[np.isfinite(v)]
            mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
            out["ours"].setdefault(key, {})[m] = {"mean": mean, "se": se, "n": int(len(v))}
            line += f"   {mean:>9.3f} ± {se:.3f}"
        print(line)
    npp = sum(x["no_predicted_positive"] for x in per["all"])
    print(f"proteins with no predicted positive (precision undefined, F1 = 0): {npp}")
    out["no_predicted_positive"] = npp
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    for key in per:
        with open(path.with_name(f"{path.stem}_per_protein_{key}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per[key][0]))
            w.writeheader()
            w.writerows(per[key])
    print(f"-> {path}")


if __name__ == "__main__":
    main()
