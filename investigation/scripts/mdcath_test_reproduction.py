"""The released V2 weights on the mdCATH test split, scored like the preprint's SI Table 6.

    python investigation/scripts/mdcath_test_reproduction.py
        [--data investigation/data/mdcath/dataset] [--splits investigation/data/mdcath]

SI Table 6 reports the dynamic-contact head on the mdCATH test split as mean +- se per
temperature (320 K: balanced accuracy 0.796 +- 0.007). mdCATH has MD at all five
temperatures, so each model output is scored against the labels of ITS OWN temperature.
Each test protein is predicted at full length (no crop); prediction = probability > 0.5;
all L x L pairs; balanced accuracy per protein, then mean and se over proteins (and pooled).
Also scores the frequency head's RMSE against the contact occupancy (SI Table 7).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

TEMPS = [320, 348, 379, 413, 450]


def bal_acc(pred, true):
    tp = int((pred & true).sum()); tn = int((~pred & ~true).sum())
    fp = int((pred & ~true).sum()); fn = int((~pred & true).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    return 0.5 * (tpr + tnr), np.array([tp, tn, fp, fn])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default="investigation/data/mdcath/dataset")
    ap.add_argument("--splits", default="investigation/data/mdcath")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="investigation/results/10_mdcath_test_reproduction.json")
    args = ap.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    ids = [x.strip() for x in open(Path(args.splits) / f"{args.split}.csv") if x.strip()]
    model = esmdynamic().cuda().eval()
    per_protein, pooled = [], {t: np.zeros(4, np.int64) for t in range(5)}
    for k, pid in enumerate(ids):
        d = Path(args.data) / pid
        seq = "".join(l.strip() for l in open(d / "consensus.fasta") if not l.startswith(">"))
        dyn = torch.load(d / "dynamic_contacts.pt").float().numpy() > 0.5      # (5, L, L)
        occ = torch.load(d / "frequency.pt").float().numpy()                   # (5, L, L)
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.no_grad():
            st = model.esmfold(aatype, mask, residx, None, None)
            st["mask"] = mask
            for h in model.heads.values():
                st = h(st, num_recycles=None)
        prob = st["dynamic_prob"][0].float().cpu().numpy()
        fval = st["frequency_value"][0].float().cpu().numpy()
        L = len(seq)
        if prob.shape != dyn.shape:
            print(f"  skip {pid}: pred {prob.shape} labels {dyn.shape}")
            continue
        rec = {"id": pid, "length": L}
        for t in range(5):
            ba, c = bal_acc(prob[t] > 0.5, dyn[t])
            pooled[t] += c
            rec[f"bal_acc_{TEMPS[t]}K"] = ba
            rec[f"occ_rmse_{TEMPS[t]}K"] = float(np.sqrt(np.mean((np.clip(fval[t], 0, 1) - occ[t]) ** 2)))
        per_protein.append(rec)
        if (k + 1) % 25 == 0 or k == 0:
            print(f"  [{k + 1:>3}/{len(ids)}] {pid} L={L}  BA@320K {rec['bal_acc_320K']:.3f}", flush=True)

    summary = {}
    for t, T in enumerate(TEMPS):
        v = np.array([r[f"bal_acc_{T}K"] for r in per_protein], float)
        o = np.array([r[f"occ_rmse_{T}K"] for r in per_protein], float)
        tp, tn, fp, fn = pooled[t]
        fin = np.isfinite(v)
        summary[f"{T}K"] = {"bal_acc_mean": float(v[fin].mean()),
                            "bal_acc_se": float(v[fin].std(ddof=1) / np.sqrt(fin.sum())),
                            "bal_acc_pooled": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                            "n_scored": int(fin.sum()),
                            "occ_rmse_mean": float(o.mean()),
                            "occ_rmse_se": float(o.std(ddof=1) / np.sqrt(len(o)))}
    print(f"\n{len(per_protein)} of {len(ids)} {args.split} proteins scored")
    print(f"{'temp':<6} {'BA mean (se)':>16} {'pooled':>8} {'occ RMSE (se)':>16}")
    for T in TEMPS:
        s = summary[f"{T}K"]
        print(f"{T}K  {s['bal_acc_mean']:>9.3f} ({s['bal_acc_se']:.3f}) {s['bal_acc_pooled']:>8.3f} "
              f"{s['occ_rmse_mean']:>9.3f} ({s['occ_rmse_se']:.3f})")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "n_proteins": len(per_protein),
                               "summary": summary}, indent=2))
    with open(out.with_name(out.stem + "_per_protein.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_protein[0]))
        w.writeheader()
        w.writerows(per_protein)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
