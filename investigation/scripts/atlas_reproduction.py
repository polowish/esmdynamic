"""Check 6: do the released V2 weights reproduce the preprint's ATLAS-test numbers?

    python investigation/scripts/atlas_reproduction.py --data investigation/data/atlas_test

The preprint (v2 p.17) reports ESMDynamic at 87% balanced accuracy for dynamic-contact
classification on the ATLAS test split (AlphaFlow's, 82 chains), averaged across proteins.
If the released weights reproduce that, the paper's main results came from the model whose
DCM blocks are untrained.

The protocol is not fully specified, so the plausible variants are all reported:
  * temperature channel: the model predicts 5 mdCATH temperatures (320-450 K); ATLAS is a
    single 300 K ensemble, so each channel is scored (320 K, the nearest, is the natural one)
  * pairs: all L x L, or the upper triangle i < j
  * aggregation: per-protein balanced accuracy averaged over proteins (the paper's stated
    convention for mdCATH), and pooled over all pairs of all proteins
Prediction = sigmoid(logit) > 0.5, as in train.py's metrics_dynamic_batch. Also reports the
frequency head's RMSE against the ATLAS contact occupancy.

Writes investigation/results/06_atlas_reproduction.json and a per-protein csv.
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
    return 0.5 * (tpr + tnr), (tp, tn, fp, fn)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="investigation/data/atlas_test")
    p.add_argument("--out", default="investigation/results/06_atlas_reproduction.json")
    args = p.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    data = Path(args.data)
    rows = list(csv.DictReader(open(data / "atlas_test.csv")))
    model = esmdynamic().cuda().eval()

    per_protein = []
    pooled = {(t, tri): np.zeros(4, np.int64) for t in range(5) for tri in (False, True)}
    for k, r in enumerate(rows):
        name, seq = r["name"], r["seqres"]
        dyn = np.load(data / f"{name}_dyn_cont.npy") > 0.5
        freq = np.load(data / f"{name}_freq.npy")
        aatype, mask, residx, _, _ = batch_encode_sequences([seq], 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.no_grad():
            st = model.esmfold(aatype, mask, residx, None, None)
            st["mask"] = mask
            for h in model.heads.values():
                st = h(st, num_recycles=None)
        prob = st["dynamic_prob"][0].float().cpu().numpy()           # (5, L, L)
        fval = st["frequency_value"][0].float().cpu().numpy()        # (5, L, L)
        L = len(seq)
        if prob.shape[-1] != L or dyn.shape != (L, L):
            print(f"  skip {name}: shapes pred {prob.shape} target {dyn.shape} L={L}")
            continue
        iu = np.triu_indices(L, 1)
        rec = {"name": name, "length": L, "dynamic_fraction": float(dyn.mean())}
        for t in range(5):
            pred = prob[t] > 0.5
            for tri in (False, True):
                pp, tt = (pred[iu], dyn[iu]) if tri else (pred.ravel(), dyn.ravel())
                ba, counts = bal_acc(pp, tt)
                pooled[(t, tri)] += np.array(counts)
                rec[f"bal_acc_{TEMPS[t]}K_{'triu' if tri else 'all'}"] = ba
            rec[f"freq_rmse_{TEMPS[t]}K"] = float(np.sqrt(np.mean((np.clip(fval[t], 0, 1) - freq) ** 2)))
        per_protein.append(rec)
        print(f"  [{k + 1:>2}/{len(rows)}] {name} L={L:>4} dyn {dyn.mean():.3f}  "
              f"BA@320K all {rec['bal_acc_320K_all']:.3f}  triu {rec['bal_acc_320K_triu']:.3f}")

    summary = {}
    for t in range(5):
        for tri in (False, True):
            key = f"{TEMPS[t]}K_{'triu' if tri else 'all'}"
            vals = np.array([r[f"bal_acc_{key}"] for r in per_protein], float)
            tp, tn, fp, fn = pooled[(t, tri)]
            summary[key] = {
                "per_protein_mean": float(np.nanmean(vals)),
                "per_protein_se": float(np.nanstd(vals, ddof=1) / np.sqrt(np.isfinite(vals).sum())),
                "pooled": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
                "recall_pooled": tp / (tp + fn), "precision_pooled": tp / (tp + fp) if tp + fp else float("nan"),
            }
        summary[f"{TEMPS[t]}K_freq_rmse_mean"] = float(np.mean([r[f"freq_rmse_{TEMPS[t]}K"] for r in per_protein]))
    print(f"\n{len(per_protein)} proteins scored. Preprint: balanced accuracy 87% (ATLAS test).")
    print(f"{'channel':<8} {'pairs':<6} {'per-protein mean (se)':>22} {'pooled':>8} {'recall':>7} {'precision':>9}")
    for t in TEMPS:
        for pr in ("all", "triu"):
            s = summary[f"{t}K_{pr}"]
            print(f"{t}K    {pr:<6} {s['per_protein_mean']:>15.3f} ({s['per_protein_se']:.3f}) "
                  f"{s['pooled']:>8.3f} {s['recall_pooled']:>7.3f} {s['precision_pooled']:>9.3f}")
    for t in TEMPS:
        print(f"frequency RMSE {t}K: {summary[f'{t}K_freq_rmse_mean']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"n_proteins": len(per_protein), "summary": summary,
                               "preprint": {"balanced_accuracy": 0.87}}, indent=2))
    with open(out.with_name("06_atlas_reproduction_per_protein.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_protein[0]))
        w.writeheader()
        w.writerows(per_protein)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
