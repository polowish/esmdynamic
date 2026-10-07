"""The released V2 weights on the ATLAS / mdCATH test splits, under the published protocol.

    python investigation/scripts/eval_protocol.py --dataset mdcath   [--crop-seeds 5]
    python investigation/scripts/eval_protocol.py --dataset atlas

Targets: Nat Commun (2026) 17:9623 Table 1 (mdCATH test, 320 K), Table 2 (ATLAS test), and
SI Tables 6-7 (mdCATH test, all five temperatures). Every number there is mean +- se over
proteins (n = 270 mdCATH, n = 82 ATLAS; Fig. 2 legend: "independent samples").

What is fixed by the paper or the repository's code, and followed here:
  - labels: the Data Bank files as distributed (dynamic_contacts.pt / frequency.pt per mdCATH
    domain, *_dyn_cont.npy / *_freq.npy per ATLAS chain); sequence = consensus.fasta line 2
    (DynContactDataset._load_sequence);
  - pairs: all L x L, diagonal included -- the training code's length_mask_2d and
    metrics_*_batch score [:L, :L] with nothing excluded;
  - threshold 0.5; balanced accuracy, precision, recall, F1, AUROC, occupancy RMSE;
  - occupancy = `frequency_pred` = sigmoid((x + x^T) / 2), the head's documented output, what
    predict.py writes and what the training loss / metrics_frequency_batch score. (Our earlier
    notes 06 and 10 used clip(frequency_value, 0, 1), the raw unsymmetrised value; it is kept
    here as the `occ_raw_clip` variant only to show what that changed.)
  - mdCATH: each of the 5 outputs against the same-temperature labels; ATLAS (300 K MD):
    the 320 K output.

What the paper does not state, so both readings are run:
  - CROP. The only crop in the paper and code is the training one: "Proteins were randomly
    cropped to 256 residues" (Methods, Pretraining; SI Fig. 19 "supporting this crop size for
    training"), DynContactDataset: if L > 256, start = torch.randint(0, L - 256), labels
    sliced [start:start+256]. Nothing says the test set was cropped, and the comparison with
    BioEmu / AlphaFlow (full-length ensembles) only makes sense at full length -- but the
    training script's validation pass is the one evaluation loop in the repository, and it
    crops. So: `full` (whole chain), and `crop256` with --crop-seeds independent random crops
    per protein exactly as DynContactDataset draws them (chains <= 256 are never cropped).
  - THRESHOLDED QUANTITY. predict.py thresholds `dynamic_prob` (sigmoid, then symmetrised);
    metrics_dynamic_batch thresholds sigmoid(`dynamic_logits`), unsymmetrised. Both are kept:
    `sym` and `unsym` (AUROC from the same score).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

TEMPS = [320, 348, 379, 413, 450]
CROP = 256

PUBLISHED = {
    "mdcath": {  # SI Tables 6-7, test split (320 K = main-text Table 1)
        320: {"bal_acc": (0.796, 0.007), "precision": (0.511, 0.012), "recall": (0.767, 0.010),
              "f1": (0.569, 0.008), "auroc": (0.889, 0.006), "rmse": (0.076, 0.002)},
        348: {"bal_acc": (0.772, 0.007), "precision": (0.533, 0.013), "recall": (0.809, 0.009),
              "f1": (0.596, 0.009), "auroc": (0.873, 0.007), "rmse": (0.074, 0.002)},
        379: {"bal_acc": (0.734, 0.008), "precision": (0.576, 0.014), "recall": (0.844, 0.009),
              "f1": (0.639, 0.010), "auroc": (0.841, 0.008), "rmse": (0.072, 0.002)},
        413: {"bal_acc": (0.657, 0.008), "precision": (0.646, 0.016), "recall": (0.918, 0.006),
              "f1": (0.720, 0.012), "auroc": (0.796, 0.008), "rmse": (0.068, 0.002)},
        450: {"bal_acc": (0.542, 0.007), "precision": (0.791, 0.016), "recall": (0.994, 0.001),
              "f1": (0.853, 0.012), "auroc": (0.729, 0.007), "rmse": (0.057, 0.002)},
    },
    "atlas": {  # main-text Table 2
        320: {"bal_acc": (0.872, 0.010), "precision": (0.284, 0.013), "recall": (0.890, 0.011),
              "f1": (0.413, 0.016), "auroc": (0.942, 0.007), "rmse": (0.063, 0.003)},
    },
}
METRICS = ["bal_acc", "precision", "recall", "f1", "auroc", "rmse", "rmse_raw_clip"]


def auroc(scores, labels):
    """Mann-Whitney AUROC with average ranks for ties (equals sklearn's roc_auc_score)."""
    npos = int(labels.sum()); nneg = labels.size - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    avg_rank = np.cumsum(counts) - (counts - 1) / 2.0
    r = avg_rank[inv]
    return float((r[labels].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def classification(score, y):
    pred = score > 0.5
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    tnr = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp + fp) and tp else 0.0
    return {"bal_acc": 0.5 * (rec + tnr), "precision": prec, "recall": rec, "f1": f1,
            "auroc": auroc(score.ravel(), y.ravel())}


def load_items(args):
    """-> list of (id, sequence, dyn [T, L, L] bool, occ [T, L, L]); T = 5 (mdCATH) or 1 (ATLAS)."""
    items = []
    if args.dataset == "mdcath":
        root = Path(args.data or "investigation/data/mdcath/dataset")
        ids = [x.strip() for x in open(Path(args.splits) / "test.csv") if x.strip()]
        for pid in ids:
            d = root / pid
            seq = open(d / "consensus.fasta").readlines()[1].strip()
            dyn = torch.load(d / "dynamic_contacts.pt").float().numpy() > 0.5
            occ = torch.load(d / "frequency.pt").float().numpy()
            items.append((pid, seq, dyn, occ))
    else:
        root = Path(args.data or "investigation/data/atlas_test")
        for r in csv.DictReader(open(root / "atlas_test.csv")):
            dyn = (np.load(root / f"{r['name']}_dyn_cont.npy") > 0.5)[None]
            occ = np.load(root / f"{r['name']}_freq.npy")[None]
            items.append((r["name"], r["seqres"], dyn, occ))
    return items


def predict(model, seq, encode):
    aatype, mask, residx, _, _ = encode([seq], 512, "G" * 25)
    aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
    with torch.no_grad():
        st = model.esmfold(aatype, mask, residx, None, None)
        st["mask"] = mask
        for h in model.heads.values():
            st = h(st, num_recycles=None)
    g = lambda k: st[k][0].float().cpu().numpy()
    return {"sym": g("dynamic_prob"), "unsym": 1 / (1 + np.exp(-g("dynamic_logits"))),
            "occ": g("frequency_pred"), "occ_raw": np.clip(g("frequency_value"), 0, 1)}


def score(out, dyn, occ, temp_index):
    """Rows for one (protein, crop): one per (threshold variant, temperature)."""
    rows = []
    for t, T in enumerate(temp_index):
        rmse = float(np.sqrt(np.mean((out["occ"][T] - occ[t]) ** 2)))
        rmse_raw = float(np.sqrt(np.mean((out["occ_raw"][T] - occ[t]) ** 2)))
        for v in ("sym", "unsym"):
            rows.append({"variant": v, "temp": TEMPS[T], **classification(out[v][T], dyn[t]),
                         "rmse": rmse, "rmse_raw_clip": rmse_raw,
                         "dynamic_fraction": float(dyn[t].mean())})
    return rows


def mean_se(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), int(len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["mdcath", "atlas"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default="investigation/data/mdcath")
    ap.add_argument("--crop-seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences

    temp_index = list(range(5)) if args.dataset == "mdcath" else [0]   # ATLAS: 320 K output
    items = load_items(args)
    model = esmdynamic().cuda().eval()
    rows = []
    n_cropped = 0
    for k, (pid, seq, dyn, occ) in enumerate(items):
        L = len(seq)
        if dyn.shape[-1] != L:
            print(f"  skip {pid}: L={L}, labels {dyn.shape}")
            continue
        full = predict(model, seq, batch_encode_sequences)
        for r in score(full, dyn, occ, temp_index):
            rows.append({"id": pid, "length": L, "protocol": "full", "seed": -1, "start": 0, **r})
        n_cropped += L > CROP
        for s in range(args.crop_seeds):
            if L > CROP:   # DynContactDataset.__getitem__, one draw per seed
                torch.manual_seed(s * 100_003 + k)
                start = torch.randint(0, L - CROP, ()).item()
                sl = slice(start, start + CROP)
                out = predict(model, seq[sl], batch_encode_sequences)
                d, o = dyn[:, sl, sl], occ[:, sl, sl]
            else:
                start, out, d, o = 0, full, dyn, occ
            for r in score(out, d, o, temp_index):
                rows.append({"id": pid, "length": L, "protocol": "crop256", "seed": s,
                             "start": start, **r})
        if (k + 1) % 25 == 0 or k == 0:
            print(f"  [{k + 1:>3}/{len(items)}] {pid} L={L}", flush=True)

    # summary: per protocol x variant x temp, per-protein mean +- se; crops: per seed, then
    # mean over seeds (se = mean of the per-seed se; seed sd reported separately)
    summary = {}
    for proto in ("full", "crop256"):
        seeds = [-1] if proto == "full" else list(range(args.crop_seeds))
        for v in ("sym", "unsym"):
            for T in [TEMPS[i] for i in temp_index]:
                cell = {}
                for m in METRICS:
                    per_seed = [mean_se([r[m] for r in rows if r["protocol"] == proto and r["seed"] == s
                                         and r["variant"] == v and r["temp"] == T]) for s in seeds]
                    means = [p[0] for p in per_seed]
                    cell[m] = {"mean": float(np.mean(means)), "se": float(np.mean([p[1] for p in per_seed])),
                               "seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else 0.0,
                               "n": per_seed[0][2]}
                summary[f"{proto}/{v}/{T}K"] = cell

    pub = PUBLISHED[args.dataset]
    n = len({r["id"] for r in rows})
    print(f"\n{args.dataset}: {n} proteins, {n_cropped} longer than {CROP} (cropped under crop256)")
    for T in pub:
        print(f"\n{T} K  {'published':>15}" + "".join(f"{c:>17}" for c in
              ("full/sym", "full/unsym", "crop256/sym", "crop256/unsym")))
        for m in METRICS:
            p = pub[T].get(m if m != "rmse_raw_clip" else "rmse")
            line = f"{m:<13} {p[0]:.3f} ± {p[1]:.3f}" if m != "rmse_raw_clip" else f"{m:<13} {'':>13}"
            for c in ("full/sym", "full/unsym", "crop256/sym", "crop256/unsym"):
                x = summary[f"{c}/{T}K"][m]
                line += f"   {x['mean']:.3f} ± {x['se']:.3f}"
            print(line)

    out = Path(args.out or f"investigation/results/12_eval_protocol_{args.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "n_proteins": n, "n_longer_than_crop": n_cropped,
                               "crop_length": CROP, "crop_seeds": args.crop_seeds,
                               "published": {str(T): v for T, v in pub.items()},
                               "summary": summary}, indent=2))
    with open(out.with_name(out.stem + "_per_protein.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
