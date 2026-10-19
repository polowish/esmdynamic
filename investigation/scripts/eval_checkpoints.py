"""Test-set evaluation of the released V2 heads and fine-tuned heads, paired on identical inputs.

    python investigation/scripts/eval_checkpoints.py --dataset mdcath \
        --model v2-released=V2 \
        --model v2ft-dcm-only-fixed-s0=~/scratch/esmdyn_retrain/runs/v2ft-dcm-only-fixed-s0/best.pt ...

ESMFold runs ONCE per protein (and per crop), in eval mode as eval_protocol.py runs it; every
model's heads then read that same structure, so model differences are paired per protein with
the ESMFold input held fixed. Run on L40S (the GPU the training cache was written on).

Two scoring protocols, both reproduced from the released weights in notes 12-13:
  si6   (primary; SI Tables 6-7) full length; symmetrised `dynamic_prob`; pairs |i - j| >= 3;
        balanced accuracy as scikit-learn computes it (a protein with no negatives scores its
        recall, no positives its specificity); AUROC; occupancy RMSE of `frequency_pred` over
        all pairs.
  t12   (Tables 1-2) 256-residue crops as DynContactDataset draws them (--crop-seeds per
        protein; chains <= 256 whole); sigmoid(`dynamic_logits`) unsymmetrised; all L x L pairs.
mdCATH: each of the 5 outputs against its own temperature. ATLAS (300 K MD): the 320 K output.

Writes results/<out>.json (per model x protocol x temperature: mean +- se over proteins, and the
paired difference to the reference model with a protein-bootstrap 95% CI and two-sided p, on
all proteins and on those whose RCSB cluster was NOT in the authors' pre-training split,
results/15_pretrain_overlap_members.csv) and a per-protein csv.
"""
import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_protocol import TEMPS, auroc, load_items  # noqa: E402

CROP = 256
METRICS = ["bal_acc", "precision", "recall", "f1", "auroc", "rmse"]


def classify(score, y, min_sep):
    L = y.shape[-1]
    i, j = np.triu_indices(L, 0) if min_sep == 0 else np.triu_indices(L, min_sep)
    if min_sep == 0:                       # all L x L pairs, as train.py's metrics
        s, t = score.ravel(), y.ravel()
    else:                                  # |i - j| >= k, both orientations
        s = np.concatenate([score[i, j], score[j, i]]); t = np.concatenate([y[i, j], y[j, i]])
    p = s > 0.5
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t).sum()); tn = int((~p & ~t).sum())
    rec = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    present = [x for x in (rec, spec) if x == x]          # scikit-learn: mean over present classes
    return {"bal_acc": float(np.mean(present)) if present else float("nan"), "precision": prec,
            "recall": rec, "f1": 2 * prec * rec / (prec + rec) if tp else 0.0, "auroc": auroc(s, t)}


def heads_out(heads, st):
    s = dict(st)
    with torch.no_grad():
        for h in heads.values():
            s = h(s, num_recycles=None)
    g = lambda k: s[k][0].float().cpu().numpy()  # noqa: E731
    return {"sym": g("dynamic_prob"), "unsym": 1 / (1 + np.exp(-g("dynamic_logits"))),
            "occ": g("frequency_pred")}


def mean_se(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))), int(len(v))


def paired(a, b, n_boot=10_000, seed=0):
    """Mean of a - b over proteins with both finite, protein-bootstrap 95% CI, two-sided p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b); d = a[ok] - b[ok]
    if len(d) < 2:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": int(len(d))}
    boot = d[np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))].mean(1)
    tail = min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "p": float(min(1.0, max(2 * tail, 1 / n_boot))), "n": int(len(d))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=["mdcath", "atlas"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--splits", default="investigation/data/mdcath")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=V2|best.pt",
                    help="repeatable; the first is the reference for paired differences")
    ap.add_argument("--crop-seeds", type=int, default=5)
    ap.add_argument("--overlap", default="investigation/results/15_pretrain_overlap_members.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from esm.esmdynamic.pretrained import esmdynamic
    from esm.esmfold.v1.misc import batch_encode_sequences as enc

    model = esmdynamic().cuda().eval()
    names, heads = [], {}
    for spec in args.model:
        name, _, src = spec.partition("=")
        h = copy.deepcopy(model.heads)
        if src != "V2":
            ck = torch.load(os.path.expanduser(src), map_location="cuda", weights_only=False)
            h.load_state_dict(ck["heads"], strict=True)
            print(f"{name}: {src} (epoch {ck.get('epoch')})", flush=True)
        else:
            print(f"{name}: released V2 heads", flush=True)
        names.append(name); heads[name] = h.eval()
    del model.heads                                   # only ESMFold is used from here on
    torch.cuda.empty_cache()

    def fold(seq):
        aatype, mask, residx, _, _ = enc([seq], 512, "G" * 25)
        aatype, mask, residx = (x.cuda() for x in (aatype, mask, residx))
        with torch.no_grad():
            st = model.esmfold(aatype, mask, residx, None, None)
        st["mask"] = mask
        return st

    temp_index = list(range(5)) if args.dataset == "mdcath" else [0]
    items = load_items(args)
    in_train = {r["id"] for r in csv.DictReader(open(args.overlap))
                if r["dataset"] == args.dataset and r["in_rcsb_train"] == "1"}
    rows = []
    for k, (pid, seq, dyn, occ) in enumerate(items):
        L = len(seq)
        if dyn.shape[-1] != L:
            print(f"  skip {pid}: L={L}, labels {dyn.shape}"); continue
        base = {"id": pid, "length": L, "in_rcsb_train": int(pid in in_train)}
        st = fold(seq)
        for name in names:                            # si6: full length, sym, |i-j| >= 3
            out = heads_out(heads[name], st)
            for t, T in enumerate(temp_index):
                rmse = float(np.sqrt(np.mean((out["occ"][T] - occ[t]) ** 2)))
                rows.append({**base, "model": name, "protocol": "si6", "seed": -1, "temp": TEMPS[T],
                             **classify(out["sym"][T], dyn[t], 3), "rmse": rmse})
        for s in range(args.crop_seeds):              # t12: crops, unsym, all pairs
            if L > CROP:
                torch.manual_seed(s * 100_003 + k)
                start = torch.randint(0, L - CROP, ()).item(); sl = slice(start, start + CROP)
                stc, d, o = fold(seq[sl]), dyn[:, sl, sl], occ[:, sl, sl]
            elif s == 0:
                stc, d, o = st, dyn, occ
            for name in names:
                if L <= CROP and s > 0:               # uncropped: identical to seed 0
                    rows.extend({**r, "seed": s} for r in [r for r in rows if r["id"] == pid and
                                r["model"] == name and r["protocol"] == "t12" and r["seed"] == 0])
                    continue
                out = heads_out(heads[name], stc)
                for t, T in enumerate(temp_index):
                    rmse = float(np.sqrt(np.mean((out["occ"][T] - o[t]) ** 2)))
                    rows.append({**base, "model": name, "protocol": "t12", "seed": s, "temp": TEMPS[T],
                                 **classify(out["unsym"][T], d[t], 0), "rmse": rmse})
        if (k + 1) % 25 == 0 or k == 0:
            print(f"  [{k + 1:>3}/{len(items)}] {pid} L={L}", flush=True)

    # per protein: average t12 over crop seeds, so both protocols give one value per protein
    def per_protein(model, proto, T, m, subset=None):
        vals = {}
        for r in rows:
            if r["model"] == model and r["protocol"] == proto and r["temp"] == T and \
                    (subset is None or r["in_rcsb_train"] == subset):
                vals.setdefault(r["id"], []).append(r[m])
        ids = sorted(vals)
        return ids, [float(np.nanmean(vals[i])) if np.isfinite(vals[i]).any() else float("nan") for i in ids]

    ref = names[0]
    summary = {}
    for proto in ("si6", "t12"):
        for T in [TEMPS[i] for i in temp_index]:
            for subset_name, subset in (("all", None), ("not_in_rcsb_train", 0)):
                key = f"{proto}/{T}K/{subset_name}"
                summary[key] = {}
                for name in names:
                    cell = {}
                    for m in METRICS:
                        ids, v = per_protein(name, proto, T, m, subset)
                        mu, se, n = mean_se(v)
                        cell[m] = {"mean": mu, "se": se, "n": n}
                        if name != ref:
                            ids0, v0 = per_protein(ref, proto, T, m, subset)
                            assert ids0 == ids
                            cell[m]["vs_" + ref] = paired(v, v0)
                    summary[key][name] = cell

    n = len({r["id"] for r in rows}); n_unseen = len({r["id"] for r in rows if not r["in_rcsb_train"]})
    print(f"\n{args.dataset}: {n} proteins ({n_unseen} not in RCSB pre-training train split)")
    for proto in ("si6", "t12"):
        for T in [TEMPS[i] for i in temp_index]:
            c = summary[f"{proto}/{T}K/all"]
            print(f"\n{proto} {T} K, all proteins" + "".join(f"{nm[:24]:>27}" for nm in names))
            for m in METRICS:
                line = f"  {m:<9}"
                for nm in names:
                    x = c[nm][m]
                    line += f"   {x['mean']:.3f}±{x['se']:.3f}"
                    line += f" ({x['vs_' + ref]['diff']:+.3f})" if nm != ref else "         "
                print(line)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "models": dict(m.partition("=")[::2] for m in args.model),
                               "reference": ref, "n_proteins": n, "n_not_in_rcsb_train": n_unseen,
                               "crop_seeds": args.crop_seeds, "summary": summary}, indent=2))
    with open(out.with_name(out.stem + "_per_protein.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
