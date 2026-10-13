"""Do the released weights score better on test proteins that were also in RCSB pre-training?

    python investigation/scripts/leakage_effect.py [--out investigation/results/15_leakage_effect.json]

Membership: results/15_pretrain_overlap_members.csv (check_pretrain_overlap.py): an mdCATH test
domain is "in pre-training" if the authors' mdcath_to_rcsb_mapping.csv maps it to a cluster in
the RCSB train split; an ATLAS chain if its sequence equals, contains or is contained in an
RCSB-train consensus sequence (a lower bound: no alignment search).

Scores: the released weights' per-protein results under the protocols that reproduce the paper
(notes 12-13): mdCATH test, pairs |i - j| >= 3, symmetrised probability, scikit-learn-style
balanced accuracy, all five temperatures; ATLAS test, train.py's protocol (256-residue crops,
5 draws per long chain averaged per chain, raw logits), 320 K output.

Reports per group: n, mean, and the difference (in - out) with a protein-level bootstrap 95%
interval, plus length and dynamic-pair fraction per group, since the groups can differ in
those too: a difference here is an association, not proof that pre-training on a protein is
what raised its score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPS = [320, 348, 379, 413, 450]


def boot_diff(a, b, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    d = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def summarise(df, metrics, extra):
    inn, out = df[df.in_rcsb_train == 1], df[df.in_rcsb_train == 0]
    res = dict(n_in=int(len(inn)), n_out=int(len(out)))
    for m in metrics + extra:
        diff, ci = boot_diff(inn[m], out[m])
        res[m] = dict(mean_in=float(np.nanmean(inn[m])), mean_out=float(np.nanmean(out[m])),
                      diff_in_minus_out=diff, ci95=ci)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--members", default="investigation/results/15_pretrain_overlap_members.csv")
    ap.add_argument("--mdcath", default="investigation/results/13_eval_pairset_mdcath_per_protein.csv")
    ap.add_argument("--atlas", default="investigation/results/12_eval_protocol_atlas_per_protein.csv")
    ap.add_argument("--out", default="investigation/results/15_leakage_effect.json")
    args = ap.parse_args()
    mem = pd.read_csv(args.members)
    out = {}

    md = pd.read_csv(args.mdcath)
    md = md[(md.variant == "sym") & (md.k == 3)].copy()
    md["bal_acc"] = np.where(md.n_negative == 0, md.recall, md.bal_acc)        # scikit-learn style
    allp = pd.read_csv(args.mdcath)
    dynfrac = allp[(allp.variant == "sym") & (allp.k == 3)]
    m = mem[mem.dataset == "mdcath"][["id", "in_rcsb_train"]]
    for T in TEMPS:
        d = md[md.temp == T].merge(m, on="id")
        d["dynamic_fraction"] = 1 - d.n_negative / d.n_pairs
        out[f"mdcath_{T}K"] = summarise(d, ["bal_acc", "auroc", "precision", "recall", "f1"],
                                        ["length", "dynamic_fraction"])

    # the control: test domains WITH an RCSB cluster that was NOT in pre-training (val/test split, or a
    # cluster without dynamic contacts, excluded from all splits). Domains with an RCSB cluster are
    # proteins with several experimental structures, which ESMFold (trained on the PDB) may simply
    # handle better; comparing pre-trained against these separates exposure from protein type.
    m3 = mem[mem.dataset == "mdcath"][["id", "rcsb_split"]]
    groups = {"pretrained": ["train"], "cluster_not_pretrained": ["val", "test", "not_in_any_split"],
              "no_cluster": ["unmapped"]}
    for T in TEMPS:
        d = md[md.temp == T].merge(m3, on="id")
        g = {k: d[d.rcsb_split.isin(v)] for k, v in groups.items()}
        res = {k: dict(n=int(len(x)), bal_acc=float(x.bal_acc.mean()), auroc=float(np.nanmean(x.auroc)),
                       length=float(x.length.mean())) for k, x in g.items()}
        for a_, b_ in (("pretrained", "cluster_not_pretrained"), ("cluster_not_pretrained", "no_cluster")):
            for mname in ("bal_acc", "auroc"):
                diff, ci = boot_diff(g[a_][mname], g[b_][mname])
                res[f"{mname}_{a_}_minus_{b_}"] = dict(diff=diff, ci95=ci)
        out[f"mdcath_{T}K_three_groups"] = res

    at = pd.read_csv(args.atlas)
    at = at[(at.protocol == "crop256") & (at.variant == "unsym") & (at.temp == 320)]
    at = at.groupby("id", as_index=False)[["bal_acc", "auroc", "precision", "recall", "f1", "rmse",
                                           "length", "dynamic_fraction"]].mean()
    a = at.merge(mem[mem.dataset == "atlas"][["id", "in_rcsb_train"]], on="id")
    out["atlas_320K"] = summarise(a, ["bal_acc", "auroc", "precision", "recall", "f1", "rmse"],
                                  ["length", "dynamic_fraction"])

    Path(args.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k.endswith("three_groups"):
            print(f"{k:<26} " + "  ".join(f"{g} n={v[g]['n']} bal_acc {v[g]['bal_acc']:.3f}" for g in groups)
                  + "  | pretrained - cluster_not_pretrained: {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]".format(
                      diff=v["bal_acc_pretrained_minus_cluster_not_pretrained"]["diff"],
                      lo=v["bal_acc_pretrained_minus_cluster_not_pretrained"]["ci95"][0],
                      hi=v["bal_acc_pretrained_minus_cluster_not_pretrained"]["ci95"][1]))
            continue
        line = f"{k:<12} in {v['n_in']:>3} / out {v['n_out']:>3}  "
        for mname in ("bal_acc", "auroc", "length", "dynamic_fraction"):
            r = v[mname]
            line += (f"{mname} {r['mean_in']:.3f} vs {r['mean_out']:.3f} "
                     f"(diff {r['diff_in_minus_out']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}])  ")
        print(line)


if __name__ == "__main__":
    main()
