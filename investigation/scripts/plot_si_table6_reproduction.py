"""Figure: the released V2 weights reproduce SI Table 6 (mdCATH test, all five temperatures).

    python investigation/scripts/plot_si_table6_reproduction.py
        [--out investigation/figures/reproduction/13_si_table6_reproduction.png]

From results/13_eval_pairset_mdcath_per_protein.csv (eval_pairset.py, job 2199271): the
released weights run once per test domain, full length, then scored two ways:

  all residue pairs           every L x L pair, specificity-undefined proteins dropped
  the paper's protocol        pairs with |i - j| >= 3, symmetrised probability > 0.5, balanced
                              accuracy per protein as sklearn computes it (a protein with no
                              non-dynamic pairs scores its recall alone); AUROC over the
                              proteins where it is defined

Both use dynamic_prob (symmetrised). Published values: SI Table 6, test split. Note 13.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
PUB, OURS, ALL = "#1D3BE2", "#1DE2C4", "#a8a7a2"
TEMPS = [320, 348, 379, 413, 450]
PUBLISHED = {  # SI Table 6, test split: mean, se
    "bal_acc": [(0.796, .007), (.772, .007), (.734, .008), (.657, .008), (.542, .007)],
    "auroc": [(.889, .006), (.873, .007), (.841, .008), (.796, .008), (.729, .007)],
    "precision": [(.511, .012), (.533, .013), (.576, .014), (.646, .016), (.791, .016)],
    "recall": [(.767, .010), (.809, .009), (.844, .009), (.918, .006), (.994, .001)],
    "f1": [(.569, .008), (.596, .009), (.639, .010), (.720, .012), (.853, .012)],
}
TITLES = {"bal_acc": "balanced accuracy", "auroc": "AUROC", "precision": "precision",
          "recall": "recall", "f1": "F1"}


def mean_se(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default="investigation/results/13_eval_pairset_mdcath_per_protein.csv")
    ap.add_argument("--noevo-csv", default="investigation/results/13_eval_pairset_mdcath_noevoformer_per_protein.csv",
                    help="same run with the DCM Evoformer blocks removed (drawn if present)")
    ap.add_argument("--out", default="investigation/figures/reproduction/13_si_table6_reproduction.png")
    args = ap.parse_args()
    d = pd.read_csv(args.csv)
    d = d[d.variant == "sym"]

    def paper_protocol(df):
        """pairs |i-j| >= 3, symmetrised; balanced accuracy as sklearn computes it"""
        res = {}
        for m in PUBLISHED:
            res[m] = []
            for T in TEMPS:
                p = df[(df.k == 3) & (df.temp == T)]
                if m == "bal_acc":   # sklearn: a protein with no negatives scores its recall alone
                    res[m].append(mean_se(np.where(p.n_negative == 0, p.recall, p.bal_acc)))
                else:
                    res[m].append(mean_se(p[m]))
        return res

    ours = paper_protocol(d)
    noevo = None
    if Path(args.noevo_csv).exists():
        dn = pd.read_csv(args.noevo_csv)
        noevo = paper_protocol(dn[dn.variant == "sym"])
    allp, nonneg = {}, []
    for m in PUBLISHED:
        allp[m] = [mean_se(d[(d.k == 0) & (d.temp == T)][m]) for T in TEMPS]
    for T in TEMPS:
        nonneg.append(int((d[(d.k == 3) & (d.temp == T)].n_negative == 0).sum()))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, grid = plt.subplots(2, 3, figsize=(14, 9.4), facecolor="white")
    fig.subplots_adjust(top=0.84, bottom=0.07, left=0.06, right=0.985, wspace=0.26, hspace=0.38)
    axes = grid.ravel()[:len(PUBLISHED)]
    x = np.arange(len(TEMPS))
    for i, (ax, m) in enumerate(zip(axes, PUBLISHED)):
        pm, ps = zip(*PUBLISHED[m])
        om, os_ = zip(*ours[m])
        # three points per temperature, side by side: published, released, blocks removed
        ax.errorbar(x - 0.16, pm, yerr=ps, color=PUB, lw=2.2, marker="o", ms=8, capsize=3, zorder=3)
        ax.errorbar(x, om, yerr=os_, color=OURS, lw=0, elinewidth=1.4, marker="D", ms=8,
                    capsize=3, zorder=4, mec="white", mew=0.9)
        if noevo is not None:
            nm, ns = zip(*noevo[m])
            ax.errorbar(x + 0.16, nm, yerr=ns, color=OURS, lw=0, elinewidth=1.2, marker="D", ms=8,
                        mfc="white", mec=OURS, mew=1.6, capsize=3, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{T}" for T in TEMPS], fontsize=10, color=INK)
        ax.set_xlim(-0.55, len(TEMPS) - 0.45)
        ax.set_xlabel("temperature (K)", color=INK_2, fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel("score", color=INK_2, fontsize=11)
        ax.set_title(TITLES[m], loc="left", fontsize=12.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=9.5, length=0)
        lo = min(min(pm), min(om))
        hi = max(max(pm), max(om))
        pad = (hi - lo) * 0.18 + 0.01
        ax.set_ylim(lo - pad, min(1.0, hi + pad) if m != "recall" else 1.005)
    from matplotlib.lines import Line2D
    leg_ax = grid.ravel()[len(PUBLISHED)]          # the free sixth cell holds the legend
    leg_ax.axis("off")
    leg_ax.legend(handles=[
        Line2D([], [], color=PUB, lw=2.2, marker="o", ms=8, label="published\n(SI Table 6, mdCATH test, 270 domains)"),
        Line2D([], [], color=OURS, lw=0, marker="D", ms=8, label="released weights,\nscored with the paper's protocol"),
        ]
        + ([Line2D([], [], color=OURS, lw=0, marker="D", ms=8, mfc="white", mew=1.6,
                   label="paper's protocol, Evoformer blocks\nremoved (not retrained)")] if noevo is not None else []),
        loc="center left", frameon=False, fontsize=11, labelcolor=INK, labelspacing=1.6, handlelength=2.4)
    fig.text(0.01, 0.99, "Reproducing ESM-Dynamic's MDCath Dynamic Contact Prediction Results Across Temperatures Using Released Weights",
             fontsize=14.5, color=INK, va="top")
    fig.text(0.01, 0.955, "Released weights, each test domain predicted at full "
             "length; mean ± standard error over domains. Paper's protocol: residue pairs ≥ 3 apart in sequence, symmetrised "
             "probability > 0.5,\nbalanced accuracy per protein as scikit-learn computes it (a domain with no non-dynamic "
             f"pairs, {', '.join(str(n) for n in nonneg)} of 270 at 320–450 K, scores its recall alone). "
             "\nHollow: the Evoformer blocks deleted from every head, not retrained: identical to the solid diamonds for every domain at every temperature, to the last digit. Note 13, jobs 2199271 and 2200605.",
             fontsize=8.8, color=INK_2, va="top", linespacing=1.45)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
