"""Figure: test proteins that were also in RCSB pre-training, and whether that raised their scores.

    python investigation/scripts/plot_leakage.py            # captioned -> figures/leakage/
    python investigation/scripts/plot_leakage.py --clean    # transparent, no title/caption, larger type
                                                            #   -> figures/transparent/

A  each test set split three ways: in RCSB pre-training (an RCSB-train cluster), an RCSB cluster
   that was not in pre-training (val / test split, or no dynamic contacts so in no split), and no
   RCSB match. mdCATH: the authors' mdcath_to_rcsb_mapping.csv; ATLAS: exact or contained sequence
   match to an RCSB consensus (a lower bound).
B  released weights, mdCATH test, balanced accuracy per group at each temperature (the paper's
   protocol: |i - j| >= 3, symmetrised probability, scikit-learn style), mean +- se over domains.
C  released weights, ATLAS test, balanced accuracy for the same three groups (train.py protocol).
Data: results/15_pretrain_overlap_members.csv, 13_eval_pairset_mdcath_per_protein.csv,
12_eval_protocol_atlas_per_protein.csv; analysis in leakage_effect.py / results/15_leakage_effect.json.
Arial (Open Sans is installed only as a variable font, which matplotlib cannot load).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
PRE, CLU, NONE = "#1D3BE2", "#1DE2C4", "#a8a7a2"
TEMPS = [320, 348, 379, 413, 450]
GROUPS = [("pre", "in RCSB pre-training", PRE),
          ("clu", "RCSB cluster, not in pre-training", CLU),
          ("none", "no RCSB match", NONE)]
SPLIT_TO_GROUP = {"train": "pre", "train,val": "pre", "val": "clu", "test": "clu", "not_in_any_split": "clu",
                  "no_split": "clu", "unmapped": "none", "no_match": "none"}


def mean_se(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ("investigation/figures/transparent/15_pretrain_leakage.png" if args.clean
                            else "investigation/figures/leakage/15_pretrain_leakage.png"))
    fs = 1.45 if args.clean else 1.0

    mem = pd.read_csv("investigation/results/15_pretrain_overlap_members.csv")
    mem["group"] = mem.rcsb_split.map(SPLIT_TO_GROUP)
    assert mem.group.notna().all(), mem[mem.group.isna()]
    md = pd.read_csv("investigation/results/13_eval_pairset_mdcath_per_protein.csv")
    md = md[(md.variant == "sym") & (md.k == 3)].copy()
    md["bal_acc"] = np.where(md.n_negative == 0, md.recall, md.bal_acc)
    md = md.merge(mem[mem.dataset == "mdcath"][["id", "group"]], on="id")
    at = pd.read_csv("investigation/results/12_eval_protocol_atlas_per_protein.csv")
    at = at[(at.protocol == "crop256") & (at.variant == "unsym") & (at.temp == 320)]
    at = at.groupby("id", as_index=False)["bal_acc"].mean().merge(
        mem[mem.dataset == "atlas"][["id", "group"]], on="id")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(15, 6.2 if args.clean else 7.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.25, 0.9], wspace=0.34,
                          left=0.12 if args.clean else 0.10, right=0.985, top=0.80 if args.clean else 0.72, bottom=0.12)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    def style(ax, grid="y"):
        ax.grid(axis=grid, color=GRID, lw=0.7, zorder=0); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=10 * fs, length=0)
        if args.clean:
            ax.set_facecolor("none")

    # A: the overlap
    for row, (ds, name) in enumerate((("mdcath", "mdCATH test"), ("atlas", "ATLAS test"))):
        d = mem[mem.dataset == ds]
        n = len(d); left = 0
        for g, _, col in GROUPS:
            k = int((d.group == g).sum())
            axA.barh(row, k / n * 100, left=left, color=col, height=0.55, zorder=3, edgecolor="none")
            if k:
                narrow = k / n < 0.15         # small segment: smaller type so the label stays inside
                axA.text(left + k / n * 50, row, f"{k}\n({100 * k / n:.0f}%)", ha="center", va="center",
                         fontsize=10.5 * fs * (0.78 if narrow else 1.0),
                         color="white" if g == "pre" else INK, linespacing=1.1)
            left += k / n * 100
        axA.text(-2, row, f"{name}\n{n} proteins", ha="right", va="center", fontsize=11 * fs, color=INK)
    axA.set_yticks([]); axA.set_ylim(-0.6, 1.6); axA.invert_yaxis()
    axA.set_xlim(0, 100); axA.set_xticks([0, 25, 50, 75, 100])
    axA.set_xlabel("test proteins (%)", fontsize=11 * fs, color=INK_2)
    axA.set_title("overlap with pre-training", loc="left", fontsize=12.5 * fs, color=INK, pad=10)
    style(axA, "x")

    # B: mdCATH balanced accuracy per group and temperature
    x = np.arange(len(TEMPS))
    for j, (g, _, col) in enumerate(GROUPS):
        stats = [mean_se(md[(md.temp == T) & (md.group == g)].bal_acc) for T in TEMPS]
        mu, se, nn = zip(*stats)
        axB.errorbar(x + (j - 1) * 0.15, mu, yerr=se, color=col, lw=2.0, marker="o", ms=7, capsize=3,
                     zorder=3 + j, label=f"n = {nn[0]}")
    axB.set_xticks(x); axB.set_xticklabels([str(T) for T in TEMPS])
    axB.set_xlim(-0.5, len(TEMPS) - 0.5)
    axB.set_xlabel("temperature (K)", fontsize=11 * fs, color=INK_2)
    axB.set_ylabel("balanced accuracy", fontsize=11 * fs, color=INK_2)
    axB.set_title("mdCATH test, released weights", loc="left", fontsize=12.5 * fs, color=INK, pad=10)
    style(axB)

    # C: ATLAS balanced accuracy for the same three groups
    for i, (g, lab, col) in enumerate(GROUPS):
        mu, se, nn = mean_se(at[at.group == g].bal_acc)
        axC.bar(i, mu, width=0.7, color=col, zorder=3, edgecolor="none")
        axC.errorbar(i, mu, yerr=se, color=INK, capsize=4, elinewidth=1.1, zorder=4)
        axC.text(i, mu + se + 0.015, f"{mu:.3f}", ha="center", va="bottom", fontsize=10.5 * fs, color=INK)
        axC.text(i, 0.03, f"n = {nn}", ha="center", va="bottom", fontsize=10 * fs,
                 color="white" if g == "pre" else INK)
    axC.set_xticks([])
    axC.set_ylim(0, 1); axC.set_xlim(-0.6, 2.6)
    axC.set_ylabel("balanced accuracy", fontsize=11 * fs, color=INK_2)
    axC.set_title("ATLAS test" if args.clean else "ATLAS test, released weights", loc="left", fontsize=12.5 * fs, color=INK, pad=10)
    style(axC)

    fig.legend(handles=[Line2D([], [], color=c, marker="s", lw=0, ms=12, label=l) for _, l, c in GROUPS],
               loc="upper center", bbox_to_anchor=(0.5, 0.97 if args.clean else 0.86), ncol=3, frameon=False,
               fontsize=11 * fs, labelcolor=INK, columnspacing=2.2, handletextpad=0.5)
    if not args.clean:
        fig.text(0.01, 0.99, "Half of each test set was also in ESMDynamic's RCSB pre-training, but those proteins "
                 "score no better than comparable ones that were not", fontsize=14.5, color=INK, va="top")
        fig.text(0.01, 0.945, "A: mdCATH test domains mapped by the authors' mdcath_to_rcsb_mapping.csv; ATLAS chains by "
                 "exact or contained sequence match to an RCSB consensus (a lower bound). B, C: released weights under the "
                 "protocols that reproduce the paper.\nIn pre-training (dark blue) vs an RCSB cluster not in pre-training "
                 "(teal): no difference at any temperature (320 K: +0.008, 95% CI -0.034 to +0.053). The gap to proteins with "
                 "no RCSB match (grey) is protein type\n(well studied, many PDB structures), not exposure. "
                 "ATLAS: no ordering by exposure (the teal group is 10 chains). "
                 "Mean ± standard error over proteins. results/15_leakage_effect.json.",
                 fontsize=9, color=INK_2, va="top", linespacing=1.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.clean:
        fig.savefig(out, dpi=170, transparent=True)
    else:
        fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
