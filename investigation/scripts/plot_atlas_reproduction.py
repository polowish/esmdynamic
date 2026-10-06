"""Figure for check 6: the released weights reproduce the preprint's ATLAS-test result.

    python investigation/scripts/plot_atlas_reproduction.py
        [--json investigation/results/06_atlas_reproduction.json]
        [--csv investigation/results/06_atlas_reproduction_per_protein.csv]
        [--out investigation/figures/reproduction/06_atlas_reproduction.png]

A  per-protein balanced accuracy (mean +- se over 82 chains) for each of the model's five
   temperature outputs, against the preprint's 87%
B  the per-protein values behind the 320 K mean, by chain length
"""
import argparse
import csv
import json
from pathlib import Path

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
ORANGE = "#eb6834"
TEMPS = [320, 348, 379, 413, 450]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", default="investigation/results/06_atlas_reproduction.json")
    p.add_argument("--csv", default="investigation/results/06_atlas_reproduction_per_protein.csv")
    p.add_argument("--out", default="investigation/figures/reproduction/06_atlas_reproduction.png")
    args = p.parse_args()
    d = json.loads(Path(args.json).read_text())
    rows = list(csv.DictReader(open(args.csv)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.0), facecolor="white",
                               gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28})
    means = [d["summary"][f"{t}K_all"]["per_protein_mean"] for t in TEMPS]
    ses = [d["summary"][f"{t}K_all"]["per_protein_se"] for t in TEMPS]
    a.axhline(0.87, color=INK_2, lw=1.3, ls=(0, (6, 3)), zorder=1)
    a.text(4.35, 0.875, "preprint: 87%", ha="right", va="bottom", fontsize=8.5, color=INK_2)
    a.errorbar(range(5), means, yerr=ses, fmt="o", ms=8, color=ORANGE, ecolor=ORANGE,
               elinewidth=1.4, capsize=3, zorder=3, markeredgecolor="white")
    for i, (m, s) in enumerate(zip(means, ses)):
        a.annotate(f"{m:.3f}", (i, m), xytext=(10, -2), textcoords="offset points",
                   fontsize=8.2, color=INK_2, va="center")
    a.set_xticks(range(5))
    a.set_xticklabels([f"{t} K" for t in TEMPS], fontsize=8.5, color=INK)
    a.set_xlim(-0.5, 4.6)
    a.set_ylim(0.5, 0.95)
    a.set_xlabel("model temperature output (ATLAS ensembles are at 300 K)", color=INK_2, fontsize=9)
    a.set_ylabel("balanced accuracy, mean over 82 proteins (+-1 se)", color=INK_2, fontsize=9)
    a.set_title("A   Released V2 weights on the ATLAS test split", loc="left", color=INK,
                fontsize=10.5, pad=8)
    style(a)

    L = [int(r["length"]) for r in rows]
    ba = [float(r["bal_acc_320K_all"]) for r in rows]
    b.axhline(0.87, color=INK_2, lw=1.3, ls=(0, (6, 3)), zorder=1)
    b.axhline(d["summary"]["320K_all"]["per_protein_mean"], color=ORANGE, lw=1.3, zorder=1)
    b.scatter(L, ba, s=26, color=ORANGE, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
    b.text(max(L), d["summary"]["320K_all"]["per_protein_mean"] + 0.006,
           f"mean {d['summary']['320K_all']['per_protein_mean']:.3f}", ha="right", va="bottom",
           fontsize=8.2, color=ORANGE)
    b.text(max(L), 0.87 - 0.006, "preprint 87%", ha="right", va="top", fontsize=8.2, color=INK_2)
    b.set_ylim(0.45, 1.0)
    b.set_xlabel("chain length (residues)", color=INK_2, fontsize=9)
    b.set_ylabel("balanced accuracy at 320 K, per protein", color=INK_2, fontsize=9)
    b.set_title("B   The 82 per-protein values behind the 320 K mean", loc="left", color=INK,
                fontsize=10.5, pad=8)
    style(b)

    fig.text(0.01, 1.04, "ESMDynamic's published ATLAS result is reproduced by the released weights, "
             "whose DCM blocks are untrained", color=INK, fontsize=12.5, ha="left", va="bottom")
    fig.text(0.01, 0.995, "Dynamic-contact classification, prediction = probability > 0.5, all residue "
             "pairs (upper-triangle-only gives the same to 3 decimals). Data: Illinois Data Bank "
             "atlas_test.zip.", color=INK_2, fontsize=8.4, ha="left", va="bottom")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
