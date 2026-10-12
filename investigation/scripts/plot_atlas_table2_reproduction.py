"""Figure: the released V2 weights reproduce Table 2 (ATLAS test, 82 chains) under every scoring protocol.

    python investigation/scripts/plot_atlas_table2_reproduction.py
        [--out investigation/figures/reproduction/13_atlas_table2_reproduction.png]

ATLAS has one MD temperature (300 K; scored against the model's 320 K output), so instead of
temperature the x-axis is the scoring protocol. Each panel: the published value as a band
(± 1 se), the released weights under each protocol as points (± 1 se):

  full / sym           whole chain, symmetrised probability > 0.5, all residue pairs
  full / logits        whole chain, sigmoid of the unsymmetrised logits > 0.5
  crop / sym, logits   train.py's 256-residue random crop (5 draws per long chain)
  full / sym, ≥3       whole chain, pairs |i - j| >= 3 (the mdCATH SI Table 6 protocol, note 13)
  full / logits, ≥3

From results/12_eval_protocol_atlas.json (note 12, job 2158167) and
results/13_eval_pairset_atlas.json (note 13, job 2199270). The pair-set runs did not score
occupancy RMSE, so that panel has the first four protocols only.
"""
import argparse
import json
from pathlib import Path

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
PUB, OURS, PUB_BAND = "#2a78d6", "#eb6834", "#dbe7f7"
METRICS = [("bal_acc", "balanced accuracy"), ("precision", "precision"), ("recall", "recall"),
           ("f1", "F1"), ("auroc", "AUROC"), ("rmse", "occupancy RMSE")]
PROTOCOLS = [  # label, file, key
    ("1", "12", "full/sym/320K"),
    ("2", "12", "full/unsym/320K"),
    ("3", "12", "crop256/sym/320K"),
    ("4", "12", "crop256/unsym/320K"),
    ("5", "13", "sym/k3/320K"),
    ("6", "13", "unsym/k3/320K"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="investigation/results")
    ap.add_argument("--out", default="investigation/figures/reproduction/13_atlas_table2_reproduction.png")
    args = ap.parse_args()
    res = Path(args.results)
    r12 = json.loads((res / "12_eval_protocol_atlas.json").read_text())
    r13 = json.loads((res / "13_eval_pairset_atlas.json").read_text())
    src = {"12": r12["summary"], "13": r13["summary"]}
    pub = r12["published"]["320"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 6, figsize=(18, 4.8), facecolor="white")
    fig.subplots_adjust(top=0.74, bottom=0.22, left=0.035, right=0.995, wspace=0.34)
    for ax, (m, title) in zip(axes, METRICS):
        pm, ps = pub[m]
        ax.axhspan(pm - ps, pm + ps, color=PUB_BAND, zorder=0, lw=0)
        ax.axhline(pm, color=PUB, lw=2, zorder=1)
        xs, ys = [], []
        for i, (lab, f, key) in enumerate(PROTOCOLS):
            cell = src[f].get(key, {}).get(m)
            if cell is None:
                continue
            ax.errorbar(i, cell["mean"], yerr=cell["se"], color=OURS, marker="D", ms=6.5, lw=0,
                        elinewidth=1.3, capsize=3, mec="white", mew=0.8, zorder=3)
            xs.append(i); ys.append(cell["mean"])
        ax.set_xticks(range(len(PROTOCOLS)))
        ax.set_xticklabels([p[0] for p in PROTOCOLS], fontsize=9.5, color=INK, rotation=0)
        ax.set_xlim(-0.6, len(PROTOCOLS) - 0.4)
        lo, hi = min(ys + [pm - ps]), max(ys + [pm + ps])
        pad = (hi - lo) * 0.45 + 0.004
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.text(0.98, 0.98, f"published\n{pm:.3f} ± {ps:.3f}", transform=ax.transAxes, fontsize=8, color=PUB,
                va="top", ha="right", bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.3, length=0)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[Line2D([], [], color=PUB, lw=2, label="published (Table 2, ATLAS test, 82 chains)"),
                        Patch(color=PUB_BAND, label="± 1 standard error"),
                        Line2D([], [], color=OURS, lw=0, marker="D", ms=6.5,
                               label="released V2 weights, per scoring protocol (± 1 se)")],
               loc="upper left", bbox_to_anchor=(0.035, 0.885), ncol=3, frameon=False, fontsize=9.2,
               labelcolor=INK)
    fig.text(0.01, 0.985, "The downloaded ESMDynamic weights reproduce Table 2 (ATLAS): scored as train.py scores, all six metrics match",
             fontsize=14.5, color=INK, va="top")
    fig.text(0.01, 0.935, "Released V2 weights (Illinois Data Bank 7odsk, sha256 b41a1a22…), 320 K output against the "
             "300 K ATLAS ensembles; mean ± standard error over 82 chains. Balanced accuracy and AUROC match under every protocol; "
             "precision, recall and F1 depend on it, and protocol 4\n(train.py's own validation: crops, raw logits) matches "
             "all six. "
             "Excluding near-neighbour pairs (5, 6) barely matters here, unlike mdCATH at high temperature: few ATLAS pairs "
             "are dynamic. Notes 12 and 13, jobs 2158167 and 2199270.",
             fontsize=8.8, color=INK_2, va="top", linespacing=1.45)
    fig.text(0.035, 0.035, "scoring protocols:   1  full length, symmetrised probability    2  full length, raw logits    "
             "3  256-residue crops, symmetrised    4  256-residue crops, raw logits (train.py's own validation)    "
             "5  full, symmetrised, pairs ≥ 3 apart    6  full, logits, pairs ≥ 3 apart", fontsize=8.8, color=INK_2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
