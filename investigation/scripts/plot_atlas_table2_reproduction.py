"""Figure: the released V2 weights reproduce Table 2 (ATLAS test, 82 chains).

    python investigation/scripts/plot_atlas_table2_reproduction.py
        [--out investigation/figures/reproduction/13_atlas_table2_reproduction.png]

Bars per Table 2 metric, in the same colours as the mdCATH figure
(plot_si_table6_reproduction.py):

  blue    published, Table 2
  orange  released weights, scored with the protocol that reproduces the paper. For ATLAS that is
          train.py's own validation scoring: 256-residue random crops (5 draws per long chain),
          sigmoid of the unsymmetrised logits > 0.5, all pairs of the crop -- the only protocol
          that matches all six metrics (note 12)
  grey    released weights, full-length chains, symmetrised probability > 0.5, all residue pairs
          (the same definition as the mdCATH figure's grey line)

Mean ± standard error over the 82 chains; 320 K output against the 300 K ATLAS ensembles.
From results/12_eval_protocol_atlas.json (note 12, job 2158167).
"""
import argparse
import json
from pathlib import Path

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
PUB, OURS, ALL = "#2a78d6", "#eb6834", "#a8a7a2"     # as in the mdCATH figure
METRICS = [("bal_acc", "balanced accuracy"), ("precision", "precision"), ("recall", "recall"),
           ("f1", "F1"), ("auroc", "AUROC"), ("rmse", "occupancy RMSE")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="investigation/results/12_eval_protocol_atlas.json")
    ap.add_argument("--noevo", default="investigation/results/12_eval_protocol_atlas_noevoformer.json",
                    help="same run with the DCM Evoformer blocks removed (drawn if present)")
    ap.add_argument("--out", default="investigation/figures/reproduction/13_atlas_table2_reproduction.png")
    args = ap.parse_args()
    r = json.loads(Path(args.results).read_text())
    pub = r["published"]["320"]
    paper_protocol = r["summary"]["crop256/unsym/320K"]
    all_pairs = r["summary"]["full/sym/320K"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 6, figsize=(17, 4.6), facecolor="white")
    fig.subplots_adjust(top=0.70, bottom=0.08, left=0.035, right=0.995, wspace=0.38)
    series = [(PUB, None, lambda m: tuple(pub[m])),
              (OURS, None, lambda m: (paper_protocol[m]["mean"], paper_protocol[m]["se"]))]
    noevo = None
    if Path(args.noevo).exists():
        noevo = json.loads(Path(args.noevo).read_text())["summary"]["crop256/unsym/320K"]
        series.append(("white", OURS, lambda m: (noevo[m]["mean"], noevo[m]["se"])))
    for ax, (m, title) in zip(axes, METRICS):
        vals = [get(m) for _, _, get in series]
        top = max(v + se for v, se in vals)
        for i, ((col, hatch_col, _), (v, se)) in enumerate(zip(series, vals)):
            if hatch_col:   # no-Evoformer: the orange protocol, hatched
                ax.bar(i, v, width=0.72, color="white", zorder=3, edgecolor=hatch_col, linewidth=1.4, hatch="////")
            else:
                ax.bar(i, v, width=0.72, color=col, zorder=3, edgecolor="white", linewidth=1.5)
            ax.errorbar(i, v, yerr=se, color=INK, capsize=3, elinewidth=1, zorder=4)
            ax.text(i, v + se + top * 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8.6, color=INK)
        ax.set_ylim(0, top * 1.16)
        ax.set_xticks([])
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.3, length=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=PUB, label="published (Table 2, ATLAS test, 82 chains)"),
                        Patch(color=OURS, label="released V2 weights, scored with the protocol that reproduces the paper"),
                        ]
               + ([Patch(fc="white", ec=OURS, hatch="////", label="protocol as orange, Evoformer blocks removed (not retrained)")]
                  if noevo is not None else []),
               loc="upper left", bbox_to_anchor=(0.035, 0.86), ncol=4 if noevo is not None else 3, frameon=False,
               fontsize=8.6 if noevo is not None else 9.2, labelcolor=INK)
    fig.text(0.01, 0.985, "The downloaded ESMDynamic weights reproduce Table 2 (ATLAS)", fontsize=14.5, color=INK, va="top")
    fig.text(0.01, 0.93, "Released V2 weights (Illinois Data Bank 7odsk, sha256 b41a1a22…), 320 K output against the 300 K "
             "ATLAS ensembles; mean ± standard error over 82 chains. Orange: scored as train.py scores (256-residue random "
             "crops,\nraw logits > 0.5), the only protocol that matches all six metrics within one standard error. "
             "Hatched: the Evoformer blocks deleted from every head, not retrained: identical to orange for every chain, to the last digit. Note 12, jobs 2158167 and 2200604.",
             fontsize=8.8, color=INK_2, va="top", linespacing=1.45)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
