"""Figure: how much of each released head the authors actually trained (transparent, Arial).

    python investigation/scripts/plot_param_pies.py
        [--counts investigation/results/15_param_counts.json]
        [--out investigation/figures/transparent/15_trained_vs_untrained_params.png]

One pie per head of the released V2 weights: trained parameters (teal) vs parameters still at
their initial values (dark blue: every DCM Evoformer block tensor, plus the occupancy head's
single-state path, which reaches its loss only through the blocks). Counts from
param_counts.py (results/15_param_counts.json). Same colours, font and transparent style as the
figures/transparent reproduction figures; no figure title or caption.
"""
import argparse
import json
import math
from pathlib import Path

INK, INK_2 = "#0b0b0b", "#52514e"
UNTRAINED, TRAINED = "#1D3BE2", "#1DE2C4"
HEADS = [("dynamic", "Dynamic contact head"), ("kinetic", "Kinetics head"), ("frequency", "Occupancy head")]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--counts", default="investigation/results/15_param_counts.json")
    ap.add_argument("--out", default="investigation/figures/transparent/15_trained_vs_untrained_params.png")
    args = ap.parse_args()
    c = json.loads(Path(args.counts).read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    fig.subplots_adjust(left=0.01, right=0.90, top=0.80, bottom=0.17, wspace=0.62)
    for ax, (key, name) in zip(axes, HEADS):
        h = c[key]
        vals = [h["trained"], h["untrained"]]
        # trained slice first, centred at 3 o'clock (so its label sits to the right, clear of the titles)
        half = 180 * h["trained"] / h["total"]
        wedges, _ = ax.pie(vals, colors=[TRAINED, UNTRAINED], startangle=half, counterclock=False,
                           wedgeprops=dict(linewidth=0))
        ax.set_aspect("equal")
        for w, v, col in zip(wedges, vals, ("trained", "untrained")):
            pct = 100 * v / h["total"]
            label = f"{v:,}\n({pct:.1f}%)" if pct >= 1 else f"{v:,}\n({pct:.2f}%)"
            mid = math.radians((w.theta1 + w.theta2) / 2)
            x, y = math.cos(mid), math.sin(mid)
            if col == "untrained":          # large slice: label inside, white
                ax.text(0.5 * x, 0.5 * y, label, ha="center", va="center", fontsize=15,
                        color="white", fontweight="bold")
            else:                           # small slice: label outside with a leader line
                ax.annotate(label, xy=(0.98 * x, 0.98 * y), xytext=(1.22, 0.0),
                            ha="left", va="center", fontsize=15, color=INK, annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=INK_2, lw=1.2, shrinkA=0, shrinkB=0))
        ax.set_title(name, fontsize=20, color=INK, pad=36)
        ax.text(0.5, 1.03, f"{h['total']:,} parameters", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=15, color=INK_2)
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    fig.legend(handles=[Patch(color=TRAINED, label="trained"),
                        Patch(color=UNTRAINED, label="untrained (still at initialisation)")],
               loc="lower center", ncol=2, frameon=False, fontsize=16, labelcolor=INK,
               handlelength=1.6, columnspacing=2.4, bbox_to_anchor=(0.5, 0.02))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, transparent=True)
    print(out)


if __name__ == "__main__":
    main()
