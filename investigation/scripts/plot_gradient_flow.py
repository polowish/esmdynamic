"""Figure for check 1: which parameters get gradient, condition by condition.

    python investigation/scripts/plot_gradient_flow.py
        [--json investigation/results/01_minimal_autocast_repro.json]
        [--out investigation/figures/gradient_flow/01_minimal_autocast_repro.png]

One row per test condition, one column per parameter; each cell is the gradient state after
one backward pass (gets gradient / exactly zero / no gradient), labelled in words so the
colours are never the only cue.
"""
import argparse
import json
from pathlib import Path

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
STATE = {"nonzero": ("#1baf7a", "white", "gets gradient"),
         "zero": ("#cfcdc8", INK, "exactly 0"),
         "none": ("#52514e", "white", "no gradient")}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--json", default="investigation/results/01_minimal_autocast_repro.json")
    p.add_argument("--out", default="investigation/figures/gradient_flow/01_minimal_autocast_repro.png")
    args = p.parse_args()
    d = json.loads(Path(args.json).read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    a_rows = [(f"{r['dtype']}, cache {'on' if r['cache_enabled'] else 'off'}, "
               f"{'no_grad use first' if r['no_grad_use_first'] else 'no prior use'}",
               [r["weight"], r["bias"]]) for r in d["A"]]
    b_params = list(next(iter(d["B_C"].values())))
    b_rows = [(k, [v[pn] for pn in b_params]) for k, v in d["B_C"].items()]

    # stacked, with one shared column on the left for the row labels: side by side, B's long
    # condition names ran into A's grid
    fig = plt.figure(figsize=(12.5, 10.5), facecolor="white")
    axa = fig.add_axes([0.40, 0.50, 0.22, 0.38])
    axb = fig.add_axes([0.40, 0.07, 0.57, 0.30])

    def grid(ax, rows, cols, title, highlight):
        for i, (label, states) in enumerate(rows):
            for j, s in enumerate(states):
                fc, tc, word = STATE[s]
                ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=fc,
                                       edgecolor="white", lw=2))
                ax.text(j + 0.5, i + 0.5, word, ha="center", va="center", fontsize=7.6, color=tc)
        ax.set_xlim(0, len(cols))
        ax.set_ylim(len(rows), 0)
        ax.set_xticks([j + 0.5 for j in range(len(cols))])
        ax.set_xticklabels(cols, fontsize=8, color=INK)
        ax.xaxis.tick_top()
        ax.set_yticks([i + 0.5 for i in range(len(rows))])
        ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
        for i in highlight:
            ax.get_yticklabels()[i].set_fontweight("bold")
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0)
        ax.set_title(title, loc="left", fontsize=10.2, color=INK, pad=26)

    grid(axa, a_rows, ["weight", "bias"],
         "A   One nn.Linear: a no_grad use, then a grad use,\n      in one autocast region",
         [i for i, r in enumerate(d["A"]) if r["cache_enabled"] and r["no_grad_use_first"]])
    grid(axb, b_rows, [pn.replace(".", "\n") for pn in b_params],
         "B   The DCM's recycling pattern: 4 passes, no_grad on the first 3,\n      "
         "zero-initialised output layer   ·   C   and three fixes", [0])

    fig.text(0.02, 0.985, "The autocast weight cache stops gradient after a no_grad use",
             color=INK, fontsize=13.5, ha="left", va="top")
    fig.text(0.02, 0.957, f"Pure PyTorch, no ESMDynamic code  ·  torch {d['torch']}  ·  "
             f"{d['device']}", color=INK_2, fontsize=9, ha="left", va="top")
    fig.text(0.02, 0.02, "Bold rows reproduce ESMDynamic's training setup. In B/C, 'exactly 0' for "
             "inp and the LayerNorm is expected on step 1:\nthey sit behind the zero-initialised "
             "output layer and move once it does.", color=INK_2, fontsize=8.4, ha="left",
             va="bottom")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
