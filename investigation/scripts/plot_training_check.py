"""Figure for check 7: the matched short training, autocast cache on (the bug) vs off (the fix).

    python investigation/scripts/plot_training_check.py
        [--on investigation/results/07_training_check_cache_on.json]
        [--off investigation/results/07_training_check_cache_off.json]
        [--out investigation/figures/training_check/07_training_check.png]

A  Frobenius norm of every DCM block's zero-initialised output layers, per step
B  training loss per step (same batches in both runs), with a 25-step running mean
C  validation before and after: each head's loss (relative to its starting value) and the
   dynamic head's balanced accuracy
"""
import argparse
import json
from pathlib import Path

import numpy as np

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BUG, FIX = "#eb6834", "#1baf7a"     # orange = as released (cache on), green = fixed (cache off)


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


def running_mean(x, k=25):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full_like(x, np.nan)
    out[k - 1:] = (c[k:] - c[:-k]) / k
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--on", default="investigation/results/07_training_check_cache_on.json")
    p.add_argument("--off", default="investigation/results/07_training_check_cache_off.json")
    p.add_argument("--out", default="investigation/figures/training_check/07_training_check.png")
    args = p.parse_args()
    runs = {"as released: autocast cache on": (json.loads(Path(args.on).read_text()), BUG),
            "fixed: cache_enabled=False": (json.loads(Path(args.off).read_text()), FIX)}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(16.5, 4.9), facecolor="white",
                                  gridspec_kw={"width_ratios": [1, 1.15, 1.25], "wspace": 0.32})
    for label, (r, col) in runs.items():
        steps = [t["step"] for t in r["train"]]
        a.plot([0] + steps, [r["before"]["block_output_norm"]] + [t["block_output_norm"] for t in r["train"]],
               color=col, lw=2, label=label, zorder=3)
        loss = [t["loss"] for t in r["train"]]
        b.plot(steps, loss, color=col, lw=0.8, alpha=0.25, zorder=2)
        b.plot(steps, running_mean(loss), color=col, lw=2, label=label, zorder=3)
    a.set_xlabel("optimiser step", color=INK_2, fontsize=9)
    a.set_ylabel("norm of the DCM blocks' zero-initialised output layers", color=INK_2, fontsize=9)
    a.set_title("A   Do the blocks leave initialisation?", loc="left", color=INK, fontsize=10.5, pad=8)
    a.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    style(a)
    b.set_xlabel("optimiser step (identical batches in both runs)", color=INK_2, fontsize=9)
    b.set_ylabel("training loss (thin: per step; thick: 25-step mean)", color=INK_2, fontsize=9)
    b.set_title("B   Training loss", loc="left", color=INK, fontsize=10.5, pad=8)
    style(b)

    keys = [("val_loss_dynamic_logits", "dynamic loss"), ("val_loss_kinetic_logits", "kinetic loss"),
            ("val_loss_frequency_pred", "frequency loss")]
    names = [k[1] for k in keys] + ["dynamic bal. acc."]
    x = np.arange(len(names))
    w = 0.36
    for k, (label, (r, col)) in enumerate(runs.items()):
        vals = [100 * (r["after"][kk] / r["before"][kk] - 1) for kk, _ in keys]
        vals.append(100 * (r["after"]["val_dynamic_bal_acc"] - r["before"]["val_dynamic_bal_acc"]))
        bars = c.bar(x + (k - 0.5) * w, vals, width=w - 0.03, color=col, label=label, zorder=3)
        for bar, v in zip(bars, vals):
            c.annotate(f"{v:+.1f}", (bar.get_x() + bar.get_width() / 2, v),
                       xytext=(0, 3 if v >= 0 else -10), textcoords="offset points",
                       ha="center", fontsize=7.8, color=INK_2)
    c.axhline(0, color=MUTED, lw=1)
    c.set_xticks(x)
    c.set_xticklabels(names, fontsize=8.5, color=INK)
    c.set_ylabel("change after training: loss in %, balanced accuracy in points", color=INK_2, fontsize=9)
    c.set_title(f"C   Validation, before vs after ({runs['fixed: cache_enabled=False'][0]['steps']} steps)",
                loc="left", color=INK, fontsize=10.5, pad=8)
    style(c)

    r0 = runs["fixed: cache_enabled=False"][0]
    fig.text(0.01, 1.05, "With the fix, ESMDynamic's DCM blocks start training; as released they never "
             "move", color=INK, fontsize=12.5, ha="left", va="bottom")
    fig.text(0.01, 1.005, f"Both runs start from the released V2 weights and see identical mdCATH "
             f"batches ({r0['batch']} x {r0['accum']} accumulated samples per step, 256-residue crops); "
             f"Adam 1e-4, bf16 autocast, ESMFold frozen; the only difference is autocast's weight cache. "
             f"{r0['device']}.", color=INK_2, fontsize=8.4, ha="left", va="bottom")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
