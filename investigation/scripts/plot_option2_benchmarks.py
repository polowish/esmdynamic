"""Figures: option 2 (all heads fine-tuned from V2, bug fixed vs bug kept) on the mdCATH and ATLAS tests.

    python investigation/scripts/plot_option2_benchmarks.py            # captioned, figures/retraining/
    python investigation/scripts/plot_option2_benchmarks.py --clean    # transparent, no title/caption,
                                                                       # Arial, score axes 0-1 (figures/transparent/)

From results/17_eval_checkpoints_{mdcath,atlas}.json (eval_checkpoints.py): ESMFold once per test
protein, every model's heads on that same structure, so the differences are paired per protein.
Each benchmark is drawn under the protocol that reproduces the paper's own numbers from the
released weights (notes 12-13):
  mdCATH (270 domains, all 5 temperatures)  si6: full length, symmetrised probability, pairs
                                            |i - j| >= 3 (SI Tables 6-7)
  ATLAS  (82 chains, 300 K MD; 320 K output) t12: 256-residue crops (5 per long chain), logits,
                                            all pairs (Table 2)
Top: mean +- se over proteins. Bottom: each fine-tune minus the released V2 weights, per protein,
with a protein-bootstrap 95% CI; solid = all test proteins, hollow = only those whose RCSB
cluster was not in the authors' pre-training split. Occupancy RMSE: lower is better.
"""
import argparse
import json
from pathlib import Path

import numpy as np

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
MODELS = [  # (key in the results, legend label, colour)
    ("v2-released", "released V2 (bug in training)", "#a8a7a2"),
    ("v2ft-all-heads-bug-s0", "fine-tuned, bug kept (control)", "#1DE2C4"),
    ("v2ft-all-heads-fixed-s0", "fine-tuned, bug fixed", "#1D3BE2"),
]
REF = "v2-released"
TEMPS = [320, 348, 379, 413, 450]
TITLES = {"auroc": "AUROC", "bal_acc": "balanced accuracy", "rmse": "occupancy RMSE (lower is better)",
          "precision": "precision", "recall": "recall", "f1": "F1"}


def load(path):
    return json.loads(Path(path).read_text())["summary"]


def style(ax, fs, clean):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK, labelsize=10 * fs, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def mdcath_figure(S, out, clean, fs):
    import matplotlib.pyplot as plt
    metrics = ["auroc", "bal_acc", "rmse"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2 if not clean else 8.6), facecolor="white",
                             gridspec_kw=dict(height_ratios=[1.25, 1]))
    fig.subplots_adjust(top=0.83 if not clean else 0.93, bottom=0.08, left=0.07, right=0.985,
                        hspace=0.42, wspace=0.26)
    x = np.arange(len(TEMPS))
    for c, m in enumerate(metrics):
        ax = axes[0, c]
        for k, (key, label, col) in enumerate(MODELS):
            mu = [S[f"si6/{T}K/all"][key][m]["mean"] for T in TEMPS]
            se = [S[f"si6/{T}K/all"][key][m]["se"] for T in TEMPS]
            ax.errorbar(x + (k - 1) * 0.12, mu, yerr=se, color=col, marker="o", ms=6, lw=2.2,
                        capsize=3, label=label, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels(TEMPS)
        ax.set_xlabel("temperature (K)", color=INK_2, fontsize=10.5 * fs)
        ax.set_title(TITLES[m], loc="left", fontsize=12.5 * fs, color=INK, pad=10 if clean else 6)
        if m != "rmse":
            ax.set_ylim(0, 1) if clean else ax.set_ylim(0.45, 1.0)
        else:
            ax.set_ylim(0, 0.09)
        style(ax, fs, clean)
        # bottom: paired difference vs released V2
        ax = axes[1, c]
        for k, (key, label, col) in enumerate(MODELS[1:]):
            for sub, fill, dx in (("all", col, -0.09), ("not_in_rcsb_train", "none", 0.09)):
                d = [S[f"si6/{T}K/{sub}"][key][m][f"vs_{REF}"] for T in TEMPS]
                ax.errorbar(x + dx + (k - 0.5) * 0.26, [v["diff"] for v in d],
                            yerr=[[v["diff"] - v["lo"] for v in d], [v["hi"] - v["diff"] for v in d]],
                            fmt="o", ms=6.5, color=col, mfc=fill, mew=1.8, capsize=3, lw=1.8, zorder=3)
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_xticks(x); ax.set_xticklabels(TEMPS)
        ax.set_xlabel("temperature (K)", color=INK_2, fontsize=10.5 * fs)
        ax.set_title(f"{'change in ' if not clean else 'Δ '}{TITLES[m].split(' (')[0]} vs released V2",
                     loc="left", fontsize=11.5 * fs, color=INK, pad=8)
        style(ax, fs, clean)
    axes[0, 0].set_ylabel("score (mean ± se over domains)", color=INK_2, fontsize=11 * fs)
    axes[1, 0].set_ylabel("paired difference (95% CI)", color=INK_2, fontsize=11 * fs)
    from matplotlib.lines import Line2D
    h, l = axes[0, 0].get_legend_handles_labels()
    h += [Line2D([], [], color=INK_2, marker="o", ls="", ms=7, label="all 270 test domains"),
          Line2D([], [], color=INK_2, marker="o", mfc="none", ls="", ms=7, mew=1.8,
                 label="domains not in RCSB pre-training")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.53, 0.995 if clean else 0.905), ncol=5,
               frameon=False, fontsize=10 * fs, labelcolor=INK, handlelength=2.0, columnspacing=1.6)
    if not clean:
        fig.text(0.07, 0.985, "Fine-tuning ESMDynamic with working DCM blocks: mdCATH test",
                 fontsize=15, color=INK, va="top")
        fig.text(0.07, 0.95, "All heads fine-tuned from the released V2 weights on mdCATH train, seed 0, "
                 "best-validation checkpoint; identical data, start and schedule in both arms, the only "
                 "difference being the autocast\nweight-cache bug. Scored as SI Table 6 (full length, "
                 "symmetrised probability, |i − j| ≥ 3), each output against its own temperature. Bottom: "
                 "per-domain paired difference, protein bootstrap.", fontsize=9, color=INK_2, va="top",
                 linespacing=1.45)
    save(fig, out, clean)


def atlas_figure(S, out, clean, fs):
    import matplotlib.pyplot as plt
    metrics = ["auroc", "bal_acc", "precision", "recall", "f1", "rmse"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2 if not clean else 5.8), facecolor="white",
                             gridspec_kw=dict(width_ratios=[1.35, 1]))
    fig.subplots_adjust(top=0.74 if not clean else 0.84, bottom=0.17, left=0.06, right=0.985, wspace=0.18)
    key0 = "t12/320K"
    x = np.arange(len(metrics)); w = 0.26
    ax = axes[0]
    for k, (key, label, col) in enumerate(MODELS):
        mu = [S[f"{key0}/all"][key][m]["mean"] for m in metrics]
        se = [S[f"{key0}/all"][key][m]["se"] for m in metrics]
        ax.bar(x + (k - 1) * w, mu, w, yerr=se, color=col, label=label, capsize=2.5,
               error_kw=dict(ecolor=INK_2, lw=1), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([TITLES[m].split(" (")[0].replace("occupancy ", "occupancy\n")
                                          for m in metrics], fontsize=10 * fs)
    ax.set_ylim(0, 1)
    ax.set_ylabel("score (mean ± se over chains)", color=INK_2, fontsize=11 * fs)
    ax.set_title("ATLAS test, 82 chains", loc="left", fontsize=12.5 * fs, color=INK, pad=8)
    style(ax, fs, clean)
    ax = axes[1]
    for k, (key, label, col) in enumerate(MODELS[1:]):
        for sub, fill, dx in (("all", col, -0.09), ("not_in_rcsb_train", "none", 0.09)):
            d = [S[f"{key0}/{sub}"][key][m][f"vs_{REF}"] for m in metrics]
            ax.errorbar(x + dx + (k - 0.5) * 0.26, [v["diff"] for v in d],
                        yerr=[[v["diff"] - v["lo"] for v in d], [v["hi"] - v["diff"] for v in d]],
                        fmt="o", ms=6.5, color=col, mfc=fill, mew=1.8, capsize=3, lw=1.8, zorder=3)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([TITLES[m].split(" (")[0].replace("occupancy ", "occupancy\n")
                                          for m in metrics], fontsize=10 * fs)
    ax.set_ylabel("paired difference vs released V2 (95% CI)", color=INK_2, fontsize=11 * fs)
    n_un = S[f"{key0}/not_in_rcsb_train"][REF]["auroc"]["n"]
    ax.set_title(f"change vs released V2", loc="left", fontsize=12.5 * fs, color=INK, pad=8)
    style(ax, fs, clean)
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color=INK_2, marker="o", ls="", ms=7, label="all 82 chains"),
          Line2D([], [], color=INK_2, marker="o", mfc="none", ls="", ms=7, mew=1.8,
                 label=f"{n_un} chains not in RCSB pre-training")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.52, 0.99 if clean else 0.86), ncol=5,
               frameon=False, fontsize=10 * fs, labelcolor=INK, handlelength=2.0, columnspacing=1.4)
    if not clean:
        fig.text(0.06, 0.985, "Fine-tuning ESMDynamic with working DCM blocks: ATLAS test",
                 fontsize=15, color=INK, va="top")
        fig.text(0.06, 0.935, "Same three models as the mdCATH figure. ATLAS is 300 K MD, scored with the "
                 "320 K output, as Table 2 (256-residue crops, 5 per long chain; unsymmetrised logits; all "
                 "pairs). ATLAS is outside the training data\nof every model here (mdCATH only). Right: "
                 "per-chain paired difference, protein bootstrap. Occupancy RMSE: lower is better.",
                 fontsize=9, color=INK_2, va="top", linespacing=1.45)
    save(fig, out, clean)


def save(fig, out, clean):
    import matplotlib.pyplot as plt
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, transparent=clean, facecolor=None if clean else "white")
    plt.close(fig)
    print("->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default="investigation/results")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial" if args.clean else "Open Sans"
    fs = 1.45 if args.clean else 1.0
    folder = Path("investigation/figures") / ("transparent" if args.clean else "retraining")
    R = Path(args.results)
    mdcath_figure(load(R / "17_eval_checkpoints_mdcath.json"), folder / "17_option2_mdcath.png", args.clean, fs)
    atlas_figure(load(R / "17_eval_checkpoints_atlas.json"), folder / "17_option2_atlas.png", args.clean, fs)


if __name__ == "__main__":
    main()
