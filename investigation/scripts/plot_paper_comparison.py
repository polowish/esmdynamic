"""Bar plot: our reproduction with the released weights vs the preprint's reported numbers.

    python investigation/scripts/plot_paper_comparison.py
        [--out investigation/figures/reproduction/11_paper_comparison.png]

A  ATLAS test split (82 chains): preprint main text (87%) vs the released V2 weights.
   The preprint reports no ablation on ATLAS.
B  mdCATH test split (270 domains), 320 K: the preprint's full model and its two ablations
   (SI Tables 6, 13, 17) vs the released V2 weights.
Ours come from results/06_atlas_reproduction.json and results/10_mdcath_test_reproduction.json.
"""
import argparse
import json
from pathlib import Path

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
PAPER, OURS = "#2a78d6", "#eb6834"            # validated categorical pair
ABL1, ABL2 = "#8fb4e3", "#c5d8f1"             # lighter steps of the paper's hue: same source

# the preprint's numbers (v2 main text p.17; SI Tables 6, 13, 17: mean +- se, 320 K)
PAPER_ATLAS = 0.87
PAPER_MDCATH = [("preprint:\nfull model", 0.796, 0.007, PAPER),
                ("preprint ablation:\nno auxiliary inputs", 0.771, 0.006, ABL1),
                ("preprint ablation:\nno DCM", 0.764, 0.006, ABL2)]


def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5, length=0)


def bars(ax, items, ylim):
    for i, (label, v, se, col) in enumerate(items):
        ax.bar(i, v, width=0.62, color=col, zorder=3, edgecolor="white", linewidth=2)
        if se is not None:
            ax.errorbar(i, v, yerr=se, color=INK, elinewidth=1.2, capsize=4, zorder=4)
        ax.text(i, v + (se or 0) + 0.006, f"{v:.3f}" + (f"\n± {se:.3f}" if se else "\n(no error\nreported)"),
                ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.15)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=8.3, color=INK)
    ax.set_ylim(*ylim)
    ax.set_ylabel("balanced accuracy, dynamic-contact classification", color=INK_2, fontsize=9)
    style(ax)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--atlas", default="investigation/results/06_atlas_reproduction.json")
    ap.add_argument("--mdcath", default="investigation/results/10_mdcath_test_reproduction.json")
    ap.add_argument("--out", default="investigation/figures/reproduction/11_paper_comparison.png")
    args = ap.parse_args()
    at = json.loads(Path(args.atlas).read_text())["summary"]["320K_all"]
    md = json.loads(Path(args.mdcath).read_text())
    m3 = md["summary"]["320K"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13.5, 9.6), facecolor="white")
    a = fig.add_axes([0.07, 0.45, 0.25, 0.42])
    b = fig.add_axes([0.42, 0.45, 0.55, 0.42])
    bars(a, [("preprint\n(main text)", PAPER_ATLAS, None, PAPER),
             ("ours: released\nV2 weights", at["per_protein_mean"], at["per_protein_se"], OURS)],
         (0.5, 0.95))
    a.set_title(f"A   ATLAS test split ({json.loads(Path(args.atlas).read_text())['n_proteins']} chains)",
                loc="left", color=INK, fontsize=11, pad=10)
    a.set_xlabel("(the preprint reports no ablation on ATLAS)", color=INK_2, fontsize=8.5, labelpad=10)
    bars(b, PAPER_MDCATH + [("ours: released\nV2 weights", m3["bal_acc_mean"], m3["bal_acc_se"], OURS)],
         (0.5, 0.95))
    b.set_title(f"B   mdCATH test split ({md['n_proteins']} domains), 320 K: the full model and its ablations",
                loc="left", color=INK, fontsize=11, pad=10)

    fig.text(0.01, 0.985, "Reproducing ESMDynamic's reported dynamic-contact accuracy with the released weights",
             color=INK, fontsize=14, ha="left", va="top")
    fig.text(0.01, 0.953, "Error bars: ± 1 standard error over proteins. The preprint states none for ATLAS.",
             color=INK_2, fontsize=9, ha="left", va="top")

    explain = [
        ("What is measured.",
         "For every pair of residues (i, j) in a protein, the model predicts the probability that the pair is a "
         "dynamic contact. A pair is labelled dynamic in a molecular-dynamics ensemble if its Cα–Cα distance is "
         "below 7.5 Å in at least one frame and above 8.5 Å in another, i.e. the contact forms and breaks "
         "(preprint Methods). A pair is predicted dynamic when the dynamic head's probability exceeds 0.5."),
        ("Balanced accuracy",
         "= ½ (true-positive rate + true-negative rate), computed over all L × L residue pairs of one protein, then "
         "averaged over proteins. Only ~1–7% of pairs are dynamic, so plain accuracy would reward predicting "
         "\"never dynamic\"; balanced accuracy weights both classes equally (0.5 = chance)."),
        ("Which output.",
         "ESMDynamic predicts each pair at the five mdCATH simulation temperatures (320–450 K). mdCATH is scored "
         "at 320 K, against its 320 K simulations. ATLAS has one 300 K ensemble per protein, so it is scored "
         "against the nearest output, 320 K."),
        ("The ablations (panel B, preprint SI).",
         "\"No auxiliary inputs\": the transition layers that add ESMFold's pLDDT, language-model, pTM and "
         "distogram logits are removed. \"No DCM\": ESMFold's own pair representation replaces the Dynamic "
         "Contact Module's output. All three models were trained with the same code, in which the DCM's "
         "Evoformer blocks never receive gradient, so no bar here reflects trained DCM blocks."),
        ("What matches and what does not.",
         "Both headline numbers reproduce with the released weights. On mdCATH the match holds at 320 K only: at "
         "348–450 K our full-length scores are higher than SI Table 6 (0.79–0.75 vs 0.77–0.54), and the "
         "occupancy RMSE differs (0.109 vs 0.076), most likely because the paper's evaluation protocol "
         "(e.g. cropping) differs from our full-length scoring. Proteins: ATLAS from the Illinois Data Bank "
         "atlas_test.zip; mdCATH test split from mdcath.zip (test.csv)."),
    ]
    y = 0.33
    for head, body in explain:
        fig.text(0.01, y, head, color=INK, fontsize=9.2, ha="left", va="top", fontweight="bold")
        fig.text(0.215, y, body, color=INK_2, fontsize=8.8, ha="left", va="top", wrap=True)
        y -= 0.062

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
