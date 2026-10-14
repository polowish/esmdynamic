"""Figure: ablation and rescue -- the autocast-cache bug is what keeps the DCM blocks untrained.

    python investigation/scripts/plot_ablation_rescue.py            # captioned -> figures/gradient_flow/
    python investigation/scripts/plot_ablation_rescue.py --clean    # transparent, no title/caption, larger type
                                                                    #   -> figures/transparent/

A  ablation (notes 05, 12, 13; jobs 2200604 / 2200605): the released V2 model scored with its
   Evoformer blocks deleted from every head, against the intact model, per test protein:
   balanced accuracy for the 270 mdCATH test domains (320 K; pairs |i - j| >= 3, symmetrised,
   scikit-learn style) and the 82 ATLAS test chains (train.py protocol). Every score is identical
   (largest difference over every metric, temperature and protocol: 0.0), because the untrained
   blocks are identity maps.
B  rescue (results/07_training_check_cache_{on,off}.json, note 07, jobs 2157527 / 2157528): 500
   fine-tuning steps of ESMDynamic itself from the released V2 weights, identical batches and
   crops, differing only in autocast(cache_enabled=...). The norm of the blocks' zero-initialised
   output layers stays exactly 0 with the bug and grows from step 1 with the fix.
Colours as the parameter pies: teal = trains, dark blue = stays untrained. Arial.
"""
import argparse
import json
from pathlib import Path

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BUG, FIX = "#1D3BE2", "#1DE2C4"          # dark blue: stays untrained; teal: trains
RES = Path("investigation/results")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or ("investigation/figures/transparent/ablation_rescue.png" if args.clean
                            else "investigation/figures/gradient_flow/ablation_rescue.png"))
    fs = 1.45 if args.clean else 1.0

    import numpy as np
    import pandas as pd

    def mdcath(path):
        d = pd.read_csv(path)
        d = d[(d.variant == "sym") & (d.k == 3) & (d.temp == 320)].copy()
        d["bal_acc"] = np.where(d.n_negative == 0, d.recall, d.bal_acc)
        return d.set_index("id").bal_acc

    def atlas(path):
        d = pd.read_csv(path)
        d = d[(d.protocol == "crop256") & (d.variant == "unsym") & (d.temp == 320)]
        return d.groupby("id").bal_acc.mean()

    md = pd.concat([mdcath(RES / "13_eval_pairset_mdcath_per_protein.csv"),
                    mdcath(RES / "13_eval_pairset_mdcath_noevoformer_per_protein.csv")], axis=1, keys=["full", "noevo"])
    at = pd.concat([atlas(RES / "12_eval_protocol_atlas_per_protein.csv"),
                    atlas(RES / "12_eval_protocol_atlas_noevoformer_per_protein.csv")], axis=1, keys=["full", "noevo"])
    max_diff = max(float((md.full - md.noevo).abs().max()), float((at.full - at.noevo).abs().max()))
    on = json.loads((RES / "07_training_check_cache_on.json").read_text())["train"]
    off = json.loads((RES / "07_training_check_cache_off.json").read_text())["train"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(14, 6.0 if args.clean else 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.28, left=0.07, right=0.98,
                          top=0.86 if args.clean else 0.74, bottom=0.11)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # A: ablation -- delete the Evoformer blocks from the released model
    lw = 1.2 if args.clean else 1
    a.plot([0, 1], [0, 1], color=GRID, lw=1.4, zorder=1)
    a.scatter(md.full, md.noevo, s=34 * lw, color=BUG, alpha=0.85, edgecolors="none", zorder=3,
              label=f"mdCATH test, {len(md)} domains")
    a.scatter(at.full, at.noevo, s=46 * lw, facecolors="none", edgecolors=BUG, linewidths=1.4, zorder=4,
              label=f"ATLAS test, {len(at)} chains")
    a.set_xlim(0.4, 1.0); a.set_ylim(0.4, 1.0); a.set_aspect("equal")
    a.set_xticks([0.4, 0.6, 0.8, 1.0]); a.set_yticks([0.4, 0.6, 0.8, 1.0])
    a.set_xlabel("balanced accuracy, released model", fontsize=11 * fs, color=INK_2)
    a.set_ylabel("balanced accuracy, blocks removed", fontsize=11 * fs, color=INK_2)
    a.text(0.98, 0.43, f"{len(md) + len(at)} test proteins:\nevery score identical\n(largest difference {max_diff:.1f})",
           ha="right", va="bottom", fontsize=10.5 * fs, color=INK, linespacing=1.25)
    a.legend(loc="upper left", frameon=False, fontsize=10 * fs, labelcolor=INK, handletextpad=0.3, borderaxespad=0.2)
    a.grid(color=GRID, lw=0.7, zorder=0); a.set_axisbelow(True)
    for side in ("top", "right"):
        a.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        a.spines[side].set_color(GRID)
    a.tick_params(colors=INK_2, labelsize=10 * fs, length=0)

    # B: rescue in ESMDynamic's own training
    s = [r["step"] for r in on]
    b.plot(s, [r["block_output_norm"] for r in off], color=FIX, lw=2.6 * (1.2 if args.clean else 1), label="fixed (cache off)")
    b.plot(s, [r["block_output_norm"] for r in on], color=BUG, lw=2.6 * (1.2 if args.clean else 1), label="as released (cache on)")
    b.text(s[-1], off[-1]["block_output_norm"] + 0.9, f"{off[-1]['block_output_norm']:.1f}", ha="right",
           va="bottom", fontsize=11 * fs, color=INK)
    b.text(s[-1], 1.0, "0.000 at every step", ha="right", va="bottom", fontsize=11 * fs, color=BUG)
    b.set_xlim(0, s[-1] + 5); b.set_ylim(-0.5, 30)
    b.set_xlabel("training step", fontsize=11 * fs, color=INK_2)
    b.set_ylabel("norm of the blocks' output layers", fontsize=11 * fs, color=INK_2)
    b.grid(axis="y", color=GRID, lw=0.7, zorder=0); b.set_axisbelow(True)
    for side in ("top", "right"):
        b.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        b.spines[side].set_color(GRID)
    b.tick_params(colors=INK_2, labelsize=10 * fs, length=0)
    b.legend(loc="upper left", frameon=False, fontsize=11 * fs, labelcolor=INK)

    # panel titles at one height above both panels (A's top tick labels would collide with set_title)
    a.apply_aspect()                      # A is square: its frame is only known after the aspect is applied
    y_title = max(a.get_position().y1, b.get_position().y1) + 0.03
    for ax, t in ((a, "ablation: Evoformer blocks deleted"), (b, "rescue: ESMDynamic training, bug vs fix")):
        fig.text(ax.get_position().x0, y_title, t, fontsize=13 * fs, color=INK, va="bottom", ha="left")
    if args.clean:
        for ax in (a, b):
            ax.set_facecolor("none")
    else:
        fig.text(0.01, 0.99, "ESMDynamic's Evoformer blocks do nothing in the released model, and the autocast-cache bug "
                 "is why: removing it rescues training", fontsize=14.5, color=INK, va="top")
        fig.text(0.01, 0.945, "Left (ablation): the released model with its Evoformer blocks deleted from every head, scored "
                 "against the intact model on every test protein. Balanced accuracy (mdCATH: 320 K, pairs |i - j| >= 3, "
                 "symmetrised; ATLAS: train.py protocol)\nis identical for all 352 proteins, as is every other metric at every "
                 "temperature (notes 05, 12, 13; jobs 2200604 / 2200605). Right (rescue): 500 fine-tuning steps of ESMDynamic "
                 "from the released weights, identical batches\nand crops, differing only in autocast(cache_enabled=...). With "
                 "the bug the blocks' zero-initialised output layers stay at exactly 0; with the fix they train from step 1 "
                 "(note 07, jobs 2157527 / 2157528).",
                 fontsize=9, color=INK_2, va="top", linespacing=1.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.clean:
        fig.savefig(out, dpi=170, transparent=True)
    else:
        fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
