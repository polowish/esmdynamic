"""Figure: ablation and rescue -- the autocast-cache bug is what keeps the DCM blocks untrained.

    python investigation/scripts/plot_ablation_rescue.py            # captioned -> figures/gradient_flow/
    python investigation/scripts/plot_ablation_rescue.py --clean    # transparent, no title/caption, larger type
                                                                    #   -> figures/transparent/

A  ablation (results/01_minimal_autocast_repro.json, note 04, job 2156988, L40S, bf16): a linear
   layer used the way the DCM uses its blocks, with each of the two suspected ingredients
   removed in turn -- the autocast weight cache, and a no_grad (recycling) pass before the
   gradient pass. The weight gets no gradient only when both are present, as in train.py.
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

    rep = json.loads((RES / "01_minimal_autocast_repro.json").read_text())
    cells = {(r["cache_enabled"], r["no_grad_use_first"]): r["weight"] for r in rep["A"] if r["dtype"] == "bfloat16"}
    on = json.loads((RES / "07_training_check_cache_on.json").read_text())["train"]
    off = json.loads((RES / "07_training_check_cache_off.json").read_text())["train"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Patch
    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(14, 6.0 if args.clean else 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.30, left=0.14, right=0.98,
                          top=0.78 if args.clean else 0.66, bottom=0.11)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # A: 2 x 2 ablation grid
    cols = [(True, "autocast\nweight cache on"), (False, "autocast weight\ncache off (the fix)")]
    rows = [(True, "no_grad pass first\n(as in recycling)"), (False, "no no_grad\npass first")]
    for i, (ng, rlab) in enumerate(rows):
        for j, (ce, clab) in enumerate(cols):
            got = cells[(ce, ng)]
            trains = got != "none"
            a.add_patch(FancyBboxPatch((j + 0.06, 1 - i + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.06",
                                       fc=FIX if trains else BUG, ec="none"))
            txt = "gradient reaches\nthe weight" if trains else "no gradient:\nweight never\nmoves"
            a.text(j + 0.5, 1 - i + 0.5, txt, ha="center", va="center", fontsize=11 * fs * (0.88 if args.clean else 1),
                   color=INK if trains else "white", fontweight="bold" if not trains else "normal", linespacing=1.15)
            if ce and ng:
                a.text(j + 0.5, 1 - i + 0.90, "as released", ha="center", va="top",
                       fontsize=10 * fs, color="white", style="italic")
    a.set_xlim(0, 2); a.set_ylim(0, 2); a.set_aspect("equal")
    a.set_xticks([0.5, 1.5]); a.set_xticklabels([c[1] for c in cols], fontsize=11 * fs, color=INK)
    a.set_yticks([1.5, 0.5]); a.set_yticklabels([r[1] for r in rows], fontsize=11 * fs, color=INK)
    a.tick_params(length=0)
    a.xaxis.set_ticks_position("top")
    for side in a.spines.values():
        side.set_visible(False)

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
    top = 0.78 if args.clean else 0.66
    for ax, t in ((a, "ablation: remove each ingredient"), (b, "rescue: ESMDynamic training, bug vs fix")):
        x0 = ax.get_position().x0 if ax is b else 0.02
        fig.text(x0, top + (0.105 if args.clean else 0.09), t, fontsize=13 * fs, color=INK, va="bottom", ha="left")
    if args.clean:
        for ax in (a, b):
            ax.set_facecolor("none")
    else:
        fig.text(0.01, 0.99, "The autocast-cache bug is what keeps ESMDynamic's Evoformer blocks untrained: "
                 "removing it rescues training", fontsize=14.5, color=INK, va="top")
        fig.text(0.01, 0.945, "Left (ablation): a linear layer used as the DCM uses its blocks, bf16 autocast, each suspected "
                 "ingredient removed in turn. The weight gets no gradient only when the weight cache and a no_grad recycling pass "
                 "\nare both present, the setup train.py runs (note 04, job 2156988). Right (rescue): 500 fine-tuning steps of "
                 "ESMDynamic from the released weights, identical batches and crops, differing only in "
                 "autocast(cache_enabled=...).\nWith the bug the blocks' zero-initialised output layers stay at exactly 0; with "
                 "the fix they train from step 1 (note 07, jobs 2157527 / 2157528).",
                 fontsize=9, color=INK_2, va="top", linespacing=1.45)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.clean:
        fig.savefig(out, dpi=170, transparent=True)
    else:
        fig.savefig(out, dpi=170, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
