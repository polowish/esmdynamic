"""Figure: ablation and rescue -- the autocast-cache bug is what keeps the DCM blocks untrained.

    python investigation/scripts/plot_ablation_rescue.py            # captioned -> figures/gradient_flow/
    python investigation/scripts/plot_ablation_rescue.py --clean    # transparent, no title/caption, larger type
                                                                    #   -> figures/transparent/

A  ablation (note 05, results/05_dcm_identity.json, scripts/dcm_identity_test.py, job 2156988):
   the released V2 model on three proteins (ubiquitin, GB1, hen lysozyme), all 11 head outputs
   compared bit for bit (torch.equal) with the intact model: a rerun (determinism check), every
   Evoformer block deleted from all three heads, and a control in which one block's
   zero-initialised output layer is set to 1e-3 so that block computes something. Deleting the
   blocks changes nothing; the control changes that head's outputs, so the test would see a
   block that did anything.
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

    ident = json.loads((RES / "05_dcm_identity.json").read_text())["proteins"]
    on = json.loads((RES / "07_training_check_cache_on.json").read_text())["train"]
    off = json.loads((RES / "07_training_check_cache_off.json").read_text())["train"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Arial"

    fig = plt.figure(figsize=(14, 6.0 if args.clean else 6.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25, left=0.09, right=0.98,
                          top=0.86 if args.clean else 0.72, bottom=0.11)
    a, b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # A: ablation -- delete the Evoformer blocks from the released model, three proteins
    names = {"ubiquitin": "ubiquitin\n(76 res.)", "GB1": "GB1\n(56 res.)", "HEWL": "lysozyme\n(129 res.)"}
    conds = [("rerun (determinism check)", "rerun", "#a8a7a2"),
             ("all Evoformer blocks deleted", "no_blocks", BUG),
             ("control: one block output layer\nset to 1e-3 instead of 0", "perturbed", FIX)]
    w = 0.26
    for i, (lab, key, col) in enumerate(conds):
        xs = [k + (i - 1) * w for k in range(len(names))]
        vals = [sum(o["equal"] for o in ident[p][key].values()) for p in names]
        a.bar(xs, vals, width=w * 0.9, color=col, zorder=3, label=lab, edgecolor="none")
        for xx, v in zip(xs, vals):
            a.text(xx, v + 0.25, f"{v}", ha="center", va="bottom", fontsize=10.5 * fs, color=INK)
    a.set_xticks(range(len(names)))
    a.set_xticklabels(names.values(), fontsize=10.5 * fs, color=INK)
    a.set_ylim(0, 15.5)
    a.set_yticks([0, 4, 8, 11])
    a.set_ylabel("outputs bit-for-bit identical\nto the released model (of 11)", fontsize=11 * fs, color=INK_2)
    a.legend(loc="upper left", frameon=False, fontsize=9.5 * fs, labelcolor=INK, ncol=1,
             handlelength=1.2, borderaxespad=0.1, labelspacing=0.35)
    a.grid(axis="y", color=GRID, lw=0.7, zorder=0); a.set_axisbelow(True)
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
    y_title = max(a.get_position().y1, b.get_position().y1) + 0.03
    for ax, t in ((a, "ablation: Evoformer blocks deleted, 3 proteins"), (b, "rescue: ESMDynamic training, bug vs fix")):
        fig.text(ax.get_position().x0, y_title, t, fontsize=13 * fs, color=INK, va="bottom", ha="left")
    if args.clean:
        for ax in (a, b):
            ax.set_facecolor("none")
    else:
        fig.text(0.01, 0.99, "ESMDynamic's Evoformer blocks do nothing in the released model, and the autocast-cache bug "
                 "is why: removing it rescues training", fontsize=14.5, color=INK, va="top")
        fig.text(0.01, 0.945, "Left (ablation): the released model on three proteins with every Evoformer block deleted "
                 "from all three heads. All 11 head outputs (dynamic, kinetic and occupancy logits, probabilities, "
                 "predictions,\nconfidences) stay bit-for-bit identical, as in a plain rerun; the control (one block "
                 "output layer set to 1e-3) changes them, so the test would detect a block that computed anything "
                 "(note 05, job 2156988).\nRight (rescue): 500 fine-tuning steps of ESMDynamic from the released weights, "
                 "identical batches and crops, differing only in autocast(cache_enabled=...). With the bug the blocks' "
                 "zero-initialised output\nlayers stay at exactly 0; with the fix they train from step 1 "
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
