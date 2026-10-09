"""The evidence figures for EVIDENCE.md: one claim per figure, each titled with its finding.

    python investigation/scripts/plot_evidence.py [--out investigation/figures/evidence]

Reads this branch's results/ where they exist; numbers from the dyn_models diagnostics
(esmdyn_checkpoint_audit.py, esmdyn_update_scale.py job 2079787, esmdyn_block_grad.py job
2153909, esmdyn_grad_check.py job 2153891) are embedded with that provenance.

  E1  the released block weights are at their initial values            (checkpoint audit)
  E2  their output layers are exactly zero, unlike trained Evoformer blocks  (job 2079787)
  E3  deleting the blocks leaves every output bit-for-bit identical      (note 05)
  E4  the mechanism in 20 lines of PyTorch: autocast cache + no_grad     (note 04)
  E5  in ESMDynamic's own training step, the blocks get no gradient      (jobs 2153891, 2153909)
  E6  with the one-line fix, the blocks train                            (note 07)
  E7  the published numbers come from these weights                     (note 12)
  E8  the paper's DCM ablation cannot measure the blocks                (note 09)
"""
import argparse
import json
from pathlib import Path

import numpy as np

INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
BLUE, ORANGE, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"     # validated set
PALE_BLUE, PALE_ORANGE = "#e3edf9", "#fdebe1"
RES = Path("investigation/results")

# E1 -- (tensors at init, tensors) per component, esmdyn_checkpoint_audit.py on each release
AUDIT = [
    ("Evoformer blocks inside the DCM", (480, 480), (160, 160)),
    ("seq / pair transitions", (6, 36), (6, 12)),
    ("DCM positional & recycling\nembeddings, norms", (2, 18), (2, 6)),
    ("prediction & confidence layers", (0, 24), (0, 4)),
]
# E2 -- Frobenius norm of the zero-initialised output layers (job 2079787)
NORM_LAYERS = ["seq attention\noutput", "seq MLP\noutput", "pair MLP\noutput", "triangle-mult.\noutput"]
NORMS = [("ESMFold block 0", [68.441, 120.765, 42.240, 23.297]),
         ("ESMFold block 23", [42.667, 26.894, 43.879, 24.130]),
         ("ESMFold block 47", [134.507, 256.099, 100.297, 155.225])]
# E5 -- gradient norm per dynamic-head component, one train.py step (job 2153891)
GRADS = [("prediction layer", 59.92), ("confidence head", 7.68), ("pair transition", 0.373),
         ("seq transition", 0.0301), ("recycling z norm", 0.0300), ("recycling distogram emb.", 0.0174),
         ("positional embedding", 0.00699), ("recycling s norm", 0.00203), ("Evoformer blocks (DCM)", 0.0)]
# E5 -- tensors of one DynamicModule (166) that get nonzero / zero / no gradient (job 2153909)
CENSUS = [("as train.py\nruns: bf16,\ncache on", 6, 44, 116),
          ("one-line fix:\ncache off", 38, 128, 0),
          ("no\nautocast", 38, 128, 0)]
# E8 -- SI Tables 6, 13, 17: mdCATH test, 320 K, dynamic-contact balanced accuracy
ABLATION = [("full model", 0.796, 0.007, (1, 1, 1, 0)),
            ("no auxiliary inputs", 0.771, 0.006, (1, 0, 1, 0)),
            ("no DCM", 0.764, 0.006, (1, 0, 0, 0))]
ABLATION_ROWS = ["ESMFold (frozen)", "auxiliary-input transitions", "DCM non-block terms\n(positional, recycling)",
                 "Evoformer blocks that\nactually compute"]


def style(ax, grid="y"):
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def titled(fig, title, sub):
    fig.text(0.01, 0.995, title, fontsize=13.5, color=INK, va="top", ha="left")
    fig.text(0.01, 0.935, sub, fontsize=9, color=INK_2, va="top", ha="left", linespacing=1.4)


def save(fig, out, name):
    path = out / name
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    print(path)


def e1(plt, out):
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor="white")
    fig.subplots_adjust(top=0.74, left=0.27, right=0.97)
    y = np.arange(len(AUDIT))[::-1]
    for k, (lab, col, off) in enumerate((("V2 weights (2026-04, what every\nentry point downloads)", ORANGE, -0.18),
                                         ("V1 weights (2025-06)", VIOLET, 0.18))):
        vals = [100 * a[1 + k][0] / a[1 + k][1] for a in AUDIT]
        ax.barh(y - off, vals, height=0.34, color=col, zorder=3, label=lab.replace("\n", " "))
        for yy, v, a in zip(y - off, vals, AUDIT):
            n0, n = a[1 + k]
            ax.text(v + 1.2, yy, f"{n0} of {n}", va="center", fontsize=8.6, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([a[0] for a in AUDIT], fontsize=9.5, color=INK)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("tensors still at their initial value (%)", color=INK_2, fontsize=9)
    style(ax, "x")
    ax.legend(loc="lower right", frameon=False, fontsize=8.6, labelcolor=INK)
    titled(fig, "E1  Every Evoformer-block tensor in both released weight files is at its initial value",
           "Each tensor tested against its initialiser: LayerNorm weights exactly 1, zero-initialised layers exactly 0, "
           "the rest matching their random initialiser's statistics.\nOutside the blocks, only the occupancy head's per-residue path is untouched (it reaches no loss once the blocks are dead). "
           "Source: dyn_models/scripts/diagnostics/esmdyn_checkpoint_audit.py on both Illinois Data Bank files.")
    save(fig, out, "E1_block_weights_at_init.png")


def e2(plt, out):
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="white")
    fig.subplots_adjust(top=0.74, left=0.08, right=0.98)
    x = np.arange(len(NORM_LAYERS))
    w = 0.2
    cols = ["#9bbbe8", "#5d93dd", BLUE]
    for i, (lab, v) in enumerate(NORMS):
        ax.bar(x + (i - 1.5) * w, v, width=w * 0.92, color=cols[i], zorder=3, label=lab)
    for xx in x:
        ax.plot(xx + 1.5 * w, 3, marker="o", ms=7, mfc="white", mec=ORANGE, mew=1.8, zorder=4)
        ax.text(xx + 1.5 * w, 12, "0.000", ha="center", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.plot([], [], marker="o", ls="", ms=7, mfc="white", mec=ORANGE, mew=1.8,
            label="ESMDynamic: all 6 DCM blocks (3 heads × 2)")
    ax.set_xticks(x)
    ax.set_xticklabels(NORM_LAYERS, fontsize=9.3, color=INK)
    ax.set_ylabel("Frobenius norm of the layer's weights", color=INK_2, fontsize=9)
    style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6, labelcolor=INK, ncol=2)
    ax.set_ylim(0, 300)
    titled(fig, "E2  The blocks' output layers are exactly zero; in trained Evoformer blocks they are not",
           "These layers are initialised to zero by design (each block adds  out(…)  to its input), so they are the first "
           "thing training moves.\nESMFold's trained trunk blocks shown for comparison. Source: esmdyn_update_scale.py, "
           "job 2079787, released V2 weights.")
    save(fig, out, "E2_output_layers_zero.png")


def e3(plt, out):
    d = json.loads((RES / "05_dcm_identity.json").read_text())["proteins"]
    names = {"ubiquitin": "ubiquitin\n(L = 76)", "GB1": "GB1\n(L = 56)", "HEWL": "hen lysozyme\n(L = 129)"}
    conds = [("rerun (determinism check)", "rerun", MUTED),
             ("all Evoformer blocks deleted", "no_blocks", ORANGE),
             ("control: one block output layer\nset to 1e-3 instead of 0", "perturbed", BLUE)]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor="white")
    fig.subplots_adjust(top=0.74, left=0.08, right=0.72)
    x = np.arange(len(names))
    w = 0.25
    for i, (lab, key, col) in enumerate(conds):
        vals = [sum(o["equal"] for o in d[p][key].values()) for p in names]
        diffs = [max((o["max_abs_diff"] or 0) for o in d[p][key].values()) for p in names]
        ax.bar(x + (i - 1) * w, vals, width=w * 0.9, color=col, zorder=3, label=lab)
        for xx, v, df in zip(x + (i - 1) * w, vals, diffs):
            ax.text(xx, v + 0.2, f"{v}/11", ha="center", fontsize=8.6, color=INK)
            ax.text(xx, 0.35, "max diff\n0" if df == 0 else f"max diff\n{df:.1e}", ha="center",
                    fontsize=7, color="white", linespacing=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(names.values(), fontsize=9.3, color=INK)
    ax.set_ylim(0, 12.5)
    ax.set_yticks([0, 4, 8, 11])
    ax.set_ylabel("model outputs bit-for-bit identical\nto the released model (of 11)", color=INK_2, fontsize=9)
    style(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.8, labelcolor=INK,
              labelspacing=1.1)
    fig.text(0.74, 0.30, "11 outputs: dynamic logits, probability,\nprediction, confidence; kinetic logits,\n"
             "probability, class, confidence; occupancy\nvalue, prediction, residual. Compared\nwith torch.equal.",
             fontsize=8.2, color=INK_2, va="top", linespacing=1.35)
    titled(fig, "E3  Deleting every Evoformer block from the released model changes no output, bit for bit",
           "Blocks removed from all three heads (each DCM keeps its embeddings and recycling terms). The control shows "
           "the test would see a block\nthat computed anything: a 1e-3 change in one layer of one block changes that "
           "head's logits. Source: scripts/dcm_identity_test.py, job 2156988.")
    save(fig, out, "E3_blocks_deleted_identical.png")


def e4(plt, out):
    d = json.loads((RES / "01_minimal_autocast_repro.json").read_text())["B_C"]
    rows = [("as ESMDynamic trains:\nbf16 autocast, cache on (default)", "as ESMDynamic trains: one bf16 autocast region, cache on"),
            ("fix 1:  autocast(cache_enabled=False)", "fix 1: autocast(cache_enabled=False)"),
            ("fix 2:  no_grad passes outside autocast", "fix 2: no_grad passes outside the autocast region"),
            ("fix 3:  clear_autocast_cache() before\nthe gradient pass", "fix 3: torch.clear_autocast_cache() before the grad pass"),
            ("no autocast (float32)", "no autocast (fp32)")]
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor="white")
    fig.subplots_adjust(top=0.74, left=0.33, right=0.97, bottom=0.06)
    for i, (lab, key) in enumerate(rows):
        y = len(rows) - 1 - i
        ok = d[key]["out.weight"] == "nonzero"
        ax.add_patch(plt.Rectangle((0, y - 0.38), 1, 0.76, fc=PALE_BLUE if ok else PALE_ORANGE,
                                   ec=BLUE if ok else ORANGE, lw=1.4))
        ax.text(0.03, y, ("✓  receives gradient: the block can learn" if ok else
                          "✗  no gradient at all: the block can never change"),
                va="center", fontsize=10, color=BLUE if ok else ORANGE, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9.3, color=INK)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    titled(fig, "E4  The mechanism, reproduced in plain PyTorch with no ESMDynamic code",
           "A residual block  x + out(relu(inp(LayerNorm(x))))  with  out  zero-initialised, recycled 4 times with "
           "torch.no_grad() on all but the last pass, exactly\nas DynamicModule.forward does. Shown: does the "
           "block's output layer get a gradient? Source: scripts/minimal_autocast_repro.py, torch 2.8.0, job 2156988.")
    save(fig, out, "E4_mechanism_minimal_repro.png")


def e5(plt, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white", gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(top=0.72, left=0.17, right=0.98, wspace=0.55)
    y = np.arange(len(GRADS))[::-1]
    for yy, (lab, g) in zip(y, GRADS):
        if g > 0:
            a.barh(yy, g, height=0.62, color=BLUE, zorder=3, left=1e-4)
            a.text(g * 1.3, yy, f"{g:g}", va="center", fontsize=8.3, color=INK)
        else:
            a.plot(1.6e-4, yy, marker="o", ms=7, mfc="white", mec=ORANGE, mew=1.8, zorder=4)
            a.text(2.6e-4, yy, "exactly 0", va="center", fontsize=8.8, color=ORANGE, fontweight="bold")
    a.set_xscale("log")
    a.set_xlim(1e-4, 600)
    a.set_yticks(y)
    a.set_yticklabels([g[0] for g in GRADS], fontsize=9, color=INK)
    a.set_xlabel("gradient norm after one training step (log)", color=INK_2, fontsize=8.8)
    style(a, "x")
    a.set_title("A  one training-mode step (train.py setup), dynamic head", loc="left", fontsize=10, color=INK)
    x = np.arange(len(CENSUS))
    for i, (lab, nz, z, none) in enumerate(CENSUS):
        bottom = 0
        for v, col in ((nz, BLUE), (z, "#9bbbe8"), (none, ORANGE)):
            if v:
                b.bar(i, v, bottom=bottom, width=0.6, color=col, zorder=3, edgecolor="white", linewidth=1.5)
                b.text(i, bottom + v / 2, str(v), ha="center", va="center", fontsize=8.6,
                       color="white" if col != "#9bbbe8" else INK)
            bottom += v
    b.set_xticks(x)
    b.set_xticklabels([c[0] for c in CENSUS], fontsize=8.4, color=INK)
    b.set_ylabel("tensors of one DynamicModule (166)", color=INK_2, fontsize=8.8)
    style(b)
    b.set_title("B  which tensors get any gradient", loc="left", fontsize=10, color=INK)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=BLUE, label="nonzero gradient"),
                      Patch(color="#9bbbe8", label="gradient exactly 0\n(behind a zero layer on step 1: expected)"),
                      Patch(color=ORANGE, label="no gradient (not in the graph)")],
             loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=8, labelcolor=INK)
    fig.subplots_adjust(right=0.80)
    titled(fig, "E5  In ESMDynamic's training setup, every trained part gets gradient except the Evoformer blocks",
           "A: one training-mode step in train.py's setup (train mode, ESMFold frozen, bf16 autocast) on ubiquitin + GB1, with a stand-in "
           "loss Σ mean(output²) over the head outputs train.py's losses read\n(esmdyn_grad_check.py, job 2153891). B: one "
           "training-mode backward through a DynamicModule on random inputs, cache on vs off (esmdyn_block_grad.py, job 2153909). "
           "Whether a\nparameter gets gradient depends on the graph, not the loss; E6 confirms it with ESMDynamic's real losses. "
           "Diagnostic: the output layers get None, so they never move.")
    save(fig, out, "E5_training_step_gradients.png")


def e6(plt, out):
    on = json.loads((RES / "07_training_check_cache_on.json").read_text())
    off = json.loads((RES / "07_training_check_cache_off.json").read_text())
    s = np.array([r["step"] for r in on["train"]])
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    fig.subplots_adjust(top=0.72, left=0.07, right=0.98, wspace=0.28)
    a.plot(s, [r["block_output_norm"] for r in off["train"]], color=BLUE, lw=2, label="fixed (autocast cache off)")
    a.plot(s, [r["block_output_norm"] for r in on["train"]], color=ORANGE, lw=2, label="as released (cache on)")
    a.text(s[-1], off["train"][-1]["block_output_norm"] + 0.8, f"{off['train'][-1]['block_output_norm']:.1f}",
           ha="right", fontsize=8.6, color=BLUE)
    a.text(s[-1], 1.0, "0.000 at every step", ha="right", fontsize=8.6, color=ORANGE)
    a.set_xlabel("optimiser step", color=INK_2, fontsize=8.8)
    a.set_ylabel("norm of the blocks' output layers", color=INK_2, fontsize=8.8)
    style(a)
    a.legend(loc="upper left", frameon=False, fontsize=8.6, labelcolor=INK)
    a.set_title("A  do the blocks move?", loc="left", fontsize=10, color=INK)
    diff = np.array([f["loss"] - n["loss"] for f, n in zip(off["train"], on["train"])])
    edges = [(1, 100), (101, 250), (251, 400), (401, 500)]
    for i, (lo, hi) in enumerate(edges):
        v = diff[(s >= lo) & (s <= hi)]
        m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        b.bar(i, m, width=0.6, color=BLUE, zorder=3)
        b.errorbar(i, m, yerr=se, color=INK, capsize=4, elinewidth=1.2, zorder=4)
        b.text(i, m - se - 12, f"{m:.0f} ± {se:.0f}", ha="center", va="top", fontsize=8.5, color=INK)
    b.axhline(0, color=INK_2, lw=0.8)
    b.set_xticks(range(len(edges)))
    b.set_xticklabels([f"steps {lo}–{hi}" for lo, hi in edges], fontsize=8.6, color=INK)
    b.set_ylabel("training loss: fixed − as released\n(same batches, same crops)", color=INK_2, fontsize=8.8)
    b.set_ylim(-260, 30)
    style(b)
    b.set_title("B  does the model fit better? (lower = better)", loc="left", fontsize=10, color=INK)
    titled(fig, "E6  With the one-line fix the blocks train, and the model fits its training data better",
           "Two 500-step fine-tuning runs from the released V2 weights on the authors' mdCATH training split, identical except "
           "autocast(cache_enabled=…): same samples, same crops,\nbit-identical first loss. Held-out after 500 steps is mixed (balanced accuracy +2.6 "
           "points, occupancy loss −7%, dynamic focal loss +5%); a real answer needs full retraining. Note 07, jobs 2157527/8.")
    save(fig, out, "E6_fix_trains_blocks.png")


def e7(plt, out):
    """Published vs released weights, every metric, under the training code's evaluation path."""
    metrics = [("bal_acc", "balanced\naccuracy"), ("precision", "precision"), ("recall", "recall"),
               ("f1", "F1"), ("auroc", "AUROC"), ("rmse", "occupancy\nRMSE")]
    sets = [("atlas", "ATLAS test (82 chains), Table 2"), ("mdcath", "mdCATH test (270 domains), 320 K, Table 1")]
    fig, axes = plt.subplots(2, 6, figsize=(13, 6.4), facecolor="white",
                             gridspec_kw={"hspace": 0.75, "wspace": 0.45})
    fig.subplots_adjust(top=0.76, left=0.05, right=0.99, bottom=0.06)
    for r, (ds, title) in enumerate(sets):
        d = json.loads((RES / f"12_eval_protocol_{ds}.json").read_text())
        pub, ours = d["published"]["320"], d["summary"]["crop256/unsym/320K"]
        for c, (m, ml) in enumerate(metrics):
            ax = axes[r, c]
            vals = [(pub[m][0], pub[m][1], BLUE), (ours[m]["mean"], ours[m]["se"], ORANGE)]
            top = max(v + se for v, se, _ in vals)
            for i, (v, se, col) in enumerate(vals):
                ax.bar(i, v, width=0.7, color=col, zorder=3)
                ax.errorbar(i, v, yerr=se, color=INK, capsize=3, elinewidth=1, zorder=4)
                ax.text(i, v + se + top * 0.03, f"{v:.3f}", ha="center", fontsize=8.2, color=INK)
            ax.set_ylim(0, top * 1.22)
            ax.set_xticks([])
            ax.set_xlabel(ml, fontsize=8.8, color=INK, labelpad=4)
            style(ax)
            ax.tick_params(labelsize=7.5)
        axes[r, 0].set_title(title, loc="left", fontsize=10.5, color=INK, pad=12, x=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label="published (Nat. Commun. 2026)"),
                        Patch(color=ORANGE, label="released V2 weights, our run")],
               loc="upper right", bbox_to_anchor=(0.99, 0.875), ncol=2, frameon=False, fontsize=9, labelcolor=INK)
    titled(fig, "E7  Every published number at 320 K reproduces with the released weights, whose blocks never trained",
           "Per-protein mean ± standard error. Scored as ESMDynamic's training code scores (train.py: 256-residue random "
           "crop, threshold 0.5 on the unsymmetrised logits,\nall residue pairs; occupancy = the frequency head's "
           "frequency_pred). So the paper's results describe a model whose Evoformer blocks are identity maps. Note 12.")
    save(fig, out, "E7_published_numbers_reproduce.png")


def e8(plt, out):
    fig = plt.figure(figsize=(10, 5.4), facecolor="white")
    a = fig.add_axes([0.30, 0.44, 0.66, 0.30])
    t = fig.add_axes([0.30, 0.03, 0.66, 0.36])
    x = np.arange(len(ABLATION))
    for i, (lab, v, se, _) in enumerate(ABLATION):
        a.bar(i, v - 0.70, bottom=0.70, width=0.5, color=BLUE, zorder=3)
        a.errorbar(i, v, yerr=se, color=INK, capsize=4, elinewidth=1.2, zorder=4)
        a.text(i, v + se + 0.004, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    a.set_xticks(x)
    a.set_xticklabels([r[0] for r in ABLATION], fontsize=9.5, color=INK)
    a.set_xlim(-0.5, len(ABLATION) - 0.5)
    a.set_ylim(0.70, 0.82)
    a.set_ylabel("balanced accuracy\n(axis starts at 0.70)", color=INK_2, fontsize=8.6)
    style(a)
    t.set_xlim(-0.5, len(ABLATION) - 0.5)
    t.set_ylim(-0.5, len(ABLATION_ROWS) - 0.5)
    t.axis("off")
    for r, lab in enumerate(ABLATION_ROWS):
        y = len(ABLATION_ROWS) - 1 - r
        fig.text(0.29, 0.03 + 0.36 * (y + 0.5) / len(ABLATION_ROWS), lab, ha="right", va="center",
                 fontsize=8.8, color=ORANGE if r == 3 else INK, fontweight="bold" if r == 3 else "normal")
        for i, (_, _, _, has) in enumerate(ABLATION):
            yes = bool(has[r])
            t.add_patch(plt.Rectangle((i - 0.24, y - 0.36), 0.48, 0.72, fc=PALE_BLUE if yes else PALE_ORANGE,
                                      ec=BLUE if yes else ORANGE, lw=1.2))
            t.text(i, y, "yes" if yes else "no", ha="center", va="center", fontsize=9,
                   color=BLUE if yes else ORANGE, fontweight="bold")
    fig.text(0.01, 0.995, "E8  The paper's DCM ablation cannot measure what the Evoformer blocks contribute",
             fontsize=13.5, color=INK, va="top")
    fig.text(0.01, 0.94, "mdCATH test, 320 K (SI Tables 6, 13, 17). All three variants were trained with the same code, "
             "so no variant has blocks that compute.\nThe full model's ~3-point lead can only come from the auxiliary "
             "transitions, the DCM's non-block terms or run-to-run variation. \"No DCM\" per Methods: ESMFold's pair\nstate is used in "
             "place of the DCM output, which likely also drops the pair transition (the ablation code is not public).",
             fontsize=9, color=INK_2, va="top", linespacing=1.4)
    save(fig, out, "E8_ablation_cannot_measure_blocks.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="investigation/figures/evidence")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (e1, e2, e3, e4, e5, e6, e7, e8):
        f(plt, out)
        plt.close("all")


if __name__ == "__main__":
    main()
