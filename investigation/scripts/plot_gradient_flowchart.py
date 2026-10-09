"""Flowchart: where ESMDynamic's missing gradient starts and how it propagates through a head.

    python investigation/scripts/plot_gradient_flowchart.py
        [--out investigation/figures/gradient_flow/02_gradient_flowchart.png]

A diagram, not data: every claim in it is measured elsewhere --
  autocast cache + no_grad pass            notes/04, scripts/minimal_autocast_repro.py, notebook §10
  116 Linear tensors None, 44 LayerNorm 0  dyn_models esmdyn_block_grad.py (job 2153909)
  every other head part gets gradient      dyn_models esmdyn_grad_check.py (job 2153891)
  occupancy head's s path exactly 0        same job; notebook §8
  blocks stay at 0 over 500 steps          notes/07; with the fix they train
"""
import argparse
from pathlib import Path

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8985"
CAUSE = ("#f4f3f0", "#8a8985")          # neutral: the setup
NONE_ = ("#fbd9c6", "#c4521f")          # no gradient at all (.grad is None)
ZERO = ("#fdebe1", "#eb6834")           # in the graph, gradient exactly 0
OK = ("#e3edf9", "#2a78d6")             # receives gradient, trains
FIX = ("#e3f5ee", "#1b8f63")            # the fix


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="investigation/figures/gradient_flow/02_gradient_flowchart.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig = plt.figure(figsize=(14, 16.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(19.5, 140)
    ax.axis("off")

    def box(x, y, w, h, c, title, body="", tsize=10.5, bsize=9.2, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    fc=c[0], ec=c[1], lw=lw, zorder=2))
        ax.text(x + 1.1, y + h - 1.0, title, fontsize=tsize, color=INK, va="top", fontweight="bold", zorder=3)
        if body:
            ax.text(x + 1.1, y + h - 3.6, body, fontsize=bsize, color=INK_2, va="top", linespacing=1.4, zorder=3)

    def arrow(x0, y0, x1, y1, color=INK_2, ls="-", lw=1.5, text=None, tx=0, ty=0, rad=0.0):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color,
                                     lw=lw, ls=ls, zorder=1, connectionstyle=f"arc3,rad={rad}"))
        if text:
            ax.text((x0 + x1) / 2 + tx, (y0 + y1) / 2 + ty, text, fontsize=8.6, color=color, ha="center",
                    va="center", zorder=3, bbox=dict(fc="white", ec="none", pad=1.2))

    def section(y, n, label):
        ax.text(2, y, f"{n}", fontsize=15, color=MUTED, fontweight="bold", va="center")
        ax.text(5.5, y, label, fontsize=12, color=INK, va="center", fontweight="bold")

    # ---- title ----------------------------------------------------------------------------------
    ax.text(2, 138.6, "How the missing gradient starts, and how it propagates through an ESMDynamic head",
            fontsize=16, color=INK, va="top")
    ax.text(2, 136.0, "One training step as train.py runs it (bf16 autocast, cache on). Notation: W a Linear layer's "
            "weight, Ŵ the bf16 copy autocast uses, O a block's zero-initialised output layer,\nf(·) the block's "
            "inner computation, g = ∂L/∂y the gradient arriving at the block's output, sg(·) stop-gradient "
            "(a value with no autograd link to where it came from).", fontsize=9.6, color=INK_2, va="top",
            linespacing=1.45)

    # ---- 1. where it starts ---------------------------------------------------------------------
    section(128.5, 1, "Where it starts: the forward pass")
    box(4, 115, 28, 11, CAUSE, "train.py",
        "with torch.autocast(bf16):\n    out = model(seq)\ncache_enabled not set → on (default)")
    box(36, 115, 28, 11, CAUSE, "DynamicModule recycles 4×",
        "passes 1–3:  with torch.no_grad()\npass 4:  gradients on\n(dynamic_module.py, the recycle loop)")
    box(68, 115, 28, 11, NONE_, "Pass 1 fills autocast's cache",
        "each block Linear is cast once:  Ŵ = sg( bf16(W) )\nmade under no_grad → no link back to W;\ncached for the whole autocast region")
    arrow(32.4, 120.5, 35.6, 120.5)
    arrow(64.4, 120.5, 67.6, 120.5)
    box(36, 102.5, 60, 8.5, NONE_, "Pass 4 (the only pass with gradients) reuses the cached copies",
        "y = x + Ô · f(x; Ŵ₁, Ŵ₂, …)     every Ŵ is a constant to autograd, and Ô = 0 (output layers start at zero)\n"
        "so the block's output is exactly its input:  y = x")
    arrow(82, 114.6, 82, 111.4)

    # ---- 2. backward through one block -----------------------------------------------------------
    section(98, 2, "Backward through each Evoformer block (2 per head × 3 heads)")
    box(4, 79, 29, 16, NONE_, "Linear weights & biases",
        "58 per block (attention, gates, MLPs,\ntriangle updates, output layers O)\n\n"
        "∂L/∂W is never computed: W is not in\nthe graph, only the constant Ŵ is\n\n→  W.grad = None")
    box(36, 79, 29, 16, ZERO, "LayerNorm weights & biases",
        "22 per block. LayerNorm stays fp32, so it\nis not cached and stays in the graph, but\nits only route to the loss runs through O:\n\n"
        "∂L/∂γ  ∝  Ôᵀ g  =  0\n\n→  γ.grad = 0 (exactly)")
    box(68, 79, 28, 16, OK, "The block's input x",
        "the residual path skips the block:\n\n∂L/∂x  =  g  +  (…) · Ôᵀ g  =  g\n\n"
        "gradient passes straight through,\nunchanged, to whatever produced x")
    for x in (18.5, 50.5, 82):
        arrow(66, 102.1, x, 95.4, color=MUTED, lw=1.1)

    # ---- 3. around the blocks --------------------------------------------------------------------
    section(74.5, 3, "Around the blocks: who still gets gradient")
    box(4, 55, 44, 16.5, OK, "Pair-state (z) path → trains",
        "the prediction layer reads z, and g flows back along the residual\nto everything added into z before the blocks:\n"
        "  • prediction layer\n  • pair transition (ESMFold's pTM + distogram logits)\n"
        "  • positional embedding, recycling z-norm, distance bias\n"
        "these Linears run outside the no_grad passes, so their cached\ncopies keep their link → real gradients")
    box(52, 55, 44, 16.5, ZERO, "Sequence-state (s) path → cut off from the predictions",
        "s reaches z only INSIDE the blocks (sequence_to_pair, whose\noutput layer is 0), so for the main predictions  ∂L_pred/∂s = 0\n"
        "  • dynamic and kinetics heads: s still trains, but only\n    through the confidence head\n"
        "  • occupancy head (no confidence head): seq transition and\n    recycling s-norm get exactly 0 and stay at init")
    arrow(82, 78.6, 30, 71.9, color=OK[1], lw=1.4, text="g reaches the z path", tx=-6, ty=1.6, rad=0.06)
    arrow(82, 78.6, 74, 71.9, color=ZERO[1], lw=1.4)

    # ---- 4. optimiser ----------------------------------------------------------------------------
    section(50.5, 4, "The optimiser step (Adam): colours as above")
    box(4, 39, 29, 8.5, NONE_, "grad = None", "parameter skipped entirely  →  no update")
    box(36, 39, 29, 8.5, ZERO, "grad = 0", "moments stay 0; step = 0 / (√0 + ε)  →  no update")
    box(68, 39, 28, 8.5, OK, "grad ≠ 0", "normal update: transitions, embeddings,\nprediction and confidence layers learn")

    # ---- 5. next step ----------------------------------------------------------------------------
    section(34.5, 5, "Next step, and every step after: nothing has changed for the blocks")
    box(4, 22, 61, 9.5, NONE_, "O is still exactly 0  →  step 2 is identical to step 1  →  … forever",
        "the loss still falls (the blue parts learn), so training looks normal. In both released checkpoints:\n"
        "all 480 block tensors at their initial values; every block is an exact identity map.")
    box(68, 22, 28, 9.5, FIX, "The fix: autocast(cache_enabled=False)",
        "Ŵ is re-cast with its link on pass 4, so\n∂L/∂O = g · f(x)ᵀ ≠ 0 on step 1 → O moves →\n"
        "from step 2 the whole block gets gradient")
    arrow(18.5, 38.6, 18.5, 31.9, color=NONE_[1], lw=1.3)
    arrow(50.5, 38.6, 50.5, 31.9, color=ZERO[1], lw=1.3)
    arrow(64.4, 26.5, 67.6, 26.5, color=FIX[1], ls=(0, (3, 2)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(out)


if __name__ == "__main__":
    main()
