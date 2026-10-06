"""Check 1: a minimal, ESMDynamic-free reproduction of the autocast weight-cache bug.

    python investigation/scripts/minimal_autocast_repro.py [--out investigation/results/01_minimal_autocast_repro.json]

`torch.autocast` caches the low-precision copy of each weight the first time it is cast,
for reuse until the autocast region exits. If that first cast happens under
`torch.no_grad()`, the cached copy has no autograd link to the parameter, and a later
grad-enabled use inside the same region silently reuses it: the parameter gets no gradient.
ESMDynamic's DynamicModule runs its first recycling passes under no_grad and trains under
bf16 autocast, which is exactly this pattern.

Part A: one nn.Linear, used once under no_grad and once with grad, in one autocast region.
Part B: a miniature of the DCM recycling loop -- a residual block whose output layer is
        zero-initialised, recycled 4 times with no_grad on all but the last pass.
Part C: three candidate fixes on Part B.
Needs a CUDA device (the cache is CUDA autocast's).
"""
import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn as nn


def grad_state(p):
    if p.grad is None:
        return "none"
    return "zero" if float(p.grad.abs().max()) == 0 else "nonzero"


def part_a(dtype, cache, noprior):
    torch.manual_seed(0)
    lin = nn.Linear(64, 64).cuda()
    # The input carries gradient but is NOT a leaf, as the DCM's input is (it comes out of
    # the trained transitions). Autocast caches the casts of every fp32 LEAF that requires
    # grad -- parameters, and also a leaf input -- so a leaf input would be cached too and
    # leave the output with no grad_fn at all (backward() then raises: the bug at its most
    # visible). A non-leaf input is cast fresh each time and keeps the graph alive.
    leaf = torch.randn(8, 64, device="cuda", requires_grad=True)
    x = leaf * 1.0
    with torch.autocast("cuda", dtype=dtype, cache_enabled=cache):
        if not noprior:
            with torch.no_grad():
                lin(x)                          # first cast happens here, without grad
        y = lin(x)                              # grad-enabled use of the same weight
    y.float().pow(2).sum().backward()
    return {"weight": grad_state(lin.weight), "bias": grad_state(lin.bias)}


class Block(nn.Module):
    """x + out(relu(inp(LayerNorm(x)))), with `out` zero-initialised (the Evoformer pattern)."""

    def __init__(self, d=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, 4 * d)
        self.out = nn.Linear(4 * d, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return x + self.out(torch.relu(self.inp(self.norm(x))))


def recycle(block, x0, n_passes=4, autocast_ctx=None, clear_cache=False):
    """The DynamicModule loop: no_grad on every pass except the last."""
    x = x0
    for i in range(n_passes):
        last = i == n_passes - 1
        if last and clear_cache:
            torch.clear_autocast_cache()
        ctx = autocast_ctx(last) if autocast_ctx else ExitStack()
        with ctx:
            with ExitStack() if last else torch.no_grad():
                x = block(x0 + x.detach() * 0.1)
    return x


def part_bc(variant):
    torch.manual_seed(0)
    block = Block().cuda()
    leaf = torch.randn(8, 64, device="cuda", requires_grad=True)
    x0 = leaf * 1.0                    # non-leaf, as above
    if variant == "as ESMDynamic trains: one bf16 autocast region, cache on":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = recycle(block, x0)
    elif variant == "fix 1: autocast(cache_enabled=False)":
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            y = recycle(block, x0)
    elif variant == "fix 2: no_grad passes outside the autocast region":
        y = recycle(block, x0, autocast_ctx=lambda last: torch.autocast(
            "cuda", dtype=torch.bfloat16) if last else ExitStack())
    elif variant == "fix 3: torch.clear_autocast_cache() before the grad pass":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = recycle(block, x0, clear_cache=True)
    elif variant == "no autocast (fp32)":
        y = recycle(block, x0)
    else:
        raise ValueError(variant)
    y.float().pow(2).sum().backward()
    return {n: grad_state(p) for n, p in block.named_parameters()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="investigation/results/01_minimal_autocast_repro.json")
    args = p.parse_args()
    assert torch.cuda.is_available(), "needs CUDA: the weight cache is CUDA autocast's"
    out = {"torch": torch.__version__, "cuda": torch.version.cuda,
           "device": torch.cuda.get_device_name(0), "A": [], "B_C": {}}
    print(f"torch {torch.__version__}  cuda {torch.version.cuda}  {out['device']}\n")

    print("A  one nn.Linear: a no_grad use, then a grad use, in one autocast region")
    for dtype in (torch.bfloat16, torch.float16):
        for cache in (True, False):
            for noprior in (False, True):
                r = part_a(dtype, cache, noprior)
                row = {"dtype": str(dtype).split(".")[1], "cache_enabled": cache,
                       "no_grad_use_first": not noprior, **r}
                out["A"].append(row)
                print(f"   {row['dtype']:<9} cache={str(cache):<5} no_grad use first={str(not noprior):<5}"
                      f" -> weight grad: {r['weight']:<8} bias grad: {r['bias']}")

    print("\nB/C  DCM-style recycling (zero-init output layer, no_grad on passes 1-3 of 4)")
    for v in ("as ESMDynamic trains: one bf16 autocast region, cache on",
              "fix 1: autocast(cache_enabled=False)",
              "fix 2: no_grad passes outside the autocast region",
              "fix 3: torch.clear_autocast_cache() before the grad pass",
              "no autocast (fp32)"):
        r = part_bc(v)
        out["B_C"][v] = r
        print(f"   {v}")
        print("      " + "  ".join(f"{n}: {s}" for n, s in r.items()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
