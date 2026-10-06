# 04 — Minimal, ESMDynamic-free reproduction of the mechanism (check 1)

Script: `scripts/minimal_autocast_repro.py` · result: `results/01_minimal_autocast_repro.json`
· figure: `figures/gradient_flow/01_minimal_autocast_repro.png` · job 2156988 (L40S,
torch 2.8.0+cu129 — the version ESMDynamic's install instructions pin).

**`torch.autocast` caches the low-precision cast of every fp32 leaf tensor that requires
grad (weights, biases, and leaf inputs) the first time it is made, until the autocast
region exits. A cast made under `torch.no_grad()` is cached with no autograd link, and a
later grad-enabled use in the same region reuses it, so the parameter gets no gradient.**

## A — one `nn.Linear`: a `no_grad` use, then a grad use, in one autocast region

| dtype | `cache_enabled` | `no_grad` use first | weight grad | bias grad |
|---|---|---|---|---|
| bfloat16 | True (default) | **yes** | **none** | **none** |
| bfloat16 | True | no | nonzero | nonzero |
| bfloat16 | False | yes | nonzero | nonzero |
| float16 | True (default) | **yes** | **none** | **none** |
| float16 | True | no | nonzero | nonzero |
| float16 | False | yes | nonzero | nonzero |

The prior `no_grad` use is necessary and sufficient (with the cache on) for the gradient to
vanish, in both half-precision types.

## B/C — the DCM's recycling pattern, and three fixes

A residual block `x + out(relu(inp(LayerNorm(x))))` with `out` zero-initialised (the
Evoformer pattern), recycled 4 passes with `no_grad` on all but the last — exactly
`DynamicModule.forward`.

| setup | `inp` | `out` |
|---|---|---|
| **as ESMDynamic trains: one bf16 autocast region, cache on** | **none** | **none** |
| fix 1: `torch.autocast(..., cache_enabled=False)` | zero | **nonzero** |
| fix 2: `no_grad` passes outside the autocast region | zero | **nonzero** |
| fix 3: `torch.clear_autocast_cache()` before the grad pass | zero | **nonzero** |
| no autocast (fp32) | zero | **nonzero** |

With any fix the zero-initialised output layer gets gradient on the first step, which is
what lets the block start learning (`inp` and the LayerNorm are behind a zero layer on step
1, so exactly 0 is correct there; they move once `out` does). Without a fix nothing moves,
ever: the block stays at initialisation for the whole of training — what both released
checkpoints show.

## Two details worth knowing

- **Leaf inputs are cached too.** A first version of this test used a leaf input
  (`torch.randn(..., requires_grad=True)`); its cast was cached under `no_grad` as well, the
  output had no `grad_fn` at all, and `backward()` raised. The DCM's input is a non-leaf
  (it comes from the trained transitions), so in ESMDynamic the graph survives and the
  failure is silent — the loss goes down through the other layers and nothing errors.
- **The silence is the danger.** Training runs, the loss falls, checkpoints save: only the
  block parameters are missing from the update, and nothing in a loss curve shows it.

## Suggested upstream fix

In `train.py`, the two `torch.autocast(...)` calls gain `cache_enabled=False` (one
line each). Equivalent alternatives: run the DCM's `no_grad` recycling passes outside the
autocast context, or call `torch.clear_autocast_cache()` before the grad-enabled pass. Any
of them needs the heads retrained.
