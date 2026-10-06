# 02 — Did the V1-era training code have the same bug? (check 2)

**Yes. Both ingredients of the mechanism are in the code as of the V1 release, unchanged since.**

| ingredient | June 2025 (`b51b35f`, 2025-06-22, the V1 release window) | today (`b3c7f04`) |
|---|---|---|
| bfloat16 autocast around the training forward | `train.py` l.157: `with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast_enabled): output = model.forward_from_seq(sequences)` | `train.py` l.470, same call |
| early recycling passes under `no_grad` | `dynamic_module.py` l.113: `with ExitStack() if recycle_idx == no_recycles - 1 else torch.no_grad():` | identical |
| `autocast(cache_enabled=...)` set | no (defaults to on) | no |

- `git diff b51b35f HEAD -- esm/esmdynamic/dynamic_module.py esm/esmfold/v1/tri_self_attn_block.py`
  is **empty**: the DCM and its blocks have not changed since June 2025. `dynamic_module.py`
  dates from the first implementation (2023-04) with one change in 2023-11.
- So the V1 weights (Data Bank V1, 2025-06) and the V2 weights (2026-04) were both produced
  by code in which no gradient reaches the DCM blocks — consistent with both checkpoints
  having every block tensor at its initial value (`dyn_models/scripts/diagnostics/esmdyn_checkpoint_audit.py`).
- The June 2025 optimiser was `torch.optim.Adam(model.parameters(), lr=lr)` — everything,
  ESMFold included (frozen by `requires_grad_(False)` in the model constructor). The current
  one is `Adam(model.heads[h].parameters())`. Either way the blocks are in the optimiser;
  they simply never receive a gradient.

Commands: `git show b51b35f:esm/esmdynamic/training/train.py`,
`git show b51b35f:esm/esmdynamic/dynamic_module.py`, `git log -- esm/esmdynamic/dynamic_module.py`.
