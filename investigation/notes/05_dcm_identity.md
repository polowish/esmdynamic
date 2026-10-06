# 05 — Deleting every DCM block changes nothing, bit for bit (check 5)

Script: `scripts/dcm_identity_test.py` · result: `results/05_dcm_identity.json` · job
2156988 (L40S). Released V2 weights, eval mode, upstream's default precision.

ESMFold runs once per protein; the three heads then run on identical copies of its output.
All 11 head outputs (`dynamic_logits/_prob/_pred/_confidence`,
`kinetic_logits/_prob/_pred_class/_confidence`, `frequency_value/_pred/_residual_pred`) are
compared with `torch.equal`.

| protein (L) | rerun (determinism) | **all DCM blocks removed** | control: one block output layer set to 1e-3 |
|---|---|---|---|
| ubiquitin (76) | 11/11 identical, max diff 0 | **11/11 identical, max diff 0** | 8/11 identical, max diff 3.7e-3 |
| GB1 (56) | 11/11, 0 | **11/11, 0** | 8/11, 3.6e-3 |
| hen lysozyme (129) | 11/11, 0 | **11/11, 0** | 8/11, 3.6e-3 |

- Removing the blocks = `head.dynamic_module.blocks = nn.ModuleList()` in all three heads, so
  each DCM keeps only its positional embedding and recycling terms.
- The control perturbs one pair-MLP output layer in the dynamic head's first block; the 3
  outputs that change are that head's logits (max diff ~3.6e-3), probabilities (~8e-4) and
  binary predictions (some pairs cross 0.5). Its confidence reads the per-residue state,
  which this perturbation does not reach, and the other two heads are separate modules — so
  the comparison detects a block that computes anything, exactly where it should.

**So in the released model the DCM's Evoformer blocks are, exactly, dead code**: the
model with them and the model without them are the same function. Anything the DCM
contributes comes from its non-block terms (positional embedding, recycling norms and the
recycling distogram embedding) — which matters for reading the preprint's DCM ablation
(see `01_preprint_review.md`).
