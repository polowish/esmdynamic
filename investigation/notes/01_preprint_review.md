# 01 — What the preprint says about the DCM, its training and its ablation

Sources: bioRxiv 10.1101/2025.08.20.671365, **v1** (35 pp) and **v2** (49 pp, posted
2026-04-21), read in full text. The supplement (Tables S4–S23, including the ablation
numbers) could not be downloaded: bioRxiv serves it behind a captcha
(`.../early/2026/04/21/2025.08.20.671365/DC1/embed/media-1.pdf`). Quotes below are v2 unless
marked.

## How the DCM is described

- "processed by three independent Dynamic Contact Modules (DCMs), each corresponding to a
  distinct prediction head" (v2 p.11); v1: "the Dynamic Contact Module (DCM), composed of
  two additional Evoformer blocks" (v1 p.7).
- Figure 1C (v2): ESMFold's last Evoformer block → pair/seq transitions with auxiliary
  outputs (LM, pLDDT, distogram and pTM logits) → "Dynamic contact module (3x heads):
  Evoformer (2 blocks)" with row & column attention, edge multiplicative update, triangle
  attention → prediction layer (MLP).
- "The Evoformer architecture is well-suited for these tasks, as it models complex,
  context-dependent relationships between residues" (p.11).

So the paper presents the DCM's blocks as a functioning, trained part of the model.

## How training is described (Methods, v2 p.40; v1 pp.26–27 give the same recipe)

- "All training was performed on a single NVIDIA RTX 4090 GPU. ESMFold weights were kept
  frozen throughout."
- Pretraining on RCSB clusters (classification head only): 10,000 train / 1,000 val samples
  per epoch, random 256-residue crops weighted by length, Adam lr 1e-4, focal loss α 0.99
  γ 2, **batch size 64**, early stopping after 10 epochs without validation improvement.
- Finetuning on mdCATH: same, with 1,000 / 100 samples per epoch, α 0.85; MSE for frequency;
  weighted cross-entropy for kinetics; confidence and residual heads enabled after 10 epochs.
- **Numerical precision is never mentioned** (no "mixed precision", "bf16", "autocast",
  "fp16" anywhere in either version). Training a 3B-LM + ESMFold forward with batch 64 on a
  24 GB card makes mixed precision very likely, which matches the bf16 autocast in
  `train.py` — consistent with, though not proof of, the mechanism we found.

## The ablation — the claim that conflicts with the weights

- v2 p.15: "we also evaluated ablated variants lacking auxiliary inputs or the DCM ... The
  reduced models generally show marginal to moderate degradation in performance ... the
  model without specialized Evoformer blocks can still attain good performance. However ...
  **the addition of auxiliary features and a dedicated DCM does improve the performance**,
  indicating the need for additional parameters to correctly predict difficult examples. An
  interesting exception is the on-time kinetics prediction, where the model with no DCM
  performs marginally better than the full model, although the difference is less than 1%."
- v2 p.16: "The full ESMDynamic model shows faster decay than ablated versions, indicating
  that its dynamic contact predictions are more tightly concentrated around the ground truth."
- v1 makes the same claims (p.9): "marginal degradation" without the DCM; "a dedicated DCM
  does improve the performance".
- **Definition of the DCM ablation** (v2 p.40): "the pairwise representation from the final
  Evoformer block of ESMFold was used in place of the DCM output."

## How the claim and the weights can both be true

The paper attributes the ablation's gap to "specialized Evoformer blocks" and "additional
parameters". In the released weights the blocks are identity maps, but **the DCM is not
empty**: outside its blocks it adds trained terms to the pair state — a learned positional
embedding, LayerNorm'd recycling of the previous pass, and a recycling distogram embedding
(`dynamic_module.py`; all trained, per our checkpoint audit). The no-DCM ablation removes
those too, because it feeds ESMFold's raw pair state to the prediction layer. So:

- **full model** = prediction( z_ESMFold + pair transition + positional + recycling terms )
- **no-DCM ablation** = prediction( z_ESMFold [+ transitions?] )

The reported "marginal to moderate" gain can therefore come entirely from the DCM's
non-block terms (or from run-to-run variation), with the blocks contributing nothing. The
ablation as designed cannot distinguish "the Evoformer blocks help" from "a few trained
embeddings help". This is the most important point to put to the authors: the paper's
interpretation of its own ablation rests on blocks that, in the released weights, do not
compute anything.

## Open items this raises

- The supplement's parameter table and ablation numbers: now read, see
  `09_supplement_ablation.md`.
- If the ablation models were trained with the same `train.py` (autocast on), their DCM-less
  variant is unaffected by the bug, and the comparison is "dead blocks + embeddings" vs
  "nothing" — check 6 (reproducing the ATLAS numbers) tells us whether the paper's main
  numbers came from these weights.
