# 12 — The published tables under the original evaluation protocol

Script: `scripts/eval_protocol.py` · results: `results/12_eval_protocol_{atlas,mdcath}.json`
(+ per-protein csv) · figure: `figures/evidence/E7_published_numbers_reproduce.png` · jobs
2158167 (ATLAS, 25 min) and 2158168 (mdCATH, 16 min), L40S. Targets: Nat Commun (2026)
17:9623 Tables 1–2, SI Tables 6–7. Supersedes the scoring in notes 06 and 10.

## What is fixed by the paper or the code, and followed

- labels from the Data Bank files as distributed; sequence read as `DynContactDataset` does;
- all L×L pairs, diagonal included (the training code's `length_mask_2d`, `metrics_*_batch`);
- threshold 0.5; per-protein mean ± se (Fig. 2: n = proteins, "independent samples");
- **occupancy = `frequency_pred`** = sigmoid((x + xᵀ)/2): the head's documented output, what
  `predict.py` writes and what the training loss and RMSE score. **Notes 06 and 10 used
  `clip(frequency_value, 0, 1)`, the raw unsymmetrised value — that alone caused the RMSE gap
  reported there** (mdCATH 320 K: 0.109 → 0.074; published 0.076).

## What is not stated, so both readings were run

- **crop**: whole chain, or the training code's crop (`DynContactDataset`: if L > 256,
  start = `torch.randint(0, L − 256)`), 5 random crops per protein. The paper mentions a crop
  only for training (Methods; SI Fig. 19 "supporting this crop size for training").
- **thresholded score**: `dynamic_prob` (symmetrised, what `predict.py` thresholds) or
  sigmoid(`dynamic_logits`) unsymmetrised (what `train.py`'s `metrics_dynamic_batch`
  thresholds).

## Results at 320 K (mean ± se)

| ATLAS, 82 chains | published | full / sym | full / unsym | crop256 / sym | **crop256 / unsym** |
|---|---|---|---|---|---|
| balanced acc. | 0.872 ± 0.010 | 0.880 | 0.872 | 0.882 | **0.873 ± 0.009** |
| precision | 0.284 ± 0.013 | 0.312 | 0.301 | 0.286 | **0.275 ± 0.012** |
| recall | 0.890 ± 0.011 | 0.889 | 0.876 | 0.906 | **0.893 ± 0.010** |
| F1 | 0.413 ± 0.016 | 0.447 | 0.433 | 0.420 | **0.406 ± 0.015** |
| AUROC | 0.942 ± 0.007 | 0.949 | 0.944 | 0.946 | **0.941 ± 0.007** |
| occupancy RMSE | 0.063 ± 0.003 | 0.059 | 0.059 | 0.063 | **0.063 ± 0.003** |

| mdCATH test, 270 domains | published | full / sym | full / unsym | crop256 / sym | **crop256 / unsym** |
|---|---|---|---|---|---|
| balanced acc. | 0.796 ± 0.007 | 0.806 | 0.800 | 0.806 | **0.800 ± 0.006** |
| precision | 0.511 ± 0.012 | 0.509 | 0.498 | 0.505 | **0.494 ± 0.012** |
| recall | 0.767 ± 0.010 | 0.772 | 0.764 | 0.773 | **0.765 ± 0.009** |
| F1 | 0.569 ± 0.008 | 0.570 | 0.561 | 0.567 | **0.557 ± 0.008** |
| AUROC | 0.889 ± 0.006 | 0.901 | 0.894 | 0.900 | **0.893 ± 0.006** |
| occupancy RMSE | 0.076 ± 0.002 | 0.074 | 0.074 | 0.075 | **0.075 ± 0.002** |

- **The training code's path (256 crop, unsymmetrised logits) reproduces all six ATLAS
  numbers within one standard error**, and all six mdCATH 320 K numbers within ~1.5 se. It
  is the most likely protocol behind Tables 1–2. Only 33 of 82 ATLAS chains and 22 of 270
  mdCATH domains exceed 256 residues, so the crop matters on ATLAS and barely on mdCATH.
- Every reading lands close; none changes a conclusion. The published results are the
  released weights' results.

## Open: balanced accuracy and AUROC at 348–450 K (mdCATH, SI Table 6)

Precision, recall, F1 and RMSE match SI Tables 6–7 at **every** temperature to within one
se (450 K: precision 0.790 vs 0.791, recall 0.994 vs 0.994, F1 0.853 vs 0.853, RMSE 0.056 vs
0.057). Balanced accuracy and AUROC do not, increasingly so with temperature (450 K: 0.753
vs 0.542, AUROC 0.880 vs 0.729), under every reading above. Since mean balanced accuracy =
(mean recall + mean specificity) / 2, the table implies a mean specificity of 0.090 at 450 K
against our 0.513 (0.735 vs 0.772 at 348 K). Matching TP and FP with a different TN points
at how negatives were counted in those two metrics at high temperature — a question for the
authors. It does not touch the DCM finding: the 320 K headline numbers match on all six
metrics.
