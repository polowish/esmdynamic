# 06 — The released weights reproduce the preprint's ATLAS-test result (check 6)

Script: `scripts/atlas_reproduction.py` · results: `results/06_atlas_reproduction.json`,
`results/06_atlas_reproduction_per_protein.csv` · figure:
`figures/reproduction/06_atlas_reproduction.png` · job 2156989 (L40S, ~5 min).
Data: Illinois Data Bank `atlas_test.zip` (datafile `2lewl`): 82 chains of the AlphaFlow
ATLAS test split, L 39–724, binary dynamic-contact and contact-occupancy maps.

**Preprint (v2 p.17): ESMDynamic reaches a balanced accuracy of 87% on the ATLAS test set.**

**Released V2 weights, 320 K output, per-protein mean: 0.880 ± 0.010 (82/82 proteins).**

| model output | per-protein mean (± se) | pooled | recall (pooled) | precision (pooled) |
|---|---|---|---|---|
| **320 K** | **0.880 (0.010)** | 0.906 | 0.858 | 0.265 |
| 348 K | 0.865 (0.012) | 0.917 | 0.911 | 0.184 |
| 379 K | 0.827 (0.014) | 0.912 | 0.955 | 0.121 |
| 413 K | 0.728 (0.017) | 0.849 | 0.989 | 0.061 |
| 450 K | 0.558 (0.006) | 0.598 | 1.000 | 0.023 |

- Prediction = probability > 0.5, as in `train.py`'s `metrics_dynamic_batch`. All L×L pairs;
  the upper triangle alone gives the same to 3 decimals.
- 320 K is the output nearest ATLAS's 300 K ensembles; hotter outputs predict ever more
  dynamic contacts (recall → 1, precision → 0), as expected.
- Contact-occupancy RMSE at 320 K: 0.087 (the preprint's ATLAS value for ESMDynamic is in
  Table S21, which we could not download).

**Conclusion.** The preprint's headline ATLAS number is produced by the released weights,
in which every DCM block is an identity map (check 5). So the published accuracy was
achieved without the DCM's Evoformer blocks contributing anything: it comes from ESMFold
plus the heads' shallow trained terms. It also means the reported performance is a lower
bound on what the intended architecture could do: whether a DCM with trained blocks
does better is open, and is what check 7 addresses.
