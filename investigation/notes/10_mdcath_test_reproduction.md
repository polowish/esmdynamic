# 10 — The released weights on the mdCATH test split, against SI Table 6

Script: `scripts/mdcath_test_reproduction.py` · results: `results/10_mdcath_test_reproduction.json`
(+ per-protein csv) · figure: `figures/reproduction/11_paper_comparison.png` (with ATLAS and
the ablations) · job 2158082 (L40S, 8 min). Data: Data Bank `mdcath.zip`, `test.csv` (270
domains), each scored at full length, every model temperature against the same-temperature
MD labels, probability > 0.5, all L×L pairs, mean ± se over proteins.

| temperature | ours: bal. acc. (± se) | preprint SI Table 6 (test) | ours: occupancy RMSE | preprint SI Table 7 |
|---|---|---|---|---|
| **320 K** | **0.806 ± 0.006** | **0.796 ± 0.007** | 0.109 ± 0.002 | 0.076 ± 0.002 |
| 348 K | 0.793 ± 0.006 | 0.772 ± 0.007 | 0.108 | 0.074 |
| 379 K | 0.778 ± 0.006 | 0.734 ± 0.008 | 0.107 | 0.072 |
| 413 K | 0.758 ± 0.008 | 0.657 ± 0.008 | 0.103 | 0.068 |
| 450 K | 0.753 ± 0.011 | 0.542 ± 0.007 | 0.094 | 0.057 |

- **320 K reproduces** (within ~1 se), as does the ATLAS headline (note 06): the paper's
  headline numbers come from these weights.
- **The other temperatures and the occupancy RMSE do not.** Our full-length scores are higher
  than the table's at 348–450 K, increasingly so, and our RMSE is ~0.035 higher throughout.
  The preprint does not specify the evaluation protocol beyond the metrics; the training
  code's own validation scores 256-residue crops pooled per batch, which would change both
  numbers. Not resolved here; worth asking the authors which protocol produced SI Tables 6–7.
  It does not affect the DCM finding: every number in these tables comes from models whose
  DCM blocks never trained.
