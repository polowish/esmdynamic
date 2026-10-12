# 13 — SI Table 6's high-temperature numbers: pairs within 2 residues were excluded

Script: `scripts/eval_pairset.py` · results: `results/13_eval_pairset_{atlas,mdcath}.json` (+
per-protein csv) · jobs 2199270 (ATLAS, 14 min) and 2199271 (mdCATH, 9 min), L40S, released V2
weights, full length, per-protein mean ± se. Resolves the open item in note 12.

## The question

Note 12 found that at 348–450 K our precision, recall and F1 matched SI Table 6 to the third
decimal while balanced accuracy and AUROC did not (450 K: published 0.542 / 0.729 vs our
0.753 / 0.880), implying a published specificity of 0.09 against our 0.51. Hypothesis: the
paper scored only pairs with |i − j| ≥ k, whereas we scored all L × L pairs. Pairs close in
sequence are always in contact, hence never dynamic, and the model gets them right; at high
temperature (median 90% of pairs dynamic at 450 K) they are a large share of the negatives.

Test: the same predictions, scored on |i − j| ≥ k for k = 0 (all pairs), 1, 2, 3, 4, 6, 12.

## Result: k = 3, symmetrised probabilities, reproduces SI Table 6

mdCATH test (270 domains), `dynamic_prob` (symmetrised) thresholded at 0.5:

| | 320 K | 348 K | 379 K | 413 K | 450 K |
|---|---|---|---|---|---|
| **AUROC, published** | 0.889 ± 0.006 | 0.873 ± 0.007 | 0.841 ± 0.008 | 0.796 ± 0.008 | 0.729 ± 0.007 |
| AUROC, all pairs (k = 0) | 0.901 | 0.893 | 0.880 | 0.876 | 0.880 |
| **AUROC, \|i − j\| ≥ 3** | **0.889 ± 0.006** | **0.873 ± 0.007** | **0.841 ± 0.008** | **0.795 ± 0.008** | **0.728 ± 0.007** |
| **balanced acc., published** | 0.796 ± 0.007 | 0.772 ± 0.007 | 0.734 ± 0.008 | 0.657 ± 0.008 | 0.542 ± 0.007 |
| balanced acc., k = 0 | 0.806 | 0.793 | 0.778 | 0.758 | 0.753 |
| **balanced acc., \|i − j\| ≥ 3** | **0.796 ± 0.007** | **0.771 ± 0.008** | **0.732 ± 0.008** | 0.645 ± 0.008 | 0.517 ± 0.003 |
| specificity, published (implied) | 0.825 | 0.735 | 0.624 | 0.396 | 0.090 |
| specificity, \|i − j\| ≥ 3 | 0.821 | 0.729 | 0.616 | 0.373 | 0.040 |
| precision / recall / F1 | unchanged from k = 0 (matched already, note 12) | | | | |

- **AUROC matches to the third decimal at every temperature, standard errors included.**
  Balanced accuracy matches exactly at 320–379 K and falls 0.012 / 0.025 short at 413 / 450 K.
- **The step is between k = 2 and k = 3**: summed |published − ours| over the five temperatures
  drops from 0.283 to 0.039 (balanced accuracy) and from 0.233 to 0.003 (AUROC). k = 4, 6, 12
  are progressively worse. So the excluded pairs are a residue with itself and its two nearest
  chain neighbours (|i − j| ≤ 2), whose Cα–Cα distances (0, ~3.8, ≤ ~7 Å) never exceed 8.5 Å.
- **Precision, recall and F1 do not move** from k = 0 to k = 3: no pair within 2 residues is a
  true or predicted positive. That is why those three matched under every protocol.
- The symmetrised scores fit better than the unsymmetrised logits here (summed AUROC error 0.003
  vs 0.054), the opposite of what note 12's crop-and-logits reading suggested for Table 2.
  Full length here vs crops there; the two notes together say the published mdCATH numbers come
  from full-length (or near-full-length) predictions with the symmetrised probability and the
  near-diagonal pairs excluded.
- **ATLAS barely depends on k** (balanced accuracy 0.880 → 0.877 symmetrised; 0.872 → 0.868
  unsymmetrised): its proteins have few dynamic pairs, so negatives are plentiful and distant.

## What it means

- **The open discrepancy is explained.** It was an evaluation-protocol difference, not a
  different model: the released weights reproduce SI Table 6 once the same pairs are scored.
- **It also says something about the model.** On the non-trivial pairs at 450 K, the released
  model's specificity is 0.04 and its balanced accuracy 0.517, close to chance: it labels nearly
  every pair at least 3 residues apart as dynamic. Scoring all L × L pairs hides this (0.753),
  because the easy near-diagonal negatives prop up specificity.
- Nothing here touches the DCM finding.

## The remaining balanced-accuracy gap: proteins with no negatives, scored by scikit-learn

With the pair set fixed, AUROC matched exactly but balanced accuracy was still 0.012 / 0.025
short at 413 / 450 K. AUROC is threshold-free and matched with its standard errors, so the
predictions and pairs are right and the difference had to be in how balanced accuracy is
computed per protein or averaged.

At 450 K, 14 of the 270 test domains (small ones: 51–76 residues against a median of 119) have
**no non-dynamic pair at |i − j| ≥ 3**: every scored pair forms and breaks contact. Specificity
is undefined for them; this script drops them. `sklearn.metrics.balanced_accuracy_score` does
not: it averages recall over the classes present, so for these proteins it returns the recall
alone (verified: an all-positive protein with recall 0.96 scores 0.960, with only a
"y_pred contains classes not in y_true" warning). The model calls every pair of these domains
dynamic, so each scores **1.000**. `roc_auc_score` is undefined on the same proteins, so a
NaN-skipping mean leaves them out of AUROC, which is why AUROC matched without this.

Scoring them the scikit-learn way:

| | 320 K | 348 K | 379 K | 413 K | 450 K |
|---|---|---|---|---|---|
| proteins with no negatives at \|i − j\| ≥ 3 | 0 | 0 | 1 | 7 | 14 |
| balanced acc., published | 0.796 ± 0.007 | 0.772 ± 0.007 | 0.734 ± 0.008 | 0.657 ± 0.008 | 0.542 ± 0.007 |
| ours, undefined proteins dropped | 0.796 ± 0.007 | 0.771 ± 0.008 | 0.732 ± 0.008 | 0.645 ± 0.008 | 0.517 ± 0.003 |
| **ours, scikit-learn-style** | **0.796 ± 0.007** | **0.771 ± 0.008** | **0.733 ± 0.008** | **0.655 ± 0.009** | **0.542 ± 0.007** |
| AUROC proteins (defined) | 270 | 270 | 269 | 263 | 256 |

**SI Table 6 is now reproduced in full**, mean and standard error, at every temperature, by the
released weights scored on |i − j| ≥ 3 with symmetrised probabilities and scikit-learn's
per-protein balanced accuracy.

What it means for the 450 K number: 14 fully-dynamic small domains, each a trivially perfect
1.0, lift the mean from 0.517 to 0.542. On the 256 proteins where balanced accuracy is defined,
the released model is at 0.517, essentially chance, because it labels almost every pair at
least 3 apart as dynamic at that temperature. Both effects inflate the high-temperature
numbers in the same direction; neither is mentioned in the paper.
