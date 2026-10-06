# 08 — Human-proteome predictions vs the released weights: skipped (check 8)

**Decision: not run.** The question it answers — were the published human-proteome
predictions (Data Bank `human_proteome_preds_*.tar.xz`, 2.5–40 GB each) made by the released
weights? — is already settled well enough by two cheaper checks:

- Check 4 (`03_public_weight_sources.md`): every public entry point loads one file, Data Bank
  V2 `7odsk`. No other V2-era weight file has been published.
- Check 6 (`06_atlas_reproduction.md`): those weights reproduce the preprint's own ATLAS
  result (0.880 ± 0.010 vs 87%), so the paper's results were produced by them.

Comparing a few proteins against the proteome archive would add a third confirmation at
the cost of a multi-gigabyte download, and could not change any conclusion: the archives
can only have been made by this model or by an unpublished one, and an unpublished model
would not change what users of the released one get. If the authors state the proteome was
made with a different checkpoint, this check becomes worth doing (smallest archive:
`human_proteome_preds_19.tar.xz`, 2.5 GB, Data Bank datafile `nufzh`).
