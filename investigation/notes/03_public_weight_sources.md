# 03 — Which weights do the public entry points load? (check 4)

**All of them load the same file: Illinois Data Bank datafile `7odsk`,
`esmdynamic_model_weights_V2.pt` (383,731,643 bytes; sha256 `b41a1a22e9afd664…`), whose DCM
blocks are untrained.**

| entry point | how it gets weights | code it runs |
|---|---|---|
| `esm.pretrained.esmdynamic()` (`esm/esmdynamic/pretrained.py`) | `torch.hub.load_state_dict_from_url("https://databank.illinois.edu/datafiles/7odsk/download")`, `strict=False` after checking no non-ESMFold key is missing | ShuklaGroup |
| `run_esmdynamic` (`esm/esmdynamic/predict.py` l.626) | `esm.pretrained.esmdynamic()` | ShuklaGroup |
| Dockerfile | installs `git+https://github.com/ShuklaGroup/esmdynamic.git`; weights fetched by the line above on first use | ShuklaGroup |
| Colab notebook (`examples/esmdynamic/esmdynamic.ipynb`, cell 2) | `aria2c https://databank.illinois.edu/datafiles/7odsk/download --out esmdynamic.pt`; `load_state_dict(..., strict=False)` | `git+https://github.com/diegoeduardok/esmdynamic.git` |

- `diegoeduardok/esmdynamic` (not a GitHub fork; pushed 2026-06-26) is at the same commit as
  `ShuklaGroup/esmdynamic` (`b3c7f04`), with the same `autocast` / `no_grad` lines.
- The Data Bank's V1 dataset (IDB-3773897, 2025-06) holds a different, smaller file,
  `esmdynamic_model_weights.pt` (126,488,478 bytes, datafile `jx4ui`) for the single-head V1
  model; no current code path loads it. Its DCM blocks are also untrained (160/160 at init).
- Our cluster install: the V2 file above, identical checksum; code at upstream HEAD.

So the issue is not specific to any install path: every current user of ESMDynamic gets
the untrained-block model.
