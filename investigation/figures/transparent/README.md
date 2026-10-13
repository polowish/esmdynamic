# Transparent figures

Presentation versions of figures: transparent background, no figure title or caption, text in
Arial, and the same 0-1 y axis in every panel (so panels compare directly; the captioned
originals zoom each panel to its data). Each is made by its plot script's `--clean` option. The captioned originals live in the
other folders and are the reference for what each figure shows.

| file | script | captioned original |
|---|---|---|
| `13_si_table6_reproduction.png` | `python investigation/scripts/plot_si_table6_reproduction.py --clean` | `figures/reproduction/13_si_table6_reproduction.png` (mdCATH test, SI Table 6; note 13) |
| `13_atlas_table2_reproduction.png` | `python investigation/scripts/plot_atlas_table2_reproduction.py --clean` | `figures/reproduction/13_atlas_table2_reproduction.png` (ATLAS test, Table 2; note 12) |
| `15_trained_vs_untrained_params.png` | `python investigation/scripts/plot_param_pies.py` (counts: `param_counts.py` → `results/15_param_counts.json`) | none: made for this folder. Per head of the released V2 weights, parameters trained (teal) vs still at initialisation (dark blue: every DCM block tensor, plus the occupancy head's single-state path) |
