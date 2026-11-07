# Next steps

Decisions for the steps after the package refactor, with the DR2 measurements
behind them. Every step appends columns (or, for Stokes V, edits metadata only)
and takes its parameters from SPICE-RACS data alone.

## Pairs used for the error model

Rows sharing a `cat_id` form pairs. DR2 has 4,513,331 overlap pairs (different
`tile_id`) and 516,464 repeat pairs (same `tile_id`, different SBID); no pair
shares both tile and SBID.

- Repeat pairs sit at the same position in the same tile's beam, so anything
  fixed to beam position (residual leakage, beam model) cancels. They measure
  only what varies between observations.
- Overlap pairs see both, but are concentrated near tile edges: half their
  members lie beyond 2.9 deg, and only 145 goodRM pairs have both members
  within 1.5 deg.

MADstd(dRM / sigma_dRM), goodRM in both, Faraday complex rows excluded, by the
larger separation of the pair:

| Separation (deg) | Overlap | Repeat |
| ---------------- | ------- | ------ |
| 1.5 to 2         | 1.27    | 1.16   |
| 2.5 to 3         | 1.21    | 1.13   |
| 3 to 3.5         | 1.26    | 1.21   |
| 3.5 to 4         | 1.45    | 1.14   |
| above 4          | 1.77    | 1.17   |

## fit-error-model

`spice-racs-fix fit-error-model CATALOGUE PARAMS.json`: fits the error model
from pairs and writes a versioned parameter file (fitted values, bootstrap
errors, pair counts, binned diagnostics). Adds no columns.

- Two components. From repeat pairs: the part that varies between
  observations, sigma^2 = (k e)^2 + a^2, applied everywhere. From overlap pairs,
  with that held fixed: the beam position part, a function of separation plus
  `c_L * leakage / fracpol` using the catalogue's `leakage` column.
- Separation resolved. Nothing fitted at the edge is applied to inner rows.
  Where overlap pairs are too sparse (under about 2.5 deg) the beam position
  part comes from the leakage term alone. Rows outside the supported range get
  NaN, and a header card gives the range.
- Supporting data: overlap MADstd rises with `leakage / fracpol` from 1.05
  (under 0.01) to 1.35 (0.05 to 0.1), 1.76 (0.1 to 0.2) and 3.34 (above 0.2).
- Spectral index: pairs with goodI and `stokesI_model_order >= 1` in both.
  Compare at the pair's mean `reffreq_I`, using
  `alpha + 2 beta log10(nu / reffreq_I)` with beta the third to last value of
  `stokesI_model_coef` (order 2 only). Some order 2 fits have extreme curvature
  (shift p99 0.30, max 23.8), so use a robust loss and report how many pairs
  moved by more than their error.
- Spectral index floor: about 0.107 per measurement for bright pairs, the same
  for repeat and overlap pairs and at all separations. Fit one floor per model
  order (about 0.088 for order 1, 0.120 for order 2). A per SBID offset explains
  only about 0.04 of it.

## add-systematic-errors

`spice-racs-fix add-systematic-errors IN PARAMS.json OUT` appends:

- `rm_err_sys`, kept separate from `rm_err`. Header and README say it may
  include real RM variability (the paper treats the excess as a lower limit on
  reliability).
- `spectral_index_err_sys`: the floor for the row's model order; NaN for flat
  (order 0) models.

## add-observation-times

`start_time`, `int_time` and `epoch` are wrong for 41.5% of DR2 rows. In
Arrakis `makecat`, a single SBID run overwrites `sbid`, `field_name` and
`tile_id` but keeps `start_time` from the PyBDSF component, and
`get_integration_time` is called without the SBID, so re-observed fields get
the highest SBID's value. `tile_id` and `sbid` are right for every row.

- Append `start_time_corr`, `int_time_corr` and `epoch_corr` from
  `field_data.csv` (`SCAN_START`, `SCAN_TINT`) keyed on (`tile_id`, `sbid`), as
  `paper_plots/time.ipynb` in spice-racs-dr2-plots already does.
- Also fix Arrakis upstream (draft for review before posting).

## Stokes V metadata

`stokesV` and `stokesV_err` are NaN in every row. Set their UCDs to
`phys.polarization.stokes.V` (with `stat.error` for the error column) and their
unit to `Jy beam-1`, recording the change in a header card. Data bytes stay
identical.

## Validation of rotate-to-local-north

Run it on DR2, then repeat the Fig. 7 calibrator comparison (PB13, TL24) with
`polangle_corr` and `derot_polangle_corr`. That tests the zero point and
handedness, which tile pairs cannot.

## Dropped

- No flag columns. The de-duplication flag is out of scope, out of range
  `fracpol` is already excluded by goodRM, and the spectral index is not
  flatter at tile edges (farther minus nearer member: -0.024 to +0.004).
- `beamdist` equals `separation_tile_centre` by construction; nothing to do.
