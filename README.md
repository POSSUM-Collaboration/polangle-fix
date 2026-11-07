# spice-racs-fix

Corrections to the SPICE-RACS DR2 catalogue.

Each step appends new columns to a copy of the catalogue. The original rows are
copied byte for byte, so every original column keeps its exact values and
layout. Nothing is ever changed in place.

## Install

As a command line tool:

```bash
uv tool install git+https://github.com/POSSUM-Collaboration/spice-racs-fix
```

Or run it once without installing:

```bash
uvx --from git+https://github.com/POSSUM-Collaboration/spice-racs-fix spice-racs-fix --help
```

As a dependency of your own project:

```bash
uv add git+https://github.com/POSSUM-Collaboration/spice-racs-fix
```

## Steps

### polframe

Catalogue polarisation angles are measured from north at the tile boresight, not
from north at each source. `polframe` adds local-north copies of the affected
columns with a `_corr` suffix, plus `pol_frame_gamma`, the rotation applied
(chi_true = chi + gamma):

| Original                     | Corrected                              |
| ---------------------------- | -------------------------------------- |
| `polangle`, `derot_polangle` | `polangle_corr`, `derot_polangle_corr` |
| `stokesQ`, `stokesU`         | `stokesQ_corr`, `stokesU_corr`         |
| `stokesQ_bkg`, `stokesU_bkg` | `stokesQ_bkg_corr`, `stokesU_bkg_corr` |
| `stokesQ_err`, `stokesU_err` | `stokesQ_err_corr`, `stokesU_err_corr` |

```bash
gunzip spice-racs.dr2.fits.gz
spice-racs-fix polframe spice-racs.dr2.fits spice-racs.dr2.polframe.fits
```

Boresights come from the RACS-low3 field list shipped with the package
(`spice_racs_fix/data/field_data.csv`); pass `--boresights` to use another. If
the catalogue has `separation_tile_centre`, every row is checked against its
boresight and the run stops if any row is more than 1 arcsec off.

From Python:

```python
from pathlib import Path

from spice_racs_fix.polframe import PolframeOptions, run_polframe

result = run_polframe(
    Path("spice-racs.dr2.fits"),
    Path("spice-racs.dr2.polframe.fits"),
    PolframeOptions(),
)
```

Same-source tile pairs confirm the sign of gamma relative to the measured
angles. A zero point or handedness shared by all tiles has not been checked
against a source of known absolute angle.

## Development

```bash
uv sync
uv run pytest
uv run prek run --hook-stage manual --all-files
```

`uv run prek install` sets up the git hook. The version comes from git tags
through hatch-vcs, so a release is a `vX.Y.Z` tag on `main`.

## License

MIT, see [LICENSE](LICENSE).
