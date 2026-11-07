"""Local-north polarisation columns for the SPICE-RACS DR2 catalogue

ASKAP catalogue angles are measured from north at the observing boresight, not
from north at each source. This step adds corrected copies of the affected
columns (suffix ``_corr``) and leaves every original column untouched:

    polangle, derot_polangle        -> + gamma, wrapped to the original range
    stokesQ/U, stokesQ/U_bkg        -> (Q + iU) exp(2i gamma)
    stokesQ_err, stokesU_err        -> rotated errors (Q and U independent)

plus ``pol_frame_gamma`` (deg). Frame-independent columns (rm, polint, fracpol,
stokesI/V, polangle_err, ...) are not touched.

Boresights come from a CSV with FIELD_NAME, RA_DEG, DEC_DEG, looked up by each
row's tile. If the catalogue has a ``separation_tile_centre`` column, every
row's separation from its boresight is checked against it. This catches a CSV
from a different RACS epoch, and rows whose tile does not name the field they
were measured in.

Sign convention: chi_true = chi_measured + gamma. Same-source tile pairs fix the
sign of gamma relative to the measured angles (the dependence of the angle
difference on the gamma difference goes from slope -0.90 to +0.01). They cannot
fix a zero point or handedness shared by all tiles; check that against a source
of known absolute angle before release.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy import units as u
from astropy.table import Table
from numpy.typing import ArrayLike, NDArray

from spice_racs_fix.fits_io import (
    CatalogueError,
    HeaderCard,
    NewColumn,
    StepResult,
    WriteOptions,
    append_columns,
    check_output,
    column_f64,
    open_catalogue,
    require_columns,
)
from spice_racs_fix.geometry import (
    gamma_deg,
    rotate_qu,
    rotate_qu_errors,
    sep_deg,
    wrap_angle,
)
from spice_racs_fix.logging import logger


@dataclass(frozen=True, kw_only=True, slots=True)
class PolframeOptions:
    """Options for the polarisation frame correction"""

    boresights: Path = Path(__file__).parent / "data" / "field_data.csv"
    """CSV with FIELD_NAME, RA_DEG, DEC_DEG; defaults to the RACS-low3 fields"""
    tile_col: str = "tile_id"
    """Column naming each row's tile"""
    sep_col: str = "separation_tile_centre"
    """Column used to cross-check boresights"""
    angle_cols: tuple[str, ...] = ("polangle", "derot_polangle")
    """Angle columns to rotate"""
    qu_cols: tuple[tuple[str, str], ...] = (
        ("stokesQ", "stokesU"),
        ("stokesQ_bkg", "stokesU_bkg"),
    )
    """Q/U column pairs to rotate"""
    qu_err_cols: tuple[str, str] = ("stokesQ_err", "stokesU_err")
    """Q/U error columns to propagate"""
    gamma_col: str = "pol_frame_gamma"
    """Name of the new gamma column"""
    angle_min: float | None = None
    """Lower edge of the angle range, 0 or -90; None infers it from the data"""
    sep_tol_arcsec: float = 1.0
    """Largest allowed difference from separation_tile_centre; inf skips it"""


class TileBoresights(NamedTuple):
    """Boresight of every row"""

    ra0: NDArray[np.float64]
    """Boresight right ascension per row (deg), NaN for an empty tile name"""
    dec0: NDArray[np.float64]
    """Boresight declination per row (deg), NaN for an empty tile name"""


class PolframeColumns(NamedTuple):
    """New columns and diagnostics from the frame correction"""

    step: StepResult
    """Columns and keywords to append"""
    gamma: NDArray[np.float64]
    """Rotation per row (deg)"""
    max_sep_diff_arcsec: float | None
    """Worst boresight cross-check difference, or None if not checked"""


class PolframeResult(NamedTuple):
    """Outcome of the polarisation frame correction"""

    n_rows: int
    """Rows written"""
    n_no_gamma: int
    """Rows with no finite gamma (their _corr columns are NaN)"""
    max_sep_diff_arcsec: float | None
    """Worst boresight cross-check difference, or None if not checked"""


def tile_names(col: ArrayLike) -> tuple[NDArray[np.str_], NDArray[np.intp]]:
    """Distinct stripped tile names, and each row's index into them.

    Names are decoded once per distinct value rather than once per row, which
    matters for catalogues with millions of rows.

    Args:
        col: Tile name column (bytes or str).

    Returns:
        Distinct names and each row's index into them.
    """
    uniq, inv = np.unique(np.asarray(col), return_inverse=True)
    names = [
        (x.decode("ascii", "replace") if isinstance(x, bytes) else str(x)).strip()
        for x in uniq.tolist()
    ]
    return np.array(names, dtype=str), np.asarray(inv, dtype=np.intp).reshape(-1)


def read_boresights(path: Path) -> dict[str, tuple[float, float]]:
    """Field centres by name; the first row wins for re-observed fields.

    Args:
        path: CSV with FIELD_NAME, RA_DEG, DEC_DEG.

    Returns:
        (RA, Dec) in degrees by field name.
    """
    centres: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            name = row["FIELD_NAME"].strip()
            centres.setdefault(name, (float(row["RA_DEG"]), float(row["DEC_DEG"])))
    return centres


def boresights_for_rows(table: Table, options: PolframeOptions) -> TileBoresights:
    """Look up each row's boresight from its tile name.

    Args:
        table: The catalogue table.
        options: Polframe options.

    Returns:
        Per-row boresights.

    Raises:
        CatalogueError: If a non-empty tile name has no boresight.
    """
    names, idx = tile_names(table[options.tile_col])
    centres = read_boresights(options.boresights)
    unknown = [n for n in names if n and n not in centres]
    if unknown:
        msg = f"{options.boresights}: no boresight for {len(unknown)} tiles, e.g. {unknown[:5]}"
        raise CatalogueError(msg)
    lookup = np.array(
        [centres.get(n, (np.nan, np.nan)) for n in names], dtype=np.float64
    )
    return TileBoresights(ra0=lookup[idx, 0], dec0=lookup[idx, 1])


def check_separations(
    table: Table,
    ra: NDArray[np.float64],
    dec: NDArray[np.float64],
    bore: TileBoresights,
    options: PolframeOptions,
) -> float | None:
    """Compare each row's separation from its boresight with the catalogue's.

    Args:
        table: The catalogue table.
        ra: Source right ascension (deg).
        dec: Source declination (deg).
        bore: Per-row boresights.
        options: Polframe options (separation column and tolerance).

    Returns:
        Worst difference (arcsec), or None if the catalogue cannot be checked.

    Raises:
        CatalogueError: If any row differs by more than the tolerance, or
            the separation column has a non-angular unit.
    """
    sep_col, tol_arcsec = options.sep_col, options.sep_tol_arcsec
    if sep_col not in table.colnames:
        logger.info("no %s column; boresights not cross-checked", sep_col)
        return None
    sep_cat = column_f64(table, sep_col)
    unit = table[sep_col].unit
    if unit is not None:
        try:
            sep_cat = sep_cat * unit.to(u.deg)
        except u.UnitConversionError as err:
            msg = f"{sep_col} has non-angular unit {unit}"
            raise CatalogueError(msg) from err
    diff = 3600.0 * np.abs(sep_deg(ra, dec, bore.ra0, bore.dec0) - sep_cat)
    ok = np.isfinite(diff)
    if not ok.any():
        logger.info("no finite %s values; boresights not cross-checked", sep_col)
        return None
    worst = float(np.max(diff[ok]))
    logger.info("boresight check vs %s: max |diff| %.3g arcsec", sep_col, worst)
    n_bad = int((diff[ok] > tol_arcsec).sum())
    if n_bad:
        msg = (
            f"{n_bad} rows are more than {tol_arcsec:g} arcsec from their {sep_col} "
            f"(worst {worst:.1f} arcsec). The boresight CSV probably comes from a "
            "different RACS epoch, or the tile column does not name the field each "
            "row was measured in. Set sep_tol_arcsec to inf to skip this check."
        )
        raise CatalogueError(msg)
    return worst


def angle_lower_edge(
    chi: NDArray[np.float64], override: float | None, name: str
) -> float:
    """Lower edge of the angle range (0 or -90), checking the data fit in it.

    Args:
        chi: Angles (deg).
        override: Lower edge to use, or None to infer it.
        name: Column name, for messages.

    Returns:
        The lower edge (deg).

    Raises:
        CatalogueError: If the angles do not fit the range.
    """
    lo_obs, hi_obs = float(np.nanmin(chi)), float(np.nanmax(chi))
    lo = override if override is not None else (-90.0 if lo_obs < 0 else 0.0)
    logger.info(
        "%s: observed [%.3f, %.3f] -> wrap to [%g, %g)",
        name,
        lo_obs,
        hi_obs,
        lo,
        lo + 180,
    )
    if lo_obs < lo - 1e-6 or hi_obs > lo + 180.0 + 1e-6:
        msg = (
            f"{name} does not fit [{lo:g}, {lo + 180:g}]; set angle_min or check units"
        )
        raise CatalogueError(msg)
    return lo


def _corrected(table: Table, src: str, values: NDArray[np.float64]) -> NewColumn:
    """A _corr column borrowing unit and description from its source column."""
    unit = table[src].unit
    return NewColumn(
        name=f"{src}_corr",
        values=values,
        unit=None if unit is None else unit.to_string(),
        description=f"{table[src].description or src} (local-north frame)",
    )


def polframe_columns(table: Table, options: PolframeOptions) -> PolframeColumns:
    """Compute the local-north columns for every row.

    Args:
        table: The catalogue table.
        options: Polframe options.

    Returns:
        The new columns, provenance keywords and diagnostics.

    Raises:
        CatalogueError: If columns are missing, a tile has no boresight, the
            boresight cross-check fails, or angles do not fit their range.
    """
    require_columns(
        table,
        [
            "ra",
            "dec",
            options.tile_col,
            *options.angle_cols,
            *(c for pr in options.qu_cols for c in pr),
            *options.qu_err_cols,
        ],
    )
    bore = boresights_for_rows(table, options)
    ra, dec = column_f64(table, "ra"), column_f64(table, "dec")
    max_diff = check_separations(table, ra, dec, bore, options)
    # chi_true = chi_measured + sign * gamma; tile pairs give +1
    sign = 1.0
    gamma = sign * gamma_deg(ra, dec, bore.ra0, bore.dec0)

    columns: list[NewColumn] = []
    for name in options.angle_cols:
        chi = column_f64(table, name)
        lo = angle_lower_edge(chi, options.angle_min, name)
        columns.append(_corrected(table, name, wrap_angle(chi + gamma, lo)))
    for q_name, u_name in options.qu_cols:
        q_rot, u_rot = rotate_qu(
            column_f64(table, q_name), column_f64(table, u_name), gamma
        )
        columns += [_corrected(table, q_name, q_rot), _corrected(table, u_name, u_rot)]
    q_err_name, u_err_name = options.qu_err_cols
    q_err, u_err = column_f64(table, q_err_name), column_f64(table, u_err_name)
    if np.isfinite(q_err).any() and np.isfinite(u_err).any():
        rotated = rotate_qu_errors(q_err, u_err, gamma)
        columns += [
            _corrected(table, q_err_name, rotated.stokes_q_err),
            _corrected(table, u_err_name, rotated.stokes_u_err),
        ]
    else:
        logger.info(
            "%s/%s have no finite values; error columns skipped", q_err_name, u_err_name
        )
    columns.append(
        NewColumn(
            name=options.gamma_col,
            values=gamma,
            unit="deg",
            description="Rotation applied to reach local north (chi_true = chi + gamma)",
        )
    )
    cards = (
        HeaderCard(
            "POLFRAME",
            "LOCAL_NORTH_COLS_ADDED",
            "_corr columns added by spice-racs-fix",
        ),
        HeaderCard("POLFSRC", f"file:{options.boresights.name}", "boresight source"),
        HeaderCard("POLFSIGN", sign, "chi_true = chi + sign*gamma"),
    )
    return PolframeColumns(
        step=StepResult(columns=tuple(columns), cards=cards),
        gamma=gamma,
        max_sep_diff_arcsec=max_diff,
    )


def run_polframe(
    catalogue: Path,
    output: Path,
    options: PolframeOptions | None = None,
    write_options: WriteOptions | None = None,
) -> PolframeResult:
    """Write a copy of the catalogue with local-north columns appended.

    Args:
        catalogue: Uncompressed input FITS file.
        output: Output FITS file.
        options: Polframe options.
        write_options: Write options.

    Returns:
        Summary of what was written.

    Raises:
        CatalogueError: If anything about the input or output is unsafe.
            Nothing is written in that case.
    """
    options = options or PolframeOptions()
    write_options = write_options or WriteOptions()
    check_output(catalogue, output, write_options.overwrite)
    cat = open_catalogue(catalogue)
    if options.gamma_col in cat.table.colnames:
        msg = f"already has {options.gamma_col}; refusing to run twice"
        raise CatalogueError(msg)
    result = polframe_columns(cat.table, options)
    n_rows = append_columns(cat, output, [result.step], write_options)
    n_no_gamma = int((~np.isfinite(result.gamma)).sum())
    if n_no_gamma:
        logger.info(
            "%d rows have no finite gamma; their _corr columns are NaN", n_no_gamma
        )
    return PolframeResult(
        n_rows=n_rows,
        n_no_gamma=n_no_gamma,
        max_sep_diff_arcsec=result.max_sep_diff_arcsec,
    )
