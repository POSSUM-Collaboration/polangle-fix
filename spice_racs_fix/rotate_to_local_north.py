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
from astropy import units
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
    column_as_float64,
    open_catalogue,
    require_columns,
)
from spice_racs_fix.geometry import (
    angular_separation_deg,
    local_north_rotation_deg,
    rotate_stokes_qu,
    rotate_stokes_qu_errors,
    wrap_polarisation_angle,
)
from spice_racs_fix.logging import logger


@dataclass(frozen=True, kw_only=True, slots=True)
class RotateToLocalNorthOptions:
    """Options for rotating polarisation columns to local north"""

    boresight_csv: Path = Path(__file__).parent / "data" / "field_data.csv"
    """CSV with FIELD_NAME, RA_DEG, DEC_DEG; defaults to the RACS-low3 fields"""
    tile_column: str = "tile_id"
    """Column naming each row's tile"""
    separation_column: str = "separation_tile_centre"
    """Column used to cross-check boresights"""
    angle_columns: tuple[str, ...] = ("polangle", "derot_polangle")
    """Angle columns to rotate"""
    stokes_qu_columns: tuple[tuple[str, str], ...] = (
        ("stokesQ", "stokesU"),
        ("stokesQ_bkg", "stokesU_bkg"),
    )
    """Stokes Q/U column pairs to rotate"""
    stokes_qu_error_columns: tuple[str, str] = ("stokesQ_err", "stokesU_err")
    """Stokes Q/U error columns to propagate"""
    rotation_column: str = "pol_frame_gamma"
    """Name of the new column holding the rotation applied"""
    angle_lower_edge_deg: float | None = None
    """Lower edge of the angle range, 0 or -90; None infers it from the data"""
    separation_tolerance_arcsec: float = 1.0
    """Largest allowed difference from the separation column; inf skips it"""


class RowBoresights(NamedTuple):
    """Boresight of every row"""

    boresight_ra_deg: NDArray[np.float64]
    """Boresight right ascension per row (deg), NaN for an empty tile name"""
    boresight_dec_deg: NDArray[np.float64]
    """Boresight declination per row (deg), NaN for an empty tile name"""


class RotateToLocalNorthColumns(NamedTuple):
    """New columns and diagnostics from the rotation"""

    step: StepResult
    """Columns and keywords to append"""
    rotation_deg: NDArray[np.float64]
    """Rotation per row (deg)"""
    worst_separation_difference_arcsec: float | None
    """Worst boresight cross-check difference, or None if not checked"""


class RotateToLocalNorthResult(NamedTuple):
    """Outcome of rotating polarisation columns to local north"""

    rows_written: int
    """Rows written"""
    rows_without_rotation: int
    """Rows with no finite rotation (their _corr columns are NaN)"""
    worst_separation_difference_arcsec: float | None
    """Worst boresight cross-check difference, or None if not checked"""


def distinct_tile_names(
    tile_column_values: ArrayLike,
) -> tuple[NDArray[np.str_], NDArray[np.intp]]:
    """Distinct stripped tile names, and each row's index into them.

    Names are decoded once per distinct value rather than once per row, which
    matters for catalogues with millions of rows.

    Args:
        tile_column_values: Tile name column (bytes or str).

    Returns:
        Distinct names and each row's index into them.
    """
    distinct_values, row_index = np.unique(
        np.asarray(tile_column_values), return_inverse=True
    )
    names = [
        (
            value.decode("ascii", "replace") if isinstance(value, bytes) else str(value)
        ).strip()
        for value in distinct_values.tolist()
    ]
    return np.array(names, dtype=str), np.asarray(row_index, dtype=np.intp).reshape(-1)


def read_boresight_csv(path: Path) -> dict[str, tuple[float, float]]:
    """Field centres by name; the first row wins for re-observed fields.

    Args:
        path: CSV with FIELD_NAME, RA_DEG, DEC_DEG.

    Returns:
        (RA, Dec) in degrees by field name.
    """
    centres: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            name = row["FIELD_NAME"].strip()
            centres.setdefault(name, (float(row["RA_DEG"]), float(row["DEC_DEG"])))
    return centres


def boresights_for_rows(
    table: Table, options: RotateToLocalNorthOptions
) -> RowBoresights:
    """Look up each row's boresight from its tile name.

    Args:
        table: The catalogue table.
        options: Rotation options.

    Returns:
        Per-row boresights.

    Raises:
        CatalogueError: If a non-empty tile name has no boresight.
    """
    names, row_index = distinct_tile_names(table[options.tile_column])
    centres = read_boresight_csv(options.boresight_csv)
    unknown = [name for name in names if name and name not in centres]
    if unknown:
        message = (
            f"{options.boresight_csv}: no boresight for {len(unknown)} tiles, "
            f"e.g. {unknown[:5]}"
        )
        raise CatalogueError(message)
    centre_per_name = np.array(
        [centres.get(name, (np.nan, np.nan)) for name in names], dtype=np.float64
    )
    return RowBoresights(
        boresight_ra_deg=centre_per_name[row_index, 0],
        boresight_dec_deg=centre_per_name[row_index, 1],
    )


def check_separations_against_catalogue(
    table: Table,
    source_ra_deg: NDArray[np.float64],
    source_dec_deg: NDArray[np.float64],
    boresights: RowBoresights,
    options: RotateToLocalNorthOptions,
) -> float | None:
    """Compare each row's separation from its boresight with the catalogue's.

    Args:
        table: The catalogue table.
        source_ra_deg: Source right ascension (deg).
        source_dec_deg: Source declination (deg).
        boresights: Per-row boresights.
        options: Rotation options (separation column and tolerance).

    Returns:
        Worst difference (arcsec), or None if the catalogue cannot be checked.

    Raises:
        CatalogueError: If any row differs by more than the tolerance, or
            the separation column has a non-angular unit.
    """
    separation_column = options.separation_column
    tolerance_arcsec = options.separation_tolerance_arcsec
    if separation_column not in table.colnames:
        logger.info("no %s column; boresights not cross-checked", separation_column)
        return None
    catalogue_separation_deg = column_as_float64(table, separation_column)
    unit = table[separation_column].unit
    if unit is not None:
        try:
            catalogue_separation_deg = catalogue_separation_deg * unit.to(units.deg)
        except units.UnitConversionError as error:
            message = f"{separation_column} has non-angular unit {unit}"
            raise CatalogueError(message) from error
    computed_separation_deg = angular_separation_deg(
        source_ra_deg,
        source_dec_deg,
        boresights.boresight_ra_deg,
        boresights.boresight_dec_deg,
    )
    difference_arcsec = 3600.0 * np.abs(
        computed_separation_deg - catalogue_separation_deg
    )
    finite = np.isfinite(difference_arcsec)
    if not finite.any():
        logger.info(
            "no finite %s values; boresights not cross-checked", separation_column
        )
        return None
    worst_arcsec = float(np.max(difference_arcsec[finite]))
    logger.info(
        "boresight check vs %s: max |diff| %.3g arcsec", separation_column, worst_arcsec
    )
    rows_beyond_tolerance = int((difference_arcsec[finite] > tolerance_arcsec).sum())
    if rows_beyond_tolerance:
        message = (
            f"{rows_beyond_tolerance} rows are more than {tolerance_arcsec:g} arcsec "
            f"from their {separation_column} (worst {worst_arcsec:.1f} arcsec). The "
            "boresight CSV probably comes from a different RACS epoch, or the tile "
            "column does not name the field each row was measured in. Set "
            "separation_tolerance_arcsec to inf to skip this check."
        )
        raise CatalogueError(message)
    return worst_arcsec


def angle_range_lower_edge(
    angles_deg: NDArray[np.float64],
    lower_edge_override_deg: float | None,
    column_name: str,
) -> float:
    """Lower edge of the angle range (0 or -90), checking the data fit in it.

    Args:
        angles_deg: Angles (deg).
        lower_edge_override_deg: Lower edge to use, or None to infer it.
        column_name: Column name, for messages.

    Returns:
        The lower edge (deg).

    Raises:
        CatalogueError: If the angles do not fit the range.
    """
    observed_min = float(np.nanmin(angles_deg))
    observed_max = float(np.nanmax(angles_deg))
    if lower_edge_override_deg is not None:
        lower_edge = lower_edge_override_deg
    else:
        lower_edge = -90.0 if observed_min < 0 else 0.0
    logger.info(
        "%s: observed [%.3f, %.3f] -> wrap to [%g, %g)",
        column_name,
        observed_min,
        observed_max,
        lower_edge,
        lower_edge + 180,
    )
    if observed_min < lower_edge - 1e-6 or observed_max > lower_edge + 180.0 + 1e-6:
        message = (
            f"{column_name} does not fit [{lower_edge:g}, {lower_edge + 180:g}]; "
            "set angle_lower_edge_deg or check units"
        )
        raise CatalogueError(message)
    return lower_edge


def corrected_column(
    table: Table, source_column: str, values: NDArray[np.float64]
) -> NewColumn:
    """A _corr column borrowing unit and description from its source column."""
    unit = table[source_column].unit
    return NewColumn(
        name=f"{source_column}_corr",
        values=values,
        unit=None if unit is None else unit.to_string(),
        description=(
            f"{table[source_column].description or source_column} (local-north frame)"
        ),
    )


def rotate_to_local_north_columns(
    table: Table, options: RotateToLocalNorthOptions
) -> RotateToLocalNorthColumns:
    """Compute the local-north columns for every row.

    Args:
        table: The catalogue table.
        options: Rotation options.

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
            options.tile_column,
            *options.angle_columns,
            *(name for pair in options.stokes_qu_columns for name in pair),
            *options.stokes_qu_error_columns,
        ],
    )
    boresights = boresights_for_rows(table, options)
    source_ra_deg = column_as_float64(table, "ra")
    source_dec_deg = column_as_float64(table, "dec")
    worst_separation_difference_arcsec = check_separations_against_catalogue(
        table, source_ra_deg, source_dec_deg, boresights, options
    )
    # chi_true = chi_measured + sign * gamma; tile pairs give +1
    sign = 1.0
    rotation_deg = sign * local_north_rotation_deg(
        source_ra_deg,
        source_dec_deg,
        boresights.boresight_ra_deg,
        boresights.boresight_dec_deg,
    )

    columns: list[NewColumn] = []
    for angle_column in options.angle_columns:
        angles_deg = column_as_float64(table, angle_column)
        lower_edge = angle_range_lower_edge(
            angles_deg, options.angle_lower_edge_deg, angle_column
        )
        columns.append(
            corrected_column(
                table,
                angle_column,
                wrap_polarisation_angle(angles_deg + rotation_deg, lower_edge),
            )
        )
    for q_column, u_column in options.stokes_qu_columns:
        rotated_q, rotated_u = rotate_stokes_qu(
            column_as_float64(table, q_column),
            column_as_float64(table, u_column),
            rotation_deg,
        )
        columns += [
            corrected_column(table, q_column, rotated_q),
            corrected_column(table, u_column, rotated_u),
        ]
    q_error_column, u_error_column = options.stokes_qu_error_columns
    q_error = column_as_float64(table, q_error_column)
    u_error = column_as_float64(table, u_error_column)
    if np.isfinite(q_error).any() and np.isfinite(u_error).any():
        rotated_errors = rotate_stokes_qu_errors(q_error, u_error, rotation_deg)
        columns += [
            corrected_column(table, q_error_column, rotated_errors.stokes_q_error),
            corrected_column(table, u_error_column, rotated_errors.stokes_u_error),
        ]
    else:
        logger.info(
            "%s/%s have no finite values; error columns skipped",
            q_error_column,
            u_error_column,
        )
    columns.append(
        NewColumn(
            name=options.rotation_column,
            values=rotation_deg,
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
        HeaderCard("POLFSRC", f"file:{options.boresight_csv.name}", "boresight source"),
        HeaderCard("POLFSIGN", sign, "chi_true = chi + sign*gamma"),
    )
    return RotateToLocalNorthColumns(
        step=StepResult(columns=tuple(columns), cards=cards),
        rotation_deg=rotation_deg,
        worst_separation_difference_arcsec=worst_separation_difference_arcsec,
    )


def run_rotate_to_local_north(
    catalogue_path: Path,
    output_path: Path,
    options: RotateToLocalNorthOptions | None = None,
    write_options: WriteOptions | None = None,
) -> RotateToLocalNorthResult:
    """Write a copy of the catalogue with local-north columns appended.

    Args:
        catalogue_path: Uncompressed input FITS file.
        output_path: Output FITS file.
        options: Rotation options.
        write_options: Write options.

    Returns:
        Summary of what was written.

    Raises:
        CatalogueError: If anything about the input or output is unsafe.
            Nothing is written in that case.
    """
    options = options or RotateToLocalNorthOptions()
    write_options = write_options or WriteOptions()
    check_output(catalogue_path, output_path, write_options.overwrite)
    catalogue = open_catalogue(catalogue_path)
    if options.rotation_column in catalogue.table.colnames:
        message = f"already has {options.rotation_column}; refusing to run twice"
        raise CatalogueError(message)
    result = rotate_to_local_north_columns(catalogue.table, options)
    rows_written = append_columns(catalogue, output_path, [result.step], write_options)
    rows_without_rotation = int((~np.isfinite(result.rotation_deg)).sum())
    if rows_without_rotation:
        logger.info(
            "%d rows have no finite rotation; their _corr columns are NaN",
            rows_without_rotation,
        )
    return RotateToLocalNorthResult(
        rows_written=rows_written,
        rows_without_rotation=rows_without_rotation,
        worst_separation_difference_arcsec=result.worst_separation_difference_arcsec,
    )
