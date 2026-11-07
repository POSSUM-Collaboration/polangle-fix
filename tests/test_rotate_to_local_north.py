"""Tests for the rotate-to-local-north step. These check the algebra and the
plumbing. They cannot check the zero point of the sign convention
(chi_true = chi + gamma); that needs a source of known absolute angle."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from conftest import TILE_CENTRES_DEG, SyntheticCatalogue, write_boresight_csv
from numpy.typing import NDArray
from spice_racs_fix import fits_io, geometry, rotate_to_local_north
from spice_racs_fix.rotate_to_local_north import (
    RotateToLocalNorthOptions,
    run_rotate_to_local_north,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def wrap_to_plus_minus_180(angle_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle_deg + 180.0) % 360.0 - 180.0


def expected_rotation_deg(table: Table) -> NDArray[np.float64]:
    boresight_ra = np.array([TILE_CENTRES_DEG[name][0] for name in table["tile_id"]])
    boresight_dec = np.array([TILE_CENTRES_DEG[name][1] for name in table["tile_id"]])
    return geometry.local_north_rotation_deg(
        np.asarray(table["ra"]), np.asarray(table["dec"]), boresight_ra, boresight_dec
    )


def raw_row_bytes(table_hdu: fits.BinTableHDU, row_count: int) -> NDArray[np.uint8]:
    return np.frombuffer(np.asarray(table_hdu.data).tobytes(), np.uint8).reshape(
        row_count, -1
    )


def test_end_to_end_preserves_everything(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    catalogue = make_catalogue(rows_per_tile=250)
    output_path = tmp_path / "out.fits"
    result = run_rotate_to_local_north(
        catalogue.path,
        output_path,
        RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
        fits_io.WriteOptions(rows_per_chunk=77),  # uneven chunks
    )
    row_count = len(catalogue.table)
    assert result.rows_written == row_count
    assert result.rows_without_rotation == 0

    with fits.open(catalogue.path) as input_hdus, fits.open(output_path) as output_hdus:
        input_header, output_header = input_hdus[1].header, output_hdus[1].header
        # stale checksums are dropped rather than copied
        assert "CHECKSUM" in input_header
        assert "CHECKSUM" not in output_header
        assert "DATASUM" not in output_header
        assert "CHECKSUM" not in output_hdus[0].header
        new_column_count = 9
        assert output_header["TFIELDS"] == input_header["TFIELDS"] + new_column_count
        last_comment = output_header[
            f"TCOMM{input_header['TFIELDS'] + new_column_count}"
        ]
        assert last_comment.startswith("Rotation applied")
        # the original rows are copied byte for byte
        input_rows = raw_row_bytes(input_hdus[1], row_count)
        output_rows = raw_row_bytes(output_hdus[1], row_count)
        assert (output_rows[:, : input_rows.shape[1]] == input_rows).all()
    output_table, input_table = Table.read(output_path), Table.read(catalogue.path)

    for column_name in input_table.colnames:
        input_values = np.ma.filled(input_table[column_name])
        output_values = np.ma.filled(output_table[column_name])
        assert np.array_equal(
            input_values, output_values, equal_nan=input_values.dtype.kind == "f"
        ), column_name
        assert output_table[column_name].unit == input_table[column_name].unit
        assert (
            output_table[column_name].description
            == input_table[column_name].description
        )
    new_columns = [
        "polangle_corr", "derot_polangle_corr", "stokesQ_corr", "stokesU_corr",
        "stokesQ_bkg_corr", "stokesU_bkg_corr", "stokesQ_err_corr",
        "stokesU_err_corr", "pol_frame_gamma",
    ]  # fmt: skip
    assert output_table.colnames == input_table.colnames + new_columns
    assert output_table["polangle_corr"].unit == "deg"
    assert "local-north" in output_table["polangle_corr"].description

    rotation = expected_rotation_deg(catalogue.table)
    assert np.abs(np.asarray(output_table["pol_frame_gamma"]) - rotation).max() < 1e-6
    assert np.abs(rotation).max() > 10  # the polar tile exercises the effect
    expected_angle = geometry.wrap_polarisation_angle(
        np.asarray(catalogue.table["derot_polangle"]) + rotation, 0.0
    )
    angle_difference = wrap_to_plus_minus_180(
        2 * (np.asarray(output_table["derot_polangle_corr"]) - expected_angle)
    )
    assert np.abs(angle_difference).max() < 1e-5
    # a null stays null
    assert np.isnan(np.ma.filled(output_table["polangle_corr"], np.nan)[3])

    with pytest.raises(fits_io.CatalogueError, match="twice"):
        run_rotate_to_local_north(
            output_path,
            tmp_path / "again.fits",
            RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
        )


def test_unknown_tile_aborts(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    catalogue = make_catalogue(rows_per_tile=20)
    tile_name, (centre_ra, centre_dec) = next(iter(TILE_CENTRES_DEG.items()))
    # drop the second tile
    write_boresight_csv(catalogue.boresight_csv, [(tile_name, centre_ra, centre_dec)])
    with pytest.raises(fits_io.CatalogueError, match="no boresight"):
        run_rotate_to_local_north(
            catalogue.path,
            tmp_path / "o.fits",
            RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
        )


def test_layout_change_stops_without_writing(
    tmp_path: Path,
    make_catalogue: Callable[..., SyntheticCatalogue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalogue = make_catalogue(rows_per_tile=20)
    output_path = tmp_path / "out.fits"
    monkeypatch.setattr(fits_io, "match_layout", lambda *_: "TFORM3 differs")
    with pytest.raises(fits_io.CatalogueError, match="TFORM3 differs"):
        run_rotate_to_local_north(
            catalogue.path,
            output_path,
            RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
        )
    assert not output_path.exists()


def test_awkward_columns_copied_verbatim(tmp_path: Path) -> None:
    """Integer nulls, scaled unsigned ints, booleans, empty strings, extra cards."""
    row_count = 6
    columns = [
        fits.Column("ra", "D", unit="deg", array=np.linspace(180, 181, row_count)),
        fits.Column("dec", "D", unit="deg", array=np.full(row_count, -30.0)),
        fits.Column(
            "tile_id",
            "14A",
            array=np.array([b"RACS_1200-30  "] * (row_count - 1) + [b""]),
        ),
        fits.Column(
            "comment", "8A", array=np.array([b"", b"ok", b"", b"x", b"", b"y"])
        ),
        fits.Column(
            "n_comp", "J", null=-1, array=np.array([1, 2, -1, 4, -1, 6], dtype=">i4")
        ),
        fits.Column(
            "count16",
            "I",
            bzero=32768,
            array=np.array([0, 1, 2, 60000, 5, 6], dtype=np.uint16),
        ),
        # uint32 stored with TZERO, as nn_rm_count in DR2
        fits.Column(
            "count32",
            "J",
            bzero=2147483648,
            array=np.array([0, 1, 2, 4_000_000_000, 5, 6], dtype=np.uint32),
        ),
        fits.Column("flag", "L", array=np.array([True, False] * 3)),
        fits.Column(
            "snr",
            "E",
            disp="F8.2",
            array=np.array([1.5, np.nan, 3, 4, 5, 6], dtype=">f4"),
        ),
    ]
    options = RotateToLocalNorthOptions()
    for angle_column in options.angle_columns:
        columns.append(
            fits.Column(
                angle_column, "D", unit="deg", array=np.linspace(10, 170, row_count)
            )
        )
    stokes_columns = [
        name for pair in options.stokes_qu_columns for name in pair
    ] + list(options.stokes_qu_error_columns)
    for stokes_column in stokes_columns:
        columns.append(
            fits.Column(
                stokes_column,
                "D",
                unit="Jy/beam",
                array=np.linspace(0.1, 1, row_count),
            )
        )
    table_hdu = fits.BinTableHDU.from_columns(columns)
    table_hdu.header["TCOMM1"] = "Right ascension"
    table_hdu.header["TUCD1"] = "pos.eq.ra;meta.main"
    table_hdu.header["SURVEY"] = ("SPICE-RACS", "survey name")
    input_path, output_path = tmp_path / "in.fits", tmp_path / "out.fits"
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(input_path)
    boresight_csv = write_boresight_csv(
        tmp_path / "bore.csv", [("RACS_1200-30", 180.0, -30.0)]
    )

    run_rotate_to_local_north(
        input_path, output_path, RotateToLocalNorthOptions(boresight_csv=boresight_csv)
    )
    with fits.open(input_path) as input_hdus, fits.open(output_path) as output_hdus:
        for keyword in (
            "TCOMM1", "TUCD1", "SURVEY", "TNULL5", "TZERO6", "TZERO7", "TDISP9",
        ):  # fmt: skip
            assert output_hdus[1].header[keyword] == input_hdus[1].header[keyword], (
                keyword
            )
        for column_name in input_hdus[1].columns.names:
            input_values = np.asarray(input_hdus[1].data.field(column_name))
            output_values = np.asarray(output_hdus[1].data.field(column_name))
            assert input_values.dtype == output_values.dtype, column_name
            assert input_values.tobytes() == output_values.tobytes(), column_name
        rotation = np.asarray(output_hdus[1].data.field("pol_frame_gamma"))
        assert np.isfinite(rotation[:-1]).all()
        assert np.isnan(rotation[-1])  # empty tile_id


def test_separation_check(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    catalogue = make_catalogue(rows_per_tile=40, with_separation=True)
    output_path = tmp_path / "out.fits"
    result = run_rotate_to_local_north(
        catalogue.path,
        output_path,
        RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
    )
    assert result.worst_separation_difference_arcsec is not None
    assert result.worst_separation_difference_arcsec < 1e-3
    # a boresight 10 arcsec off, as a CSV from another epoch might give
    with catalogue.boresight_csv.open() as csv_file:
        boresight_rows = [
            (row["FIELD_NAME"], float(row["RA_DEG"]), float(row["DEC_DEG"]))
            for row in csv.DictReader(csv_file)
        ]
    shifted_rows = [
        (
            name,
            centre_ra,
            centre_dec + 10 / 3600 if name == "RACS_1200-30" else centre_dec,
        )
        for name, centre_ra, centre_dec in boresight_rows
    ]
    write_boresight_csv(catalogue.boresight_csv, shifted_rows)
    overwrite = fits_io.WriteOptions(overwrite=True)
    with pytest.raises(fits_io.CatalogueError, match="different RACS epoch"):
        run_rotate_to_local_north(
            catalogue.path,
            output_path,
            RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
            overwrite,
        )
    skip_check = RotateToLocalNorthOptions(
        boresight_csv=catalogue.boresight_csv,
        separation_tolerance_arcsec=float("inf"),
    )
    # the check can be skipped
    run_rotate_to_local_north(catalogue.path, output_path, skip_check, overwrite)


def test_refuses_to_overwrite_input(
    make_catalogue: Callable[..., SyntheticCatalogue],
) -> None:
    catalogue = make_catalogue(rows_per_tile=20)
    before = catalogue.path.read_bytes()
    with pytest.raises(fits_io.CatalogueError, match="input"):
        run_rotate_to_local_north(
            catalogue.path,
            catalogue.path,
            RotateToLocalNorthOptions(boresight_csv=catalogue.boresight_csv),
            fits_io.WriteOptions(overwrite=True),
        )
    assert catalogue.path.read_bytes() == before


def test_distinct_tile_names_decodes_each_value_once() -> None:
    tile_column = np.array([b"RACS_A  ", b"RACS_B", b"RACS_A  ", b"", b"RACS_B"])
    names, row_index = rotate_to_local_north.distinct_tile_names(tile_column)
    assert list(names[row_index]) == ["RACS_A", "RACS_B", "RACS_A", "", "RACS_B"]
    assert len(names) == 3


def test_default_boresight_csv_covers_racs_names() -> None:
    centres = rotate_to_local_north.read_boresight_csv(
        RotateToLocalNorthOptions().boresight_csv
    )
    assert len(centres) > 1000
    assert all(name.startswith("RACS_") for name in centres)
