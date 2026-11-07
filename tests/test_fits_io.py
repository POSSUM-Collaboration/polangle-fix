from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from spice_racs_fix import fits_io


def test_match_layout_keeps_input_null_markers() -> None:
    input_header, output_header = fits.Header(), fits.Header()
    for header in (input_header, output_header):
        header["NAXIS1"] = 8
        header["TFIELDS"] = 2
        header["TTYPE1"], header["TFORM1"] = "n_comp", "J"
        header["TTYPE2"], header["TFORM2"] = "count", "J"
    input_header["TNULL1"] = -1  # astropy < 6.0 writes its default instead
    output_header["TNULL1"] = 999999
    output_header["TNULL2"] = 999999  # and may add one the input did not have
    output_header["NAXIS1"] = 16
    output_header["TFIELDS"] = 3
    output_header["TTYPE3"], output_header["TFORM3"] = "pol_frame_gamma", "D"
    assert fits_io.match_layout(input_header, output_header, 2, 1) is None
    assert output_header["TNULL1"] == -1
    assert "TNULL2" not in output_header
    output_header["TFORM2"] = "K"
    problem = fits_io.match_layout(input_header, output_header, 2, 1)
    assert problem is not None
    assert "TFORM2" in problem


def test_open_refuses_gzip(tmp_path: Path) -> None:
    with pytest.raises(fits_io.CatalogueError, match="gzipped"):
        fits_io.open_catalogue(tmp_path / "x.fits.gz")


def test_open_refuses_primary_with_data(tmp_path: Path) -> None:
    path = tmp_path / "p.fits"
    table_hdu = fits.BinTableHDU.from_columns([fits.Column("a", "D", array=np.ones(3))])
    fits.HDUList([fits.PrimaryHDU(np.zeros(4)), table_hdu]).writeto(path)
    with pytest.raises(fits_io.CatalogueError, match="bare primary"):
        fits_io.open_catalogue(path)


def test_check_output(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "in.fits", tmp_path / "out.fits"
    input_path.write_bytes(b"x")
    fits_io.check_output(input_path, output_path, overwrite=False)  # absent is fine
    output_path.write_bytes(b"y")
    with pytest.raises(fits_io.CatalogueError, match="exists"):
        fits_io.check_output(input_path, output_path, overwrite=False)
    fits_io.check_output(input_path, output_path, overwrite=True)
    with pytest.raises(fits_io.CatalogueError, match="input"):
        fits_io.check_output(input_path, input_path, overwrite=True)


def test_append_writes_column_comments(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "in.fits", tmp_path / "out.fits"
    table_hdu = fits.BinTableHDU.from_columns(
        [fits.Column("a", "D", array=np.arange(5.0))]
    )
    table_hdu.header["TCOMM1"] = "Original"
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(input_path)
    step = fits_io.StepResult(
        columns=(fits_io.NewColumn("b", np.arange(5.0) * 2, "deg", "Doubled"),),
        cards=(fits_io.HeaderCard("TESTKEY", 1.0, "a provenance card"),),
    )
    catalogue = fits_io.open_catalogue(input_path)
    rows_written = fits_io.append_columns(
        catalogue, output_path, [step], fits_io.WriteOptions()
    )
    assert rows_written == 5
    with fits.open(output_path) as hdu_list:
        header = hdu_list[1].header
        assert header["TCOMM1"] == "Original"
        assert header["TCOMM2"] == "Doubled"
        assert header["TESTKEY"] == 1.0
        assert np.array_equal(hdu_list[1].data["b"], np.arange(5.0) * 2)


def test_append_refuses_existing_column_and_empty(tmp_path: Path) -> None:
    input_path, output_path = tmp_path / "in.fits", tmp_path / "o.fits"
    table_hdu = fits.BinTableHDU.from_columns(
        [fits.Column("a", "D", array=np.arange(3.0))]
    )
    fits.HDUList([fits.PrimaryHDU(), table_hdu]).writeto(input_path)
    catalogue = fits_io.open_catalogue(input_path)
    clashing_step = fits_io.StepResult(
        columns=(fits_io.NewColumn("a", np.zeros(3), None, "x"),), cards=()
    )
    with pytest.raises(fits_io.CatalogueError, match="twice"):
        fits_io.append_columns(
            catalogue, output_path, [clashing_step], fits_io.WriteOptions()
        )
    with pytest.raises(fits_io.CatalogueError, match="no columns"):
        fits_io.append_columns(catalogue, output_path, [], fits_io.WriteOptions())
    assert not output_path.exists()
