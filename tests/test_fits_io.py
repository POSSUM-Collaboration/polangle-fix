from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from spice_racs_fix import fits_io


def test_match_layout_keeps_input_null_markers() -> None:
    old, new = fits.Header(), fits.Header()
    for h in (old, new):
        h["NAXIS1"] = 8
        h["TFIELDS"] = 2
        h["TTYPE1"], h["TFORM1"] = "n_comp", "J"
        h["TTYPE2"], h["TFORM2"] = "count", "J"
    old["TNULL1"] = -1  # astropy < 6.0 writes its default instead
    new["TNULL1"] = 999999
    new["TNULL2"] = 999999  # and may add one the input did not have
    new["NAXIS1"] = 16
    new["TFIELDS"] = 3
    new["TTYPE3"], new["TFORM3"] = "pol_frame_gamma", "D"
    assert fits_io.match_layout(old, new, 2, 1) is None
    assert new["TNULL1"] == -1
    assert "TNULL2" not in new
    new["TFORM2"] = "K"
    problem = fits_io.match_layout(old, new, 2, 1)
    assert problem is not None
    assert "TFORM2" in problem


def test_open_refuses_gzip(tmp_path: Path) -> None:
    with pytest.raises(fits_io.CatalogueError, match="gzipped"):
        fits_io.open_catalogue(tmp_path / "x.fits.gz")


def test_open_refuses_primary_with_data(tmp_path: Path) -> None:
    path = tmp_path / "p.fits"
    hdu = fits.BinTableHDU.from_columns([fits.Column("a", "D", array=np.ones(3))])
    fits.HDUList([fits.PrimaryHDU(np.zeros(4)), hdu]).writeto(path)
    with pytest.raises(fits_io.CatalogueError, match="bare primary"):
        fits_io.open_catalogue(path)


def test_check_output(tmp_path: Path) -> None:
    src, out = tmp_path / "in.fits", tmp_path / "out.fits"
    src.write_bytes(b"x")
    fits_io.check_output(src, out, overwrite=False)  # absent is fine
    out.write_bytes(b"y")
    with pytest.raises(fits_io.CatalogueError, match="exists"):
        fits_io.check_output(src, out, overwrite=False)
    fits_io.check_output(src, out, overwrite=True)
    with pytest.raises(fits_io.CatalogueError, match="input"):
        fits_io.check_output(src, src, overwrite=True)


def test_append_writes_column_comments(tmp_path: Path) -> None:
    path, out = tmp_path / "in.fits", tmp_path / "out.fits"
    hdu = fits.BinTableHDU.from_columns([fits.Column("a", "D", array=np.arange(5.0))])
    hdu.header["TCOMM1"] = "Original"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    step = fits_io.StepResult(
        columns=(fits_io.NewColumn("b", np.arange(5.0) * 2, "deg", "Doubled"),),
        cards=(fits_io.HeaderCard("TESTKEY", 1.0, "a provenance card"),),
    )
    cat = fits_io.open_catalogue(path)
    assert fits_io.append_columns(cat, out, [step], fits_io.WriteOptions()) == 5
    with fits.open(out) as hdul:
        h = hdul[1].header
        assert h["TCOMM1"] == "Original"
        assert h["TCOMM2"] == "Doubled"
        assert h["TESTKEY"] == 1.0
        assert np.array_equal(hdul[1].data["b"], np.arange(5.0) * 2)


def test_append_refuses_existing_column_and_empty(tmp_path: Path) -> None:
    path = tmp_path / "in.fits"
    hdu = fits.BinTableHDU.from_columns([fits.Column("a", "D", array=np.arange(3.0))])
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    cat = fits_io.open_catalogue(path)
    clash = fits_io.StepResult(
        columns=(fits_io.NewColumn("a", np.zeros(3), None, "x"),), cards=()
    )
    with pytest.raises(fits_io.CatalogueError, match="twice"):
        fits_io.append_columns(
            cat, tmp_path / "o.fits", [clash], fits_io.WriteOptions()
        )
    with pytest.raises(fits_io.CatalogueError, match="no columns"):
        fits_io.append_columns(cat, tmp_path / "o.fits", [], fits_io.WriteOptions())
    assert not (tmp_path / "o.fits").exists()
