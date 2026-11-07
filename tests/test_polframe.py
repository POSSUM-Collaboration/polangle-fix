"""Tests for the polarisation frame step. These check the algebra and the
plumbing. They cannot check the zero point of the sign convention
(chi_true = chi + gamma); that needs a source of known absolute angle."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from conftest import CENTRES, SyntheticCatalogue, write_boresights
from numpy.typing import NDArray
from spice_racs_fix import fits_io, geometry
from spice_racs_fix import polframe as pf

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def wrap180(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x + 180.0) % 360.0 - 180.0


def expected_gamma(t: Table) -> NDArray[np.float64]:
    r0 = np.array([CENTRES[x][0] for x in t["tile_id"]])
    d0 = np.array([CENTRES[x][1] for x in t["tile_id"]])
    return geometry.gamma_deg(np.asarray(t["ra"]), np.asarray(t["dec"]), r0, d0)


def raw_rows(hdu: fits.BinTableHDU, n: int) -> NDArray[np.uint8]:
    return np.frombuffer(np.asarray(hdu.data).tobytes(), np.uint8).reshape(n, -1)


def test_end_to_end_preserves_everything(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    cat = make_catalogue(n_per_tile=250)
    out = tmp_path / "out.fits"
    res = pf.run_polframe(
        cat.path,
        out,
        pf.PolframeOptions(boresights=cat.boresights),
        fits_io.WriteOptions(chunk_rows=77),  # uneven chunks
    )
    assert res.n_rows == len(cat.table)
    assert res.n_no_gamma == 0

    with fits.open(cat.path) as a, fits.open(out) as b:
        ha, hb = a[1].header, b[1].header
        # stale checksums are dropped rather than copied
        assert "CHECKSUM" in ha
        assert "CHECKSUM" not in hb
        assert "DATASUM" not in hb
        assert "CHECKSUM" not in b[0].header
        assert hb["TFIELDS"] == ha["TFIELDS"] + 9
        assert hb[f"TCOMM{ha['TFIELDS'] + 9}"].startswith("Rotation applied")
        # the original rows are copied byte for byte
        ra_, rb_ = raw_rows(a[1], len(cat.table)), raw_rows(b[1], len(cat.table))
        assert (rb_[:, : ra_.shape[1]] == ra_).all()
    got, base = Table.read(out), Table.read(cat.path)

    for c in base.colnames:
        a_, b_ = np.ma.filled(base[c]), np.ma.filled(got[c])
        assert np.array_equal(a_, b_, equal_nan=a_.dtype.kind == "f"), c
        assert got[c].unit == base[c].unit
        assert got[c].description == base[c].description
    new = [
        "polangle_corr", "derot_polangle_corr", "stokesQ_corr", "stokesU_corr",
        "stokesQ_bkg_corr", "stokesU_bkg_corr", "stokesQ_err_corr",
        "stokesU_err_corr", "pol_frame_gamma",
    ]  # fmt: skip
    assert got.colnames == base.colnames + new
    assert got["polangle_corr"].unit == "deg"
    assert "local-north" in got["polangle_corr"].description

    g = expected_gamma(cat.table)
    assert np.abs(np.asarray(got["pol_frame_gamma"]) - g).max() < 1e-6
    assert np.abs(g).max() > 10  # the polar tile exercises the effect
    exp = geometry.wrap_angle(np.asarray(cat.table["derot_polangle"]) + g, 0.0)
    diff = wrap180(2 * (np.asarray(got["derot_polangle_corr"]) - exp))
    assert np.abs(diff).max() < 1e-5
    assert np.isnan(np.ma.filled(got["polangle_corr"], np.nan)[3])  # null stays null

    with pytest.raises(fits_io.CatalogueError, match="twice"):
        pf.run_polframe(
            out, tmp_path / "again.fits", pf.PolframeOptions(boresights=cat.boresights)
        )


def test_unknown_tile_aborts(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    cat = make_catalogue(n_per_tile=20)
    name, (r0, d0) = next(iter(CENTRES.items()))
    write_boresights(cat.boresights, [(name, r0, d0)])  # drop the second tile
    with pytest.raises(fits_io.CatalogueError, match="no boresight"):
        pf.run_polframe(
            cat.path, tmp_path / "o.fits", pf.PolframeOptions(boresights=cat.boresights)
        )


def test_layout_change_stops_without_writing(
    tmp_path: Path,
    make_catalogue: Callable[..., SyntheticCatalogue],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cat = make_catalogue(n_per_tile=20)
    out = tmp_path / "out.fits"
    monkeypatch.setattr(fits_io, "match_layout", lambda *_: "TFORM3 differs")
    with pytest.raises(fits_io.CatalogueError, match="TFORM3 differs"):
        pf.run_polframe(cat.path, out, pf.PolframeOptions(boresights=cat.boresights))
    assert not out.exists()


def test_awkward_columns_copied_verbatim(tmp_path: Path) -> None:
    """Integer nulls, scaled unsigned ints, booleans, empty strings, extra cards."""
    n = 6
    cols = [
        fits.Column("ra", "D", unit="deg", array=np.linspace(180, 181, n)),
        fits.Column("dec", "D", unit="deg", array=np.full(n, -30.0)),
        fits.Column(
            "tile_id", "14A", array=np.array([b"RACS_1200-30  "] * (n - 1) + [b""])
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
    opts = pf.PolframeOptions()
    for nm in opts.angle_cols:
        cols.append(fits.Column(nm, "D", unit="deg", array=np.linspace(10, 170, n)))
    for nm in [c for pr in opts.qu_cols for c in pr] + list(opts.qu_err_cols):
        cols.append(fits.Column(nm, "D", unit="Jy/beam", array=np.linspace(0.1, 1, n)))
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["TCOMM1"] = "Right ascension"
    hdu.header["TUCD1"] = "pos.eq.ra;meta.main"
    hdu.header["SURVEY"] = ("SPICE-RACS", "survey name")
    path, out = tmp_path / "in.fits", tmp_path / "out.fits"
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    bore = write_boresights(tmp_path / "bore.csv", [("RACS_1200-30", 180.0, -30.0)])

    pf.run_polframe(path, out, pf.PolframeOptions(boresights=bore))
    with fits.open(path) as a, fits.open(out) as b:
        for k in ("TCOMM1", "TUCD1", "SURVEY", "TNULL5", "TZERO6", "TZERO7", "TDISP9"):
            assert b[1].header[k] == a[1].header[k], k
        for name in a[1].columns.names:
            x, y = np.asarray(a[1].data.field(name)), np.asarray(b[1].data.field(name))
            assert x.dtype == y.dtype, name
            assert x.tobytes() == y.tobytes(), name
        gamma = np.asarray(b[1].data.field("pol_frame_gamma"))
        assert np.isfinite(gamma[:-1]).all()
        assert np.isnan(gamma[-1])  # empty tile_id


def test_separation_check(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    cat = make_catalogue(n_per_tile=40, with_sep=True)
    out = tmp_path / "out.fits"
    res = pf.run_polframe(cat.path, out, pf.PolframeOptions(boresights=cat.boresights))
    assert res.max_sep_diff_arcsec is not None
    assert res.max_sep_diff_arcsec < 1e-3
    # a boresight 10 arcsec off, as a CSV from another epoch might give
    with cat.boresights.open() as f:
        rows = [
            (r["FIELD_NAME"], float(r["RA_DEG"]), float(r["DEC_DEG"]))
            for r in csv.DictReader(f)
        ]
    rows = [(nm, r, d + 10 / 3600 if nm == "RACS_1200-30" else d) for nm, r, d in rows]
    write_boresights(cat.boresights, rows)
    overwrite = fits_io.WriteOptions(overwrite=True)
    with pytest.raises(fits_io.CatalogueError, match="different RACS epoch"):
        pf.run_polframe(
            cat.path, out, pf.PolframeOptions(boresights=cat.boresights), overwrite
        )
    skip = pf.PolframeOptions(boresights=cat.boresights, sep_tol_arcsec=float("inf"))
    pf.run_polframe(cat.path, out, skip, overwrite)  # check can be skipped


def test_refuses_to_overwrite_input(
    make_catalogue: Callable[..., SyntheticCatalogue],
) -> None:
    cat = make_catalogue(n_per_tile=20)
    before = cat.path.read_bytes()
    with pytest.raises(fits_io.CatalogueError, match="input"):
        pf.run_polframe(
            cat.path,
            cat.path,
            pf.PolframeOptions(boresights=cat.boresights),
            fits_io.WriteOptions(overwrite=True),
        )
    assert cat.path.read_bytes() == before


def test_tile_names_decodes_each_distinct_value_once() -> None:
    col = np.array([b"RACS_A  ", b"RACS_B", b"RACS_A  ", b"", b"RACS_B"])
    names, idx = pf.tile_names(col)
    assert list(names[idx]) == ["RACS_A", "RACS_B", "RACS_A", "", "RACS_B"]
    assert len(names) == 3


def test_default_boresights_cover_racs_names() -> None:
    centres = pf.read_boresights(pf.PolframeOptions().boresights)
    assert len(centres) > 1000
    assert all(name.startswith("RACS_") for name in centres)
