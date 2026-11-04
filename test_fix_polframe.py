"""Tests for fix_polframe. These check the algebra and the plumbing. They cannot
check the sign convention (chi_true = chi + gamma); that needs a source of known
absolute angle."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table

import fix_polframe as fp

RNG = np.random.default_rng(1)


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def random_field(n: int, max_off: float = 5.0):
    ra0 = RNG.uniform(0, 360, n)
    dec0 = RNG.uniform(-89, 60, n)
    cs = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(
        RNG.uniform(0, 360, n) * u.deg, RNG.uniform(0, max_off, n) * u.deg
    )
    return cs.ra.deg, cs.dec.deg, ra0, dec0


def test_gamma_matches_vector_projection():
    ra, dec, ra0, dec0 = random_field(5000)

    def north(a, d):
        a, d = np.radians(a), np.radians(d)
        return np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])

    def east(a):
        a = np.radians(a)
        return np.array([-np.sin(a), np.cos(a), np.zeros_like(a)])

    d = north(ra0, dec0)
    ref = np.degrees(np.arctan2((d * east(ra)).sum(0), (d * north(ra, dec)).sum(0)))
    assert np.abs(wrap180(fp.gamma_deg(ra, dec, ra0, dec0) - ref)).max() < 1e-8


def test_gamma_zero_on_meridian_and_small_offset_limit():
    g = fp.gamma_deg(10.0, -50.0, 10.0, -60.0)
    assert g == pytest.approx(0, abs=1e-12)
    g = fp.gamma_deg(4.0, -84.0, 0.0, -84.0)
    assert g == pytest.approx(4.0 * np.sin(np.radians(-84.0)), rel=0.01)


def test_rotation_preserves_p_and_shifts_angle_by_gamma():
    n = 5000
    gamma = RNG.uniform(-60, 60, n)
    q, uu = RNG.normal(size=n), RNG.normal(size=n)
    q2, u2 = fp.rotate_qu(q, uu, gamma)
    assert np.allclose(np.hypot(q, uu), np.hypot(q2, u2))
    dchi = 0.5 * np.degrees(np.arctan2(u2, q2) - np.arctan2(uu, q)) - gamma
    assert np.abs((dchi + 90) % 180 - 90).max() < 1e-8


def test_cov_matches_matrix_form():
    n = 1000
    gamma = RNG.uniform(-60, 60, n)
    sq, su = RNG.uniform(0.5, 2, n), RNG.uniform(0.5, 2, n)
    cov = RNG.uniform(-0.2, 0.2, n)
    a, b, c = fp.rotate_qu_cov(sq, su, gamma, cov)
    th = np.radians(2 * gamma)
    for i in range(0, n, 97):
        R = np.array([[np.cos(th[i]), -np.sin(th[i])], [np.sin(th[i]), np.cos(th[i])]])
        C2 = R @ np.array([[sq[i] ** 2, cov[i]], [cov[i], su[i] ** 2]]) @ R.T
        assert a[i] ** 2 == pytest.approx(C2[0, 0])
        assert b[i] ** 2 == pytest.approx(C2[1, 1])
        assert c[i] == pytest.approx(C2[0, 1])
    # equal, uncorrelated errors stay uncorrelated
    _, _, c0 = fp.rotate_qu_cov(np.ones(5), np.ones(5), np.linspace(-40, 40, 5))
    assert np.allclose(c0, 0)


def test_wrap_angle():
    x = np.array([-100.0, -90.0, 0.0, 89.9, 90.0, 200.0])
    for lo in (0.0, -90.0):
        w = fp.wrap_angle(x, lo)
        assert (w >= lo).all() and (w < lo + 180).all()
        assert np.allclose(np.sin(np.radians(2 * w)), np.sin(np.radians(2 * x)))


CENTRES = {"RACS_0426-84": (66.5, -84.0), "RACS_1200-30": (180.0, -30.0)}


def make_catalogue(tmp_path, n_per_tile=300):
    """Two tiles, one near the pole, as a FITS table plus a boresight CSV."""
    ras, decs, tiles = [], [], []
    for name, (r0, d0) in CENTRES.items():
        cs = SkyCoord(r0 * u.deg, d0 * u.deg).directional_offset_by(
            RNG.uniform(0, 360, n_per_tile) * u.deg,
            RNG.uniform(0, 3, n_per_tile) * u.deg,
        )
        ras += list(cs.ra.deg)
        decs += list(cs.dec.deg)
        tiles += [name] * n_per_tile
    n = len(tiles)
    chi = RNG.uniform(0, 180, n)
    q, uu = RNG.normal(size=n), RNG.normal(size=n)
    t = Table()
    t["ra"] = Column(ras, unit="deg", description="Right Ascension [ICRS]")
    t["dec"] = Column(decs, unit="deg", description="Declination [ICRS]")
    t["tile_id"] = Column(tiles, description="Tile ID")
    t["polangle"] = Column(
        chi, unit="deg", description="Electric vector polarization angle"
    )
    t["derot_polangle"] = Column(chi, unit="deg")
    t["stokesQ"] = Column(q, unit="Jy / beam")
    t["stokesU"] = Column(uu, unit="Jy / beam")
    t["stokesQ_err"] = Column(RNG.uniform(1, 2, n), unit="Jy")
    t["stokesU_err"] = Column(RNG.uniform(1, 2, n), unit="Jy")
    t["stokesQ_bkg"] = Column(q * 0.1, unit="Jy / beam")
    t["stokesU_bkg"] = Column(uu * 0.1, unit="Jy / beam")
    t["flag"] = Column(RNG.uniform(size=n) > 0.5)
    t["name"] = Column([f"src{i}" for i in range(n)])
    t["polangle"][3] = np.nan  # a null
    path = tmp_path / "cat.fits"
    t.write(path)
    with fits.open(path) as hdul:  # add checksums, as a released file may have
        hdul.writeto(path, overwrite=True, checksum=True)

    bore = tmp_path / "bore.csv"
    rows = [(nm, r, d) for nm, (r, d) in CENTRES.items()]
    rows.append(rows[0])  # re-observation of a field: duplicate name, same centre
    pd.DataFrame(rows, columns=["FIELD_NAME", "RA_DEG", "DEC_DEG"]).to_csv(
        bore, index=False
    )
    return path, t, bore


def expected_gamma(t):
    r0 = np.array([CENTRES[x][0] for x in t["tile_id"]])
    d0 = np.array([CENTRES[x][1] for x in t["tile_id"]])
    return fp.gamma_deg(np.asarray(t["ra"]), np.asarray(t["dec"]), r0, d0)


def test_end_to_end_preserves_everything(tmp_path):
    path, orig, bore = make_catalogue(tmp_path, n_per_tile=250)
    out = tmp_path / "out.fits"
    assert fp.main([str(path), str(out), "--boresights", str(bore)]) == 0

    with fits.open(path) as a, fits.open(out) as b:
        ha, hb = a[1].header, b[1].header
        # stale checksums are dropped rather than copied
        assert "CHECKSUM" in ha and "CHECKSUM" not in hb and "DATASUM" not in hb
        assert "CHECKSUM" not in b[0].header
        assert hb["TFIELDS"] == ha["TFIELDS"] + 10
    res, base = Table.read(out), Table.read(path)

    # original columns are unchanged, values and metadata
    for c in base.colnames:
        a_, b_ = np.ma.filled(base[c]), np.ma.filled(res[c])
        assert np.array_equal(a_, b_, equal_nan=a_.dtype.kind == "f"), c
        assert res[c].unit == base[c].unit and res[c].description == base[c].description

    new = [
        "polangle_corr", "derot_polangle_corr", "stokesQ_corr", "stokesU_corr",
        "stokesQ_bkg_corr", "stokesU_bkg_corr", "stokesQ_err_corr",
        "stokesU_err_corr", "stokesQU_cov_corr", "pol_frame_gamma",
    ]  # fmt: skip
    assert res.colnames == base.colnames + new
    assert res["polangle_corr"].unit == "deg"
    assert res["stokesQU_cov_corr"].unit == u.Jy**2
    assert "local-north" in res["polangle_corr"].description

    g = expected_gamma(orig)
    assert np.abs(np.asarray(res["pol_frame_gamma"]) - g).max() < 1e-6
    assert np.abs(g).max() > 10  # the polar tile exercises the effect
    exp = fp.wrap_angle(np.asarray(orig["derot_polangle"]) + g, 0.0)
    diff = wrap180(2 * (np.asarray(res["derot_polangle_corr"]) - exp))
    assert np.abs(diff).max() < 1e-5
    assert np.isnan(np.ma.filled(res["polangle_corr"], np.nan)[3])  # null stays null

    with pytest.raises(SystemExit):  # refuses to run twice
        fp.main([str(out), str(tmp_path / "again.fits"), "--boresights", str(bore)])


def test_unknown_tile_aborts(tmp_path):
    path, _, bore = make_catalogue(tmp_path, n_per_tile=20)
    pd.read_csv(bore).iloc[:1].to_csv(bore, index=False)  # drop the second tile
    with pytest.raises(SystemExit):
        fp.main([str(path), str(tmp_path / "o.fits"), "--boresights", str(bore)])


def test_refuses_primary_with_data(tmp_path):
    path, _, bore = make_catalogue(tmp_path, n_per_tile=20)
    with fits.open(path) as hdul:
        hdul[0].data = np.zeros(4)
        hdul.writeto(path, overwrite=True)
    with pytest.raises(SystemExit):
        fp.main([str(path), str(tmp_path / "o.fits"), "--boresights", str(bore)])


def test_refuses_gzip(tmp_path):
    with pytest.raises(SystemExit):
        fp.main([str(tmp_path / "x.fits.gz"), str(tmp_path / "o.fits")])
