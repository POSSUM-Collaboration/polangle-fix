"""Tests for fix_polframe. These check the algebra and the plumbing. They cannot
check the sign convention (chi_true = chi + gamma); that needs a source of known
absolute angle."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Column, Table

import fix_polframe as fp

RNG = np.random.default_rng(1)


def random_field(n: int, max_off: float = 5.0):
    ra0 = RNG.uniform(0, 360, n)
    dec0 = RNG.uniform(-89, 60, n)
    c0 = SkyCoord(ra0 * u.deg, dec0 * u.deg)
    cs = c0.directional_offset_by(
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
    err = np.abs(fp.wrap180(fp.gamma_deg(ra, dec, ra0, dec0) - ref))
    assert err.max() < 1e-8


def test_gamma_zero_on_meridian_and_small_offset_limit():
    assert fp.gamma_deg(
        np.array([10.0]), np.array([-50.0]), np.array([10.0]), np.array([-60.0])
    )[0] == pytest.approx(0, abs=1e-12)
    g = fp.gamma_deg(
        np.array([4.0]), np.array([-84.0]), np.array([0.0]), np.array([-84.0])
    )[0]
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
        C = np.array([[sq[i] ** 2, cov[i]], [cov[i], su[i] ** 2]])
        C2 = R @ C @ R.T
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


def test_skyoffset_matches_astropy():
    ra, dec, ra0, dec0 = random_field(2000)
    lon, lat = SkyCoord(ra0 * u.deg, dec0 * u.deg).spherical_offsets_to(
        SkyCoord(ra * u.deg, dec * u.deg)
    )
    l, m = fp.offsets_skyoffset(ra, dec, ra0, dec0)
    assert np.abs(l - lon.deg).max() < 1e-9
    assert np.abs(m - lat.deg).max() < 1e-9


@pytest.mark.parametrize("conv", fp.CONVENTIONS)
def test_recover_centre_roundtrip(conv):
    ra, dec, ra0, dec0 = random_field(2000)
    if conv == "skyoffset":
        l, m = fp.offsets_skyoffset(ra, dec, ra0, dec0)
    elif conv == "flat":
        l, m = fp.wrap180(ra - ra0) * np.cos(np.radians(dec)), dec - dec0
    else:
        l, m = fp.wrap180(ra - ra0), dec - dec0
    r0, d0 = fp.recover_centre(conv, ra, dec, l, m)
    assert np.abs(fp.wrap180(r0 - ra0) * np.cos(np.radians(dec0))).max() < 1e-7
    assert np.abs(d0 - dec0).max() < 1e-7


def test_parse_tile_name():
    assert fp.parse_tile_name("RACS_0426-84") == (pytest.approx(66.5), -84.0)
    assert fp.parse_tile_name("garbage") is None


def make_catalogue(tmp_path, n_per_tile=300):
    """Two tiles, one near the pole, written as a plain FITS binary table."""
    centres = {"RACS_0426-84": (66.5, -84.0), "RACS_1200-30": (180.0, -30.0)}
    ras, decs, ls, ms, tiles = [], [], [], [], []
    for name, (r0, d0) in centres.items():
        n = n_per_tile
        cs = SkyCoord(r0 * u.deg, d0 * u.deg).directional_offset_by(
            RNG.uniform(0, 360, n) * u.deg, RNG.uniform(0, 3, n) * u.deg
        )
        l, m = fp.offsets_skyoffset(
            cs.ra.deg, cs.dec.deg, np.full(n, r0), np.full(n, d0)
        )
        ras += list(cs.ra.deg)
        decs += list(cs.dec.deg)
        ls += list(l)
        ms += list(m)
        tiles += [name] * n
    n = len(tiles)
    chi = RNG.uniform(0, 180, n)
    q, uu = RNG.normal(size=n), RNG.normal(size=n)
    t = Table()
    t["ra"] = Column(ras, unit="deg", description="Right Ascension [ICRS]")
    t["dec"] = Column(decs, unit="deg", description="Declination [ICRS]")
    t["l_tile_centre"] = Column(ls, unit="deg")
    t["m_tile_centre"] = Column(ms, unit="deg")
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
    return path, t, centres


def test_end_to_end_preserves_everything(tmp_path):
    from astropy.io import fits

    path, orig, centres = make_catalogue(tmp_path, n_per_tile=250)
    out = tmp_path / "out.fits"
    assert fp.main([str(path), str(out), "--chunk-rows", "77"]) == 0  # uneven chunks

    with fits.open(path) as a, fits.open(out) as b:
        ha, hb = a[1].header, b[1].header
        n0 = ha["TFIELDS"]
        # every original header card is still there, unchanged
        skip = {"NAXIS1", "TFIELDS"}
        for card in ha.cards:
            if card.keyword in skip or card.keyword in ("COMMENT", "HISTORY"):
                continue
            assert hb[card.keyword] == card.value, card.keyword
        assert [str(c.value) for c in ha.cards if c.keyword == "COMMENT"] == [
            str(c.value) for c in hb.cards if c.keyword == "COMMENT"
        ]
        assert hb["TFIELDS"] > n0
        assert hb["NAXIS1"] == ha["NAXIS1"] + 8 * (hb["TFIELDS"] - n0)
        # original bytes are identical, row by row
        ra = np.frombuffer(np.asarray(a[1].data).tobytes(), np.uint8).reshape(
            len(orig), -1
        )
        rb = np.frombuffer(np.asarray(b[1].data).tobytes(), np.uint8).reshape(
            len(orig), -1
        )
        assert (rb[:, : ra.shape[1]] == ra).all()

    res = Table.read(out)
    for c in (
        "polangle_corr",
        "derot_polangle_corr",
        "stokesQ_corr",
        "stokesU_corr",
        "stokesQ_err_corr",
        "stokesU_err_corr",
        "stokesQU_cov_corr",
        "stokesQ_bkg_corr",
        "stokesU_bkg_corr",
        "pol_frame_gamma",
    ):
        assert c in res.colnames
    assert res["polangle_corr"].unit == "deg"
    assert res["ra"].description == "Right Ascension [ICRS]"
    assert "local-north" in res["polangle_corr"].description

    r0 = np.array([centres[t][0] for t in orig["tile_id"]])
    d0 = np.array([centres[t][1] for t in orig["tile_id"]])
    g = fp.gamma_deg(np.asarray(orig["ra"]), np.asarray(orig["dec"]), r0, d0)
    assert np.abs(np.asarray(res["pol_frame_gamma"]) - g).max() < 1e-6
    assert np.abs(g).max() > 10  # the polar tile exercises the effect
    exp = fp.wrap_angle(np.asarray(orig["derot_polangle"]) + g, 0.0)
    assert (
        np.abs(fp.wrap180(2 * (np.asarray(res["derot_polangle_corr"]) - exp))).max()
        < 1e-5
    )
    assert np.isnan(res["polangle_corr"][3])  # null stays null

    with pytest.raises(SystemExit):  # refuses to run twice
        fp.main([str(out), str(tmp_path / "again.fits")])


def test_refuses_gzip(tmp_path):
    with pytest.raises(SystemExit):
        fp.main([str(tmp_path / "x.fits.gz"), str(tmp_path / "o.fits")])


def test_inconsistent_offsets_abort(tmp_path):
    path, t, _ = make_catalogue(tmp_path)
    t["l_tile_centre"] = RNG.uniform(-3, 3, len(t))
    t.write(path, overwrite=True)
    with pytest.raises(SystemExit):
        fp.main([str(path), str(tmp_path / "out.fits")])
