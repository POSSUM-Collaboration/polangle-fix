"""Tests for fix_polframe. These check the algebra and the plumbing. They cannot
check the sign convention (chi_true = chi + gamma); that needs a source of known
absolute angle."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
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


def test_spherical_helpers_match_astropy():
    ra, dec, ra0, dec0 = random_field(2000)
    c0, c = SkyCoord(ra0 * u.deg, dec0 * u.deg), SkyCoord(ra * u.deg, dec * u.deg)
    pa = c0.position_angle(c).deg
    assert (
        np.abs(fp.wrap180(fp.position_angle_deg(ra0, dec0, ra, dec) - pa)).max() < 1e-9
    )
    sep = c0.separation(c).deg
    r, d = fp.offset_by_deg(ra0, dec0, pa, sep)
    assert fp.sep_deg(r, d, ra, dec).max() < 1e-9
    # ARC offsets as arrakis writes them, in a rotated frame
    rot = RNG.uniform(-180, 180, len(ra))
    l, m = fp.offsets_arc(ra, dec, ra0, dec0, rot)
    assert np.allclose(np.hypot(l, m), sep, atol=1e-9)
    assert np.abs(fp.wrap180(np.degrees(np.arctan2(l, m)) + rot - pa)).max() < 1e-9


def test_arc_centre_inverts_offsets_exactly():
    # away from the poles the inversion is unique and needs no seed
    ra, dec, ra0, dec0, _ = tiled_field(300, 10, -80.0, 60.0)
    rot = RNG.uniform(-180, 180, len(ra))
    l, m = fp.offsets_arc(ra, dec, ra0, dec0, rot)
    r, d = fp._arc_centre(ra, dec, l, m, rot)
    assert fp.sep_deg(r, d, ra0, dec0).max() < 1e-9
    # within reach of the pole it can have two solutions, sometimes only a
    # fraction of a degree apart in direction; seeding with the direction to
    # the known centre (as recover_centre does) picks the right one
    ra, dec, ra0, dec0, _ = tiled_field(300, 10, -89.9, -85.0)
    rot = RNG.uniform(-180, 180, len(ra))
    l, m = fp.offsets_arc(ra, dec, ra0, dec0, rot)
    psi0 = fp.position_angle_deg(ra, dec, ra0, dec0)
    r, d = fp._arc_centre(ra, dec, l, m, rot, psi0)
    assert fp.sep_deg(r, d, ra0, dec0).max() < 1e-9


def tiled_field(
    n_tiles: int, per_tile: int, dec_lo: float, dec_hi: float, max_off: float = 5.0
):
    """Sources grouped into tiles; every source in a tile shares one centre."""
    ra0 = RNG.uniform(0, 360, n_tiles)
    dec0 = RNG.uniform(dec_lo, dec_hi, n_tiles)
    idx = np.repeat(np.arange(n_tiles), per_tile)
    n = n_tiles * per_tile
    cs = SkyCoord(ra0[idx] * u.deg, dec0[idx] * u.deg).directional_offset_by(
        RNG.uniform(0, 360, n) * u.deg, RNG.uniform(0, max_off, n) * u.deg
    )
    return cs.ra.deg, cs.dec.deg, ra0[idx], dec0[idx], idx


@pytest.mark.parametrize("conv", fp.CONVENTIONS)
@pytest.mark.parametrize(
    "dec_range", [(-60.0, 30.0), (-89.9, -85.0)], ids=["mid", "polar"]
)
def test_recover_centre_roundtrip(conv, dec_range):
    n_tiles = 60
    ra, dec, ra0, dec0, idx = tiled_field(n_tiles, 40, *dec_range)
    rot_t = RNG.uniform(-180, 180, n_tiles)
    if conv == "arc":
        l, m = fp.offsets_arc(ra, dec, ra0, dec0, rot_t[idx])
    elif conv == "skyoffset":
        l, m = fp.offsets_skyoffset(ra, dec, ra0, dec0)
    elif conv == "flat":
        l, m = fp.wrap180(ra - ra0) * np.cos(np.radians(dec)), dec - dec0
    else:
        l, m = fp.wrap180(ra - ra0), dec - dec0
    r0, d0, rot = fp.recover_centre(conv, ra, dec, l, m, idx, n_tiles)
    assert fp.sep_deg(r0, d0, ra0, dec0).max() < 1e-5
    if conv == "arc":
        assert np.abs(fp.wrap180(rot - rot_t)).max() < 1e-6
    else:
        assert rot is None


def test_arc_rotation_fit_is_robust_to_bad_rows():
    n_tiles = 10
    ra, dec, ra0, dec0, idx = tiled_field(n_tiles, 50, -70.0, 20.0)
    rot_t = RNG.uniform(-180, 180, n_tiles)
    l, m = fp.offsets_arc(ra, dec, ra0, dec0, rot_t[idx])
    bad = RNG.uniform(size=len(ra)) < 0.1  # 10% of rows have garbage offsets
    l[bad] += RNG.uniform(-1, 1, bad.sum())
    rot, c_ra, c_dec = fp.fit_arc_frame(ra, dec, l, m, idx, n_tiles)
    assert np.abs(fp.wrap180(rot - rot_t)).max() < 1e-6
    assert fp.sep_deg(c_ra, c_dec, ra0[::50], dec0[::50]).max() < 1e-6


def test_nan_centres_count_as_inconsistent():
    n_tiles = 5
    ra, dec, ra0, dec0, idx = tiled_field(n_tiles, 20, -60.0, 20.0)
    l, m = fp.offsets_skyoffset(ra, dec, ra0, dec0)
    valid = np.ones(len(ra), bool)
    good = fp.fit_centres("skyoffset", ra, dec, l, m, idx, n_tiles, valid, 1.0)
    assert good.frac_bad == 0
    l[::7] = np.nan  # NaN offsets in rows that are still marked valid
    nan = fp.fit_centres("skyoffset", ra, dec, l, m, idx, n_tiles, valid, 1.0)
    assert nan.frac_bad == pytest.approx(np.mean(np.isnan(l)))
    # the tile centres are still recovered from the finite rows, even when
    # the first row of a tile is the NaN one
    assert fp.sep_deg(nan.ra0, nan.dec0, good.ra0, good.dec0).max() < 1e-9


def test_skyoffset_has_two_roots_near_the_pole():
    """Near the pole two centres give the same (l, m); the tile resolves which."""
    n_tiles = 20
    ra, dec, ra0, dec0, _idx = tiled_field(n_tiles, 40, -89.0, -86.0)
    l, m = fp.offsets_skyoffset(ra, dec, ra0, dec0)
    roots = fp._skyoffset_roots(ra, dec, l, m)
    both = np.isfinite(roots[0][1]) & np.isfinite(roots[1][1])
    assert both.any()  # the ambiguity is real in this regime
    for rr, dd in roots:  # and every valid root reproduces (l, m)
        ok = np.isfinite(dd)
        l2, m2 = fp.offsets_skyoffset(ra[ok], dec[ok], rr[ok], dd[ok])
        assert np.abs(fp.wrap180(l2 - l[ok])).max() < 1e-6
        assert np.abs(m2 - m[ok]).max() < 1e-6


@pytest.mark.parametrize(
    "tail",
    [
        ["meta: !!omap", "- __serialized_columns__: {}"],  # astropy >= 7
        ["schema: astropy-2.0"],  # older astropy, table without meta
        [],
    ],
    ids=["meta", "schema", "bare"],
)
def test_splice_serialized_inserts_at_end_of_datatype_list(tail):
    import yaml

    full = (
        "- {name: dec, unit: deg, datatype: float64, description: 'Declination [ICRS]'}"
    )
    chunk = full[:70] + "\\"  # a 70-character piece plus the continuation mark
    assert len(chunk) == 71
    items = ["- {name: ra, unit: deg, datatype: float64}", chunk, full[70:]]
    hdr = fits.Header()
    hdr["SIMPLE"] = True
    for v in [fp.SER_BEGIN, "datatype:", *items, *tail, fp.SER_END]:
        hdr["COMMENT"] = v
    new = ["- {name: x_corr, datatype: float64}"]
    assert fp.splice_serialized(hdr, new)
    values = [str(c.value) for c in hdr.cards if c.keyword == "COMMENT"]
    assert values == [fp.SER_BEGIN, "datatype:", *items, *new, *tail, fp.SER_END]
    # and the result is what astropy will parse: join the chunks, load the YAML
    joined, cont = [], False
    for v in values[1:-1]:
        if cont:
            joined[-1] += v[:70]
        else:
            joined.append(v[:70])
        cont = len(v) == 71 and v[70] == "\\"
    doc = yaml.safe_load("\n".join(joined))
    assert [d["name"] for d in doc["datatype"]] == ["ra", "dec", "x_corr"]


def test_parse_tile_name():
    assert fp.parse_tile_name("RACS_0426-84") == (pytest.approx(66.5), -84.0)
    assert fp.parse_tile_name("garbage") is None


def make_catalogue(tmp_path, n_per_tile=300, convention="skyoffset", rot=0.0):
    """Two tiles, one near the pole, written as a plain FITS binary table."""
    centres = {"RACS_0426-84": (66.5, -84.0), "RACS_1200-30": (180.0, -30.0)}
    ras, decs, ls, ms, tiles = [], [], [], [], []
    for name, (r0, d0) in centres.items():
        n = n_per_tile
        cs = SkyCoord(r0 * u.deg, d0 * u.deg).directional_offset_by(
            RNG.uniform(0, 360, n) * u.deg, RNG.uniform(0, 3, n) * u.deg
        )
        if convention == "arc":
            l, m = fp.offsets_arc(
                cs.ra.deg, cs.dec.deg, np.full(n, r0), np.full(n, d0), rot
            )
        else:
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
    with fits.open(path) as hdul:  # add checksums, as a released file may have
        hdul.writeto(path, overwrite=True, checksum=True)
    return path, t, centres


def test_end_to_end_preserves_everything(tmp_path):
    from astropy.io import fits

    path, orig, centres = make_catalogue(tmp_path, n_per_tile=250)
    out = tmp_path / "out.fits"
    assert fp.main([str(path), str(out), "--chunk-rows", "77"]) == 0  # uneven chunks

    with fits.open(path) as a, fits.open(out) as b:
        ha, hb = a[1].header, b[1].header
        n0 = ha["TFIELDS"]
        # stale checksums are dropped rather than copied
        assert "CHECKSUM" in ha and "CHECKSUM" not in hb and "DATASUM" not in hb
        assert "CHECKSUM" not in b[0].header
        # every original header card is still there, unchanged
        skip = {"NAXIS1", "TFIELDS", "CHECKSUM", "DATASUM"}
        for card in ha.cards:
            if card.keyword in skip or card.keyword in ("COMMENT", "HISTORY"):
                continue
            assert hb[card.keyword] == card.value, card.keyword
        # original COMMENT lines are all still there, in order; only new lines were added
        ca = [str(c.value) for c in ha.cards if c.keyword == "COMMENT"]
        cb = [str(c.value) for c in hb.cards if c.keyword == "COMMENT"]
        it = iter(cb)
        assert all(any(x == y for y in it) for x in ca)
        added = len(cb) - len(ca)
        assert added > 0 and any("name: pol_frame_gamma" in x for x in cb)
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
    assert np.isnan(np.ma.filled(res["polangle_corr"], np.nan)[3])  # null stays null

    with pytest.raises(SystemExit):  # refuses to run twice
        fp.main([str(out), str(tmp_path / "again.fits")])


def test_end_to_end_arc_offsets_with_rotation(tmp_path):
    """arrakis-style offsets in a frame rotated by 45 deg + POL_AXIS."""
    import csv

    from astropy.io import fits

    path, orig, centres = make_catalogue(
        tmp_path, n_per_tile=200, convention="arc", rot=30.0
    )
    out = tmp_path / "out.fits"
    assert fp.main([str(path), str(out)]) == 0
    with fits.open(out) as b:
        assert b[1].header["POLFSRC"] == "offsets:arc"
    with open(f"{out}.tile_centres.csv", newline="") as fh:
        rows = {r["tile_id"]: r for r in csv.DictReader(fh)}
    for name, (r0, d0) in centres.items():
        assert float(rows[name]["ra0"]) == pytest.approx(r0, abs=1e-6)
        assert float(rows[name]["dec0"]) == pytest.approx(d0, abs=1e-6)
        assert float(rows[name]["arc_rot_deg"]) == pytest.approx(30.0, abs=1e-6)
    res = Table.read(out)
    r0 = np.array([centres[t][0] for t in orig["tile_id"]])
    d0 = np.array([centres[t][1] for t in orig["tile_id"]])
    g = fp.gamma_deg(np.asarray(orig["ra"]), np.asarray(orig["dec"]), r0, d0)
    assert np.abs(np.asarray(res["pol_frame_gamma"]) - g).max() < 1e-6


def test_refuses_primary_with_data(tmp_path):
    from astropy.io import fits

    path, _t, _ = make_catalogue(tmp_path, n_per_tile=20)
    with fits.open(path) as hdul:
        hdul[0].data = np.zeros(4)
        hdul.writeto(path, overwrite=True)
    with pytest.raises(SystemExit):
        fp.main([str(path), str(tmp_path / "o.fits")])


def test_refuses_gzip(tmp_path):
    with pytest.raises(SystemExit):
        fp.main([str(tmp_path / "x.fits.gz"), str(tmp_path / "o.fits")])


def test_inconsistent_offsets_abort(tmp_path):
    path, t, _ = make_catalogue(tmp_path)
    t["l_tile_centre"] = RNG.uniform(-3, 3, len(t))
    t.write(path, overwrite=True)
    with pytest.raises(SystemExit):
        fp.main([str(path), str(tmp_path / "out.fits")])
