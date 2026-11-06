from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from spice_racs_fix import geometry


def wrap180(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return (x + 180.0) % 360.0 - 180.0


def random_field(
    rng: np.random.Generator, n: int, max_off: float = 5.0
) -> tuple[NDArray[np.float64], ...]:
    ra0 = rng.uniform(0, 360, n)
    dec0 = rng.uniform(-89, 60, n)
    cs = SkyCoord(ra0 * u.deg, dec0 * u.deg).directional_offset_by(
        rng.uniform(0, 360, n) * u.deg, rng.uniform(0, max_off, n) * u.deg
    )
    return cs.ra.deg, cs.dec.deg, ra0, dec0


def test_gamma_matches_vector_projection(rng: np.random.Generator) -> None:
    ra, dec, ra0, dec0 = random_field(rng, 5000)

    def north(a: NDArray[np.float64], d: NDArray[np.float64]) -> NDArray[np.float64]:
        a, d = np.radians(a), np.radians(d)
        return np.array([-np.sin(d) * np.cos(a), -np.sin(d) * np.sin(a), np.cos(d)])

    def east(a: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.radians(a)
        return np.array([-np.sin(a), np.cos(a), np.zeros_like(a)])

    d = north(ra0, dec0)
    ref = np.degrees(np.arctan2((d * east(ra)).sum(0), (d * north(ra, dec)).sum(0)))
    assert np.abs(wrap180(geometry.gamma_deg(ra, dec, ra0, dec0) - ref)).max() < 1e-8


def test_gamma_zero_on_meridian_and_small_offset_limit() -> None:
    assert float(geometry.gamma_deg(10.0, -50.0, 10.0, -60.0)) == pytest.approx(
        0, abs=1e-12
    )
    g = float(geometry.gamma_deg(4.0, -84.0, 0.0, -84.0))
    assert g == pytest.approx(4.0 * np.sin(np.radians(-84.0)), rel=0.01)


def test_sep_matches_astropy(rng: np.random.Generator) -> None:
    ra, dec, ra0, dec0 = random_field(rng, 500)
    ref = SkyCoord(ra * u.deg, dec * u.deg).separation(
        SkyCoord(ra0 * u.deg, dec0 * u.deg)
    )
    assert np.abs(geometry.sep_deg(ra, dec, ra0, dec0) - ref.deg).max() < 1e-9


def test_rotation_preserves_p_and_shifts_angle_by_gamma(
    rng: np.random.Generator,
) -> None:
    n = 5000
    gamma = rng.uniform(-60, 60, n)
    q, uu = rng.normal(size=n), rng.normal(size=n)
    q2, u2 = geometry.rotate_qu(q, uu, gamma)
    assert np.allclose(np.hypot(q, uu), np.hypot(q2, u2))
    dchi = 0.5 * np.degrees(np.arctan2(u2, q2) - np.arctan2(uu, q)) - gamma
    assert np.abs((dchi + 90) % 180 - 90).max() < 1e-8


def test_errors_match_matrix_form(rng: np.random.Generator) -> None:
    n = 1000
    gamma = rng.uniform(-60, 60, n)
    sq, su = rng.uniform(0.5, 2, n), rng.uniform(0.5, 2, n)
    cov = rng.uniform(-0.2, 0.2, n)
    res = geometry.rotate_qu_errors(sq, su, gamma, cov)
    th = np.radians(2 * gamma)
    for i in range(0, n, 97):
        rot = np.array(
            [[np.cos(th[i]), -np.sin(th[i])], [np.sin(th[i]), np.cos(th[i])]]
        )
        c2 = rot @ np.array([[sq[i] ** 2, cov[i]], [cov[i], su[i] ** 2]]) @ rot.T
        assert res.stokes_q_err[i] ** 2 == pytest.approx(c2[0, 0])
        assert res.stokes_u_err[i] ** 2 == pytest.approx(c2[1, 1])
        assert res.cov_qu[i] == pytest.approx(c2[0, 1])
    # equal, uncorrelated errors stay uncorrelated
    same = geometry.rotate_qu_errors(np.ones(5), np.ones(5), np.linspace(-40, 40, 5))
    assert np.allclose(same.cov_qu, 0)


def test_wrap_angle() -> None:
    x = np.array([-100.0, -90.0, 0.0, 89.9, 90.0, 200.0])
    for lo in (0.0, -90.0):
        w = geometry.wrap_angle(x, lo)
        assert (w >= lo).all()
        assert (w < lo + 180).all()
        assert np.allclose(np.sin(np.radians(2 * w)), np.sin(np.radians(2 * x)))
