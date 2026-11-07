from __future__ import annotations

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from numpy.typing import NDArray
from spice_racs_fix import geometry


def wrap_to_plus_minus_180(angle_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    return (angle_deg + 180.0) % 360.0 - 180.0


def random_sources_near_boresights(
    random_generator: np.random.Generator,
    count: int,
    max_offset_deg: float = 5.0,
) -> tuple[NDArray[np.float64], ...]:
    """Source RA, Dec and boresight RA, Dec (deg)."""
    boresight_ra = random_generator.uniform(0, 360, count)
    boresight_dec = random_generator.uniform(-89, 60, count)
    sources = SkyCoord(
        boresight_ra * units.deg, boresight_dec * units.deg
    ).directional_offset_by(
        random_generator.uniform(0, 360, count) * units.deg,
        random_generator.uniform(0, max_offset_deg, count) * units.deg,
    )
    return sources.ra.deg, sources.dec.deg, boresight_ra, boresight_dec


def test_rotation_matches_vector_projection(
    random_generator: np.random.Generator,
) -> None:
    source_ra, source_dec, boresight_ra, boresight_dec = random_sources_near_boresights(
        random_generator, 5000
    )

    def north_unit_vector(
        ra_deg: NDArray[np.float64], dec_deg: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        ra, dec = np.radians(ra_deg), np.radians(dec_deg)
        return np.array(
            [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)]
        )

    def east_unit_vector(ra_deg: NDArray[np.float64]) -> NDArray[np.float64]:
        ra = np.radians(ra_deg)
        return np.array([-np.sin(ra), np.cos(ra), np.zeros_like(ra)])

    boresight_north = north_unit_vector(boresight_ra, boresight_dec)
    expected = np.degrees(
        np.arctan2(
            (boresight_north * east_unit_vector(source_ra)).sum(0),
            (boresight_north * north_unit_vector(source_ra, source_dec)).sum(0),
        )
    )
    rotation = geometry.local_north_rotation_deg(
        source_ra, source_dec, boresight_ra, boresight_dec
    )
    assert np.abs(wrap_to_plus_minus_180(rotation - expected)).max() < 1e-8


def test_rotation_zero_on_meridian_and_small_offset_limit() -> None:
    on_meridian = geometry.local_north_rotation_deg(10.0, -50.0, 10.0, -60.0)
    assert float(on_meridian) == pytest.approx(0, abs=1e-12)
    small_offset = float(geometry.local_north_rotation_deg(4.0, -84.0, 0.0, -84.0))
    assert small_offset == pytest.approx(4.0 * np.sin(np.radians(-84.0)), rel=0.01)


def test_separation_matches_astropy(random_generator: np.random.Generator) -> None:
    source_ra, source_dec, boresight_ra, boresight_dec = random_sources_near_boresights(
        random_generator, 500
    )
    expected = SkyCoord(source_ra * units.deg, source_dec * units.deg).separation(
        SkyCoord(boresight_ra * units.deg, boresight_dec * units.deg)
    )
    separation = geometry.angular_separation_deg(
        source_ra, source_dec, boresight_ra, boresight_dec
    )
    assert np.abs(separation - expected.deg).max() < 1e-9


def test_rotation_preserves_polarised_intensity_and_shifts_angle(
    random_generator: np.random.Generator,
) -> None:
    count = 5000
    rotation = random_generator.uniform(-60, 60, count)
    stokes_q = random_generator.normal(size=count)
    stokes_u = random_generator.normal(size=count)
    rotated_q, rotated_u = geometry.rotate_stokes_qu(stokes_q, stokes_u, rotation)
    assert np.allclose(np.hypot(stokes_q, stokes_u), np.hypot(rotated_q, rotated_u))
    angle_change = (
        0.5
        * np.degrees(np.arctan2(rotated_u, rotated_q) - np.arctan2(stokes_u, stokes_q))
        - rotation
    )
    assert np.abs((angle_change + 90) % 180 - 90).max() < 1e-8


def test_errors_match_matrix_form(random_generator: np.random.Generator) -> None:
    count = 1000
    rotation = random_generator.uniform(-60, 60, count)
    q_error = random_generator.uniform(0.5, 2, count)
    u_error = random_generator.uniform(0.5, 2, count)
    covariance = random_generator.uniform(-0.2, 0.2, count)
    result = geometry.rotate_stokes_qu_errors(q_error, u_error, rotation, covariance)
    double_angle = np.radians(2 * rotation)
    for index in range(0, count, 97):
        rotation_matrix = np.array(
            [
                [np.cos(double_angle[index]), -np.sin(double_angle[index])],
                [np.sin(double_angle[index]), np.cos(double_angle[index])],
            ]
        )
        input_covariance = np.array(
            [
                [q_error[index] ** 2, covariance[index]],
                [covariance[index], u_error[index] ** 2],
            ]
        )
        expected = rotation_matrix @ input_covariance @ rotation_matrix.T
        assert result.stokes_q_error[index] ** 2 == pytest.approx(expected[0, 0])
        assert result.stokes_u_error[index] ** 2 == pytest.approx(expected[1, 1])
        assert result.covariance_qu[index] == pytest.approx(expected[0, 1])
    # equal, uncorrelated errors stay uncorrelated
    equal_errors = geometry.rotate_stokes_qu_errors(
        np.ones(5), np.ones(5), np.linspace(-40, 40, 5)
    )
    assert np.allclose(equal_errors.covariance_qu, 0)


def test_wrap_polarisation_angle() -> None:
    angles = np.array([-100.0, -90.0, 0.0, 89.9, 90.0, 200.0])
    for lower_edge in (0.0, -90.0):
        wrapped = geometry.wrap_polarisation_angle(angles, lower_edge)
        assert (wrapped >= lower_edge).all()
        assert (wrapped < lower_edge + 180).all()
        assert np.allclose(
            np.sin(np.radians(2 * wrapped)), np.sin(np.radians(2 * angles))
        )
