"""Sky geometry for the polarisation frame rotation"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RotatedStokesErrors(NamedTuple):
    """Stokes Q and U errors after a rotation"""

    stokes_q_error: NDArray[np.float64]
    """Error in rotated Q"""
    stokes_u_error: NDArray[np.float64]
    """Error in rotated U"""
    covariance_qu: NDArray[np.float64]
    """Covariance of rotated Q and U"""


def as_radians(degrees: ArrayLike) -> NDArray[np.float64]:
    """Angles in degrees as a float64 array in radians."""
    return np.radians(np.asarray(degrees, dtype=np.float64))


def local_north_rotation_deg(
    source_ra_deg: ArrayLike,
    source_dec_deg: ArrayLike,
    boresight_ra_deg: ArrayLike,
    boresight_dec_deg: ArrayLike,
) -> NDArray[np.float64]:
    """Rotation (gamma) from boresight north to local north.

    gamma is the angle, east of local north at the source, of the
    boresight-north direction projected onto the sky:

        gamma = atan2(sin(dec0) sin(ra - ra0),
                      cos(dec) cos(dec0) + sin(dec) sin(dec0) cos(ra - ra0))

    Args:
        source_ra_deg: Source right ascension (deg).
        source_dec_deg: Source declination (deg).
        boresight_ra_deg: Boresight right ascension (deg).
        boresight_dec_deg: Boresight declination (deg).

    Returns:
        gamma (deg).
    """
    source_ra, source_dec = as_radians(source_ra_deg), as_radians(source_dec_deg)
    boresight_ra = as_radians(boresight_ra_deg)
    boresight_dec = as_radians(boresight_dec_deg)
    ra_offset = source_ra - boresight_ra
    numerator = np.sin(boresight_dec) * np.sin(ra_offset)
    denominator = np.cos(source_dec) * np.cos(boresight_dec) + np.sin(
        source_dec
    ) * np.sin(boresight_dec) * np.cos(ra_offset)
    return np.asarray(np.degrees(np.arctan2(numerator, denominator)), dtype=np.float64)


def angular_separation_deg(
    first_ra_deg: ArrayLike,
    first_dec_deg: ArrayLike,
    second_ra_deg: ArrayLike,
    second_dec_deg: ArrayLike,
) -> NDArray[np.float64]:
    """Great-circle separation between two points (haversine formula).

    Args:
        first_ra_deg: Right ascension of the first point (deg).
        first_dec_deg: Declination of the first point (deg).
        second_ra_deg: Right ascension of the second point (deg).
        second_dec_deg: Declination of the second point (deg).

    Returns:
        Separation (deg).
    """
    first_ra, first_dec = as_radians(first_ra_deg), as_radians(first_dec_deg)
    second_ra, second_dec = as_radians(second_ra_deg), as_radians(second_dec_deg)
    haversine = (
        np.sin((first_dec - second_dec) / 2) ** 2
        + np.cos(first_dec)
        * np.cos(second_dec)
        * np.sin((first_ra - second_ra) / 2) ** 2
    )
    return np.asarray(
        np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))),
        dtype=np.float64,
    )


def wrap_polarisation_angle(
    angle_deg: ArrayLike, lower_edge_deg: float
) -> NDArray[np.float64]:
    """Wrap polarisation angles into [lower_edge_deg, lower_edge_deg + 180).

    Args:
        angle_deg: Angles (deg).
        lower_edge_deg: Lower edge of the range (deg).

    Returns:
        Wrapped angles (deg).
    """
    angles = np.asarray(angle_deg, dtype=np.float64)
    return np.asarray(
        (angles - lower_edge_deg) % 180.0 + lower_edge_deg, dtype=np.float64
    )


def rotate_stokes_qu(
    stokes_q: ArrayLike, stokes_u: ArrayLike, rotation_deg: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate Stokes Q and U by (Q + iU) exp(2i rotation).

    Args:
        stokes_q: Stokes Q.
        stokes_u: Stokes U.
        rotation_deg: Rotation (deg).

    Returns:
        Rotated (Q, U).
    """
    linear_polarisation = (np.asarray(stokes_q) + 1j * np.asarray(stokes_u)) * np.exp(
        2j * as_radians(rotation_deg)
    )
    return (
        np.asarray(linear_polarisation.real, dtype=np.float64),
        np.asarray(linear_polarisation.imag, dtype=np.float64),
    )


def rotate_stokes_qu_errors(
    stokes_q_error: ArrayLike,
    stokes_u_error: ArrayLike,
    rotation_deg: ArrayLike,
    covariance_qu: ArrayLike | None = None,
) -> RotatedStokesErrors:
    """Propagate Stokes Q and U errors through the rotation, C' = R C R^T.

    Args:
        stokes_q_error: Error in Q.
        stokes_u_error: Error in U.
        rotation_deg: Rotation (deg).
        covariance_qu: Q/U covariance. Q and U are independent if None.

    Returns:
        Rotated errors and covariance.
    """
    q_error = np.asarray(stokes_q_error, dtype=np.float64)
    u_error = np.asarray(stokes_u_error, dtype=np.float64)
    covariance = (
        np.zeros_like(q_error)
        if covariance_qu is None
        else np.asarray(covariance_qu, dtype=np.float64)
    )
    double_angle = as_radians(2.0 * np.asarray(rotation_deg, dtype=np.float64))
    cosine, sine = np.cos(double_angle), np.sin(double_angle)
    q_variance, u_variance = q_error**2, u_error**2
    return RotatedStokesErrors(
        stokes_q_error=np.sqrt(
            cosine**2 * q_variance
            + sine**2 * u_variance
            - 2 * cosine * sine * covariance
        ),
        stokes_u_error=np.sqrt(
            sine**2 * q_variance
            + cosine**2 * u_variance
            + 2 * cosine * sine * covariance
        ),
        covariance_qu=cosine * sine * (q_variance - u_variance)
        + (cosine**2 - sine**2) * covariance,
    )
