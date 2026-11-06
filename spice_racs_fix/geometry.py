"""Sky geometry for the polarisation frame correction"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RotatedErrors(NamedTuple):
    """Q/U errors after a rotation"""

    stokes_q_err: NDArray[np.float64]
    """Error in rotated Q"""
    stokes_u_err: NDArray[np.float64]
    """Error in rotated U"""
    cov_qu: NDArray[np.float64]
    """Covariance of rotated Q and U"""


def gamma_deg(
    ra: ArrayLike, dec: ArrayLike, ra0: ArrayLike, dec0: ArrayLike
) -> NDArray[np.float64]:
    """Frame rotation from boresight north to local north.

    gamma is the angle, east of local north at (ra, dec), of the
    boresight-north direction projected onto the sky:

        gamma = atan2(sin(dec0) sin(ra - ra0),
                      cos(dec) cos(dec0) + sin(dec) sin(dec0) cos(ra - ra0))

    Args:
        ra: Source right ascension (deg).
        dec: Source declination (deg).
        ra0: Boresight right ascension (deg).
        dec0: Boresight declination (deg).

    Returns:
        gamma (deg).
    """
    a, d, a0, d0 = (
        np.radians(np.asarray(x, dtype=np.float64)) for x in (ra, dec, ra0, dec0)
    )
    num = np.sin(d0) * np.sin(a - a0)
    den = np.cos(d) * np.cos(d0) + np.sin(d) * np.sin(d0) * np.cos(a - a0)
    return np.asarray(np.degrees(np.arctan2(num, den)), dtype=np.float64)


def sep_deg(
    ra: ArrayLike, dec: ArrayLike, ra0: ArrayLike, dec0: ArrayLike
) -> NDArray[np.float64]:
    """Great-circle separation (haversine).

    Args:
        ra: Right ascension of the first point (deg).
        dec: Declination of the first point (deg).
        ra0: Right ascension of the second point (deg).
        dec0: Declination of the second point (deg).

    Returns:
        Separation (deg).
    """
    a, d, a0, d0 = (
        np.radians(np.asarray(x, dtype=np.float64)) for x in (ra, dec, ra0, dec0)
    )
    h = np.sin((d - d0) / 2) ** 2 + np.cos(d) * np.cos(d0) * np.sin((a - a0) / 2) ** 2
    return np.asarray(
        np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))), dtype=np.float64
    )


def wrap_angle(chi: ArrayLike, lo: float) -> NDArray[np.float64]:
    """Wrap polarisation angles into [lo, lo + 180).

    Args:
        chi: Angles (deg).
        lo: Lower edge of the range (deg).

    Returns:
        Wrapped angles (deg).
    """
    return np.asarray(
        (np.asarray(chi, dtype=np.float64) - lo) % 180.0 + lo, dtype=np.float64
    )


def rotate_qu(
    stokes_q: ArrayLike, stokes_u: ArrayLike, gamma: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotate Q and U by (Q + iU) exp(2i gamma).

    Args:
        stokes_q: Stokes Q.
        stokes_u: Stokes U.
        gamma: Rotation (deg).

    Returns:
        Rotated (Q, U).
    """
    z = (np.asarray(stokes_q) + 1j * np.asarray(stokes_u)) * np.exp(
        2j * np.radians(np.asarray(gamma, dtype=np.float64))
    )
    return np.asarray(z.real, dtype=np.float64), np.asarray(z.imag, dtype=np.float64)


def rotate_qu_errors(
    stokes_q_err: ArrayLike,
    stokes_u_err: ArrayLike,
    gamma: ArrayLike,
    cov_qu: ArrayLike | None = None,
) -> RotatedErrors:
    """Propagate Q/U errors through the rotation, C' = R C R^T.

    Args:
        stokes_q_err: Error in Q.
        stokes_u_err: Error in U.
        gamma: Rotation (deg).
        cov_qu: Q/U covariance. Q and U are independent if None.

    Returns:
        Rotated errors and covariance.
    """
    sq = np.asarray(stokes_q_err, dtype=np.float64)
    su = np.asarray(stokes_u_err, dtype=np.float64)
    cov = np.zeros_like(sq) if cov_qu is None else np.asarray(cov_qu, dtype=np.float64)
    th = np.radians(2.0 * np.asarray(gamma, dtype=np.float64))
    c, s = np.cos(th), np.sin(th)
    vq, vu = sq**2, su**2
    return RotatedErrors(
        stokes_q_err=np.sqrt(c * c * vq + s * s * vu - 2 * c * s * cov),
        stokes_u_err=np.sqrt(s * s * vq + c * c * vu + 2 * c * s * cov),
        cov_qu=c * s * (vq - vu) + (c * c - s * s) * cov,
    )
