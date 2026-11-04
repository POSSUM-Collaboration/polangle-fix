#!/usr/bin/env python3
"""Add local-north polarisation columns to the SPICE-RACS DR2 catalogue (streaming).

ASKAP catalogue angles are measured from north at the observing boresight, not
from north at each source. This script keeps every original column and every
header card exactly as it is and appends corrected copies with a ``_corr``
suffix, row by row:

    polangle, derot_polangle        -> + gamma, wrapped to the original range
    stokesQ/U, stokesQ/U_bkg        -> (Q + iU) * exp(2i gamma)
    stokesQ_err, stokesU_err        -> rotated covariance, plus stokesQU_cov_corr

plus a diagnostic column ``pol_frame_gamma`` (degrees).

Not corrected (the originals remain in the frame of the boresight):
  * rm, rm_err, polint*, fracpol*, stokesI*, stokesV*, polangle_err and
    derot_polangle_err are frame-independent.
  * sigma_add and sigma_add_err are Faraday-complexity metrics, left alone.
  * Spectra, FDFs, cubelets and RM-CLEAN components are not catalogue columns.

gamma is the angle, east of local north, of the boresight-north direction
projected onto the sky at the source:

    gamma = atan2( sin(dec0) sin(ra - ra0),
                   cos(dec) cos(dec0) + sin(dec) sin(dec0) cos(ra - ra0) )

How it works, and why metadata is preserved
  * The input must be an uncompressed FITS file with one binary table (unzip
    first). It is memory-mapped; the table is never loaded whole.
  * Pass 1 reads only ra, dec, l/m offsets, tile_id and the angle columns to
    recover per-tile centres.
  * Pass 2 streams the rows in chunks. Each output row is the original row
    bytes followed by the new columns (big-endian float64). Nothing about the
    original columns is re-encoded.
  * The output header is the input header verbatim (including TCOMM cards and
    the astropy-serialised column block) with NAXIS1/TFIELDS updated and cards
    added for the new columns. The new columns get TUNIT/TCOMM cards and an
    entry (unit, description) appended to the serialised block, so astropy
    reads them back with metadata. They have no UCD.
  * New columns are appended at the end of the table.

Boresight (assumptions to be aware of)
  * ``tile_id`` names the field whose cube supplied each row's Q/U. arrakis
    computes the offset columns per ``field_name``; if that column exists and
    disagrees with ``tile_id`` for any row the script says so (use --tile-col).
  * The pointing centre of each tile is recovered from ``l_tile_centre`` and
    ``m_tile_centre`` (angular offsets of the source from the tile centre).
    Every candidate offset convention is tested and the one with consistent
    centres within each tile is used; the script stops if none is. The
    conventions are:
      arc        l = sep sin(PA - rot), m = sep cos(PA - rot), with a frame
                 rotation ``rot`` fitted per tile. This is what arrakis
                 ``update_tile_separations`` writes since 2024-07-15, with
                 rot = 45 deg + POL_AXIS of the field (0 for POL_AXIS = -45).
      skyoffset  astropy ``spherical_offsets_to`` from the centre
      flat       l = (ra - ra0) cos(dec), m = dec - dec0
      radec      l = ra - ra0, m = dec - dec0
    Catalogues built with arrakis between 2024-06-25 and 2024-07-15 hold
    ``spherical_offsets_to`` from the *source* to the centre (sign reversed);
    no convention fits those, so supply --boresights.
  * The recovered centre is the tile centre, which may differ from the true
    ASKAP boresight of the SBID. Use --boresights to supply true centres
    (RA_DEG/DEC_DEG of the field in the RACS database is the right source).
  * Every tile that has rows must end up with a finite centre; the script
    stops otherwise, so no row can silently receive NaN corrections.
  * CHECKSUM/DATASUM cards are dropped from both output headers because the
    data they describe change. The primary HDU must carry no data.

Sign convention (FRAME_SIGN): chi_true = chi_measured + gamma. This follows the
report but is not verified. Check it once against a source of known absolute
angle (for example 3C 286 placed off-axis) before release.

Memory: roughly 3 GB for pass 1 on the full DR2 table, then a few hundred MB.
Disk: the output is about the size of the input plus ~1 GB.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from astropy import units as au
from astropy.io import fits
from astropy.table import Column, Table
from numpy.typing import NDArray

F64 = NDArray[np.float64]
BoolArr = NDArray[np.bool_]
IntArr = NDArray[np.intp]

# chi_true = chi_measured + FRAME_SIGN * gamma. Verify against a known source.
FRAME_SIGN: Final[float] = 1.0
SUFFIX: Final[str] = "_corr"
GAMMA_COL: Final[str] = "pol_frame_gamma"
COV_COL: Final[str] = "stokesQU_cov_corr"
BLOCK: Final[int] = 2880


@dataclass(frozen=True)
class Columns:
    """Names of the DR2 catalogue columns used by this script."""

    ra: str = "ra"
    dec: str = "dec"
    l_off: str = "l_tile_centre"
    m_off: str = "m_tile_centre"
    tile: str = "tile_id"
    sep: str = "separation_tile_centre"
    angles: tuple[str, ...] = ("polangle", "derot_polangle")
    q: str = "stokesQ"
    u: str = "stokesU"
    q_err: str = "stokesQ_err"
    u_err: str = "stokesU_err"
    q_bkg: str = "stokesQ_bkg"
    u_bkg: str = "stokesU_bkg"

    def required(self, with_offsets: bool) -> list[str]:
        out = [
            self.ra,
            self.dec,
            self.tile,
            self.q,
            self.u,
            self.q_err,
            self.u_err,
            self.q_bkg,
            self.u_bkg,
            *self.angles,
        ]
        return out + ([self.l_off, self.m_off] if with_offsets else [])


# --------------------------------------------------------------------------
# Pure maths
# --------------------------------------------------------------------------
def wrap180(x: F64) -> F64:
    """Wrap angles in degrees into [-180, 180)."""
    return (x + 180.0) % 360.0 - 180.0


def wrap_angle(chi: F64, lo: float) -> F64:
    """Wrap polarisation angles (deg) into [lo, lo + 180)."""
    return (chi - lo) % 180.0 + lo


def gamma_deg(ra: F64, dec: F64, ra0: F64, dec0: F64) -> F64:
    """Frame rotation gamma in degrees, from the boresight-north feed vector."""
    ra_r, dec_r, ra0_r, dec0_r = (np.radians(x) for x in (ra, dec, ra0, dec0))
    dra = ra_r - ra0_r
    num = np.sin(dec0_r) * np.sin(dra)
    den = np.cos(dec_r) * np.cos(dec0_r) + np.sin(dec_r) * np.sin(dec0_r) * np.cos(dra)
    return FRAME_SIGN * np.degrees(np.arctan2(num, den))


def rotate_qu(q: F64, u: F64, gamma: F64) -> tuple[F64, F64]:
    """(Q + iU) * exp(2 i gamma)."""
    c, s = np.cos(np.radians(2.0 * gamma)), np.sin(np.radians(2.0 * gamma))
    return c * q - s * u, s * q + c * u


def rotate_qu_cov(
    sq: F64, su: F64, gamma: F64, cov: F64 | None = None
) -> tuple[F64, F64, F64]:
    """Propagate Q/U errors through the rotation: C' = R C R^T.

    Returns (sigma_Q', sigma_U', cov_QU'). Q and U are assumed independent
    (cov = 0) unless ``cov`` is given.
    """
    c, s = np.cos(np.radians(2.0 * gamma)), np.sin(np.radians(2.0 * gamma))
    vq, vu = sq**2, su**2
    cqu = np.zeros_like(vq) if cov is None else cov
    vq2 = c * c * vq + s * s * vu - 2.0 * c * s * cqu
    vu2 = s * s * vq + c * c * vu + 2.0 * c * s * cqu
    cov2 = c * s * (vq - vu) + (c * c - s * s) * cqu
    return np.sqrt(vq2), np.sqrt(vu2), cov2


# --------------------------------------------------------------------------
# Tile-centre recovery from (l, m) offsets
# --------------------------------------------------------------------------
def offsets_skyoffset(ra: F64, dec: F64, ra0: F64, dec0: F64) -> tuple[F64, F64]:
    """Offsets of (ra, dec) in the sky-offset frame centred on (ra0, dec0).

    Matches astropy's ``SkyCoord.spherical_offsets_to``. All values in degrees.
    """
    a, d, a0, d0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    x = np.cos(d) * np.cos(a - a0)
    y = np.cos(d) * np.sin(a - a0)
    z = np.sin(d)
    xx = x * np.cos(d0) + z * np.sin(d0)
    zz = -x * np.sin(d0) + z * np.cos(d0)
    return np.degrees(np.arctan2(y, xx)), np.degrees(np.arcsin(np.clip(zz, -1.0, 1.0)))


def position_angle_deg(ra0: F64, dec0: F64, ra: F64, dec: F64) -> F64:
    """Position angle (deg, east of north) of (ra, dec) seen from (ra0, dec0).

    Matches astropy's ``SkyCoord.position_angle``.
    """
    a0, d0, a, d = (np.radians(x) for x in (ra0, dec0, ra, dec))
    dra = a - a0
    x = np.sin(d) * np.cos(d0) - np.cos(d) * np.sin(d0) * np.cos(dra)
    y = np.sin(dra) * np.cos(d)
    return np.degrees(np.arctan2(y, x))


def offset_by_deg(ra: F64, dec: F64, pa: F64, sep: F64) -> tuple[F64, F64]:
    """Point ``sep`` degrees from (ra, dec) along position angle ``pa``.

    Matches astropy's ``SkyCoord.directional_offset_by``. All in degrees.
    """
    d, p, s = (np.radians(x) for x in (dec, pa, sep))
    cos_b = np.sin(d) * np.cos(s) + np.cos(d) * np.sin(s) * np.cos(p)
    xsin = np.sin(s) * np.sin(p) * np.cos(d)
    xcos = np.cos(s) - cos_b * np.sin(d)
    da = np.arctan2(xsin, xcos)
    at_pole = np.cos(d) < 1e-12  # ra is arbitrary there; copy astropy's choice
    da = np.where(at_pole, np.where(d > 0, np.pi - p, p), da)
    return (ra + np.degrees(da)) % 360.0, np.degrees(
        np.arcsin(np.clip(cos_b, -1.0, 1.0))
    )


def offsets_arc(
    ra: F64, dec: F64, ra0: F64, dec0: F64, rot: F64 | float = 0.0
) -> tuple[F64, F64]:
    """ARC offsets of (ra, dec) from (ra0, dec0) in a frame rotated by ``rot``.

    l = sep sin(PA - rot), m = sep cos(PA - rot), all in degrees, where sep
    and PA are the separation and position angle of the source seen from the
    centre. This is what arrakis ``update_tile_separations`` writes, with
    rot = 45 deg + POL_AXIS of the field.
    """
    sep = sep_deg(ra, dec, ra0, dec0)
    t = np.radians(position_angle_deg(ra0, dec0, ra, dec) - rot)
    return sep * np.sin(t), sep * np.cos(t)


def _arc_centre(
    ra: F64,
    dec: F64,
    l_off: F64,
    m_off: F64,
    rot: F64,
    psi0: F64 | None = None,
    n_iter: int = 20,
) -> tuple[F64, F64]:
    """Invert ``offsets_arc`` row by row, given the frame rotation (deg).

    The centre lies ``sep`` away from the source along some direction psi
    (the back-azimuth), chosen so that the source is at position angle PA as
    seen from the centre. psi is found by Newton's method with a numerical
    derivative, starting from ``psi0`` or from PA + 180 (exact on a plane).
    Within ``sep`` of a pole there can be two solutions; pass ``psi0`` (for
    example the direction towards an independently known centre) to pick.
    """
    sep = np.hypot(l_off, m_off)
    pa = np.degrees(np.arctan2(l_off, m_off)) + rot  # source seen from centre
    psi = pa + 180.0 if psi0 is None else np.array(psi0, dtype=np.float64)
    h = 1e-4  # deg, finite-difference step

    def miss(p: F64) -> F64:
        r0, d0 = offset_by_deg(ra, dec, p, sep)
        return wrap180(position_angle_deg(r0, d0, ra, dec) - pa)

    for _ in range(n_iter):
        g = miss(psi)
        with np.errstate(invalid="ignore", divide="ignore"):
            step = g * h / (miss(psi + h) - g)
        step = np.where(np.isfinite(step), np.clip(step, -90.0, 90.0), 0.0)
        psi = psi - step
        if np.nanmax(np.abs(step), initial=0.0) < 1e-12:
            break
    ra0, dec0 = offset_by_deg(ra, dec, psi, sep)
    at_source = sep < 1e-12  # psi is meaningless; the centre is the source
    return np.where(at_source, ra, ra0), np.where(at_source, dec, dec0)


def _tile_mean_dir(ra0: F64, dec0: F64, idx: IntArr, n: int) -> tuple[F64, F64]:
    """Mean direction (deg) of the finite (ra0, dec0) in each group 0..n-1."""
    r, d = np.radians(ra0), np.radians(dec0)
    v = np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])
    ok = np.isfinite(v).all(axis=0)
    x, y, z = (np.bincount(idx[ok], weights=v[k][ok], minlength=n) for k in range(3))
    norm = np.sqrt(x * x + y * y + z * z)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.degrees(np.arcsin(np.clip(z / norm, -1.0, 1.0)))
    return np.degrees(np.arctan2(y, x)) % 360.0, dec


def tile_median_centre(ra0_r: F64, dec0_r: F64, idx: IntArr, n: int) -> tuple[F64, F64]:
    """Per-group median of per-row centres (deg), robust to RA wrap and NaNs.

    RA is taken relative to the first finite row of the group so that a group
    straddling RA = 0 does not average to 180.
    """
    fin = np.isfinite(ra0_r) & np.isfinite(dec0_r)
    ref = np.full(n, np.nan)
    present, first = np.unique(idx[fin], return_index=True)
    ref[present] = ra0_r[fin][first]
    rel = wrap180(ra0_r - ref[idx])
    ra0_t = (ref + group_reduce(rel, idx, n)) % 360.0
    return ra0_t, group_reduce(dec0_r, idx, n)


def fit_arc_frame(
    ra: F64,
    dec: F64,
    l_off: F64,
    m_off: F64,
    tile_idx: IntArr,
    n_tiles: int,
    max_rows: int = 300,
    n_iter: int = 15,
    max_stages: int = 8,
) -> tuple[F64, F64, F64]:
    """Frame rotation and centre of the ``arc`` convention, per tile (deg).

    hypot(l, m) is the separation from the centre whatever the rotation, so
    the centre is found first, by least-squares trilateration on the sphere
    (Gauss-Newton from the centroid of the tile's sources, on up to
    ``max_rows`` rows per tile). Rows whose residual is far outside the
    robust scatter are dropped and the fit repeated until the kept set is
    stable, so a minority of rows with wrong offsets cannot bias it. The
    rotation is then the median over kept rows of
    PA(centre -> source) - atan2(l, m).

    Returns (rotation, centre ra, centre dec); NaN for tiles without rows.
    """
    order = np.argsort(tile_idx, kind="stable")
    starts = np.searchsorted(tile_idx[order], np.arange(n_tiles + 1))
    rank = np.arange(len(order)) - starts[tile_idx[order]]
    keep = order[rank < max_rows]
    r, d, l, m, idx = ra[keep], dec[keep], l_off[keep], m_off[keep], tile_idx[keep]
    sep = np.hypot(l, m)
    use = np.isfinite(sep)
    c_ra, c_dec = _tile_mean_dir(r, d, idx, n_tiles)

    def residual() -> tuple[F64, F64]:
        """Distance residual (deg) and position angle (rad) of each row."""
        f = sep_deg(c_ra[idx], c_dec[idx], r, d) - sep
        return f, np.radians(position_angle_deg(c_ra[idx], c_dec[idx], r, d))

    for _stage in range(max_stages):
        for _ in range(n_iter):
            f, th = residual()
            w = use.astype(np.float64)
            sx, cx = np.sin(th), np.cos(th)  # unit vector centre -> source
            sxx = np.bincount(idx, weights=w * sx * sx, minlength=n_tiles)
            sxy = np.bincount(idx, weights=w * sx * cx, minlength=n_tiles)
            syy = np.bincount(idx, weights=w * cx * cx, minlength=n_tiles)
            bx = np.bincount(idx, weights=w * sx * f, minlength=n_tiles)
            by = np.bincount(idx, weights=w * cx * f, minlength=n_tiles)
            det = sxx * syy - sxy * sxy
            with np.errstate(invalid="ignore", divide="ignore"):
                dx = (syy * bx - sxy * by) / det  # east component of the move
                dy = (sxx * by - sxy * bx) / det  # north component
            step = np.hypot(dx, dy)
            ok = np.isfinite(step) & (step > 0)
            n_ra, n_dec = offset_by_deg(
                c_ra, c_dec, np.degrees(np.arctan2(dx, dy)), step
            )
            c_ra, c_dec = np.where(ok, n_ra, c_ra), np.where(ok, n_dec, c_dec)
            if not ok.any() or step[ok].max() < 1e-13:
                break
        f, _ = residual()  # drop rows that do not fit, refit without them
        scale = 1.4826 * group_reduce(np.abs(np.where(use, f, np.nan)), idx, n_tiles)
        still = use & (np.abs(f) <= 5.0 * scale[idx] + 1e-7)
        if (still == use).all():
            break
        use = still
    rot_row = wrap180(np.degrees(residual()[1]) - np.degrees(np.arctan2(l, m)))
    rot_row = np.where(use, rot_row, np.nan)
    ref = np.full(n_tiles, np.nan)
    present, first = np.unique(idx[use], return_index=True)
    ref[present] = rot_row[use][first]
    rot = (ref + group_reduce(wrap180(rot_row - ref[idx]), idx, n_tiles)) % 360.0
    empty = np.bincount(idx[use], minlength=n_tiles) == 0
    return rot, np.where(empty, np.nan, c_ra), np.where(empty, np.nan, c_dec)


def _skyoffset_roots(
    ra: F64, dec: F64, l_off: F64, m_off: F64
) -> list[tuple[F64, F64]]:
    """Both tile centres consistent with a source's sky-offset (l, m).

    With w the source's unit vector in the offset frame, the forward map gives
    w_y = cos(dec) sin(ra - ra0) and sin(dec) = w_x sin(dec0) + w_z cos(dec0).
    The second equation has two roots for dec0, and both reproduce (l, m)
    exactly. Roots outside [-90, 90] are returned as NaN. Away from the poles
    only one root is valid; near the poles (|dec0| above ~85 deg) both can be.
    """
    lr, mr = np.radians(l_off), np.radians(m_off)
    wx, wy, wz = np.cos(mr) * np.cos(lr), np.cos(mr) * np.sin(lr), np.sin(mr)
    phi = np.degrees(np.arctan2(wx, wz))
    amp = np.hypot(wx, wz)
    theta = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec)) / amp, -1.0, 1.0)))
    roots: list[tuple[F64, F64]] = []
    for dec0 in (wrap180(phi + theta), wrap180(phi - theta)):
        ok = np.abs(dec0) <= 90.0 + 1e-9
        d0 = np.radians(dec0)
        xp = wx * np.cos(d0) - wz * np.sin(d0)
        ra0 = (ra - np.degrees(np.arctan2(wy, xp))) % 360.0
        roots.append((np.where(ok, ra0, np.nan), np.where(ok, dec0, np.nan)))
    return roots


def _resolve_roots(
    roots: Sequence[tuple[F64, F64]],
    tile_idx: IntArr,
    n_tiles: int,
    tol_deg: float = 0.01,
) -> tuple[F64, F64]:
    """Choose one root per row using the fact that a tile has a single centre.

    The two roots of each tile's first row with a finite root are the
    hypotheses. The one that most rows of the tile can reproduce (with either
    of their roots) wins, and every row then takes the root nearest to it.
    """
    fin = np.isfinite(roots[0][1]) | np.isfinite(roots[1][1])
    present, first = np.unique(tile_idx[fin], return_index=True)
    hyps: list[tuple[F64, F64]] = []
    scores: list[F64] = []
    for rr, dd in roots:
        hra, hdec = np.full(n_tiles, np.nan), np.full(n_tiles, np.nan)
        hra[present], hdec[present] = rr[fin][first], dd[fin][first]
        near = np.fmin(
            *(sep_deg(r, d, hra[tile_idx], hdec[tile_idx]) for r, d in roots)
        )
        hyps.append((hra, hdec))
        scores.append(
            np.bincount(tile_idx, weights=(near < tol_deg), minlength=n_tiles)
        )
    use_b = scores[1] > scores[0]
    hra = np.where(use_b, hyps[1][0], hyps[0][0])[tile_idx]
    hdec = np.where(use_b, hyps[1][1], hyps[0][1])[tile_idx]
    sa = np.nan_to_num(sep_deg(*roots[0], hra, hdec), nan=np.inf)
    sb = np.nan_to_num(sep_deg(*roots[1], hra, hdec), nan=np.inf)
    pick_b = sb < sa
    return (
        np.where(pick_b, roots[1][0], roots[0][0]),
        np.where(pick_b, roots[1][1], roots[0][1]),
    )


def recover_centre(
    convention: str,
    ra: F64,
    dec: F64,
    l_off: F64,
    m_off: F64,
    tile_idx: IntArr,
    n_tiles: int,
) -> tuple[F64, F64, F64 | None]:
    """Invert an offset convention to get the per-row tile centre (deg).

    Conventions:
      radec      l = ra - ra0,                m = dec - dec0
      flat       l = (ra - ra0) cos(dec),     m = dec - dec0
      skyoffset  l, m = astropy spherical_offsets_to
      arc        l = sep sin(PA - rot), m = sep cos(PA - rot)  (arrakis)

    ``tile_idx`` (values below ``n_tiles``) says which rows share a centre.
    ``skyoffset`` uses it to pick between the two polar roots; ``arc`` uses
    it to fit the frame rotation of each tile, which is returned as the third
    element (per tile, degrees; None for the other conventions).
    """
    if convention == "radec":
        return (ra - l_off) % 360.0, dec - m_off, None
    if convention == "flat":
        return (ra - l_off / np.cos(np.radians(dec))) % 360.0, dec - m_off, None
    if convention == "skyoffset":
        roots = _skyoffset_roots(ra, dec, l_off, m_off)
        return (*_resolve_roots(roots, tile_idx, n_tiles), None)
    if convention == "arc":
        rot, c_ra, c_dec = fit_arc_frame(ra, dec, l_off, m_off, tile_idx, n_tiles)
        # seed each row's inversion with the direction to its tile's centre,
        # which picks the right solution for rows within reach of a pole
        psi0 = position_angle_deg(ra, dec, c_ra[tile_idx], c_dec[tile_idx])
        return (*_arc_centre(ra, dec, l_off, m_off, rot[tile_idx], psi0), rot)
    raise ValueError(f"unknown convention {convention!r}")


CONVENTIONS: Final[tuple[str, ...]] = ("arc", "skyoffset", "flat", "radec")


def group_reduce(values: F64, idx: IntArr, n: int, fn: str = "median") -> F64:
    """Median or max of the finite ``values`` in each group 0..n-1 (NaN if none)."""
    order = np.argsort(idx, kind="stable")
    sv = values[order]
    bounds = np.searchsorted(idx[order], np.arange(n + 1))
    f = np.nanmedian if fn == "median" else np.nanmax
    out = np.full(n, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN groups
        for i in range(n):
            if bounds[i + 1] > bounds[i]:
                out[i] = f(sv[bounds[i] : bounds[i + 1]])
    return out


@dataclass(frozen=True)
class CentreFit:
    convention: str
    ra0: F64  # per tile
    dec0: F64  # per tile
    dev_arcsec: F64  # per valid row: its own centre vs its tile's
    frac_bad: float  # fraction of valid rows deviating by more than tol, or NaN
    rot: F64 | None = None  # per tile frame rotation (arc convention only)


def fit_centres(
    convention: str,
    ra: F64,
    dec: F64,
    l_off: F64,
    m_off: F64,
    tile_idx: IntArr,
    n_tiles: int,
    valid: BoolArr,
    tol_arcsec: float,
) -> CentreFit:
    """Recover per-tile centres under one convention and measure consistency.

    A row whose recovered centre is not finite counts as inconsistent, so a
    convention cannot score well by failing to produce centres.
    """
    idx = tile_idx[valid]
    ra0_r, dec0_r, rot = recover_centre(
        convention, ra[valid], dec[valid], l_off[valid], m_off[valid], idx, n_tiles
    )
    ra0_t, dec0_t = tile_median_centre(ra0_r, dec0_r, idx, n_tiles)
    dev = 3600.0 * np.hypot(
        wrap180(ra0_r - ra0_t[idx]) * np.cos(np.radians(dec0_t[idx])),
        dec0_r - dec0_t[idx],
    )
    frac_bad = float(np.mean(~(dev <= tol_arcsec))) if len(dev) else float("nan")
    return CentreFit(convention, ra0_t, dec0_t, dev, frac_bad, rot)


def sep_deg(ra: F64, dec: F64, ra0: F64, dec0: F64) -> F64:
    """Great-circle separation in degrees (haversine)."""
    a, d, a0, d0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    h = np.sin((d - d0) / 2) ** 2 + np.cos(d) * np.cos(d0) * np.sin((a - a0) / 2) ** 2
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


_TILE_NAME = re.compile(r"RACS_(\d{2})(\d{2})([+-])(\d{2})")


def parse_tile_name(name: str) -> tuple[float, float] | None:
    """Coarse (ra, dec) in degrees from a name like RACS_0426-84, or None."""
    mt = _TILE_NAME.search(name)
    if mt is None:
        return None
    hh, mm, sign, dd = mt.groups()
    return (int(hh) + int(mm) / 60.0) * 15.0, (-1.0 if sign == "-" else 1.0) * int(dd)


def tile_index(raw_tile: NDArray[np.bytes_]) -> tuple[NDArray[np.str_], IntArr]:
    """Unique stripped tile names and the per-row index into them."""
    uniq, inv = np.unique(np.ascontiguousarray(raw_tile), return_inverse=True)
    stripped = [b.decode("ascii", "replace").strip() for b in uniq.tolist()]
    names, remap = np.unique(np.array(stripped, dtype=str), return_inverse=True)
    return names, remap[inv.reshape(-1)]


def choose_centres(
    ra: F64,
    dec: F64,
    l_off: F64,
    m_off: F64,
    sep: F64 | None,
    names: NDArray[np.str_],
    idx: IntArr,
    tol_arcsec: float,
    max_bad_frac: float = 1e-3,
) -> tuple[CentreFit, BoolArr]:
    """Fit all conventions, report, and return the best (exits if none fits)."""
    valid = (
        np.isfinite(ra)
        & np.isfinite(dec)
        & np.isfinite(l_off)
        & np.isfinite(m_off)
        & (names[idx] != "")
    )
    fits_ = [
        fit_centres(c, ra, dec, l_off, m_off, idx, len(names), valid, tol_arcsec)
        for c in CONVENTIONS
    ]
    for f in fits_:
        print(
            f"  convention {f.convention:10s}: {100 * f.frac_bad:7.3f}% rows > {tol_arcsec:g} arcsec from tile centre"
        )
    best = min(fits_, key=lambda f: np.nan_to_num(f.frac_bad, nan=np.inf))
    if not best.frac_bad <= max_bad_frac:
        sys.exit(
            "no offset convention gives consistent tile centres (see --tol-arcsec "
            "and --max-bad-frac); supply --boresights"
        )
    print(f"using convention: {best.convention}")
    n_rows = np.bincount(idx[valid], minlength=len(names))
    has_rows = n_rows > 0
    no_centre = has_rows & ~(np.isfinite(best.ra0) & np.isfinite(best.dec0))
    if no_centre.any():
        sys.exit(
            f"no finite centre for {int(no_centre.sum())} tiles with rows, "
            f"e.g. {list(names[no_centre][:5])}; supply --boresights"
        )
    if best.rot is not None:
        rot = wrap180(best.rot[has_rows])
        print(
            f"  arc frame rotation per tile: median {np.median(rot):.3f} deg, "
            f"range [{rot.min():.3f}, {rot.max():.3f}] (expect 45 deg + POL_AXIS)"
        )
        few = has_rows & (n_rows < 5)
        if few.any():
            print(
                f"  warning: rotation fitted from fewer than 5 rows for {int(few.sum())} tiles, "
                f"e.g. {list(names[few][:5])}; check the audit CSV or supply --boresights"
            )
    if sep is not None:
        v = valid & np.isfinite(sep)
        mine = sep_deg(ra[v], dec[v], best.ra0[idx[v]], best.dec0[idx[v]])
        diff = np.abs(mine - sep[v])
        print(
            f"  check vs separation column: median |diff| {np.median(diff):.2e} deg, max {diff.max():.2e} deg"
        )
    return best, valid


def write_audit(
    path: Path,
    names: NDArray[np.str_],
    ra0: F64,
    dec0: F64,
    idx_valid: IntArr,
    dev: F64,
    rot: F64 | None = None,
) -> None:
    """CSV of recovered centres, with tile-name positions for comparison."""
    n = len(names)
    count = np.bincount(idx_valid, minlength=n)
    max_dev = group_reduce(dev, idx_valid, n, "max")
    rot_t = np.full(n, np.nan) if rot is None else wrap180(rot)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "tile_id",
                "ra0",
                "dec0",
                "n_rows",
                "max_dev_arcsec",
                "arc_rot_deg",
                "name_ra",
                "name_dec",
            ]
        )
        for i, nm in enumerate(names):
            p = parse_tile_name(nm) or (float("nan"), float("nan"))
            w.writerow(
                [nm, ra0[i], dec0[i], int(count[i]), max_dev[i], rot_t[i], p[0], p[1]]
            )


def centres_from_file(path: str, names: NDArray[np.str_]) -> tuple[F64, F64]:
    """Per-tile centres from a CSV with columns tile_id, ra0, dec0 (deg)."""
    with open(path, newline="") as fh:
        look = {
            r["FIELD_NAME"].strip(): (float(r["RA_DEG"]), float(r["DEC_DEG"]))
            for r in csv.DictReader(fh)
        }
    missing = [n for n in names if n and n not in look]
    if missing:
        sys.exit(f"{path}: no boresight for {len(missing)} tiles, e.g. {missing[:5]}")
    nan = (float("nan"), float("nan"))
    return (
        np.array([look.get(n, nan)[0] for n in names]),
        np.array([look.get(n, nan)[1] for n in names]),
    )


# --------------------------------------------------------------------------
# New columns
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NewColumn:
    name: str
    source: str | None  # original column whose unit/description is reused
    description: str | None = None
    unit: str | None = None
    product_with: str | None = None  # unit becomes unit(source) * unit(product_with)


def plan_columns(cols: Columns, has_err: bool) -> list[NewColumn]:
    """The columns to add, in output order."""
    plan = [NewColumn(n + SUFFIX, n) for n in cols.angles]
    plan += [NewColumn(n + SUFFIX, n) for n in (cols.q, cols.u, cols.q_bkg, cols.u_bkg)]
    if has_err:
        plan += [
            NewColumn(cols.q_err + SUFFIX, cols.q_err),
            NewColumn(cols.u_err + SUFFIX, cols.u_err),
            NewColumn(
                COV_COL,
                cols.u_err,
                "Q-U covariance of the error, local-north frame",
                product_with=cols.q_err,
            ),
        ]
    plan.append(
        NewColumn(
            GAMMA_COL,
            None,
            "Rotation applied to reach local north (chi_true = chi + gamma)",
            "deg",
        )
    )
    return plan


def compute_new(
    cols: Columns,
    chunk: Mapping[str, F64],
    gamma: F64,
    lo: Mapping[str, float],
    has_err: bool,
) -> list[F64]:
    """Values of the new columns for one chunk, in ``plan_columns`` order."""
    out = [wrap_angle(chunk[n] + gamma, lo[n]) for n in cols.angles]
    for qn, un in ((cols.q, cols.u), (cols.q_bkg, cols.u_bkg)):
        out += list(rotate_qu(chunk[qn], chunk[un], gamma))
    if has_err:
        out += list(rotate_qu_cov(chunk[cols.q_err], chunk[cols.u_err], gamma))
    out.append(gamma)
    return out


def squared_unit(a: str | None, b: str | None) -> str | None:
    """FITS-style unit string for a product of two units, or None."""
    if not a or not b:
        return None
    try:
        unit = au.Unit(a) * au.Unit(b)
    except ValueError:
        return None
    try:
        return unit.to_string("fits")
    except ValueError:  # e.g. "beam" has no FITS spelling
        return unit.to_string()


SER_BEGIN: Final[str] = "--BEGIN-ASTROPY-SERIALIZED-COLUMNS--"
SER_END: Final[str] = "--END-ASTROPY-SERIALIZED-COLUMNS--"


def serialized_lines(
    entries: Sequence[tuple[str, str | None, str | None]],
) -> list[str]:
    """Lines astropy writes in its serialised-column block for these columns.

    ``entries`` are (name, unit, description). Astropy reads units and
    descriptions from this block, so the new columns need entries in it to show
    their descriptions when the file is read back into a Table.
    """
    t = Table()
    for name, unit, desc in entries:
        t[name] = Column(np.zeros(1), unit=unit or None, description=desc)
    buf = io.BytesIO()
    t.write(buf, format="fits")  # table_to_hdu alone omits the block
    buf.seek(0)
    with fits.open(buf) as hdul:
        lines = [str(c.value) for c in hdul[1].header.cards if c.keyword == "COMMENT"]
    if "datatype:" not in lines:
        return []
    start = lines.index("datatype:") + 1
    return lines[start : datatype_list_end(lines, start)]


def datatype_list_end(values: Sequence[str], start: int) -> int:
    """Index just past the last COMMENT line of the ``datatype:`` list.

    ``values`` are the COMMENT values of the serialised block and ``start``
    the line after ``datatype:``. A line belongs to the list if it starts a
    new item (``- ``), is a YAML wrap of the previous item (indented), or is
    a continuation chunk (astropy splits long lines into 70-character pieces
    ending in a backslash). What follows the list depends on the astropy
    version: ``meta: !!omap`` (7.x and later), ``schema: astropy-2.0``
    (earlier versions, when the table has no meta) or the end sentinel.
    """
    i, cont = start, False
    while i < len(values):
        v = values[i]
        if not (cont or v.startswith("- ") or v[:1].isspace()):
            break
        cont = len(v) == 71 and v.endswith("\\")
        i += 1
    return i


def splice_serialized(hdr: fits.Header, lines: Sequence[str]) -> bool:
    """Insert ``lines`` at the end of the block's ``datatype:`` list, in place.

    Returns False (header untouched) if the header has no serialised block.
    """
    com = [(i, str(c.value)) for i, c in enumerate(hdr.cards) if c.keyword == "COMMENT"]
    values = [v for _, v in com]
    if "datatype:" not in values:
        return False
    d = values.index("datatype:") + 1
    e = datatype_list_end(values, d)
    if e >= len(com):
        return False  # no card after the list to insert before
    end = com[e][0]
    for line in reversed(lines):
        hdr.insert(end, ("COMMENT", line))
    return True


def extend_header(
    hdr: fits.Header, plan: Sequence[NewColumn], source_label: str
) -> fits.Header:
    """Copy of ``hdr`` with the new columns added; every other card is untouched."""
    new = hdr.copy()
    n0 = int(hdr["TFIELDS"])
    names = [str(hdr[f"TTYPE{i}"]).strip() for i in range(1, n0 + 1)]
    cards: list[tuple[str, object, str]] = []
    resolved: list[tuple[str, str | None, str | None]] = []
    for k, nc in enumerate(plan, start=n0 + 1):
        j = names.index(nc.source) + 1 if nc.source else None
        unit = nc.unit
        if unit is None and j is not None:
            unit = hdr.get(f"TUNIT{j}")
        if nc.product_with:
            unit = squared_unit(
                hdr.get(f"TUNIT{j}"),
                hdr.get(f"TUNIT{names.index(nc.product_with) + 1}"),
            )
        desc = nc.description
        if desc is None and j is not None:
            base = hdr.get(f"TCOMM{j}") or names[j - 1]
            desc = f"{base} (local-north frame)"
        cards.append((f"TTYPE{k}", nc.name, ""))
        cards.append((f"TFORM{k}", "D", ""))
        if unit:
            cards.append((f"TUNIT{k}", str(unit).strip(), ""))
        if desc:
            cards.append((f"TCOMM{k}", desc, ""))
        resolved.append((nc.name, str(unit).strip() if unit else None, desc))
    last = max(
        i
        for i, c in enumerate(new.cards)
        if c.keyword.startswith(("TTYPE", "TFORM", "TUNIT", "TCOMM"))
    )
    for card in reversed(cards):
        new.insert(last + 1, card)
    splice_serialized(new, serialized_lines(resolved))
    new["NAXIS1"] = int(hdr["NAXIS1"]) + 8 * len(plan)
    new["TFIELDS"] = n0 + len(plan)
    strip_checksums(new)
    new["POLFRAME"] = ("LOCAL_NORTH_COLS_ADDED", "_corr columns added by fix_polframe")
    new["POLFSRC"] = (source_label, "boresight source")
    new["POLFSIGN"] = (FRAME_SIGN, "chi_true = chi + sign*gamma")
    return new


def strip_checksums(hdr: fits.Header) -> None:
    """Drop CHECKSUM/DATASUM in place; they would not match the output."""
    for key in ("CHECKSUM", "DATASUM"):
        hdr.remove(key, ignore_missing=True, remove_all=True)


def angle_lower_edge(chi: F64, override: float | None, name: str) -> float:
    """Lower edge of the angle range; checks the data fit a 180 degree span."""
    lo_obs, hi_obs = float(np.nanmin(chi)), float(np.nanmax(chi))
    lo = override if override is not None else (-90.0 if lo_obs < 0 else 0.0)
    print(
        f"  {name}: observed [{lo_obs:.3f}, {hi_obs:.3f}] deg -> wrap to [{lo:g}, {lo + 180:g})"
    )
    if lo_obs < lo - 1e-6 or hi_obs > lo + 180.0 + 1e-6:
        sys.exit(
            f"{name} does not fit [{lo:g}, {lo + 180:g}]; set --angle-min or check units"
        )
    return lo


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("catalogue", help="uncompressed FITS (gunzip first)")
    p.add_argument("output")
    p.add_argument("--tile-col", default=Columns.tile)
    p.add_argument(
        "--boresights",
        help="CSV with tile_id, ra0, dec0 (deg); overrides "
        "centres recovered from the l/m offsets",
    )
    p.add_argument(
        "--centres-out", help="audit CSV path (default: <output>.tile_centres.csv)"
    )
    p.add_argument(
        "--angle-min",
        type=float,
        default=None,
        help="lower edge of the angle range, 0 or -90 (default: inferred and printed)",
    )
    p.add_argument(
        "--tol-arcsec",
        type=float,
        default=1.0,
        help="max within-tile scatter of the recovered centre",
    )
    p.add_argument(
        "--max-bad-frac",
        type=float,
        default=1e-3,
        help="max fraction of rows allowed to exceed --tol-arcsec",
    )
    p.add_argument("--chunk-rows", type=int, default=100_000)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cols = Columns(tile=args.tile_col)
    if args.catalogue.endswith(".gz"):
        sys.exit("input is gzipped; gunzip it first so it can be memory-mapped")
    if Path(args.output).exists() and not args.overwrite:
        sys.exit(f"{args.output} exists; use --overwrite")

    with fits.open(args.catalogue, memmap=True) as hdul:
        if len(hdul) != 2 or not isinstance(hdul[1], fits.BinTableHDU):
            sys.exit("expected a primary HDU and exactly one binary table")
        hdr0, hdr1 = hdul[0].header, hdul[1].header
        if int(hdr0.get("NAXIS", 0)) != 0:
            sys.exit("primary HDU carries data; only a bare primary is supported")
        if int(hdr1.get("PCOUNT", 0)) != 0:
            sys.exit("variable-length columns (PCOUNT != 0) are not supported")
        n, naxis1 = int(hdr1["NAXIS2"]), int(hdr1["NAXIS1"])
        names_all = [
            str(hdr1[f"TTYPE{i}"]).strip() for i in range(1, int(hdr1["TFIELDS"]) + 1)
        ]
        absent = [
            c for c in cols.required(args.boresights is None) if c not in names_all
        ]
        if absent:
            sys.exit(f"missing columns: {absent}")
        clash = [
            c for c in names_all if c.endswith(SUFFIX) or c in (GAMMA_COL, COV_COL)
        ]
        if clash:
            sys.exit(f"already has output columns {clash[:4]}; refusing to run twice")

        dloc = hdul.fileinfo(1)["datLoc"]
        rec_dtype = hdul[1].data.dtype
        if rec_dtype.itemsize != naxis1:
            sys.exit(f"record size {rec_dtype.itemsize} != NAXIS1 {naxis1}")
        raw = np.memmap(
            args.catalogue, dtype=rec_dtype, mode="r", offset=dloc, shape=(n,)
        )
        rows = np.memmap(
            args.catalogue, dtype=np.uint8, mode="r", offset=dloc, shape=(n, naxis1)
        )

        def col(name: str) -> F64:
            return raw[name].astype(np.float64)

        # ---- pass 1: centres, angle ranges, error-column check ------------
        print(f"pass 1: {n} rows")
        tnames, tidx = tile_index(raw[cols.tile])
        if "field_name" in names_all and cols.tile != "field_name":
            fnames, fidx = tile_index(raw["field_name"])
            n_diff = int((tnames[tidx] != fnames[fidx]).sum())
            if n_diff:
                print(
                    f"warning: field_name differs from {cols.tile} for {n_diff} rows; "
                    "arrakis computes the l/m offsets per field_name, so consider "
                    "--tile-col field_name or --boresights"
                )
            del fnames, fidx
        ra, dec = col(cols.ra), col(cols.dec)
        if args.boresights:
            ra0_t, dec0_t = centres_from_file(args.boresights, tnames)
            source = f"file:{Path(args.boresights).name}"
        else:
            sep = col(cols.sep) if cols.sep in names_all else None
            best, valid = choose_centres(
                ra,
                dec,
                col(cols.l_off),
                col(cols.m_off),
                sep,
                tnames,
                tidx,
                args.tol_arcsec,
                args.max_bad_frac,
            )
            ra0_t, dec0_t = best.ra0, best.dec0
            audit = Path(args.centres_out or f"{args.output}.tile_centres.csv")
            write_audit(
                audit, tnames, ra0_t, dec0_t, tidx[valid], best.dev_arcsec, best.rot
            )
            print(f"wrote tile-centre audit {audit}")
            source = f"offsets:{best.convention}"
        lo = {a: angle_lower_edge(col(a), args.angle_min, a) for a in cols.angles}
        has_err = bool(
            np.isfinite(col(cols.q_err)).any() and np.isfinite(col(cols.u_err)).any()
        )
        if not has_err:
            print(
                f"note: {cols.q_err}/{cols.u_err} have no finite values; error columns skipped"
            )
        del ra, dec

        plan = plan_columns(cols, has_err)
        new_hdr = extend_header(hdr1, plan, source)
        extra = 8 * len(plan)

        # ---- pass 2: stream rows -------------------------------------------
        print(f"pass 2: writing {args.output} (+{len(plan)} columns)")
        read_cols = [
            cols.ra,
            cols.dec,
            *cols.angles,
            cols.q,
            cols.u,
            cols.q_bkg,
            cols.u_bkg,
        ]
        if has_err:
            read_cols += [cols.q_err, cols.u_err]
        written = 0
        n_nan_gamma = 0
        out_hdr0 = hdr0.copy()
        strip_checksums(out_hdr0)
        with open(args.output, "wb") as out:
            out.write(out_hdr0.tostring(padding=True).encode("ascii"))
            out.write(new_hdr.tostring(padding=True).encode("ascii"))
            for s in range(0, n, args.chunk_rows):
                e = min(s + args.chunk_rows, n)
                chunk = {c: raw[c][s:e].astype(np.float64) for c in read_cols}
                ti = tidx[s:e]
                gamma = gamma_deg(
                    chunk[cols.ra], chunk[cols.dec], ra0_t[ti], dec0_t[ti]
                )
                n_nan_gamma += int((~np.isfinite(gamma)).sum())
                new = np.stack(compute_new(cols, chunk, gamma, lo, has_err), axis=1)
                buf = np.empty((e - s, naxis1 + extra), dtype=np.uint8)
                buf[:, :naxis1] = rows[s:e]
                buf[:, naxis1:] = (
                    np.ascontiguousarray(new.astype(">f8"))
                    .view(np.uint8)
                    .reshape(e - s, extra)
                )
                out.write(buf.tobytes())
                written += buf.size
                print(f"  {e}/{n}", end="\r")
            out.write(b"\0" * (-written % BLOCK))
    print(f"\nwrote {args.output}: {n} rows, added {[c.name for c in plan]}")
    if n_nan_gamma:
        print(
            f"note: {n_nan_gamma} rows have no finite gamma (non-finite ra/dec or "
            "an empty tile name); their _corr columns are NaN"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
