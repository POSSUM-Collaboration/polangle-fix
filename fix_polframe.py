#!/usr/bin/env python3
"""Add local-north polarisation columns to the SPICE-RACS DR2 catalogue.

ASKAP catalogue angles are measured from north at the observing boresight, not
from north at each source. This appends corrected copies of the affected
columns (suffix ``_corr``) and leaves every original column untouched:

    polangle, derot_polangle        -> + gamma, wrapped to the original range
    stokesQ/U, stokesQ/U_bkg        -> (Q + iU) * exp(2i gamma)
    stokesQ_err, stokesU_err        -> rotated covariance, plus stokesQU_cov_corr

plus a diagnostic column ``pol_frame_gamma`` (degrees). Frame-independent
columns (rm, polint, fracpol, stokesI/V, polangle_err, ...) are not touched.

gamma is the angle, east of local north, of the boresight-north direction
projected onto the sky at the source:

    gamma = atan2( sin(dec0) sin(ra - ra0),
                   cos(dec) cos(dec0) + sin(dec) sin(dec0) cos(ra - ra0) )

Boresights (ra0, dec0) come from a CSV with FIELD_NAME, RA_DEG, DEC_DEG, looked
up by each row's ``tile_id`` (default: field_data.csv next to this script).

Sign convention (FRAME_SIGN): chi_true = chi_measured + gamma. This follows the
report but is not verified. Check it against a source of known absolute angle
(for example 3C 286 placed off-axis) before release.

The input must be an uncompressed FITS file with a bare primary HDU and one
binary table. Checksums are dropped from the output because the data change.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Column, Table

FRAME_SIGN = 1.0  # chi_true = chi_measured + FRAME_SIGN * gamma
SUFFIX = "_corr"
GAMMA_COL = "pol_frame_gamma"
COV_COL = "stokesQU_cov_corr"

ANGLES = ("polangle", "derot_polangle")
QU_PAIRS = (("stokesQ", "stokesU"), ("stokesQ_bkg", "stokesU_bkg"))
ERRS = ("stokesQ_err", "stokesU_err")


# --------------------------------------------------------------------------
# Maths
# --------------------------------------------------------------------------
def gamma_deg(ra, dec, ra0, dec0):
    """Frame rotation gamma (deg) from the boresight-north feed vector."""
    ra, dec, ra0, dec0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    num = np.sin(dec0) * np.sin(ra - ra0)
    den = np.cos(dec) * np.cos(dec0) + np.sin(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return FRAME_SIGN * np.degrees(np.arctan2(num, den))


def wrap_angle(chi, lo):
    """Wrap polarisation angles (deg) into [lo, lo + 180)."""
    return (chi - lo) % 180.0 + lo


def rotate_qu(q, u, gamma):
    """(Q + iU) * exp(2 i gamma)."""
    z = (q + 1j * u) * np.exp(2j * np.radians(gamma))
    return z.real, z.imag


def rotate_qu_cov(sq, su, gamma, cov=None):
    """Propagate Q/U errors through the rotation, C' = R C R^T.

    Returns (sigma_Q', sigma_U', cov_QU'). Q and U are taken as independent
    unless ``cov`` is given.
    """
    th = np.radians(2.0 * gamma)
    c, s = np.cos(th), np.sin(th)
    cov = np.zeros_like(sq) if cov is None else cov
    vq, vu = sq**2, su**2
    return (
        np.sqrt(c * c * vq + s * s * vu - 2 * c * s * cov),
        np.sqrt(s * s * vq + c * c * vu + 2 * c * s * cov),
        c * s * (vq - vu) + (c * c - s * s) * cov,
    )


# --------------------------------------------------------------------------
# Table helpers
# --------------------------------------------------------------------------
def f64(t: Table, name: str) -> np.ndarray:
    """Column as a plain float64 array, with masked/null values as NaN."""
    return np.ma.filled(np.ma.asarray(t[name]).astype(float), np.nan)


def boresights_for(tiles: np.ndarray, path: str) -> pd.DataFrame:
    """(RA_DEG, DEC_DEG) for each tile name; NaN where the name is unknown."""
    bore = (
        pd.read_csv(path, usecols=["FIELD_NAME", "RA_DEG", "DEC_DEG"])
        .assign(FIELD_NAME=lambda d: d.FIELD_NAME.str.strip())
        .drop_duplicates("FIELD_NAME")  # re-observations share one centre
        .set_index("FIELD_NAME")
    )
    return bore.reindex(tiles)


def angle_lower_edge(chi: np.ndarray, override: float | None, name: str) -> float:
    """Lower edge of the angle range (0 or -90); checks the data fit in it."""
    lo_obs, hi_obs = np.nanmin(chi), np.nanmax(chi)
    lo = override if override is not None else (-90.0 if lo_obs < 0 else 0.0)
    print(
        f"  {name}: observed [{lo_obs:.3f}, {hi_obs:.3f}] -> wrap to [{lo:g}, {lo + 180:g})"
    )
    if lo_obs < lo - 1e-6 or hi_obs > lo + 180.0 + 1e-6:
        sys.exit(
            f"{name} does not fit [{lo:g}, {lo + 180:g}]; set --angle-min or check units"
        )
    return lo


def add_column(t, name, values, src=None, unit=None, desc=None):
    """Append a column, borrowing unit and description from ``src`` if given."""
    if src is not None:
        unit = unit if unit is not None else t[src].unit
        desc = desc or f"{t[src].description or src} (local-north frame)"
    t[name] = Column(values, unit=unit, description=desc)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("catalogue", help="uncompressed FITS (gunzip first)")
    p.add_argument("output")
    p.add_argument(
        "--boresights",
        default=str(Path(__file__).with_name("field_data.csv")),
        help="CSV with FIELD_NAME, RA_DEG, DEC_DEG (default: %(default)s)",
    )
    p.add_argument("--tile-col", default="tile_id")
    p.add_argument(
        "--angle-min", type=float, help="lower edge of angle range, 0 or -90"
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.catalogue.endswith(".gz"):
        sys.exit("input is gzipped; gunzip it first")
    if Path(a.output).exists() and not a.overwrite:
        sys.exit(f"{a.output} exists; use --overwrite")

    hdr0 = fits.getheader(a.catalogue, 0)
    if hdr0.get("NAXIS", 0) != 0:
        sys.exit("primary HDU carries data; only a bare primary is supported")
    t = Table.read(a.catalogue, memmap=True)

    needed = [
        "ra",
        "dec",
        a.tile_col,
        *ANGLES,
        *(c for pr in QU_PAIRS for c in pr),
        *ERRS,
    ]
    if absent := [c for c in needed if c not in t.colnames]:
        sys.exit(f"missing columns: {absent}")
    if clash := [
        c for c in t.colnames if c.endswith(SUFFIX) or c in (GAMMA_COL, COV_COL)
    ]:
        sys.exit(f"already has output columns {clash[:4]}; refusing to run twice")

    # gamma per row, from each row's tile boresight
    tiles = np.char.strip(np.asarray(t[a.tile_col]).astype(str))
    bore = boresights_for(tiles, a.boresights)
    unknown = np.unique(tiles[bore.RA_DEG.isna() & (tiles != "")])
    if unknown.size:
        sys.exit(
            f"{a.boresights}: no boresight for {unknown.size} tiles, e.g. {list(unknown[:5])}"
        )
    gamma = gamma_deg(
        f64(t, "ra"), f64(t, "dec"), bore.RA_DEG.to_numpy(), bore.DEC_DEG.to_numpy()
    )

    # new columns, in output order
    for name in ANGLES:
        lo = angle_lower_edge(f64(t, name), a.angle_min, name)
        add_column(t, name + SUFFIX, wrap_angle(f64(t, name) + gamma, lo), src=name)
    for qn, un in QU_PAIRS:
        q2, u2 = rotate_qu(f64(t, qn), f64(t, un), gamma)
        add_column(t, qn + SUFFIX, q2, src=qn)
        add_column(t, un + SUFFIX, u2, src=un)
    qe, ue = ERRS
    if np.isfinite(f64(t, qe)).any() and np.isfinite(f64(t, ue)).any():
        sq, su, cov = rotate_qu_cov(f64(t, qe), f64(t, ue), gamma)
        add_column(t, qe + SUFFIX, sq, src=qe)
        add_column(t, ue + SUFFIX, su, src=ue)
        units = [t[c].unit for c in ERRS]
        add_column(
            t,
            COV_COL,
            cov,
            unit=units[0] * units[1] if all(units) else None,
            desc="Q-U covariance of the error, local-north frame",
        )
    else:
        print(f"note: {qe}/{ue} have no finite values; error columns skipped")
    add_column(
        t,
        GAMMA_COL,
        gamma,
        unit="deg",
        desc="Rotation applied to reach local north (chi_true = chi + gamma)",
    )

    # provenance, and drop checksums that no longer match
    t.meta["POLFRAME"] = (
        "LOCAL_NORTH_COLS_ADDED",
        "_corr columns added by fix_polframe",
    )
    t.meta["POLFSRC"] = (f"file:{Path(a.boresights).name}", "boresight source")
    t.meta["POLFSIGN"] = (FRAME_SIGN, "chi_true = chi + sign*gamma")
    for k in ("CHECKSUM", "DATASUM"):
        hdr0.remove(k, ignore_missing=True, remove_all=True)
        t.meta.pop(k, None)

    # Table.write (unlike table_to_hdu) keeps units/descriptions in the header;
    # go via memory only to put the original primary header back
    buf = io.BytesIO()
    t.write(buf, format="fits")
    buf.seek(0)
    with fits.open(buf) as tab:
        fits.HDUList([fits.PrimaryHDU(header=hdr0), tab[1]]).writeto(
            a.output, overwrite=a.overwrite
        )
    n_bad = int((~np.isfinite(gamma)).sum())
    print(f"wrote {a.output}: {len(t)} rows")
    if n_bad:
        print(f"note: {n_bad} rows have no finite gamma; their _corr columns are NaN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
