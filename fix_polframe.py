#!/usr/bin/env python3
"""Add local-north polarisation columns to the SPICE-RACS DR2 catalogue.

ASKAP catalogue angles are measured from north at the observing boresight, not
from north at each source. This appends corrected copies of the affected
columns (suffix ``_corr``) and leaves every original column untouched:

    polangle, derot_polangle        -> + gamma, wrapped to the original range
    stokesQ/U, stokesQ/U_bkg        -> (Q + iU) * exp(2i gamma)
    stokesQ_err, stokesU_err        -> rotated errors (Q and U independent)

plus a diagnostic column ``pol_frame_gamma`` (degrees). Frame-independent
columns (rm, polint, fracpol, stokesI/V, polangle_err, ...) are not touched.

gamma is the angle, east of local north, of the boresight-north direction
projected onto the sky at the source:

    gamma = atan2( sin(dec0) sin(ra - ra0),
                   cos(dec) cos(dec0) + sin(dec) sin(dec0) cos(ra - ra0) )

Boresights (ra0, dec0) come from a CSV with FIELD_NAME, RA_DEG, DEC_DEG, looked
up by each row's ``tile_id`` (default: field_data.csv next to this script).
If the catalogue has a ``separation_tile_centre`` column (arrakis computes it
from the same field database), every row's separation from its boresight is
checked against it. This catches a CSV from a different RACS epoch, and rows
whose ``tile_id`` does not name the field they were measured in.

Sign convention (FRAME_SIGN): chi_true = chi_measured + gamma. This follows the
report but is not verified. Check it against a source of known absolute angle
(for example 3C 286 placed off-axis) before release.

The input must be an uncompressed FITS file with a bare primary HDU and one
binary table. Only the output header is built with astropy; the original rows
are copied byte for byte with the new columns appended, chunk by chunk, so
memory use does not grow with the width of the table. Each original column's
null marker (TNULL) is kept from the input; if astropy would store any
original column differently in another way, the script stops rather than
alter it. Checksums are dropped from the output because the data change.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.io import fits
from astropy.table import Column, Table

FRAME_SIGN = 1.0  # chi_true = chi_measured + FRAME_SIGN * gamma
SUFFIX = "_corr"
GAMMA_COL = "pol_frame_gamma"
SEP_COL = "separation_tile_centre"

ANGLES = ("polangle", "derot_polangle")
QU_PAIRS = (("stokesQ", "stokesU"), ("stokesQ_bkg", "stokesU_bkg"))
ERRS = ("stokesQ_err", "stokesU_err")

BLOCK = 2880  # FITS block size
# header keywords that fix how a column is laid out in the row bytes; TNULL
# is handled separately (taken from the input, see match_layout)
LAYOUT_KEYS = ("TTYPE", "TFORM", "TDIM", "TSCAL", "TZERO")


# --------------------------------------------------------------------------
# Maths
# --------------------------------------------------------------------------
def gamma_deg(ra, dec, ra0, dec0):
    """Frame rotation gamma (deg) from the boresight-north feed vector."""
    ra, dec, ra0, dec0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    num = np.sin(dec0) * np.sin(ra - ra0)
    den = np.cos(dec) * np.cos(dec0) + np.sin(dec) * np.sin(dec0) * np.cos(ra - ra0)
    return FRAME_SIGN * np.degrees(np.arctan2(num, den))


def sep_deg(ra, dec, ra0, dec0):
    """Great-circle separation in degrees (haversine)."""
    ra, dec, ra0, dec0 = (np.radians(x) for x in (ra, dec, ra0, dec0))
    h = (
        np.sin((dec - dec0) / 2) ** 2
        + np.cos(dec) * np.cos(dec0) * np.sin((ra - ra0) / 2) ** 2
    )
    return np.degrees(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))))


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


def tile_names(col) -> tuple[np.ndarray, np.ndarray]:
    """Distinct stripped tile names, and each row's index into them.

    Names are decoded and stripped once per distinct value rather than once
    per row, which matters for catalogues with millions of rows.
    """
    uniq, inv = np.unique(np.asarray(col), return_inverse=True)
    names = [
        (x.decode("ascii", "replace") if isinstance(x, bytes) else str(x)).strip()
        for x in uniq.tolist()
    ]
    return np.array(names, dtype=str), inv.reshape(-1)


def boresights_for(names: np.ndarray, path: str) -> pd.DataFrame:
    """(RA_DEG, DEC_DEG) for each tile name; NaN where the name is unknown."""
    bore = (
        pd.read_csv(path, usecols=["FIELD_NAME", "RA_DEG", "DEC_DEG"])
        .assign(FIELD_NAME=lambda d: d.FIELD_NAME.str.strip())
        .drop_duplicates("FIELD_NAME")  # re-observations share one centre
        .set_index("FIELD_NAME")
    )
    return bore.reindex(names)


def check_separations(t, ra, dec, ra0, dec0, tol_arcsec: float) -> None:
    """Compare each row's separation from its boresight with SEP_COL.

    Exits if any row differs by more than ``tol_arcsec``. Does nothing (with
    a note) when the catalogue has no such column.
    """
    if SEP_COL not in t.colnames:
        print(f"note: no {SEP_COL} column; boresights not cross-checked")
        return
    sep_cat = f64(t, SEP_COL)
    unit = t[SEP_COL].unit
    if unit is not None:
        try:
            sep_cat = sep_cat * unit.to(u.deg)
        except u.UnitConversionError:
            sys.exit(f"{SEP_COL} has non-angular unit {unit}")
    diff = 3600.0 * np.abs(sep_deg(ra, dec, ra0, dec0) - sep_cat)
    ok = np.isfinite(diff)
    if not ok.any():
        print(f"note: no finite {SEP_COL} values; boresights not cross-checked")
        return
    worst = float(diff[ok].max())
    print(f"  boresight check vs {SEP_COL}: max |diff| {worst:.3g} arcsec")
    n_bad = int((diff[ok] > tol_arcsec).sum())
    if n_bad:
        sys.exit(
            f"{n_bad} rows are more than {tol_arcsec:g} arcsec from their {SEP_COL} "
            f"(worst {worst:.1f} arcsec). The boresight CSV probably comes from a "
            "different RACS epoch, or tile_id does not name the field each row "
            "was measured in. Use --sep-tol-arcsec inf to skip this check."
        )


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
# Output
# --------------------------------------------------------------------------
def table_header(t: Table) -> fits.Header:
    """The extension header astropy writes for ``t``, from a zero-row copy.

    Table.write (unlike table_to_hdu) puts units, descriptions and column
    meta in the header, so go via an in-memory file; with no rows this is
    cheap whatever the size of the table.
    """
    buf = io.BytesIO()
    t[:0].write(buf, format="fits")
    buf.seek(0)
    with fits.open(buf) as h:
        return h[1].header.copy()


def match_layout(
    old: fits.Header, new: fits.Header, n_orig: int, n_new: int
) -> str | None:
    """Make ``new`` describe the input row bytes; return a problem, or None.

    ``old`` is the input header, ``new`` the one astropy wrote for the table
    with ``n_new`` float64 columns appended; ``new`` is edited in place.
    The original rows are copied from the input, so each original column's
    null marker (TNULL) is taken from the input header: astropy before 6.0
    replaces it with its own default (999999) on write. Any other layout
    difference means astropy would store a column differently, and is
    returned as a description.
    """
    for i in range(1, n_orig + 1):
        for k in LAYOUT_KEYS:
            key = f"{k}{i}"
            if old.get(key) != new.get(key):
                return f"{key} is {old.get(key)!r} in the input, {new.get(key)!r} from astropy"
        key = f"TNULL{i}"
        if key not in old:
            new.remove(key, ignore_missing=True)
        elif key in new:
            new[key] = old[key]
        else:
            new.set(key, old[key], after=f"TFORM{i}")
    if new["NAXIS1"] != old["NAXIS1"] + 8 * n_new:
        return f"NAXIS1 is {new['NAXIS1']}, expected {old['NAXIS1'] + 8 * n_new}"
    for j in range(n_orig + 1, n_orig + n_new + 1):
        if new[f"TFORM{j}"] != "D":
            return f"TFORM{j} is {new[f'TFORM{j}']!r}, expected 'D'"
    return None


def write_streaming(src, out, hdr0, hdr, new: np.ndarray, chunk_rows: int) -> None:
    """Copy the rows of ``src`` byte for byte, appending ``new`` to each.

    ``new`` has one float64 column per new table column. Memory use is one
    chunk of rows, independent of the size of the table.
    """
    with fits.open(src, memmap=True) as hin:
        naxis1, n = hin[1].header["NAXIS1"], hin[1].header["NAXIS2"]
        offset = hin.fileinfo(1)["datLoc"]
    rows = np.memmap(src, dtype=np.uint8, mode="r", offset=offset, shape=(n, naxis1))
    new = np.ascontiguousarray(new, dtype=">f8")
    extra = new.shape[1] * 8
    hdr = hdr.copy()
    hdr["NAXIS2"] = n
    written = 0
    with open(out, "wb") as f:
        f.write(hdr0.tostring(padding=True).encode("ascii"))
        f.write(hdr.tostring(padding=True).encode("ascii"))
        for s in range(0, n, chunk_rows):
            e = min(s + chunk_rows, n)
            block = np.empty((e - s, naxis1 + extra), dtype=np.uint8)
            block[:, :naxis1] = rows[s:e]
            block[:, naxis1:] = new[s:e].view(np.uint8)
            f.write(block)
            written += block.size
        f.write(b"\0" * (-written % BLOCK))
    del rows


def same_file(a: str, b: str) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return Path(a).resolve() == Path(b).resolve()


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
    p.add_argument(
        "--sep-tol-arcsec",
        type=float,
        default=1.0,
        help=f"max allowed difference from {SEP_COL} (default: %(default)s; "
        "inf skips the check)",
    )
    p.add_argument("--chunk-rows", type=int, default=100_000)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.catalogue.endswith(".gz"):
        sys.exit("input is gzipped; gunzip it first")
    if Path(a.output).exists():
        if not a.overwrite:
            sys.exit(f"{a.output} exists; use --overwrite")
        if same_file(a.catalogue, a.output):
            sys.exit("output is the input file; write to a new file")

    with fits.open(a.catalogue, memmap=True) as hdul:
        if len(hdul) != 2 or not isinstance(hdul[1], fits.BinTableHDU):
            sys.exit("expected a primary HDU and exactly one binary table")
        hdr0, hdr_in = hdul[0].header.copy(), hdul[1].header.copy()
    if hdr0.get("NAXIS", 0) != 0:
        sys.exit("primary HDU carries data; only a bare primary is supported")
    if hdr_in.get("PCOUNT", 0) != 0:
        sys.exit("variable-length columns (PCOUNT != 0) are not supported")
    t = Table.read(a.catalogue, memmap=True)
    n_orig = len(t.colnames)

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
    if clash := [c for c in t.colnames if c.endswith(SUFFIX) or c == GAMMA_COL]:
        sys.exit(f"already has output columns {clash[:4]}; refusing to run twice")

    # gamma per row, from each row's tile boresight
    names, idx = tile_names(t[a.tile_col])
    bore = boresights_for(names, a.boresights)
    known = bore.RA_DEG.notna().to_numpy()
    unknown = names[~known & (names != "")]
    if unknown.size:
        sys.exit(
            f"{a.boresights}: no boresight for {unknown.size} tiles, e.g. {list(unknown[:5])}"
        )
    ra0, dec0 = bore.RA_DEG.to_numpy()[idx], bore.DEC_DEG.to_numpy()[idx]
    ra, dec = f64(t, "ra"), f64(t, "dec")
    check_separations(t, ra, dec, ra0, dec0, a.sep_tol_arcsec)
    gamma = gamma_deg(ra, dec, ra0, dec0)
    del ra, dec, ra0, dec0

    # new columns, in output order
    for name in ANGLES:
        chi = f64(t, name)
        lo = angle_lower_edge(chi, a.angle_min, name)
        add_column(t, name + SUFFIX, wrap_angle(chi + gamma, lo), src=name)
    for qn, un in QU_PAIRS:
        q2, u2 = rotate_qu(f64(t, qn), f64(t, un), gamma)
        add_column(t, qn + SUFFIX, q2, src=qn)
        add_column(t, un + SUFFIX, u2, src=un)
    qe, ue = ERRS
    sq, su = f64(t, qe), f64(t, ue)
    if np.isfinite(sq).any() and np.isfinite(su).any():
        sq2, su2, _ = rotate_qu_cov(sq, su, gamma)
        add_column(t, qe + SUFFIX, sq2, src=qe)
        add_column(t, ue + SUFFIX, su2, src=ue)
    else:
        print(f"note: {qe}/{ue} have no finite values; error columns skipped")
    del sq, su
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

    new_names = t.colnames[n_orig:]
    hdr = table_header(t)
    if problem := match_layout(hdr_in, hdr, n_orig, len(new_names)):
        sys.exit(
            f"astropy would store an original column differently ({problem}); "
            "stopping rather than alter the original data"
        )
    new = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in new_names])
    del t  # release the table's memory map before streaming
    write_streaming(a.catalogue, a.output, hdr0, hdr, new, a.chunk_rows)

    n_bad = int((~np.isfinite(gamma)).sum())
    print(f"wrote {a.output}: {len(gamma)} rows")
    if n_bad:
        print(f"note: {n_bad} rows have no finite gamma; their _corr columns are NaN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
