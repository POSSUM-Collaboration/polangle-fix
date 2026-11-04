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
    added for the new columns. New columns carry TUNIT and TCOMM but no entry
    in the serialised block (no UCDs).
  * New columns are appended at the end of the table.

Boresight (assumptions to be aware of)
  * ``tile_id`` names the field whose cube supplied each row's Q/U.
  * The pointing centre of each tile is recovered from ``l_tile_centre`` and
    ``m_tile_centre`` (angular offsets of the source from the tile centre).
    Every candidate offset convention is tested and the one with consistent
    centres within each tile is used; the script stops if none is.
  * The recovered centre is the tile centre, which may differ from the true
    ASKAP boresight of the SBID. Use --boresights to supply true centres.

Sign convention (FRAME_SIGN): chi_true = chi_measured + gamma. This follows the
report but is not verified. Check it once against a source of known absolute
angle (for example 3C 286 placed off-axis) before release.

Memory: roughly 3 GB for pass 1 on the full DR2 table, then a few hundred MB.
Disk: the output is about the size of the input plus ~1 GB.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np
from astropy import units as au
from astropy.io import fits
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
        out = [self.ra, self.dec, self.tile, self.q, self.u, self.q_err, self.u_err,
               self.q_bkg, self.u_bkg, *self.angles]
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


def recover_centre(
    convention: str, ra: F64, dec: F64, l_off: F64, m_off: F64, n_iter: int = 30
) -> tuple[F64, F64]:
    """Invert an offset convention to get the per-row tile centre (deg).

    Conventions:
      radec      l = ra - ra0,                m = dec - dec0
      flat       l = (ra - ra0) cos(dec),     m = dec - dec0
      skyoffset  l, m = astropy spherical_offsets_to
    """
    if convention == "radec":
        return (ra - l_off) % 360.0, dec - m_off
    flat_ra0 = (ra - l_off / np.cos(np.radians(dec))) % 360.0
    flat_dec0 = dec - m_off
    if convention == "flat":
        return flat_ra0, flat_dec0
    if convention != "skyoffset":
        raise ValueError(f"unknown convention {convention!r}")
    ra0, dec0 = flat_ra0, flat_dec0
    for _ in range(n_iter):
        lp, mp = offsets_skyoffset(ra, dec, ra0, dec0)
        ra0 = ra0 + wrap180(lp - l_off) / np.cos(np.radians(dec0))
        dec0 = dec0 + (mp - m_off)
    return ra0 % 360.0, dec0


CONVENTIONS: Final[tuple[str, ...]] = ("skyoffset", "flat", "radec")


def group_reduce(values: F64, idx: IntArr, n: int, fn: str = "median") -> F64:
    """Median or max of ``values`` for each group 0..n-1 (NaN if empty)."""
    order = np.argsort(idx, kind="stable")
    sv = values[order]
    bounds = np.searchsorted(idx[order], np.arange(n + 1))
    f = np.median if fn == "median" else np.max
    out = np.full(n, np.nan)
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
    frac_bad: float  # fraction of valid rows deviating by more than tol


def fit_centres(
    convention: str, ra: F64, dec: F64, l_off: F64, m_off: F64,
    tile_idx: IntArr, n_tiles: int, valid: BoolArr, tol_arcsec: float,
) -> CentreFit:
    """Recover per-tile centres under one convention and measure consistency."""
    ra0_r, dec0_r = recover_centre(convention, ra[valid], dec[valid], l_off[valid], m_off[valid])
    idx = tile_idx[valid]
    ref = np.full(n_tiles, np.nan)
    present, first = np.unique(idx, return_index=True)
    ref[present] = ra0_r[first]
    rel = wrap180(ra0_r - ref[idx])
    ra0_t = (ref + group_reduce(rel, idx, n_tiles)) % 360.0
    dec0_t = group_reduce(dec0_r, idx, n_tiles)
    dev = 3600.0 * np.hypot(
        wrap180(ra0_r - ra0_t[idx]) * np.cos(np.radians(dec0_t[idx])), dec0_r - dec0_t[idx]
    )
    return CentreFit(convention, ra0_t, dec0_t, dev, float(np.mean(dev > tol_arcsec)))


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
    ra: F64, dec: F64, l_off: F64, m_off: F64, sep: F64 | None,
    names: NDArray[np.str_], idx: IntArr, tol_arcsec: float,
) -> tuple[CentreFit, BoolArr]:
    """Fit all conventions, report, and return the best (exits if none fits)."""
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(l_off) & np.isfinite(m_off) & (names[idx] != "")
    fits_ = [fit_centres(c, ra, dec, l_off, m_off, idx, len(names), valid, tol_arcsec)
             for c in CONVENTIONS]
    for f in fits_:
        print(f"  convention {f.convention:10s}: {100 * f.frac_bad:7.3f}% rows > {tol_arcsec:g} arcsec from tile centre")
    best = min(fits_, key=lambda f: f.frac_bad)
    if best.frac_bad > 1e-3:
        sys.exit("no offset convention gives consistent tile centres; supply --boresights")
    print(f"using convention: {best.convention}")
    if sep is not None:
        v = valid & np.isfinite(sep)
        mine = sep_deg(ra[v], dec[v], best.ra0[idx[v]], best.dec0[idx[v]])
        diff = np.abs(mine - sep[v])
        print(f"  check vs separation column: median |diff| {np.median(diff):.2e} deg, max {diff.max():.2e} deg")
    return best, valid


def write_audit(path: Path, names: NDArray[np.str_], ra0: F64, dec0: F64,
                idx_valid: IntArr, dev: F64) -> None:
    """CSV of recovered centres, with tile-name positions for comparison."""
    n = len(names)
    count = np.bincount(idx_valid, minlength=n)
    max_dev = group_reduce(dev, idx_valid, n, "max")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_id", "ra0", "dec0", "n_rows", "max_dev_arcsec", "name_ra", "name_dec"])
        for i, nm in enumerate(names):
            p = parse_tile_name(nm) or (float("nan"), float("nan"))
            w.writerow([nm, ra0[i], dec0[i], int(count[i]), max_dev[i], p[0], p[1]])


def centres_from_file(path: str, names: NDArray[np.str_]) -> tuple[F64, F64]:
    """Per-tile centres from a CSV with columns tile_id, ra0, dec0 (deg)."""
    with open(path, newline="") as fh:
        look = {r["tile_id"].strip(): (float(r["ra0"]), float(r["dec0"])) for r in csv.DictReader(fh)}
    missing = [n for n in names if n and n not in look]
    if missing:
        sys.exit(f"{path}: no boresight for {len(missing)} tiles, e.g. {missing[:5]}")
    nan = (float("nan"), float("nan"))
    return (np.array([look.get(n, nan)[0] for n in names]),
            np.array([look.get(n, nan)[1] for n in names]))


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
        plan += [NewColumn(cols.q_err + SUFFIX, cols.q_err),
                 NewColumn(cols.u_err + SUFFIX, cols.u_err),
                 NewColumn(COV_COL, cols.u_err, "Q-U covariance of the error, local-north frame",
                           product_with=cols.q_err)]
    plan.append(NewColumn(GAMMA_COL, None,
                          "Rotation applied to reach local north (chi_true = chi + gamma)", "deg"))
    return plan


def compute_new(cols: Columns, chunk: Mapping[str, F64], gamma: F64,
                lo: Mapping[str, float], has_err: bool) -> list[F64]:
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
        return (au.Unit(a) * au.Unit(b)).to_string()
    except ValueError:
        return None


def extend_header(hdr: fits.Header, plan: Sequence[NewColumn], source_label: str) -> fits.Header:
    """Copy of ``hdr`` with the new columns added; every other card is untouched."""
    new = hdr.copy()
    n0 = int(hdr["TFIELDS"])
    names = [str(hdr[f"TTYPE{i}"]).strip() for i in range(1, n0 + 1)]
    cards: list[tuple[str, object, str]] = []
    for k, nc in enumerate(plan, start=n0 + 1):
        j = names.index(nc.source) + 1 if nc.source else None
        unit = nc.unit
        if unit is None and j is not None:
            unit = hdr.get(f"TUNIT{j}")
        if nc.product_with:
            unit = squared_unit(hdr.get(f"TUNIT{j}"),
                                hdr.get(f"TUNIT{names.index(nc.product_with) + 1}"))
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
    last = max(i for i, c in enumerate(new.cards)
               if c.keyword.startswith(("TTYPE", "TFORM", "TUNIT", "TCOMM")))
    for card in reversed(cards):
        new.insert(last + 1, card)
    new["NAXIS1"] = int(hdr["NAXIS1"]) + 8 * len(plan)
    new["TFIELDS"] = n0 + len(plan)
    new["POLFRAME"] = ("LOCAL_NORTH_COLS_ADDED", "_corr columns added by fix_polframe")
    new["POLFSRC"] = (source_label, "boresight source")
    new["POLFSIGN"] = (FRAME_SIGN, "chi_true = chi + sign*gamma")
    return new


def angle_lower_edge(chi: F64, override: float | None, name: str) -> float:
    """Lower edge of the angle range; checks the data fit a 180 degree span."""
    lo_obs, hi_obs = float(np.nanmin(chi)), float(np.nanmax(chi))
    lo = override if override is not None else (-90.0 if lo_obs < 0 else 0.0)
    print(f"  {name}: observed [{lo_obs:.3f}, {hi_obs:.3f}] deg -> wrap to [{lo:g}, {lo + 180:g})")
    if lo_obs < lo - 1e-6 or hi_obs > lo + 180.0 + 1e-6:
        sys.exit(f"{name} does not fit [{lo:g}, {lo + 180:g}]; set --angle-min or check units")
    return lo


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("catalogue", help="uncompressed FITS (gunzip first)")
    p.add_argument("output")
    p.add_argument("--tile-col", default=Columns.tile)
    p.add_argument("--boresights", help="CSV with tile_id, ra0, dec0 (deg); overrides "
                   "centres recovered from the l/m offsets")
    p.add_argument("--centres-out", help="audit CSV path (default: <output>.tile_centres.csv)")
    p.add_argument("--angle-min", type=float, default=None,
                   help="lower edge of the angle range, 0 or -90 (default: inferred and printed)")
    p.add_argument("--tol-arcsec", type=float, default=1.0,
                   help="max within-tile scatter of the recovered centre")
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
        if int(hdr1.get("PCOUNT", 0)) != 0:
            sys.exit("variable-length columns (PCOUNT != 0) are not supported")
        n, naxis1 = int(hdr1["NAXIS2"]), int(hdr1["NAXIS1"])
        names_all = [str(hdr1[f"TTYPE{i}"]).strip() for i in range(1, int(hdr1["TFIELDS"]) + 1)]
        absent = [c for c in cols.required(args.boresights is None) if c not in names_all]
        if absent:
            sys.exit(f"missing columns: {absent}")
        clash = [c for c in names_all if c.endswith(SUFFIX) or c in (GAMMA_COL, COV_COL)]
        if clash:
            sys.exit(f"already has output columns {clash[:4]}; refusing to run twice")

        dloc = hdul.fileinfo(1)["datLoc"]
        rec_dtype = hdul[1].data.dtype
        if rec_dtype.itemsize != naxis1:
            sys.exit(f"record size {rec_dtype.itemsize} != NAXIS1 {naxis1}")
        raw = np.memmap(args.catalogue, dtype=rec_dtype, mode="r", offset=dloc, shape=(n,))
        rows = np.memmap(args.catalogue, dtype=np.uint8, mode="r", offset=dloc, shape=(n, naxis1))

        def col(name: str) -> F64:
            return raw[name].astype(np.float64)

        # ---- pass 1: centres, angle ranges, error-column check ------------
        print(f"pass 1: {n} rows")
        tnames, tidx = tile_index(raw[cols.tile])
        ra, dec = col(cols.ra), col(cols.dec)
        if args.boresights:
            ra0_t, dec0_t = centres_from_file(args.boresights, tnames)
            source = f"file:{Path(args.boresights).name}"
        else:
            sep = col(cols.sep) if cols.sep in names_all else None
            best, valid = choose_centres(ra, dec, col(cols.l_off), col(cols.m_off), sep,
                                         tnames, tidx, args.tol_arcsec)
            ra0_t, dec0_t = best.ra0, best.dec0
            audit = Path(args.centres_out or f"{args.output}.tile_centres.csv")
            write_audit(audit, tnames, ra0_t, dec0_t, tidx[valid], best.dev_arcsec)
            print(f"wrote tile-centre audit {audit}")
            source = f"offsets:{best.convention}"
        lo = {a: angle_lower_edge(col(a), args.angle_min, a) for a in cols.angles}
        has_err = bool(np.isfinite(col(cols.q_err)).any() and np.isfinite(col(cols.u_err)).any())
        if not has_err:
            print(f"note: {cols.q_err}/{cols.u_err} have no finite values; error columns skipped")
        del ra, dec

        plan = plan_columns(cols, has_err)
        new_hdr = extend_header(hdr1, plan, source)
        extra = 8 * len(plan)

        # ---- pass 2: stream rows -------------------------------------------
        print(f"pass 2: writing {args.output} (+{len(plan)} columns)")
        read_cols = [cols.ra, cols.dec, *cols.angles, cols.q, cols.u, cols.q_bkg, cols.u_bkg]
        if has_err:
            read_cols += [cols.q_err, cols.u_err]
        written = 0
        with open(args.output, "wb") as out:
            out.write(hdr0.tostring(padding=True).encode("ascii"))
            out.write(new_hdr.tostring(padding=True).encode("ascii"))
            for s in range(0, n, args.chunk_rows):
                e = min(s + args.chunk_rows, n)
                chunk = {c: raw[c][s:e].astype(np.float64) for c in read_cols}
                ti = tidx[s:e]
                gamma = gamma_deg(chunk[cols.ra], chunk[cols.dec], ra0_t[ti], dec0_t[ti])
                new = np.stack(compute_new(cols, chunk, gamma, lo, has_err), axis=1)
                buf = np.empty((e - s, naxis1 + extra), dtype=np.uint8)
                buf[:, :naxis1] = rows[s:e]
                buf[:, naxis1:] = np.ascontiguousarray(new.astype(">f8")).view(np.uint8).reshape(e - s, extra)
                out.write(buf.tobytes())
                written += buf.size
                print(f"  {e}/{n}", end="\r")
            out.write(b"\0" * (-written % BLOCK))
    print(f"\nwrote {args.output}: {n} rows, added {[c.name for c in plan]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
