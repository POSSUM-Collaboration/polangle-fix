"""Command line interface"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spice_racs_fix.fits_io import CatalogueError, WriteOptions
from spice_racs_fix.logging import logger
from spice_racs_fix.polframe import (
    DEFAULT_BORESIGHTS,
    SEP_COL,
    PolframeOptions,
    run_polframe,
)


def _default(options_cls: Any, name: str) -> Any:
    """Default of an Options field, so the CLI cannot drift from it."""
    return next(f.default for f in dataclasses.fields(options_cls) if f.name == name)


def _add_write_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every step that writes a catalogue."""
    parser.add_argument("catalogue", type=Path, help="Uncompressed FITS (gunzip first)")
    parser.add_argument("output", type=Path, help="Output FITS")
    parser.add_argument(
        "--chunk-rows", type=int, default=_default(WriteOptions, "chunk_rows")
    )
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Argument parser for spice-racs-fix.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        prog="spice-racs-fix", description="Corrections to the SPICE-RACS DR2 catalogue"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pol = sub.add_parser(
        "polframe",
        help="Add local-north polarisation columns (_corr) and pol_frame_gamma",
    )
    _add_write_args(pol)
    pol.add_argument(
        "--boresights",
        type=Path,
        default=DEFAULT_BORESIGHTS,
        help="CSV with FIELD_NAME, RA_DEG, DEC_DEG (default: the RACS-low3 fields)",
    )
    pol.add_argument("--tile-col", default=_default(PolframeOptions, "tile_col"))
    pol.add_argument(
        "--angle-min", type=float, help="Lower edge of angle range, 0 or -90"
    )
    pol.add_argument(
        "--sep-tol-arcsec",
        type=float,
        default=_default(PolframeOptions, "sep_tol_arcsec"),
        help=f"Largest allowed difference from {SEP_COL} (default: %(default)s; inf skips)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run spice-racs-fix.

    Args:
        argv: Arguments; None reads sys.argv.

    Returns:
        Exit status.
    """
    args = build_parser().parse_args(argv)
    write_options = WriteOptions(chunk_rows=args.chunk_rows, overwrite=args.overwrite)
    try:
        if args.command == "polframe":
            run_polframe(
                args.catalogue,
                args.output,
                PolframeOptions(
                    boresights=args.boresights,
                    tile_col=args.tile_col,
                    angle_min=args.angle_min,
                    sep_tol_arcsec=args.sep_tol_arcsec,
                ),
                write_options,
            )
    except CatalogueError as err:
        logger.error("%s", err)
        return 1
    return 0
