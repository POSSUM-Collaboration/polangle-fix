"""Command line interface"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from spice_racs_fix.fits_io import CatalogueError, WriteOptions
from spice_racs_fix.logging import logger
from spice_racs_fix.rotate_to_local_north import (
    RotateToLocalNorthOptions,
    run_rotate_to_local_north,
)


def options_default(options_class: Any, field_name: str) -> Any:
    """Default of an Options field, so the CLI cannot drift from it."""
    return next(
        field.default
        for field in dataclasses.fields(options_class)
        if field.name == field_name
    )


def add_write_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by every step that writes a catalogue."""
    parser.add_argument("catalogue", type=Path, help="Uncompressed FITS (gunzip first)")
    parser.add_argument("output", type=Path, help="Output FITS")
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=options_default(WriteOptions, "rows_per_chunk"),
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
    subparsers = parser.add_subparsers(dest="command", required=True)

    rotate_parser = subparsers.add_parser(
        "rotate-to-local-north",
        help="Add local-north polarisation columns (_corr) and pol_frame_gamma",
    )
    add_write_arguments(rotate_parser)
    rotate_parser.add_argument(
        "--boresight-csv",
        type=Path,
        default=options_default(RotateToLocalNorthOptions, "boresight_csv"),
        help="CSV with FIELD_NAME, RA_DEG, DEC_DEG (default: the RACS-low3 fields)",
    )
    rotate_parser.add_argument(
        "--tile-column",
        default=options_default(RotateToLocalNorthOptions, "tile_column"),
    )
    rotate_parser.add_argument(
        "--angle-lower-edge-deg",
        type=float,
        help="Lower edge of the angle range, 0 or -90 (default: from the data)",
    )
    separation_column = options_default(RotateToLocalNorthOptions, "separation_column")
    rotate_parser.add_argument(
        "--separation-tolerance-arcsec",
        type=float,
        default=options_default(
            RotateToLocalNorthOptions, "separation_tolerance_arcsec"
        ),
        help=(
            f"Largest allowed difference from {separation_column} "
            "(default: %(default)s; inf skips)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run spice-racs-fix.

    Args:
        argv: Arguments; None reads sys.argv.

    Returns:
        Exit status.
    """
    arguments = build_parser().parse_args(argv)
    write_options = WriteOptions(
        rows_per_chunk=arguments.rows_per_chunk, overwrite=arguments.overwrite
    )
    try:
        if arguments.command == "rotate-to-local-north":
            run_rotate_to_local_north(
                arguments.catalogue,
                arguments.output,
                RotateToLocalNorthOptions(
                    boresight_csv=arguments.boresight_csv,
                    tile_column=arguments.tile_column,
                    angle_lower_edge_deg=arguments.angle_lower_edge_deg,
                    separation_tolerance_arcsec=arguments.separation_tolerance_arcsec,
                ),
                write_options,
            )
    except CatalogueError as error:
        logger.error("%s", error)
        return 1
    return 0
