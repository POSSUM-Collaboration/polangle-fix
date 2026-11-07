from __future__ import annotations

from typing import TYPE_CHECKING

from astropy.table import Table
from conftest import SyntheticCatalogue
from spice_racs_fix import cli

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_rotate_to_local_north_command(
    tmp_path: Path, make_catalogue: Callable[..., SyntheticCatalogue]
) -> None:
    catalogue = make_catalogue(rows_per_tile=20)
    output_path = tmp_path / "out.fits"
    arguments = [
        "rotate-to-local-north",
        str(catalogue.path),
        str(output_path),
        "--boresight-csv",
        str(catalogue.boresight_csv),
    ]
    assert cli.main([*arguments, "--rows-per-chunk", "7"]) == 0
    assert "pol_frame_gamma" in Table.read(output_path).colnames
    assert cli.main(arguments) == 1  # output exists
    assert cli.main([*arguments, "--overwrite"]) == 0


def test_errors_become_exit_status(tmp_path: Path) -> None:
    exit_status = cli.main(
        [
            "rotate-to-local-north",
            str(tmp_path / "x.fits.gz"),
            str(tmp_path / "o.fits"),
        ]
    )
    assert exit_status == 1
