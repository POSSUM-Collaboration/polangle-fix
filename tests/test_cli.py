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
    cat = make_catalogue(n_per_tile=20)
    out = tmp_path / "out.fits"
    args = [
        "rotate-to-local-north",
        str(cat.path),
        str(out),
        "--boresights",
        str(cat.boresights),
    ]
    assert cli.main([*args, "--chunk-rows", "7"]) == 0
    assert "pol_frame_gamma" in Table.read(out).colnames
    assert cli.main(args) == 1  # output exists
    assert cli.main([*args, "--overwrite"]) == 0


def test_errors_become_exit_status(tmp_path: Path) -> None:
    assert (
        cli.main(
            [
                "rotate-to-local-north",
                str(tmp_path / "x.fits.gz"),
                str(tmp_path / "o.fits"),
            ]
        )
        == 1
    )
