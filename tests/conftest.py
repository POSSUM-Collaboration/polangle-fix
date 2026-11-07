"""Fixtures shared by more than one test module."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import pytest
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

TILE_CENTRES_DEG = {"RACS_0426-84": (66.5, -84.0), "RACS_1200-30": (180.0, -30.0)}
"""Two tiles, one near the pole: (RA, Dec) in degrees by tile name"""


class SyntheticCatalogue(NamedTuple):
    """A small catalogue on disk"""

    path: Path
    """FITS file"""
    table: Table
    """The table as written"""
    boresight_csv: Path
    """Boresight CSV"""


def write_boresight_csv(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """Write a FIELD_NAME, RA_DEG, DEC_DEG CSV."""
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FIELD_NAME", "RA_DEG", "DEC_DEG"])
        writer.writerows(rows)
    return path


@pytest.fixture
def random_generator() -> np.random.Generator:
    return np.random.default_rng(1)


@pytest.fixture
def make_catalogue(
    tmp_path: Path, random_generator: np.random.Generator
) -> Callable[..., SyntheticCatalogue]:
    """Two tiles as a FITS table plus a boresight CSV.

    ``with_separation`` adds a separation_tile_centre column, as arrakis writes it.
    """

    def make(
        rows_per_tile: int = 300, with_separation: bool = False
    ) -> SyntheticCatalogue:
        source_ra_deg: list[float] = []
        source_dec_deg: list[float] = []
        tile_names: list[str] = []
        for tile_name, (centre_ra, centre_dec) in TILE_CENTRES_DEG.items():
            positions = SkyCoord(
                centre_ra * units.deg, centre_dec * units.deg
            ).directional_offset_by(
                random_generator.uniform(0, 360, rows_per_tile) * units.deg,
                random_generator.uniform(0, 3, rows_per_tile) * units.deg,
            )
            source_ra_deg += list(positions.ra.deg)
            source_dec_deg += list(positions.dec.deg)
            tile_names += [tile_name] * rows_per_tile
        row_count = len(tile_names)
        polarisation_angle = random_generator.uniform(0, 180, row_count)
        stokes_q = random_generator.normal(size=row_count)
        stokes_u = random_generator.normal(size=row_count)
        table = Table()
        table["ra"] = Column(
            source_ra_deg, unit="deg", description="Right Ascension [ICRS]"
        )
        table["dec"] = Column(
            source_dec_deg, unit="deg", description="Declination [ICRS]"
        )
        table["tile_id"] = Column(tile_names, description="Tile ID")
        table["polangle"] = Column(
            polarisation_angle,
            unit="deg",
            description="Electric vector polarization angle",
        )
        table["derot_polangle"] = Column(polarisation_angle, unit="deg")
        table["stokesQ"] = Column(stokes_q, unit="Jy / beam")
        table["stokesU"] = Column(stokes_u, unit="Jy / beam")
        table["stokesQ_err"] = Column(
            random_generator.uniform(1, 2, row_count), unit="Jy"
        )
        table["stokesU_err"] = Column(
            random_generator.uniform(1, 2, row_count), unit="Jy"
        )
        table["stokesQ_bkg"] = Column(stokes_q * 0.1, unit="Jy / beam")
        table["stokesU_bkg"] = Column(stokes_u * 0.1, unit="Jy / beam")
        table["flag"] = Column(random_generator.uniform(size=row_count) > 0.5)
        table["name"] = Column([f"src{index}" for index in range(row_count)])
        table["polangle"][3] = np.nan  # a null
        if with_separation:
            boresight_ra = np.array([TILE_CENTRES_DEG[name][0] for name in tile_names])
            boresight_dec = np.array([TILE_CENTRES_DEG[name][1] for name in tile_names])
            separation = SkyCoord(
                source_ra_deg * units.deg, source_dec_deg * units.deg
            ).separation(SkyCoord(boresight_ra * units.deg, boresight_dec * units.deg))
            table["separation_tile_centre"] = Column(separation.deg, unit="deg")
        path = tmp_path / "cat.fits"
        table.write(path)
        # add checksums, as a released file may have
        with fits.open(path) as hdu_list:
            hdu_list.writeto(path, overwrite=True, checksum=True)
        boresight_rows = [
            (name, centre_ra, centre_dec)
            for name, (centre_ra, centre_dec) in TILE_CENTRES_DEG.items()
        ]
        # re-observation of a field: duplicate name, same centre
        boresight_rows.append(boresight_rows[0])
        boresight_csv = write_boresight_csv(tmp_path / "bore.csv", boresight_rows)
        return SyntheticCatalogue(path=path, table=table, boresight_csv=boresight_csv)

    return make
