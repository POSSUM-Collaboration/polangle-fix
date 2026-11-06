"""Fixtures shared by more than one test module."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING, NamedTuple

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

CENTRES = {"RACS_0426-84": (66.5, -84.0), "RACS_1200-30": (180.0, -30.0)}
"""Two tiles, one near the pole"""


class SyntheticCatalogue(NamedTuple):
    """A small catalogue on disk"""

    path: Path
    """FITS file"""
    table: Table
    """The table as written"""
    boresights: Path
    """Boresight CSV"""


def write_boresights(path: Path, rows: list[tuple[str, float, float]]) -> Path:
    """Write a FIELD_NAME, RA_DEG, DEC_DEG CSV."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FIELD_NAME", "RA_DEG", "DEC_DEG"])
        writer.writerows(rows)
    return path


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1)


@pytest.fixture
def make_catalogue(
    tmp_path: Path, rng: np.random.Generator
) -> Callable[..., SyntheticCatalogue]:
    """Two tiles as a FITS table plus a boresight CSV.

    ``with_sep`` adds a separation_tile_centre column, as arrakis writes it.
    """

    def make(n_per_tile: int = 300, with_sep: bool = False) -> SyntheticCatalogue:
        ras: list[float] = []
        decs: list[float] = []
        tiles: list[str] = []
        for name, (r0, d0) in CENTRES.items():
            cs = SkyCoord(r0 * u.deg, d0 * u.deg).directional_offset_by(
                rng.uniform(0, 360, n_per_tile) * u.deg,
                rng.uniform(0, 3, n_per_tile) * u.deg,
            )
            ras += list(cs.ra.deg)
            decs += list(cs.dec.deg)
            tiles += [name] * n_per_tile
        n = len(tiles)
        chi = rng.uniform(0, 180, n)
        q, uu = rng.normal(size=n), rng.normal(size=n)
        t = Table()
        t["ra"] = Column(ras, unit="deg", description="Right Ascension [ICRS]")
        t["dec"] = Column(decs, unit="deg", description="Declination [ICRS]")
        t["tile_id"] = Column(tiles, description="Tile ID")
        t["polangle"] = Column(
            chi, unit="deg", description="Electric vector polarization angle"
        )
        t["derot_polangle"] = Column(chi, unit="deg")
        t["stokesQ"] = Column(q, unit="Jy / beam")
        t["stokesU"] = Column(uu, unit="Jy / beam")
        t["stokesQ_err"] = Column(rng.uniform(1, 2, n), unit="Jy")
        t["stokesU_err"] = Column(rng.uniform(1, 2, n), unit="Jy")
        t["stokesQ_bkg"] = Column(q * 0.1, unit="Jy / beam")
        t["stokesU_bkg"] = Column(uu * 0.1, unit="Jy / beam")
        t["flag"] = Column(rng.uniform(size=n) > 0.5)
        t["name"] = Column([f"src{i}" for i in range(n)])
        t["polangle"][3] = np.nan  # a null
        if with_sep:
            ra0 = np.array([CENTRES[x][0] for x in tiles])
            dec0 = np.array([CENTRES[x][1] for x in tiles])
            sep = SkyCoord(ras * u.deg, decs * u.deg).separation(
                SkyCoord(ra0 * u.deg, dec0 * u.deg)
            )
            t["separation_tile_centre"] = Column(sep.deg, unit="deg")
        path = tmp_path / "cat.fits"
        t.write(path)
        with fits.open(path) as hdul:  # add checksums, as a released file may have
            hdul.writeto(path, overwrite=True, checksum=True)
        rows = [(nm, r, d) for nm, (r, d) in CENTRES.items()]
        rows.append(rows[0])  # re-observation of a field: duplicate name, same centre
        bore = write_boresights(tmp_path / "bore.csv", rows)
        return SyntheticCatalogue(path=path, table=t, boresights=bore)

    return make
