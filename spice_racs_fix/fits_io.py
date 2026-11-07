"""Append columns to a large FITS catalogue without touching the original bytes

The input must be an uncompressed FITS file with a bare primary HDU and one
binary table. Only the output header is built with astropy; the original rows
are copied byte for byte with the new columns appended, chunk by chunk, so
memory use does not grow with the width of the table. Each original column's
null marker (TNULL) is kept from the input; if astropy would store any original
column differently in another way, nothing is written. Checksums are dropped
because the data change.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from astropy.table import Column, Table
from numpy.typing import NDArray

from spice_racs_fix.logging import logger

CHECKSUM_KEYS = ("CHECKSUM", "DATASUM")
"""Keywords that go stale when the data change"""


class CatalogueError(ValueError):
    """The catalogue or a request on it cannot be handled safely"""


class NewColumn(NamedTuple):
    """A float64 column to append"""

    name: str
    """Column name"""
    values: NDArray[np.float64]
    """One value per row"""
    unit: str | None
    """Unit, or None"""
    description: str
    """Description, written as TCOMMn and in the astropy metadata"""


class HeaderCard(NamedTuple):
    """A provenance keyword for the table header"""

    keyword: str
    """Keyword (8 characters at most)"""
    value: str | float
    """Value"""
    comment: str
    """Comment"""


class StepResult(NamedTuple):
    """What one correction step adds to the catalogue"""

    columns: tuple[NewColumn, ...]
    """Columns to append, in order"""
    cards: tuple[HeaderCard, ...]
    """Header keywords to add"""


@dataclass(frozen=True, kw_only=True, slots=True)
class WriteOptions:
    """Options for writing the output catalogue"""

    chunk_rows: int = 100_000
    """Rows copied per chunk"""
    overwrite: bool = False
    """Replace an existing output file (never the input)"""


class Catalogue(NamedTuple):
    """An opened input catalogue"""

    path: Path
    """Input file"""
    primary_header: fits.Header
    """Header of the bare primary HDU"""
    table_header: fits.Header
    """Header of the binary table"""
    table: Table
    """The table, memory mapped"""


def same_file(a: Path, b: Path) -> bool:
    """True if two paths name the same file.

    Args:
        a: First path.
        b: Second path.

    Returns:
        Whether they are the same file.
    """
    try:
        return a.samefile(b)
    except OSError:
        return a.resolve() == b.resolve()


def check_output(catalogue: Path, output: Path, overwrite: bool) -> None:
    """Refuse to write over the input, or over an output unless asked.

    Args:
        catalogue: Input file.
        output: Output file.
        overwrite: Whether an existing output may be replaced.

    Raises:
        CatalogueError: If the output may not be written.
    """
    if not output.exists():
        return
    if not overwrite:
        msg = f"{output} exists; use overwrite"
        raise CatalogueError(msg)
    if same_file(catalogue, output):
        msg = "output is the input file; write to a new file"
        raise CatalogueError(msg)


def open_catalogue(path: Path) -> Catalogue:
    """Open a catalogue and check it has the layout the writer supports.

    Args:
        path: Uncompressed FITS file.

    Returns:
        The opened catalogue.

    Raises:
        CatalogueError: If the file is gzipped or laid out unexpectedly.
    """
    if path.suffix == ".gz":
        msg = "input is gzipped; gunzip it first"
        raise CatalogueError(msg)
    with fits.open(path, memmap=True) as hdul:
        if len(hdul) != 2 or not isinstance(hdul[1], fits.BinTableHDU):
            msg = "expected a primary HDU and exactly one binary table"
            raise CatalogueError(msg)
        primary_header = hdul[0].header.copy()
        table_header = hdul[1].header.copy()
    if primary_header.get("NAXIS", 0) != 0:
        msg = "primary HDU carries data; only a bare primary is supported"
        raise CatalogueError(msg)
    if table_header.get("PCOUNT", 0) != 0:
        msg = "variable-length columns (PCOUNT != 0) are not supported"
        raise CatalogueError(msg)
    return Catalogue(
        path=path,
        primary_header=primary_header,
        table_header=table_header,
        table=Table.read(path, memmap=True),
    )


def require_columns(table: Table, names: Sequence[str]) -> None:
    """Raise if any of the named columns are missing.

    Args:
        table: The catalogue table.
        names: Column names the caller needs.

    Raises:
        CatalogueError: If any are missing.
    """
    if absent := [c for c in names if c not in table.colnames]:
        msg = f"missing columns: {absent}"
        raise CatalogueError(msg)


def column_f64(table: Table, name: str) -> NDArray[np.float64]:
    """Column as a plain float64 array, with masked or null values as NaN.

    Args:
        table: The catalogue table.
        name: Column name.

    Returns:
        The values.
    """
    return np.asarray(
        np.ma.filled(np.ma.asarray(table[name]).astype(np.float64), np.nan),
        dtype=np.float64,
    )


def build_header(
    table: Table, columns: Sequence[NewColumn], cards: Sequence[HeaderCard]
) -> fits.Header:
    """The extension header astropy writes for the table plus new columns.

    Table.write (unlike table_to_hdu) puts units, descriptions and column meta
    in the header, so this goes via an in-memory file. A zero-row copy keeps it
    cheap whatever the size of the table.

    Args:
        table: The input table.
        columns: Columns to append.
        cards: Keywords to add.

    Returns:
        The header, still to be checked by match_layout.
    """
    empty = table[:0].copy()
    for col in columns:
        empty[col.name] = Column(
            np.zeros(0, dtype=np.float64), unit=col.unit, description=col.description
        )
    for card in cards:
        empty.meta[card.keyword] = (card.value, card.comment)
    for key in CHECKSUM_KEYS:
        empty.meta.pop(key, None)
    buf = io.BytesIO()
    empty.write(buf, format="fits")
    buf.seek(0)
    with fits.open(buf) as hdul:
        return hdul[1].header.copy()


def match_layout(
    old: fits.Header, new: fits.Header, n_orig: int, n_new: int
) -> str | None:
    """Make the new header describe the input row bytes.

    The original rows are copied from the input, so each original column's null
    marker (TNULL) is taken from the input header: astropy before 6.0 replaces
    it with its own default (999999) on write. Any other layout difference means
    astropy would store a column differently. ``new`` is edited in place.

    Args:
        old: Input table header.
        new: Header astropy wrote with ``n_new`` float64 columns appended.
        n_orig: Number of input columns.
        n_new: Number of appended columns.

    Returns:
        A description of the first problem, or None.
    """
    # Keywords that fix how a column sits in the row bytes; TNULL is handled apart
    layout_keys = ("TTYPE", "TFORM", "TDIM", "TSCAL", "TZERO")
    for i in range(1, n_orig + 1):
        for k in layout_keys:
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


def add_column_comments(
    header: fits.Header, columns: Sequence[NewColumn], n_orig: int
) -> None:
    """Give each appended column a TCOMMn, as the original columns have.

    Args:
        header: Output table header, edited in place.
        columns: The appended columns.
        n_orig: Number of input columns.
    """
    for j, col in enumerate(columns, start=n_orig + 1):
        header[f"TCOMM{j}"] = col.description


def write_streaming(
    catalogue: Catalogue,
    output: Path,
    header: fits.Header,
    new: NDArray[np.float64],
    chunk_rows: int,
) -> None:
    """Copy the input rows byte for byte, appending the new values to each.

    Memory use is one chunk of rows, independent of the size of the table.

    Args:
        catalogue: The input catalogue.
        output: Output file.
        header: Output table header.
        new: One float64 column per appended table column.
        chunk_rows: Rows per chunk.
    """
    with fits.open(catalogue.path, memmap=True) as hdul:
        naxis1, n_rows = hdul[1].header["NAXIS1"], hdul[1].header["NAXIS2"]
        offset = hdul.fileinfo(1)["datLoc"]
    rows = np.memmap(
        catalogue.path, dtype=np.uint8, mode="r", offset=offset, shape=(n_rows, naxis1)
    )
    new_be = np.ascontiguousarray(new, dtype=">f8")
    extra = new_be.shape[1] * 8
    header = header.copy()
    header["NAXIS2"] = n_rows
    primary = catalogue.primary_header.copy()
    for key in CHECKSUM_KEYS:
        primary.remove(key, ignore_missing=True, remove_all=True)
    written = 0
    with output.open("wb") as f:
        f.write(primary.tostring(padding=True).encode("ascii"))
        f.write(header.tostring(padding=True).encode("ascii"))
        for start in range(0, n_rows, chunk_rows):
            end = min(start + chunk_rows, n_rows)
            block = np.empty((end - start, naxis1 + extra), dtype=np.uint8)
            block[:, :naxis1] = rows[start:end]
            block[:, naxis1:] = new_be[start:end].view(np.uint8)
            f.write(block.tobytes())
            written += block.size
        # Pad the data to a whole number of 2880 byte FITS blocks
        f.write(b"\0" * (-written % 2880))
    del rows


def append_columns(
    catalogue: Catalogue,
    output: Path,
    steps: Sequence[StepResult],
    options: WriteOptions,
) -> int:
    """Write the catalogue with every step's columns and keywords appended.

    Args:
        catalogue: The input catalogue.
        output: Output file.
        steps: Results of the correction steps, in order.
        options: Write options.

    Returns:
        Number of rows written.

    Raises:
        CatalogueError: If a column already exists, or astropy would store an
            original column differently. Nothing is written in either case.
    """
    check_output(catalogue.path, output, options.overwrite)
    columns = [col for step in steps for col in step.columns]
    if not columns:
        msg = "no columns to append"
        raise CatalogueError(msg)
    cards = [card for step in steps for card in step.cards]
    if clash := [c.name for c in columns if c.name in catalogue.table.colnames]:
        msg = f"already has output columns {clash[:4]}; refusing to run twice"
        raise CatalogueError(msg)

    n_orig = len(catalogue.table.colnames)
    header = build_header(catalogue.table, columns, cards)
    if problem := match_layout(catalogue.table_header, header, n_orig, len(columns)):
        msg = (
            f"astropy would store an original column differently ({problem}); "
            "stopping rather than alter the original data"
        )
        raise CatalogueError(msg)
    add_column_comments(header, columns, n_orig)

    new = np.column_stack([np.asarray(c.values, dtype=np.float64) for c in columns])
    write_streaming(catalogue, output, header, new, options.chunk_rows)
    n_rows = int(new.shape[0])
    logger.info("wrote %s: %d rows, %d new columns", output, n_rows, len(columns))
    return n_rows
