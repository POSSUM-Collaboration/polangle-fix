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

CHECKSUM_KEYWORDS = ("CHECKSUM", "DATASUM")
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

    rows_per_chunk: int = 100_000
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


def same_file(first_path: Path, second_path: Path) -> bool:
    """True if two paths name the same file.

    Args:
        first_path: First path.
        second_path: Second path.

    Returns:
        Whether they are the same file.
    """
    try:
        return first_path.samefile(second_path)
    except OSError:
        return first_path.resolve() == second_path.resolve()


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
        message = f"{output} exists; use overwrite"
        raise CatalogueError(message)
    if same_file(catalogue, output):
        message = "output is the input file; write to a new file"
        raise CatalogueError(message)


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
        message = "input is gzipped; gunzip it first"
        raise CatalogueError(message)
    with fits.open(path, memmap=True) as hdu_list:
        if len(hdu_list) != 2 or not isinstance(hdu_list[1], fits.BinTableHDU):
            message = "expected a primary HDU and exactly one binary table"
            raise CatalogueError(message)
        primary_header = hdu_list[0].header.copy()
        table_header = hdu_list[1].header.copy()
    if primary_header.get("NAXIS", 0) != 0:
        message = "primary HDU carries data; only a bare primary is supported"
        raise CatalogueError(message)
    if table_header.get("PCOUNT", 0) != 0:
        message = "variable-length columns (PCOUNT != 0) are not supported"
        raise CatalogueError(message)
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
    if absent := [name for name in names if name not in table.colnames]:
        message = f"missing columns: {absent}"
        raise CatalogueError(message)


def column_as_float64(table: Table, name: str) -> NDArray[np.float64]:
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
    for column in columns:
        empty[column.name] = Column(
            np.zeros(0, dtype=np.float64),
            unit=column.unit,
            description=column.description,
        )
    for card in cards:
        empty.meta[card.keyword] = (card.value, card.comment)
    for keyword in CHECKSUM_KEYWORDS:
        empty.meta.pop(keyword, None)
    buffer = io.BytesIO()
    empty.write(buffer, format="fits")
    buffer.seek(0)
    with fits.open(buffer) as hdu_list:
        return hdu_list[1].header.copy()


def match_layout(
    input_header: fits.Header,
    output_header: fits.Header,
    original_column_count: int,
    new_column_count: int,
) -> str | None:
    """Make the output header describe the input row bytes.

    The original rows are copied from the input, so each original column's null
    marker (TNULL) is taken from the input header: astropy before 6.0 replaces
    it with its own default (999999) on write. Any other layout difference means
    astropy would store a column differently. ``output_header`` is edited in
    place.

    Args:
        input_header: Input table header.
        output_header: Header astropy wrote with ``new_column_count`` float64
            columns appended.
        original_column_count: Number of input columns.
        new_column_count: Number of appended columns.

    Returns:
        A description of the first problem, or None.
    """
    # Keywords that fix how a column sits in the row bytes; TNULL is handled apart
    layout_keywords = ("TTYPE", "TFORM", "TDIM", "TSCAL", "TZERO")
    for column_number in range(1, original_column_count + 1):
        for layout_keyword in layout_keywords:
            keyword = f"{layout_keyword}{column_number}"
            if input_header.get(keyword) != output_header.get(keyword):
                return (
                    f"{keyword} is {input_header.get(keyword)!r} in the input, "
                    f"{output_header.get(keyword)!r} from astropy"
                )
        null_keyword = f"TNULL{column_number}"
        if null_keyword not in input_header:
            output_header.remove(null_keyword, ignore_missing=True)
        elif null_keyword in output_header:
            output_header[null_keyword] = input_header[null_keyword]
        else:
            output_header.set(
                null_keyword, input_header[null_keyword], after=f"TFORM{column_number}"
            )
    expected_row_bytes = input_header["NAXIS1"] + 8 * new_column_count
    if output_header["NAXIS1"] != expected_row_bytes:
        return f"NAXIS1 is {output_header['NAXIS1']}, expected {expected_row_bytes}"
    for column_number in range(
        original_column_count + 1, original_column_count + new_column_count + 1
    ):
        if output_header[f"TFORM{column_number}"] != "D":
            return (
                f"TFORM{column_number} is "
                f"{output_header[f'TFORM{column_number}']!r}, expected 'D'"
            )
    return None


def add_column_comments(
    header: fits.Header, columns: Sequence[NewColumn], original_column_count: int
) -> None:
    """Give each appended column a TCOMMn, as the original columns have.

    Args:
        header: Output table header, edited in place.
        columns: The appended columns.
        original_column_count: Number of input columns.
    """
    for column_number, column in enumerate(columns, start=original_column_count + 1):
        header[f"TCOMM{column_number}"] = column.description


def write_streaming(
    catalogue: Catalogue,
    output: Path,
    header: fits.Header,
    new_values: NDArray[np.float64],
    rows_per_chunk: int,
) -> None:
    """Copy the input rows byte for byte, appending the new values to each.

    Memory use is one chunk of rows, independent of the size of the table.

    Args:
        catalogue: The input catalogue.
        output: Output file.
        header: Output table header.
        new_values: One float64 column per appended table column.
        rows_per_chunk: Rows copied per chunk.
    """
    with fits.open(catalogue.path, memmap=True) as hdu_list:
        row_bytes = hdu_list[1].header["NAXIS1"]
        row_count = hdu_list[1].header["NAXIS2"]
        data_offset = hdu_list.fileinfo(1)["datLoc"]
    input_rows = np.memmap(
        catalogue.path,
        dtype=np.uint8,
        mode="r",
        offset=data_offset,
        shape=(row_count, row_bytes),
    )
    new_values_big_endian = np.ascontiguousarray(new_values, dtype=">f8")
    appended_bytes = new_values_big_endian.shape[1] * 8
    header = header.copy()
    header["NAXIS2"] = row_count
    primary_header = catalogue.primary_header.copy()
    for keyword in CHECKSUM_KEYWORDS:
        primary_header.remove(keyword, ignore_missing=True, remove_all=True)
    bytes_written = 0
    with output.open("wb") as output_file:
        output_file.write(primary_header.tostring(padding=True).encode("ascii"))
        output_file.write(header.tostring(padding=True).encode("ascii"))
        for chunk_start in range(0, row_count, rows_per_chunk):
            chunk_end = min(chunk_start + rows_per_chunk, row_count)
            chunk = np.empty(
                (chunk_end - chunk_start, row_bytes + appended_bytes), dtype=np.uint8
            )
            chunk[:, :row_bytes] = input_rows[chunk_start:chunk_end]
            chunk[:, row_bytes:] = new_values_big_endian[chunk_start:chunk_end].view(
                np.uint8
            )
            output_file.write(chunk.tobytes())
            bytes_written += chunk.size
        # Pad the data to a whole number of 2880 byte FITS blocks
        output_file.write(b"\0" * (-bytes_written % 2880))
    del input_rows


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
    columns = [column for step in steps for column in step.columns]
    if not columns:
        message = "no columns to append"
        raise CatalogueError(message)
    cards = [card for step in steps for card in step.cards]
    if clashing := [
        column.name for column in columns if column.name in catalogue.table.colnames
    ]:
        message = f"already has output columns {clashing[:4]}; refusing to run twice"
        raise CatalogueError(message)

    original_column_count = len(catalogue.table.colnames)
    header = build_header(catalogue.table, columns, cards)
    if problem := match_layout(
        catalogue.table_header, header, original_column_count, len(columns)
    ):
        message = (
            f"astropy would store an original column differently ({problem}); "
            "stopping rather than alter the original data"
        )
        raise CatalogueError(message)
    add_column_comments(header, columns, original_column_count)

    new_values = np.column_stack(
        [np.asarray(column.values, dtype=np.float64) for column in columns]
    )
    write_streaming(catalogue, output, header, new_values, options.rows_per_chunk)
    rows_written = int(new_values.shape[0])
    logger.info("wrote %s: %d rows, %d new columns", output, rows_written, len(columns))
    return rows_written
