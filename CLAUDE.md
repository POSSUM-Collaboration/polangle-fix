# spice-racs-fix

Corrections to the SPICE-RACS DR2 catalogue. Fully typed Python, mypy strict on
`spice_racs_fix/*`.

## Data

The catalogue files (several GB) are not in the repo. Tests build small
synthetic catalogues in `tests/conftest.py`.
`spice_racs_fix/data/field_data.csv` is the RACS-low3 field list used for
boresights.

## Environment

Use `uv run` for everything. Never call `pytest`/`mypy`/`ruff` bare.

## Verify gate (run before calling any task done)

```
uv run --extra dev pytest tests/<touched modules>
uv run --extra dev prek run --hook-stage manual --all-files
```

## Design rules

- Every step appends columns. Original rows are copied byte for byte by
  `fits_io.append_columns`; never change an original column in place.
- A step is a function `table, Options -> StepResult` (new columns plus header
  cards), with a `run_<step>` wrapper that opens, checks and writes.
- Corrected values take a `_corr` suffix. New systematic error terms take
  `_err_sys`.
- Parameters come from SPICE-RACS data alone (for example same-source tile
  pairs), not from POSSUM.
- Library code raises `CatalogueError`; only `cli.main` turns errors into an
  exit status.

## Code style

- Functions: short Google docstrings (`Args:`/`Returns:`/`Raises:` where
  useful). NamedTuple fields and comments: terse one-liners (`"""Does X"""`).
- Write like a human, not a manual. No over-explaining, no em/en dashes anywhere
  (docs, comments, commit messages, prose).
- `from __future__ import annotations` required in every module (ruff isort
  enforces).
- Python must be fully typed. mypy strict + `disallow_untyped_defs` on
  `spice_racs_fix.*`.
- Ruff ruleset is wide (see `pyproject.toml`); don't add per-file ignores
  without asking.
- Prefer plain Options/Results containers (NamedTuple/dataclass) over full OOP
  with methods and inheritance. Functions take an Options object, return a
  Results object.
