# spice-racs-fix

Corrections to the SPICE-RACS DR2 catalogue. Fully typed Python, mypy strict on
`spice_racs_fix/*`.

## Data

The catalogue files (several GB) are not in the repo. Tests build small
synthetic catalogues in `tests/conftest.py`.
`spice_racs_fix/data/field_data.csv` is the RACS-low3 field list used for
boresights.

## Environment

Use `uv run` for everything. Never call `pytest`/`mypy`/`ruff` bare. Dev tools
are in the `dev` dependency group, which `uv run` and `uv sync` install by
default. `uv.lock` is committed; change dependencies with `uv add`/`uv remove`.
Hooks run with prek (`prek.toml`), not pre-commit.

The version comes from git tags through hatch-vcs (`vX.Y.Z`). Never write a
version number into the code.

## Verify gate (run before calling any task done)

```
uv run pytest tests/<touched modules>
uv run prek run --hook-stage manual --all-files
```

## Design rules

- Every step appends columns. Original rows are copied byte for byte by
  `fits_io.append_columns`; never change an original column in place.
- A step is a function `table, Options -> StepResult` (new columns plus header
  cards), with a `run_<step>` wrapper that opens, checks and writes.
- Every name is fully self-describing: steps, modules, functions, Options
  fields, CLI flags, arguments and locals. Write words out (`separation`, not
  `sep`; `rotation_deg`, not `gamma`; `catalogue`, not `cat`). Published output
  column names and FITS keywords are data, not code, and stay as they are.
- Names describe what the thing does. A step's CLI subcommand, module and
  `run_<step>` say what it produces or changes (for example `fit-error-model`,
  not `calibrate`). Avoid jargon that means something else in radio astronomy.
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
- No module level constants unless several functions or modules genuinely share
  them. Column names and tunable values go in the step's Options; a value used
  in one place is a local (or inline) with a short comment.
- Prefer plain Options/Results containers (NamedTuple/dataclass) over full OOP
  with methods and inheritance. Functions take an Options object, return a
  Results object.
