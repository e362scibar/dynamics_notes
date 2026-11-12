# common/python/

Shared Python code for all topics: helpers for symbolic derivations (sympy), data and plotting utilities, and the modules used by the slide generator in [common/slides/](../slides/README.md).

## Layout

```text
common/python/
  pyproject.toml          # package metadata and dependency lower bounds
  README.md               # this file
  dynamics_common/        # importable package (name is a planned placeholder)
    __init__.py           # explicit __all__ listing the public API
    ...                   # one module per concern
  tests/                  # pytest tests for dynamics_common
```

- The package name `dynamics_common` is a placeholder. Rename it before the first module is committed if a better name is chosen.
- Topic notebooks and the slide generator import from `dynamics_common`. Do not copy helper code into a topic.

## Installation

The package is installed in editable mode into the `dynamics_notes` conda environment so that `import dynamics_common` works from any topic:

```bash
pip install -e common/python
```

When the first module is added, add `-e ./common/python` to the `pip:` section of `environment.yml` and run `conda env update -f environment.yml --prune`.

## Rules

- Python code in this directory follows the rules in [AGENTS.md](../../AGENTS.md): GPL header, English comments and docstrings, `black` (line length 120), `isort`, Google-style docstrings, type hints, and `pathlib.Path`.
- Each public function is tested under `tests/` with `pytest`. Use `@pytest.mark.parametrize` where it avoids duplicated tests.
- Symbolic helpers should return sympy objects and leave the choice of printing (`sympy.latex()`, MathML) to the caller.
- Keep the package free of topic-specific content. Topic code goes in `topics/<topic>/`.

## License

Source code in this directory is licensed under [GPL-3.0-or-later](../../LICENSE). See [LICENSING.md](../../LICENSING.md).
