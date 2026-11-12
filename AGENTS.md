# AGENTS.md

Guidance for AI coding agents working in this repository.

## Overview

This repository holds personal memo material on **high-energy electron beam dynamics in accelerators**. Each topic is kept in three forms:

| Form | Location | Languages | Source of truth | Output |
|---|---|---|---|---|
| Jupyter notebook | `notebooks/` | English only | Jupytext `.py` file | `.ipynb` (outputs stripped on commit) |
| LaTeX document | `docs/` | English and Japanese | `docs/main.tex` (bilingual source) | One PDF per language |
| PowerPoint slides | `slides/` | English and Japanese | LaTeX document, via a shared generator | One `.pptx` per language |

The English and Japanese LaTeX and slide versions must describe the same content. Equations and figures are shared between the two languages; only prose, labels, and titles differ.

## Language rules

- **Notebooks, Python scripts, and code comments**: English only. This includes comments, docstrings, log messages, and display strings in code.
- **LaTeX and slides**: English and Japanese versions.
- **Commit messages**: English only.
- **Chat replies to the user**: Japanese, unless the user asks otherwise.

## Git rules

- **Work on the `devel` branch.** Do not commit directly to `main`. If the current branch is `main` when you start a change, switch to `devel` first (create it from `main` if it does not exist yet).
- **Do not run `git commit` unless the user explicitly asks for a commit.** Staging, editing, and building are fine without a commit request. Approval for one commit does not extend to later changes.
- Do not push, rebase, or rewrite history unless explicitly asked.

## Environment

- **Conda environment**: `dynamics_memo`
- **Python**: 3.14
- **Definition**: `environment.yml` at the repository root is the single place where the environment is specified. Do not hand-edit an exported lock file to add packages.

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate dynamics_memo
```

Update after changing `environment.yml`:

```bash
conda env update -f environment.yml --prune
```

Always activate the environment before running any script, notebook sync, or build.

### Packages (planned)

- Scientific: `numpy`, `scipy`, `sympy`, `matplotlib`, `pandas`
- Notebooks: `jupyter` (or `jupyterlab`), `jupytext`, `nbstripout`
- Slides: `python-pptx`, `lxml`, `latex2mathml`, `mathml2omml-as` (PyPI name; imported as `mathml2omml`)
- Tests and style: `pytest`, `black`, `isort`
- LaTeX toolchain (installed outside conda): TeX Live or MacTeX providing `latexmk`, `lualatex`, `luatexja`

Version constraints in `environment.yml` should use lower bounds (`>=`) unless an exact pin is needed for reproducibility.

## Commands

### Jupytext and notebooks

Edit only the `.py` files. Sync them to the `.ipynb` files:

```bash
jupytext --sync notebooks/<topic>.py
```

Execute a notebook headlessly to check that it runs (does not commit outputs):

```bash
jupytext --to notebook --execute notebooks/<topic>.py
```

### LaTeX documents

Build both languages with `latexmk` and `lualatex`:

```bash
# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=main-en docs/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=main-ja docs/main.tex

# Clean auxiliary files
latexmk -C -outdir=docs/build docs/main.tex
```

Build outputs go to `docs/build/`. Do not commit build outputs.

### Slides

One shared generator script builds both languages:

```bash
python slides/build_slides.py --lang en
python slides/build_slides.py --lang ja
```

Generated `.pptx` files go to `slides/build/`. Do not commit them unless the user asks for a specific deck to be tracked.

### Tests

```bash
python -m pytest -q
```

## Critical rules

### Notebooks

- **Edit `.py`, never `.ipynb`.** Jupytext-paired `.ipynb` files are generated artifacts.
- Notebooks are **English only**. Do not create Japanese notebook versions.
- Use **sympy** for every symbolic expression, derivation, or algebraic transformation. Do not hand-type a transformed equation and leave it unverified. Derive it in sympy, check it numerically when practical, and then present it using `sympy.latex()` or an equivalent that matches the LaTeX source.
- Notebook outputs are **stripped on commit by `nbstripout`**. Do not rely on committed outputs. The `.gitattributes` filter must stay configured (see below).
- Never add a notebook under `notebooks/` without also adding it to `notebooks/README.md` in the same change.

### nbstripout

`.gitattributes` routes `*.ipynb` through the `nbstripout` filter. The filter driver is registered in the local git config, which must be set once per clone:

```bash
nbstripout --install
nbstripout --status   # verify
```

Do not disable or bypass the filter (for example with `git -c filter...` tricks or by committing with `--no-verify` to skip hooks). If a notebook shows outputs in `git diff`, check that the filter is installed in this clone before committing.

### LaTeX

- The source of truth is `docs/main.tex`, which handles both languages. Do not create separate full copies of the document for each language.
- Language selection is done at build time with the `\memolang` macro (`en` or `ja`), passed through `latexmk -pretex`.
- Provide bilingual text with the macros `\en{...}` and `\ja{...}` (or the `bilingual` environment, if defined). Text that is identical in both languages needs no switch.
- Equations are written once, outside any language switch, so both languages share identical math.
- Figures are shared. If a figure contains text labels, either make the labels language-neutral (symbols, axes with math only) or generate one figure per language and reference it through the same macro.
- Use `luatexja` / `luatexja-preset` for Japanese typesetting. Do not use `pdflatex` or `platex`.
- Keep package use minimal and explicit; declare every package in the preamble.

### Slides

- **One generator, two languages.** `slides/build_slides.py` is the only generator; `--lang en|ja` selects the output. Do not write separate scripts per language.
- Do not hand-edit generated `.pptx` files.
- Slide content must be derived from the LaTeX document, not written twice. The planned mechanism is that LaTeX slide-marker macros write structured entries (titles, bullet points, math strings, figure paths, speaker notes) for each language to an auxiliary JSON file during compilation, and the Python builder reads that JSON. If that mechanism is not yet implemented, say so explicitly rather than duplicating content silently.
- **Equations are native PowerPoint equations (OMML).** The pipeline is LaTeX math string → MathML (`latex2mathml`) → OMML (`mathml2omml.convert`, from the `mathml2omml-as` package) → inserted into the slide XML with `lxml`, because `python-pptx` has no equation API. Do not fall back to rendered images or mathtext unless the user asks for them.
- Each slide should carry a concise title and a small number of points. Put longer explanation in speaker notes.
- Japanese decks must set the East Asian font attribute on text runs so that the text renders in a Japanese font.

## Python code style

The global Python rules in `~/.claude/CLAUDE.md` apply. In summary, for all Python files except Jupytext-paired notebooks:

- Format with `black` (line length 120) and sort imports with `isort`.
- Google-style docstrings and type hints on all functions and classes.
- `pathlib.Path` for file operations; f-strings for formatting.
- No bare `except:`; no mutable default arguments; no global mutable state.
- `@dataclass` for fixed-field data; convert JSON/YAML to dataclasses early.
- `logging` with explicit levels; no debug `print` in library code.
- Name constants in `SCREAMING_SNAKE_CASE`; no magic numbers.
- Always write a decimal point in floating-point literals (`1.0`, not `1`).
- Comments and docstrings in English only.

Notebook `.py` files (Jupytext pairs) are exempt from the formatting rules above, but keep them readable, keep comments in English, and avoid hard-coded absolute paths.

## Commit messages

- Use [Conventional Commits](https://www.conventionalcommits.org/): `<type>(<scope>): <short summary>`.
- Write commit messages entirely in English.
- Common scopes: `notebooks`, `docs`, `slides`, `env`, `repo`.
- Commit only when the user explicitly asks (see Git rules).

## Documentation

- Write documentation, code comments, and display strings in US English, except for the Japanese versions of LaTeX and slide content.

## Licensing

The repository is split by material type. The full policy is in [LICENSING.md](LICENSING.md).

- **Source code** (Python, notebook `.py` files, `environment.yml`, build scripts): GPL-3.0-or-later.
- **Documents, slides, figures, and narrative text in notebooks**: CC BY 4.0.

Rules for agents:

- Every new source code file must start with the GPL header (copyright, `SPDX-License-Identifier: GPL-3.0-or-later`, and the warranty disclaimer). Use the exact form in LICENSING.md, with the comment syntax of the file type and the file path on the first line.
- Every new LaTeX source must carry the CC BY 4.0 header (`SPDX-License-Identifier: CC-BY-4.0`) in `%` comments.
- Every new Markdown document must end with a `## License` section that states CC BY 4.0 and links to LICENSING.md.
- Code snippets inside documents are GPL-3.0-or-later.
- Do not add a file of unknown or incompatible license. Third-party code must be checked and attributed.
- Do not change the license texts (`LICENSE`, `LICENSES/CC-BY-4.0.txt`) or the copyright line without the user's instruction.
- Headers and license sections are part of a change. Do not leave them out to save space.

## License

This document is licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Code snippets are GPL-3.0-or-later. See [LICENSING.md](LICENSING.md).
