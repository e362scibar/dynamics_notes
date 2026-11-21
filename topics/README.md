# topics/

One directory per topic. Everything about a topic lives in its directory.

## Layout of a topic directory

```
topics/<topic>/
  <topic>.py     # Jupytext notebook (English only). Source of truth; .ipynb is generated.
  main.tex       # Bilingual LaTeX document. Language is selected at build time with \memolang.
  figures/       # Figures used only by this topic.
```

- The directory name is a snake_case ASCII slug, for example `twiss_optics`. It does not contain the document number.
- Build outputs go to `build/<topic>/` at the repository root. That directory is not tracked.

## Index

| Number | Topic | Notebook | LaTeX | Slides | Status |
|---|---|---|---|---|---|
| 2026-001 | [curvilinear_coordinates](curvilinear_coordinates/) | [curvilinear_coordinates.py](curvilinear_coordinates/curvilinear_coordinates.py) | [main.tex](curvilinear_coordinates/main.tex) (en, ja) | `build/curvilinear_coordinates/2026-001-slides-{en,ja}.pptx` via `common/slides/build_slides.py` | Draft |

The number is the document number `YYYY-NNN` registered in [common/documents.toml](../common/documents.toml). It is used to cite a topic from other documents (`\cite{2026-001}` in LaTeX, `[2026-001]` in notebooks) and in the names of the output files, for example `build/curvilinear_coordinates/2026-001-en.pdf`.

## Adding a topic

1. Allocate the document number with `python common/python/dynamics_common/documents.py --next`, add it to [common/documents.toml](../common/documents.toml), and regenerate [common/documents.bib](../common/documents.bib) with `python common/python/dynamics_common/documents.py --write`. The serial number restarts at `001` every year, and a number is never reused.
2. Create `topics/<topic>/` with `<topic>.py`, `main.tex`, and `figures/`. Put `\setmemonumber{YYYY-NNN}` right after the `\input` of the preamble in `main.tex`, and write `Memo YYYY-NNN` below the title in the notebook.
3. Add the license headers described in [LICENSING.md](../LICENSING.md).
4. Cite the references the topic uses with `\cite{key}` in `main.tex` and `[key]` in the notebook. Keys come from [common/references/references.bib](../common/references/references.bib).
5. Add a row to the index above in the same change.
6. Write the Japanese text in `main.tex` in the plain form (だ・である調). Do not use the polite form (です・ます調). See [AGENTS.md](../AGENTS.md).
7. Use the Japanese technical terms from [common/glossary.toml](../common/glossary.toml). If a term is missing, add it there and regenerate [common/glossary.md](../common/glossary.md) with `python common/python/dynamics_common/glossary.py --write`.

## License

Notebook `.py` files are source code and are licensed under [GPL-3.0-or-later](../LICENSE). `main.tex` and figures are licensed under [CC BY 4.0](../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../LICENSING.md).
