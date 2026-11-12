# topics/

One directory per topic. Everything about a topic lives in its directory.

## Layout of a topic directory

```
topics/<topic>/
  <topic>.py     # Jupytext notebook (English only). Source of truth; .ipynb is generated.
  main.tex       # Bilingual LaTeX document. Language is selected at build time with \memolang.
  figures/       # Figures used only by this topic.
```

- The directory name is a snake_case ASCII slug, for example `twiss_optics`.
- Build outputs go to `build/<topic>/` at the repository root. That directory is not tracked.

## Index

| Topic | Notebook | LaTeX | Slides | Status |
|---|---|---|---|---|
| [curvilinear_coordinates](curvilinear_coordinates/) | [curvilinear_coordinates.py](curvilinear_coordinates/curvilinear_coordinates.py) | [main.tex](curvilinear_coordinates/main.tex) (en, ja) | `build/curvilinear_coordinates/slides-{en,ja}.pptx` via `common/slides/build_slides.py` | Draft |

## Adding a topic

1. Create `topics/<topic>/` with `<topic>.py`, `main.tex`, and `figures/`.
2. Add the license headers described in [LICENSING.md](../LICENSING.md).
3. Cite the references the topic uses with `\cite{key}` in `main.tex` and `[key]` in the notebook. Keys come from [common/references/references.bib](../common/references/references.bib).
4. Add a row to the index above in the same change.
5. Write the Japanese text in `main.tex` in the plain form (だ・である調). Do not use the polite form (です・ます調). See [AGENTS.md](../AGENTS.md).

## License

Notebook `.py` files are source code and are licensed under [GPL-3.0-or-later](../LICENSE). `main.tex` and figures are licensed under [CC BY 4.0](../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../LICENSING.md).
