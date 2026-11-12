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
| (none yet) | | | | |

## Adding a topic

1. Create `topics/<topic>/` with `<topic>.py`, `main.tex`, and `figures/`.
2. Add the license headers described in [LICENSING.md](../LICENSING.md).
3. Add a row to the index above in the same change.

## License

Notebook `.py` files are source code and are licensed under [GPL-3.0-or-later](../LICENSE). `main.tex` and figures are licensed under [CC BY 4.0](../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../LICENSING.md).
