# docs/

Bilingual LaTeX document. `main.tex` contains both English and Japanese text; the language is chosen at build time.

## Build

Requires TeX Live or MacTeX with `latexmk`, `lualatex`, and `luatexja`.

```bash
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=main-en main.tex
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=main-ja main.tex
```

Outputs go to `build/`, which is not tracked.

## Conventions

- Prose: `\en{...}` and `\ja{...}` macros. Text identical in both languages needs no switch.
- Equations: written once and shared.
- Figures: shared; text inside figures is language-neutral or generated per language.
- Japanese typesetting uses `luatexja`. Do not use `pdflatex` or `platex`.

## License

This document and its LaTeX sources are licensed under [CC BY 4.0](../LICENSES/CC-BY-4.0.txt). Code snippets are GPL-3.0-or-later. See [LICENSING.md](../LICENSING.md).
