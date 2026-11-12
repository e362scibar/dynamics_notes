# common/tex/

Shared LaTeX code for all topics: the preamble, the `\memolang` language switch, and the bilingual macros `\en{...}` and `\ja{...}`. Each `topics/<topic>/main.tex` loads these files with `\input`, so the preamble is not duplicated.

## Build a topic

Run from the repository root. Paths in `\input` are relative to the root.

```bash
# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=main-en -outdir=build/<topic> topics/<topic>/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=main-ja -outdir=build/<topic> topics/<topic>/main.tex

# Clean
latexmk -C -outdir=build/<topic> topics/<topic>/main.tex
```

PDFs are written to `build/<topic>/`. The Japanese build uses `luatexja`; do not use `pdflatex` or `platex`.

## Bibliography

The shared preamble loads `biblatex` with `biber` and the database `common/references/references.bib`. Cite with `\cite{key}` in `main.tex`. Do not write bibliography entries inside topic files. `latexmk` runs `biber` automatically. See [common/references/README.md](../references/README.md) for the reference policy.

## License

The files in this directory are LaTeX sources and are licensed under [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../../LICENSING.md).
