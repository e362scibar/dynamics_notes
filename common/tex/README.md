# common/tex/

Shared LaTeX code for all topics: the preamble, the `\memolang` language switch, and the bilingual macros `\en{...}` and `\ja{...}`. Each `topics/<topic>/main.tex` loads these files with `\input`, so the preamble is not duplicated.

## Build a topic

Run from the repository root. Paths in `\input` are relative to the root.

```bash
NUM=$(python common/python/dynamics_common/documents.py --number <topic>)

# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=$NUM-en -outdir=build/<topic> topics/<topic>/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=$NUM-ja -outdir=build/<topic> topics/<topic>/main.tex

# Clean
latexmk -C -jobname=$NUM-en -outdir=build/<topic> topics/<topic>/main.tex
latexmk -C -jobname=$NUM-ja -outdir=build/<topic> topics/<topic>/main.tex
```

PDFs are written to `build/<topic>/<number>-en.pdf` and `build/<topic>/<number>-ja.pdf`, where `<number>` is the document number of the topic. The Japanese build uses `luatexja`; do not use `pdflatex` or `platex`.

## Japanese text style

Write the Japanese text inside `\ja{...}` in the plain form (だ・である調). Do not use the polite form (です・ます調). For example, write 「求める。」「得られる。」「確認すること。」 and not 「求めます。」「得られます。」「確認してください。」. Keep the English text in `\en{...}` as it is.

## Document number

Each `main.tex` declares its document number with `\setmemonumber{YYYY-NNN}` right after the `\input` of the preamble. The preamble shows `Memo YYYY-NNN` in the page header, and `\memonumber` gives the number to the title (`\date{Memo~\memonumber, YYYY}`). The numbers are registered in `common/documents.toml`; see [AGENTS.md](../../AGENTS.md).

## Bibliography

The shared preamble loads `biblatex` with `biber` and two databases: `common/references/references.bib` (literature) and `common/documents.bib` (document numbers, generated from `common/documents.toml`). Cite with `\cite{key}` in `main.tex`, and cite another topic by its number, for example `\cite{2026-001}`. Do not write bibliography entries inside topic files. `latexmk` runs `biber` automatically. See [common/references/README.md](../references/README.md) for the reference policy.

## License

The files in this directory are LaTeX sources and are licensed under [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../../LICENSING.md).
