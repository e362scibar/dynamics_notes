# dynamics_notes

Personal memo notes on **high-energy electron beam dynamics in accelerators**, organized by topic. Each topic is kept as

- a Jupyter notebook (computations and derivations, with sympy; English only),
- a LaTeX document (typeset to PDF; English and Japanese), and
- PowerPoint slides (generated from the LaTeX content; English and Japanese).

Math and figures are shared between the two languages; only prose and labels differ.

## Layout

| Path | Contents |
|---|---|
| `topics/<topic>/` | One directory per topic: `<topic>.py` (notebook), `main.tex` (LaTeX), `figures/` |
| `topics/README.md` | Topic index and conventions |
| `common/tex/` | Shared LaTeX preamble and bilingual macros |
| `common/slides/` | Shared `python-pptx` generator, run once per topic and language |
| `common/python/` | Shared Python package `dynamics_common` (sympy helpers, utilities) |
| `common/references/` | Reference index (`README.md`) and BibTeX database (`references.bib`). PDFs are local only, not committed |
| `build/<topic>/` | Generated PDFs and slide decks (not tracked) |
| `environment.yml` | Conda environment `dynamics_notes` (Python 3.14) |
| `AGENTS.md` | Working rules for AI coding agents |

## Setup

```bash
conda env create -f environment.yml
conda activate dynamics_notes
```

The LaTeX build also needs a TeX distribution (TeX Live or MacTeX) providing `latexmk`, `lualatex`, and `luatexja`.

Enable the notebook output filter once per clone (`.gitattributes` already routes `*.ipynb` through it):

```bash
nbstripout --install
```

## Workflow

Replace `<topic>` with the topic directory name, for example `twiss_optics`. Run commands from the repository root.

### Notebooks

1. Edit `topics/<topic>/<topic>.py`.
2. Sync to the notebook: `jupytext --sync topics/<topic>/<topic>.py`.
3. Commit (only when asked). `nbstripout` removes outputs from the `.ipynb` at commit time.

Symbolic work uses sympy.

### LaTeX document

```bash
# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=main-en -outdir=build/<topic> topics/<topic>/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=main-ja -outdir=build/<topic> topics/<topic>/main.tex
```

PDFs are written to `build/<topic>/`.

### Slides

```bash
python common/slides/build_slides.py --topic <topic> --lang en
python common/slides/build_slides.py --topic <topic> --lang ja
```

Decks are written to `build/<topic>/`. Slide content comes from the topic's LaTeX document, so update the document first. Equations are inserted as native PowerPoint equations.

## Topics

The index is in [topics/README.md](topics/README.md). Planned topics are placeholders:

- Transverse linear optics: transfer matrices, Twiss parameters, betatron tune
- Dispersion and chromaticity
- Closed-orbit distortion and orbit correction
- Synchrotron radiation and radiation damping
- Nonlinear dynamics: sextupoles, dynamic aperture

## 日本語概要

加速器における高エネルギー電子ビームのダイナミクスに関するメモである。トピックごとにディレクトリを分け、各トピックに Jupyter Notebook（sympy による式の導出を含む、英語のみ）、LaTeX 文書（英語・日本語）、PowerPoint スライド（英語・日本語）を置く。

- トピック: `topics/<topic>/`。一覧は [topics/README.md](topics/README.md) にある。
- Notebook: `topics/<topic>/<topic>.py`。英語のみである。`.py`（Jupytext）が原本で、コミット時に `nbstripout` で出力を削除する。
- LaTeX: `topics/<topic>/main.tex` 1 ファイルで両言語を扱い、`\memolang` でビルド時に言語を選択する（`latexmk` + `lualatex`）。共通のプリアンブルは `common/tex/` にある。
- スライド: `common/slides/build_slides.py --topic <topic> --lang en|ja` で英語版・日本語版を生成する。数式は PowerPoint のネイティブ数式（OMML）として埋め込む。
- 共通の Python コード: `common/python/` の `dynamics_common` パッケージにまとめ、トピックから import する。
- 参考文献: 索引は `common/references/README.md`、BibTeX は `common/references/references.bib` にある。PDF は手元にのみ置き、git には含めない。文書では `\cite{key}` や `[key]` で引用する。
- 日本語の文章は だ・である調で書く。です・ます調は使わない（詳細は [AGENTS.md](AGENTS.md)）。
- 生成物は `build/<topic>/` に出力され、git 管理しない。
- 環境: conda 環境 `dynamics_notes`（Python 3.14）。定義は `environment.yml` である。
- コミットは明示的に依頼された場合にのみ行う。

## License

- Source code (Python, notebook `.py` files, `environment.yml`, build scripts): [GPL-3.0-or-later](LICENSE).
- Documents, slides, figures, and narrative text in notebooks: [CC BY 4.0](LICENSES/CC-BY-4.0.txt).

Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center). See [LICENSING.md](LICENSING.md) for the full policy.

日本語: ソースコードは GPL-3.0-or-later、文書・スライド・図などは CC BY 4.0 である。詳細は [LICENSING.md](LICENSING.md) を参照すること。
