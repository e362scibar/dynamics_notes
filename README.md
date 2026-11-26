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
| `common/documents.toml` | Document numbers `YYYY-NNN` of the topics. `common/documents.bib` is generated from it |
| `topics/README.md` | Topic index and conventions |
| `common/tex/` | Shared LaTeX preamble and bilingual macros |
| `common/slides/` | Shared `python-pptx` generator, run once per topic and language |
| `common/python/` | Shared Python package `dynamics_common` (sympy helpers, utilities) |
| `common/references/` | Reference index (`README.md`) and BibTeX database (`references.bib`). PDFs are local only, not committed |
| `build/<topic>/` | Generated PDFs and slide decks, named by document number, for example `2026-001-en.pdf` (not tracked) |
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

Replace `<topic>` with the topic directory name, for example `twiss_optics`. Run commands from the repository root. Every topic has a document number `YYYY-NNN` (see [common/documents.toml](common/documents.toml)); it is used to cite the topic from other documents and in the output file names.

### Notebooks

1. Edit `topics/<topic>/<topic>.py`.
2. Sync to the notebook: `jupytext --sync topics/<topic>/<topic>.py`.
3. Commit (only when asked). `nbstripout` removes outputs from the `.ipynb` at commit time.

Symbolic work uses sympy.

### LaTeX document

```bash
NUM=$(python common/python/dynamics_common/documents.py --number <topic>)

# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=$NUM-en -outdir=build/<topic> topics/<topic>/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=$NUM-ja -outdir=build/<topic> topics/<topic>/main.tex
```

PDFs are written to `build/<topic>/$NUM-en.pdf` and `build/<topic>/$NUM-ja.pdf`.

### Slides

```bash
python common/slides/build_slides.py --topic <topic> --lang en
python common/slides/build_slides.py --topic <topic> --lang ja
```

Decks are written to `build/<topic>/$NUM-slides-en.pptx` and `build/<topic>/$NUM-slides-ja.pptx`. Slide content comes from the topic's LaTeX document, so update the document first. Equations are inserted as native PowerPoint equations.

### Build every topic

```bash
for topic in topics/*/; do
  topic=$(basename "$topic")
  [ -f "topics/$topic/main.tex" ] || continue
  NUM=$(python common/python/dynamics_common/documents.py --number "$topic")
  for lang in en ja; do
    latexmk -lualatex -pretex="\def\memolang{$lang}" -usepretex \
      -jobname="$NUM-$lang" -outdir="build/$topic" "topics/$topic/main.tex"
    python common/slides/build_slides.py --topic "$topic" --lang "$lang"
  done
done
```

The LaTeX build of a language must run before its slide build, because it writes `build/<topic>/<number>-<lang>.slides`.

## Topics

The index is in [topics/README.md](topics/README.md). Current topics:

- 2026-001 `curvilinear_coordinates`: canonical equations of motion in curvilinear (Frenet-Serret) coordinates
- 2026-002 `transfer_matrix`: linear transfer matrices of a drift, a sector dipole, and a quadrupole
- 2026-003 `second_order_eom`: canonical equations of motion expanded to second order
- 2026-004 `second_order_map`: second-order transfer maps of a drift, a sector dipole, and a quadrupole
- 2026-005 `linear_dispersion`: time evolution of the dispersion function through the three basic elements

Planned topics are placeholders:

- Transverse linear optics: Twiss parameters, betatron tune
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
- 文書番号: 各トピックに `YYYY-NNN`（例: `2026-001`）の文書番号を付ける。年が変わると連番は `001` から振り直す。番号は変更も再利用もしない。台帳は `common/documents.toml` で、他の文書からは LaTeX で `\cite{2026-001}`、ノートブックで `[2026-001]` と引用する。
- 生成物は `build/<topic>/` に出力され、git 管理しない。ファイル名には文書番号を含める（例: `2026-001-ja.pdf`）。
- 環境: conda 環境 `dynamics_notes`（Python 3.14）。定義は `environment.yml` である。
- コミットは明示的に依頼された場合にのみ行う。

## License

- Source code (Python, notebook `.py` files, `environment.yml`, build scripts): [GPL-3.0-or-later](LICENSE).
- Documents, slides, figures, and narrative text in notebooks: [CC BY 4.0](LICENSES/CC-BY-4.0.txt).

Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center). See [LICENSING.md](LICENSING.md) for the full policy.

日本語: ソースコードは GPL-3.0-or-later、文書・スライド・図などは CC BY 4.0 である。詳細は [LICENSING.md](LICENSING.md) を参照すること。
