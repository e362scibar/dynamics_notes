# dynamics_memo

Personal memo notes on **high-energy electron beam dynamics in accelerators**, kept as

- Jupyter notebooks (computations and derivations, with sympy; English only),
- a LaTeX document (typeset to PDF; English and Japanese), and
- PowerPoint slides (generated from the LaTeX content; English and Japanese).

Math and figures are shared between the two languages; only prose and labels differ.

## Layout

| Path | Contents |
|---|---|
| `notebooks/` | Jupytext-paired notebooks, English only (`.py` is the source, `.ipynb` is generated) |
| `docs/` | Bilingual LaTeX document (`main.tex`), built with `latexmk` and `lualatex` |
| `slides/` | Shared `python-pptx` generator that builds English and Japanese decks from the LaTeX content |
| `environment.yml` | Conda environment `dynamics_memo` (Python 3.14) |
| `AGENTS.md` | Working rules for AI coding agents |

## Setup

```bash
conda env create -f environment.yml
conda activate dynamics_memo
```

The LaTeX build also needs a TeX distribution (TeX Live or MacTeX) providing `latexmk`, `lualatex`, and `luatexja`.

Enable the notebook output filter once per clone (`.gitattributes` already routes `*.ipynb` through it):

```bash
nbstripout --install
```

## Workflow

### Notebooks

1. Edit the `.py` file in `notebooks/`.
2. Sync to the notebook: `jupytext --sync notebooks/<topic>.py`.
3. Commit (only when asked). `nbstripout` removes outputs from the `.ipynb` at commit time.

Symbolic work uses sympy.

### LaTeX document

```bash
# English
latexmk -lualatex -pretex='\def\memolang{en}' -usepretex -jobname=main-en docs/main.tex

# Japanese
latexmk -lualatex -pretex='\def\memolang{ja}' -usepretex -jobname=main-ja docs/main.tex
```

PDFs are written to `docs/build/`.

### Slides

```bash
python slides/build_slides.py --lang en
python slides/build_slides.py --lang ja
```

Decks are written to `slides/build/`. Slide content comes from the LaTeX document, so update the document first. Equations are inserted as native PowerPoint equations.

## Planned topics

Topics are added as notebooks, sections of `docs/main.tex`, and slide sections together. The list below is a placeholder to be filled in.

- Transverse linear optics: transfer matrices, Twiss parameters, betatron tune
- Dispersion and chromaticity
- Closed-orbit distortion and orbit correction
- Synchrotron radiation and radiation damping
- Nonlinear dynamics: sextupoles, dynamic aperture

## 日本語概要

加速器における高エネルギー電子ビームのダイナミクスに関するメモです。Jupyter Notebook（sympy による式の導出を含む、英語のみ）、LaTeX 文書（英語・日本語）、PowerPoint スライド（英語・日本語）の三形式で管理します。

- Notebook: `notebooks/`。英語のみです。`.py`（Jupytext）が原本で、コミット時に `nbstripout` で出力を削除します。
- LaTeX: `docs/main.tex` 1 ファイルで両言語を扱い、`\memolang` でビルド時に言語を選択します（`latexmk` + `lualatex`）。
- スライド: `slides/` の共通スクリプト `build_slides.py` で英語版・日本語版を生成します。数式は PowerPoint のネイティブ数式（OMML）として埋め込みます。
- 環境: conda 環境 `dynamics_memo`（Python 3.14）。定義は `environment.yml` です。
- コミットは明示的に依頼された場合にのみ行います。
