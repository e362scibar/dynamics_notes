# Licensing

This repository uses two licenses, split by the kind of material:

| Material | Files | License |
|---|---|---|
| Source code | Python scripts, notebook `.py` files, `environment.yml`, other programs and build scripts | [GNU GPL v3 or later](LICENSE) (`GPL-3.0-or-later`) |
| Documents, slides, figures | Markdown files, LaTeX sources (`topics/*/main.tex`, `common/tex/`), generated PDFs and slide decks, figures and other images, narrative text in notebooks | [Creative Commons Attribution 4.0 International](LICENSES/CC-BY-4.0.txt) (`CC-BY-4.0`) |

Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center), unless stated otherwise.

## Files

- [LICENSE](LICENSE): full text of GPL v3. Applies to all source code.
- [LICENSES/CC-BY-4.0.txt](LICENSES/CC-BY-4.0.txt): full legal code of CC BY 4.0. Applies to documents and figures. The canonical text, including the Creative Commons notice, is at <https://creativecommons.org/licenses/by/4.0/legalcode>.

## Source file headers

Every source code file carries a short header with the copyright, the SPDX identifier, and a warranty disclaimer. Use the following form, with the comment syntax of the file type:

```python
# topics/twiss_optics/twiss_optics.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.
```

For LaTeX sources, use `%` comments and `SPDX-License-Identifier: CC-BY-4.0`. Use `// ` for C or C++ and `#` for YAML or shell.

## Documents, slides, and figures

Documents are licensed under CC BY 4.0. Each Markdown file ends with a `License` section that states this. LaTeX sources carry a CC BY 4.0 header. Generated slide decks and PDFs are covered by CC BY 4.0. Figures that are generated from code carry CC BY 4.0 because they are images. The code that produces them remains GPL-3.0-or-later.

Code snippets that appear inside documents are provided under GPL-3.0-or-later.

## Configuration files

Pure configuration files, such as `.gitattributes` and `.gitignore`, carry no header. They contain no program logic. `environment.yml` is treated as source code and carries the GPL header.

## Third-party material

Third-party code and data keep their own licenses. Mention their origin and license in the file that uses them. Do not copy code into this repository without checking that its license is compatible with GPL-3.0-or-later.

Reference literature is kept under `common/references/`. Its PDFs are third-party works and are **not committed**, because their redistribution terms are not controlled by this repository. Only the index (`common/references/README.md`) and the BibTeX database (`common/references/references.bib`) are tracked. Cite the works instead of redistributing them.

## License

This document is licensed under [CC BY 4.0](LICENSES/CC-BY-4.0.txt). Code snippets are GPL-3.0-or-later.
