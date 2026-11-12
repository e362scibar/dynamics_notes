# common/slides/

Shared PowerPoint generator. `build_slides.py` builds the English and the Japanese deck for one topic from that topic's LaTeX document (`topics/<topic>/main.tex`), using `python-pptx`.

## Build

Run from the repository root.

```bash
python common/slides/build_slides.py --topic <topic> --lang en
python common/slides/build_slides.py --topic <topic> --lang ja
```

Outputs go to `build/<topic>/slides-<lang>.pptx`, which is not tracked.

## Design

- The topic's LaTeX document is the single source of content. The slide-marker macros in `common/tex/slides.tex` write one `kind:payload` line per entry to `build/<topic>/main-<lang>.slides` during compilation. The kinds are `title`, `point`, `math`, `figure`, and `note`. The builder reads that file with `dynamics_common.read_slide_aux`. The format is line based instead of JSON because TeX cannot escape JSON strings reliably.
- Equations: LaTeX math string → MathML (`latex2mathml`) → OMML (`mathml2omml.convert` from `mathml2omml-as`) → inserted into the slide XML with `lxml`, using `dynamics_common.latex_to_omml`. The result is native, editable PowerPoint equations.
- Before the first build, install the shared package with `pip install -e common/python` (already listed in `environment.yml`).
- Each slide has a short title and a few points. Longer explanation goes into speaker notes.
- The deck is 16:9 (13.333 x 7.5 in). The default `python-pptx` template is 4:3, so `dynamics_common.set_widescreen_size` widens the master and layout placeholders by the slide-width ratio before any slide is added. Do not set `slide_width` alone: the placeholders would stay at their 4:3 positions. Content boxes use the full 16:9 width (`MARGIN_LEFT_IN`/`MARGIN_RIGHT_IN`), and figures are scaled down when they would exceed the body width.
- Japanese decks set the East Asian font on text runs.
- Japanese slide text uses the plain form (だ・である調), the same as the LaTeX source. Do not use the polite form (です・ます調).
- Reusable helpers (for example the OMML conversion or the slide-entry reader) belong in [common/python/](../python/README.md), not in this script.
- Comments and code are in English.

## License

The generator scripts in this directory are source code and are licensed under [GPL-3.0-or-later](../../LICENSE). Generated slide decks are documents and are licensed under [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../../LICENSING.md).
