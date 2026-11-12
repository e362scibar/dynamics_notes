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

- The topic's LaTeX document is the single source of content. Slide-marker macros in the LaTeX source write structured entries for each language to an auxiliary JSON file during compilation; the builder reads that JSON.
- Equations: LaTeX math string → MathML (`latex2mathml`) → OMML (`mathml2omml.convert` from `mathml2omml-as`) → inserted into the slide XML with `lxml`. The result is native, editable PowerPoint equations.
- Each slide has a short title and a few points. Longer explanation goes into speaker notes.
- Japanese decks set the East Asian font on text runs.
- Reusable helpers (for example the OMML conversion or the JSON reader) belong in [common/python/](../python/README.md), not in this script.
- Comments and code are in English.

## License

The generator scripts in this directory are source code and are licensed under [GPL-3.0-or-later](../../LICENSE). Generated slide decks are documents and are licensed under [CC BY 4.0](../../LICENSES/CC-BY-4.0.txt). See [LICENSING.md](../../LICENSING.md).
