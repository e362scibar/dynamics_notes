# slides/

PowerPoint decks generated with `python-pptx` from the content of `docs/main.tex`. One shared generator, `build_slides.py`, builds both the English and the Japanese deck.

## Build

```bash
python slides/build_slides.py --lang en
python slides/build_slides.py --lang ja
```

Outputs go to `slides/build/`, which is not tracked.

## Design

- The LaTeX document is the single source of content. Slide-marker macros in the LaTeX source write structured entries for each language to an auxiliary JSON file during compilation; the builder reads that JSON.
- Equations: LaTeX math string → MathML (`latex2mathml`) → OMML (`mathml2omml.convert` from `mathml2omml-as`) → inserted into the slide XML with `lxml`. The result is native, editable PowerPoint equations.
- Each slide has a short title and a few points. Longer explanation goes into speaker notes.
- Japanese decks set the East Asian font on text runs.
- Comments and code are in English.
