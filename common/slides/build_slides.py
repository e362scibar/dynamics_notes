# common/slides/build_slides.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Build the English or Japanese PowerPoint deck of one topic from its LaTeX slide markers.

The LaTeX build of ``topics/<topic>/main.tex`` writes ``build/<topic>/main-<lang>.slides``. This script reads that file
and writes ``build/<topic>/slides-<lang>.pptx``. Equations are inserted as native PowerPoint equations (OMML).

Usage (from the repository root)::

    python common/slides/build_slides.py --topic curvilinear_coordinates --lang en
"""

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from dynamics_common import Block, BlockKind, Slide, latex_to_omml, read_slide_aux
from lxml import etree
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.shapes.shapetree import SlideShapes
from pptx.slide import Slide as PptxSlide
from pptx.text.text import _Run
from pptx.util import Inches, Pt

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPICS_DIR_NAME = "topics"
BUILD_DIR_NAME = "build"
LANGUAGES = ("en", "ja")
JAPANESE_LANGUAGE = "ja"

SLIDE_WIDTH_IN = 13.333
SLIDE_HEIGHT_IN = 7.5
TITLE_ONLY_LAYOUT_INDEX = 5
BODY_LEFT_IN = 0.8
BODY_WIDTH_IN = 11.7
BODY_TOP_IN = 1.8
POINT_HEIGHT_IN = 0.7
MATH_HEIGHT_IN = 1.0
FIGURE_HEIGHT_IN = 3.6
BLOCK_GAP_IN = 0.2
POINT_FONT_SIZE_PT = 20
JAPANESE_EAST_ASIAN_FONT = "Hiragino Sans"
BULLET_PREFIX = "• "

MC_NAMESPACE = "http://schemas.openxmlformats.org/markup-compatibility/2006"
DRAWING_NAMESPACE = "http://schemas.openxmlformats.org/drawingml/2006/main"
A14_NAMESPACE = "http://schemas.microsoft.com/office/drawing/2010/main"


@dataclass(frozen=True)
class DeckPaths:
    """Input and output paths for one topic and language.

    Attributes:
        topic_dir: Directory ``topics/<topic>/``, used to resolve figure paths.
        aux: Slide auxiliary file written by the LaTeX build.
        output: Output ``.pptx`` path.
    """

    topic_dir: Path
    aux: Path
    output: Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list without the program name. ``None`` reads ``sys.argv``.

    Returns:
        The parsed arguments with ``topic`` and ``lang`` attributes.
    """
    parser = argparse.ArgumentParser(description="Build a PowerPoint deck from the LaTeX slide markers of one topic.")
    parser.add_argument(
        "--topic", required=True, help="Topic directory name under topics/, e.g. curvilinear_coordinates."
    )
    parser.add_argument("--lang", required=True, choices=LANGUAGES, help="Deck language.")
    return parser.parse_args(argv)


def deck_paths(topic: str, lang: str, repo_root: Path = REPO_ROOT) -> DeckPaths:
    """Return the input and output paths for one topic and language.

    Args:
        topic: Topic directory name.
        lang: Deck language, ``en`` or ``ja``.
        repo_root: Repository root directory.

    Returns:
        The :class:`DeckPaths` for the topic and language.
    """
    build_dir = repo_root / BUILD_DIR_NAME / topic
    return DeckPaths(
        topic_dir=repo_root / TOPICS_DIR_NAME / topic,
        aux=build_dir / f"main-{lang}.slides",
        output=build_dir / f"slides-{lang}.pptx",
    )


def _style_run(run: _Run, lang: str) -> None:
    """Set the East Asian font on a run when the deck is Japanese.

    Args:
        run: Text run to style.
        lang: Deck language. Only ``ja`` changes the run.
    """
    if lang != JAPANESE_LANGUAGE:
        return
    run_properties = run._r.get_or_add_rPr()
    run_properties.append(run_properties.makeelement(qn("a:ea"), {"typeface": JAPANESE_EAST_ASIAN_FONT}))


def _append_omml(paragraph_element: etree._Element, latex: str) -> None:
    """Append a native equation to a paragraph, with the LaTeX source as the fallback text.

    The equation is wrapped in ``mc:AlternateContent`` with an ``a14:m`` choice, the form PowerPoint uses for
    equations. The fallback is the plain LaTeX string for applications that do not read OMML.

    Args:
        paragraph_element: The ``a:p`` element that receives the equation.
        latex: LaTeX math source of the equation.

    Raises:
        ValueError: If the LaTeX cannot be converted (see :func:`dynamics_common.latex_to_omml`).
    """
    omml = latex_to_omml(latex)
    fragment = (
        f'<mc:AlternateContent xmlns:mc="{MC_NAMESPACE}" xmlns:a="{DRAWING_NAMESPACE}">'
        f'<mc:Choice xmlns:a14="{A14_NAMESPACE}" Requires="a14"><a14:m>{omml}</a14:m></mc:Choice>'
        f"<mc:Fallback><a:r><a:t>{escape(latex)}</a:t></a:r></mc:Fallback>"
        "</mc:AlternateContent>"
    )
    element = etree.fromstring(fragment)
    end_paragraph = paragraph_element.find(qn("a:endParaRPr"))
    if end_paragraph is None:
        paragraph_element.append(element)
    else:
        end_paragraph.addprevious(element)


def _add_point(shapes: SlideShapes, text: str, lang: str, top: float) -> float:
    """Add a bulleted text line and return the top position of the next block.

    Args:
        shapes: Shape collection of the slide.
        text: Point text.
        lang: Deck language, used for the font.
        top: Top position in inches.

    Returns:
        The top position of the next block in inches.
    """
    box = shapes.add_textbox(Inches(BODY_LEFT_IN), Inches(top), Inches(BODY_WIDTH_IN), Inches(POINT_HEIGHT_IN))
    frame = box.text_frame
    frame.word_wrap = True
    run = frame.paragraphs[0].add_run()
    run.text = BULLET_PREFIX + text
    run.font.size = Pt(POINT_FONT_SIZE_PT)
    _style_run(run, lang)
    return top + POINT_HEIGHT_IN + BLOCK_GAP_IN


def _add_math(shapes: SlideShapes, latex: str, top: float) -> float:
    """Add a native equation in a text box and return the top position of the next block.

    Args:
        shapes: Shape collection of the slide.
        latex: LaTeX math source.
        top: Top position in inches.

    Returns:
        The top position of the next block in inches.
    """
    box = shapes.add_textbox(Inches(BODY_LEFT_IN), Inches(top), Inches(BODY_WIDTH_IN), Inches(MATH_HEIGHT_IN))
    _append_omml(box.text_frame.paragraphs[0]._p, latex)
    return top + MATH_HEIGHT_IN + BLOCK_GAP_IN


def _add_figure(shapes: SlideShapes, relative_path: str, topic_dir: Path, top: float) -> float:
    """Add a picture from the topic directory and return the top position of the next block.

    Args:
        shapes: Shape collection of the slide.
        relative_path: Figure path relative to the topic directory.
        topic_dir: Topic directory.
        top: Top position in inches.

    Returns:
        The top position of the next block in inches.

    Raises:
        FileNotFoundError: If the figure does not exist.
    """
    path = topic_dir / relative_path
    if not path.is_file():
        logger.error("Figure not found: path=%s", path)
        raise FileNotFoundError(path)
    shapes.add_picture(str(path), Inches(BODY_LEFT_IN), Inches(top), height=Inches(FIGURE_HEIGHT_IN))
    return top + FIGURE_HEIGHT_IN + BLOCK_GAP_IN


def _add_block(shapes: SlideShapes, block: Block, lang: str, topic_dir: Path, top: float) -> float:
    """Add one content block and return the top position of the next block.

    Args:
        shapes: Shape collection of the slide.
        block: Block to add.
        lang: Deck language.
        topic_dir: Topic directory, used for figures.
        top: Top position in inches.

    Returns:
        The top position of the next block in inches.
    """
    if block.kind is BlockKind.POINT:
        return _add_point(shapes, block.text, lang, top)
    if block.kind is BlockKind.MATH:
        return _add_math(shapes, block.text, top)
    return _add_figure(shapes, block.text, topic_dir, top)


def _add_slide(presentation: Presentation, slide: Slide, lang: str, topic_dir: Path) -> None:
    """Add one slide with its title, blocks, and speaker notes.

    Args:
        presentation: Presentation to add the slide to.
        slide: Slide content read from the auxiliary file.
        lang: Deck language.
        topic_dir: Topic directory, used for figures.
    """
    layout = presentation.slide_layouts[TITLE_ONLY_LAYOUT_INDEX]
    pptx_slide: PptxSlide = presentation.slides.add_slide(layout)
    title_frame = pptx_slide.shapes.title.text_frame
    title_frame.text = slide.title
    for run in title_frame.paragraphs[0].runs:
        _style_run(run, lang)
    top = BODY_TOP_IN
    for block in slide.blocks:
        top = _add_block(pptx_slide.shapes, block, lang, topic_dir, top)
    if slide.notes:
        pptx_slide.notes_slide.notes_text_frame.text = "\n".join(slide.notes)


def build_deck(slides: Sequence[Slide], lang: str, topic_dir: Path) -> Presentation:
    """Build a presentation from the slides of one topic.

    Args:
        slides: Slides in document order.
        lang: Deck language, ``en`` or ``ja``.
        topic_dir: Topic directory, used to resolve figure paths.

    Returns:
        The built presentation. The caller saves it.
    """
    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_WIDTH_IN)
    presentation.slide_height = Inches(SLIDE_HEIGHT_IN)
    for slide in slides:
        _add_slide(presentation, slide, lang, topic_dir)
    return presentation


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deck build for the arguments given on the command line.

    Args:
        argv: Argument list without the program name. ``None`` reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 on a missing input or a malformed slide file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    paths = deck_paths(args.topic, args.lang)
    if not paths.topic_dir.is_dir():
        logger.error("Topic directory not found: topic=%s path=%s", args.topic, paths.topic_dir)
        return 1
    try:
        slides = read_slide_aux(paths.aux)
    except (OSError, ValueError) as exc:
        logger.error("Cannot read slide entries: aux=%s error=%s", paths.aux, exc)
        return 1
    try:
        presentation = build_deck(slides, args.lang, paths.topic_dir)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot build deck: topic=%s lang=%s error=%s", args.topic, args.lang, exc)
        return 1
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(paths.output)
    logger.info("Deck saved: topic=%s lang=%s slides=%d output=%s", args.topic, args.lang, len(slides), paths.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
