# common/python/tests/test_slide_geometry.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :func:`dynamics_common.set_widescreen_size`."""

import pytest
from dynamics_common import WIDESCREEN_HEIGHT_IN, WIDESCREEN_WIDTH_IN, set_widescreen_size
from pptx import Presentation
from pptx.util import Inches

DEFAULT_WIDTH_IN = 10.0
DEFAULT_HEIGHT_IN = 7.5
TITLE_ONLY_LAYOUT_INDEX = 5


def _title_placeholder(presentation: Presentation):
    """Return the title placeholder of the Title Only layout."""
    layout = presentation.slide_layouts[TITLE_ONLY_LAYOUT_INDEX]
    return next(shape for shape in layout.shapes if shape.name.startswith("Title"))


def test_sets_widescreen_slide_size() -> None:
    """The slide size becomes 16:9."""
    presentation = Presentation()
    set_widescreen_size(presentation)
    assert presentation.slide_width == Inches(WIDESCREEN_WIDTH_IN)
    assert presentation.slide_height == Inches(WIDESCREEN_HEIGHT_IN)


def test_scales_master_and_layout_geometry_horizontally() -> None:
    """Placeholders with an explicit transform are widened by the slide-width ratio."""
    presentation = Presentation()
    factor = WIDESCREEN_WIDTH_IN / DEFAULT_WIDTH_IN
    master_title = next(shape for shape in presentation.slide_master.shapes if shape.name.startswith("Title"))
    layout_title = _title_placeholder(presentation)
    master_left, master_width = master_title.left, master_title.width
    layout_left, layout_width = layout_title.left, layout_title.width

    set_widescreen_size(presentation)

    assert master_title.left == pytest.approx(master_left * factor)
    assert master_title.width == pytest.approx(master_width * factor)
    assert layout_title.left == pytest.approx(layout_left * factor)
    assert layout_title.width == pytest.approx(layout_width * factor)


def test_keeps_vertical_geometry() -> None:
    """Only the horizontal geometry changes, so the 7.5 in height is preserved."""
    presentation = Presentation()
    master_title = next(shape for shape in presentation.slide_master.shapes if shape.name.startswith("Title"))
    top, height = master_title.top, master_title.height

    set_widescreen_size(presentation)

    assert master_title.top == top
    assert master_title.height == height


def test_is_idempotent() -> None:
    """Calling the function twice does not scale the geometry twice."""
    presentation = Presentation()
    set_widescreen_size(presentation)
    layout_title = _title_placeholder(presentation)
    left, width = layout_title.left, layout_title.width

    set_widescreen_size(presentation)

    assert layout_title.left == left
    assert layout_title.width == width
