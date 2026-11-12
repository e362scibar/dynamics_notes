# common/python/dynamics_common/slide_geometry.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Set a presentation to the 16:9 widescreen size and rescale its template geometry.

The default template of ``python-pptx`` is 4:3 (10 x 7.5 in). Changing only ``slide_width`` and
``slide_height`` leaves the master and layout placeholders at their 4:3 positions, so the title and
other placeholders stay inside the left 10 in of a 13.333 in slide. This module widens the template
geometry by the same factor as the slide width.
"""

from pptx.oxml.ns import qn
from pptx.presentation import Presentation
from pptx.shapes.base import BaseShape
from pptx.util import Inches

WIDESCREEN_WIDTH_IN = 13.333
WIDESCREEN_HEIGHT_IN = 7.5

__all__ = ["WIDESCREEN_HEIGHT_IN", "WIDESCREEN_WIDTH_IN", "set_widescreen_size"]


def _scale_shape_xfrm(shape: BaseShape, factor: float) -> None:
    """Scale the horizontal offset and width of a shape that carries an explicit transform.

    Shapes without an explicit ``a:xfrm`` inherit their geometry from the master, so they are left
    untouched and follow the scaled master instead.

    Args:
        shape: Shape of the slide master or of a slide layout.
        factor: Horizontal scale factor.
    """
    shape_properties = shape._element.find(qn("p:spPr"))
    if shape_properties is None:
        return
    transform = shape_properties.find(qn("a:xfrm"))
    if transform is None:
        return
    offset = transform.find(qn("a:off"))
    extent = transform.find(qn("a:ext"))
    if offset is not None:
        offset.set("x", str(round(int(offset.get("x")) * factor)))
    if extent is not None:
        extent.set("cx", str(round(int(extent.get("cx")) * factor)))


def set_widescreen_size(presentation: Presentation) -> None:
    """Set the 16:9 slide size and widen the master and layout geometry to match.

    The horizontal scale factor is the ratio of the new slide width to the current one, so calling
    this function again on an already widescreen presentation changes nothing.

    Args:
        presentation: Presentation to resize in place.
    """
    current_width = presentation.slide_width
    target_width = Inches(WIDESCREEN_WIDTH_IN)
    factor = target_width / current_width
    presentation.slide_width = target_width
    presentation.slide_height = Inches(WIDESCREEN_HEIGHT_IN)
    if factor == 1.0:
        return
    for shape in presentation.slide_master.shapes:
        _scale_shape_xfrm(shape, factor)
    for layout in presentation.slide_layouts:
        for shape in layout.shapes:
            _scale_shape_xfrm(shape, factor)
