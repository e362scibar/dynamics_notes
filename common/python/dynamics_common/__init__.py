# common/python/dynamics_common/__init__.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Shared helpers for the dynamics_notes repository."""

from dynamics_common.glossary import (
    GLOSSARY_MARKDOWN_PATH,
    GLOSSARY_PATH,
    GlossaryTerm,
    Violation,
    find_forbidden_terms,
    load_glossary,
    render_markdown,
    scan_latex_file,
)
from dynamics_common.omml import latex_to_omml
from dynamics_common.slide_aux import Block, BlockKind, Slide, parse_slide_aux, read_slide_aux
from dynamics_common.slide_geometry import WIDESCREEN_HEIGHT_IN, WIDESCREEN_WIDTH_IN, set_widescreen_size

__all__ = [
    "GLOSSARY_MARKDOWN_PATH",
    "GLOSSARY_PATH",
    "WIDESCREEN_HEIGHT_IN",
    "WIDESCREEN_WIDTH_IN",
    "Block",
    "BlockKind",
    "GlossaryTerm",
    "Slide",
    "Violation",
    "find_forbidden_terms",
    "latex_to_omml",
    "load_glossary",
    "parse_slide_aux",
    "read_slide_aux",
    "render_markdown",
    "scan_latex_file",
    "set_widescreen_size",
]
