# common/python/dynamics_common/__init__.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Shared helpers for the dynamics_notes repository."""

from dynamics_common.omml import latex_to_omml
from dynamics_common.slide_aux import Block, BlockKind, Slide, parse_slide_aux, read_slide_aux
from dynamics_common.slide_geometry import WIDESCREEN_HEIGHT_IN, WIDESCREEN_WIDTH_IN, set_widescreen_size

__all__ = [
    "Block",
    "BlockKind",
    "Slide",
    "WIDESCREEN_HEIGHT_IN",
    "WIDESCREEN_WIDTH_IN",
    "latex_to_omml",
    "parse_slide_aux",
    "read_slide_aux",
    "set_widescreen_size",
]
