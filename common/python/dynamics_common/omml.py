# common/python/dynamics_common/omml.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Convert LaTeX math strings into Office Math Markup Language (OMML) for native PowerPoint equations."""

import re

import latex2mathml.converter
import mathml2omml

OMML_NAMESPACE = "http://schemas.openxmlformats.org/officeDocument/2006/math"
OMML_ROOT_TAG = "<m:oMath"
# latex2mathml renders an unknown control sequence as an identifier that still starts with a backslash.
UNKNOWN_COMMAND_PATTERN = re.compile(r"<mi>\\")


def latex_to_omml(latex: str) -> str:
    """Convert a LaTeX math string to an OMML ``m:oMath`` element.

    The conversion chain is LaTeX -> MathML (``latex2mathml``) -> OMML (``mathml2omml``). The returned string
    declares the ``m`` namespace on the root element, so it can be parsed on its own.

    Args:
        latex: LaTeX math source without surrounding delimiters, for example ``r"\\frac{d}{ds}"``.

    Returns:
        The OMML ``<m:oMath>`` element as a string, with the ``m`` namespace declared.

    Raises:
        ValueError: If ``latex`` is blank or cannot be converted.
    """
    stripped = latex.strip()
    if not stripped:
        raise ValueError("LaTeX math string must not be blank.")
    try:
        mathml = latex2mathml.converter.convert(stripped)
        omml = mathml2omml.convert(mathml)
    except Exception as exc:
        raise ValueError(f"Cannot convert LaTeX to OMML: latex={stripped!r}") from exc
    if UNKNOWN_COMMAND_PATTERN.search(mathml):
        raise ValueError(f"Cannot convert LaTeX to OMML, unknown command: latex={stripped!r}")
    if not omml.startswith(OMML_ROOT_TAG):
        raise ValueError(f"Unexpected OMML root element: latex={stripped!r}")
    return omml.replace(OMML_ROOT_TAG, f'{OMML_ROOT_TAG} xmlns:m="{OMML_NAMESPACE}"', 1)
