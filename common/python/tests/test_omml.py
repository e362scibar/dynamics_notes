# common/python/tests/test_omml.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :func:`dynamics_common.latex_to_omml`."""

import pytest
from dynamics_common import latex_to_omml
from dynamics_common.omml import OMML_NAMESPACE
from lxml import etree

FRACTION_LATEX = r"\frac{d\mathbf{T}}{ds}"
SQRT_LATEX = r"\sqrt{(1+\delta)^2 - p_x^2}"


@pytest.mark.parametrize(
    ("latex", "expected_tag"),
    [
        ("x", "oMath"),
        (FRACTION_LATEX, "oMath"),
        (SQRT_LATEX, "oMath"),
    ],
)
def test_latex_to_omml_returns_parsable_omml(latex: str, expected_tag: str) -> None:
    """The result parses as XML and its root is an ``m:oMath`` element in the OMML namespace."""
    root = etree.fromstring(latex_to_omml(latex))
    assert root.tag == f"{{{OMML_NAMESPACE}}}{expected_tag}"


def test_latex_to_omml_contains_fraction_element() -> None:
    """A fraction in LaTeX becomes an ``m:f`` element."""
    root = etree.fromstring(latex_to_omml(FRACTION_LATEX))
    assert root.find(f".//{{{OMML_NAMESPACE}}}f") is not None


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_latex_to_omml_rejects_blank_input(blank: str) -> None:
    """Blank input raises ``ValueError``."""
    with pytest.raises(ValueError, match="blank"):
        latex_to_omml(blank)


def test_latex_to_omml_rejects_unconvertible_input() -> None:
    """Input that the converter cannot handle raises ``ValueError`` with the original source."""
    with pytest.raises(ValueError, match="Cannot convert"):
        latex_to_omml(r"\notarealcommand{x}")
