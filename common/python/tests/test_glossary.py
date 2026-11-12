# common/python/tests/test_glossary.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :mod:`dynamics_common.glossary`."""

from pathlib import Path

import pytest
from dynamics_common import GlossaryTerm, find_forbidden_terms, load_glossary, render_markdown, scan_latex_file

GLOSSARY_TOML = """
[terms.dipole_magnet]
en = "dipole magnet"
ja = "偏向磁石"

[terms.frenet_serret_frame]
en = "Frenet-Serret frame"
ja = "Frenet-Serret 座標系"
forbidden_ja = ["標構"]
note = "Use 座標系."
"""


def _write_glossary(tmp_path: Path, content: str = GLOSSARY_TOML) -> Path:
    """Write a glossary file into a temporary directory."""
    path = tmp_path / "glossary.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_glossary_reads_terms_sorted_by_key(tmp_path: Path) -> None:
    """Terms are returned sorted by key with their optional fields."""
    terms = load_glossary(_write_glossary(tmp_path))
    assert [term.key for term in terms] == ["dipole_magnet", "frenet_serret_frame"]
    assert terms[0] == GlossaryTerm(key="dipole_magnet", en="dipole magnet", ja="偏向磁石")
    assert terms[1].forbidden_ja == ("標構",)
    assert terms[1].note == "Use 座標系."


@pytest.mark.parametrize(
    "content",
    [
        '[terms.broken]\nja = "訳"\n',
        '[terms.broken]\nen = "term"\n',
        '[terms.broken]\nen = "term"\nja = "訳"\nforbidden_ja = "標構"\n',
        '[terms.broken]\nen = "term"\nja = "訳"\nnote = 1\n',
        "not_terms = 1\n",
    ],
)
def test_load_glossary_rejects_malformed_entries(tmp_path: Path, content: str) -> None:
    """Missing or mistyped fields raise ValueError."""
    with pytest.raises(ValueError):
        load_glossary(_write_glossary(tmp_path, content))


def test_find_forbidden_terms_reports_line_numbers() -> None:
    """Each occurrence is reported with its one-based line number."""
    terms = (GlossaryTerm(key="frame", en="frame", ja="座標系", forbidden_ja=("標構",)),)
    text = "first line\nFrenet-Serret 標構と計量\n標構 again\n"
    matches = find_forbidden_terms(text, terms)
    assert [(line, term.key, forbidden) for line, term, forbidden in matches] == [
        (2, "frame", "標構"),
        (3, "frame", "標構"),
    ]


def test_find_forbidden_terms_ignores_terms_without_forbidden_forms() -> None:
    """A term without forbidden forms never matches."""
    terms = (GlossaryTerm(key="dipole", en="dipole", ja="偏向磁石"),)
    assert find_forbidden_terms("偏向磁石", terms) == []


def test_scan_latex_file_skips_comment_lines(tmp_path: Path) -> None:
    """A forbidden form in a LaTeX comment is not a violation."""
    terms = (GlossaryTerm(key="frame", en="frame", ja="座標系", forbidden_ja=("標構",)),)
    path = tmp_path / "main.tex"
    path.write_text("% 標構 in a comment\n\\section{標構と計量}\n", encoding="utf-8")
    violations = scan_latex_file(path, terms)
    assert len(violations) == 1
    assert violations[0].line_number == 2
    assert violations[0].forbidden == "標構"
    assert violations[0].term_key == "frame"


def test_render_markdown_lists_every_term() -> None:
    """The Markdown index has one row per term and marks missing optional fields."""
    terms = (
        GlossaryTerm(key="dipole", en="dipole magnet", ja="偏向磁石"),
        GlossaryTerm(key="frame", en="frame", ja="座標系", forbidden_ja=("標構",), note="Use 座標系."),
    )
    markdown = render_markdown(terms)
    assert "| `dipole` | dipole magnet | 偏向磁石 | — | — |" in markdown
    assert "| `frame` | frame | 座標系 | 標構 | Use 座標系. |" in markdown
    assert markdown.endswith("\n")
