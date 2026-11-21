# common/python/tests/test_slide_coverage.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :mod:`dynamics_common.slide_coverage`."""

from pathlib import Path

import pytest
from dynamics_common import scan_latex_file_for_slides, sections_without_slides


def test_section_with_a_slide_is_covered() -> None:
    """A slide title anywhere in the section covers it."""
    text = "\\section{A}\nBody.\n\\slidetitle{Slide}\n\\section{B}\n\\slidetitle{Slide B}\n\\printbibliography\n"
    assert sections_without_slides(text) == []


def test_section_without_a_slide_is_reported_with_its_line_number() -> None:
    """A section without a slide is reported, and the next section is not affected."""
    text = "\\section{A}\nBody.\n\\section{B}\n\\slidetitle{Slide B}\n\\end{document}\n"
    assert sections_without_slides(text) == [(1, "\\section{A}")]


def test_no_slide_note_with_a_reason_covers_the_section() -> None:
    """The note '% no slide: reason' is an explicit exception."""
    assert sections_without_slides("\\section{A}\n% no slide: only definitions\n\\end{document}\n") == []


@pytest.mark.parametrize("note", ["% no slide:", "% no slide:   ", "% noslide: reason"])
def test_no_slide_note_needs_a_reason(note: str) -> None:
    """A note without a reason, or with another spelling, does not cover the section."""
    assert len(sections_without_slides(f"\\section{{A}}\n{note}\n\\end{{document}}\n")) == 1


def test_slide_in_the_next_section_does_not_cover_the_previous_one() -> None:
    """The slide has to be inside the section, before the next section starts."""
    text = "\\section{A}\n\\section{B}\n\\slidetitle{Slide B}\n\\end{document}\n"
    assert [line for line, _ in sections_without_slides(text)] == [1]


def test_commented_slide_title_does_not_count() -> None:
    """A commented-out slide marker is not a slide."""
    assert len(sections_without_slides("\\section{A}\n% \\slidetitle{Old}\n\\end{document}\n")) == 1


def test_starred_sections_are_not_checked() -> None:
    """Unnumbered sections such as the license need no slide."""
    assert sections_without_slides("\\section{A}\n\\slidetitle{S}\n\\section*{License}\nText.\n\\end{document}\n") == []


def test_scan_latex_file_for_slides_formats_messages(tmp_path: Path) -> None:
    """The file scan returns one readable message per offending section."""
    path = tmp_path / "main.tex"
    path.write_text("\\section{A}\n\\end{document}\n", encoding="utf-8")
    messages = scan_latex_file_for_slides(path)
    assert len(messages) == 1
    assert f"{path}:1:" in messages[0]
