# common/python/dynamics_common/slide_coverage.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Check that every section of a topic document has slides.

The slide markers (``\\slidetitle`` and so on, see ``common/tex/slides.tex``) are written at the end of the section
they summarize. A numbered ``\\section`` must contain at least one ``\\slidetitle`` before the next section starts.
A section that deliberately has no slide carries a line ``% no slide: <reason>`` instead.
"""

import re
from collections.abc import Sequence
from pathlib import Path

NUMBERED_SECTION_PATTERN = re.compile(r"^\\section\{")
SECTION_BOUNDARY_PATTERN = re.compile(r"^(\\section\*?\{|\\printbibliography|\\end\{document\})")
SLIDE_TITLE_PATTERN = re.compile(r"^\\slidetitle\{")
NO_SLIDE_PATTERN = re.compile(r"^%\s*no slide:\s*\S")

__all__ = ["sections_without_slides", "scan_latex_file_for_slides"]


def sections_without_slides(text: str) -> list[tuple[int, str]]:
    """Find the numbered sections that have neither a slide nor a ``% no slide:`` note.

    Args:
        text: LaTeX source of one topic document.

    Returns:
        One ``(line_number, section_line)`` tuple per offending section, in line order. The line number is one-based.
    """
    lines: Sequence[str] = text.splitlines()
    offenders: list[tuple[int, str]] = []
    for index, line in enumerate(lines):
        if NUMBERED_SECTION_PATTERN.match(line) is None:
            continue
        end = next(
            (j for j in range(index + 1, len(lines)) if SECTION_BOUNDARY_PATTERN.match(lines[j])),
            len(lines),
        )
        body = lines[index + 1 : end]
        covered = any(SLIDE_TITLE_PATTERN.match(item) or NO_SLIDE_PATTERN.match(item) for item in body)
        if not covered:
            offenders.append((index + 1, line.strip()))
    return offenders


def scan_latex_file_for_slides(path: Path) -> list[str]:
    """Report the sections of a LaTeX file that have no slide.

    Args:
        path: LaTeX source file.

    Returns:
        One message per offending section.
    """
    return [
        f"{path}:{line_number}: section has no \\slidetitle and no '% no slide: <reason>' note: {section}"
        for line_number, section in sections_without_slides(path.read_text(encoding="utf-8"))
    ]
