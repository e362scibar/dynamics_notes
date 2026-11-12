# common/python/tests/test_slide_aux.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :func:`dynamics_common.parse_slide_aux` and :func:`dynamics_common.read_slide_aux`."""

from pathlib import Path

import pytest
from dynamics_common import Block, BlockKind, Slide, parse_slide_aux, read_slide_aux


def test_parse_groups_entries_by_title() -> None:
    """Entries after a title belong to that slide, in document order."""
    lines = [
        "title:First",
        "point:alpha",
        "math:x + y",
        "note:speaker text",
        "title:Second",
        "point:beta",
    ]
    slides = parse_slide_aux(lines)
    assert slides == [
        Slide(
            title="First",
            blocks=(Block(BlockKind.POINT, "alpha"), Block(BlockKind.MATH, "x + y")),
            notes=("speaker text",),
        ),
        Slide(title="Second", blocks=(Block(BlockKind.POINT, "beta"),), notes=()),
    ]


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("point", BlockKind.POINT), ("math", BlockKind.MATH), ("figure", BlockKind.FIGURE)],
)
def test_parse_maps_kind_names_to_block_kinds(kind: str, expected: BlockKind) -> None:
    """Each block kind name maps to its enum member."""
    slides = parse_slide_aux([f"title:T", f"{kind}:payload"])
    assert slides[0].blocks == (Block(expected, "payload"),)


def test_parse_keeps_colons_in_payload() -> None:
    """Only the first colon separates the kind from the payload."""
    slides = parse_slide_aux(["title:Ratio 1:2", "math:a:b"])
    assert slides[0].title == "Ratio 1:2"
    assert slides[0].blocks == (Block(BlockKind.MATH, "a:b"),)


def test_parse_skips_blank_lines() -> None:
    """Blank lines are ignored."""
    slides = parse_slide_aux(["", "title:T", "   ", "point:p", ""])
    assert len(slides) == 1
    assert slides[0].blocks == (Block(BlockKind.POINT, "p"),)


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["point:orphan"], "before the first title"),
        (["title:T", "bogus:x"], "Unknown slide entry kind"),
        (["title:T", "no separator here"], "without separator"),
        (["title:   "], "Blank slide title"),
    ],
)
def test_parse_rejects_malformed_input(lines: list[str], message: str) -> None:
    """Malformed input raises ``ValueError`` with a descriptive message."""
    with pytest.raises(ValueError, match=message):
        parse_slide_aux(lines)


def test_parse_empty_input_returns_no_slides() -> None:
    """An empty file yields an empty list."""
    assert parse_slide_aux([]) == []


def test_read_slide_aux_reads_utf8_file(tmp_path: Path) -> None:
    """The reader decodes UTF-8 text, which is needed for Japanese slides."""
    path = tmp_path / "main-ja.slides"
    path.write_text("title:曲線座標\npoint:基準軌道の曲率である\n", encoding="utf-8")
    slides = read_slide_aux(path)
    assert slides[0].title == "曲線座標"
    assert slides[0].blocks == (Block(BlockKind.POINT, "基準軌道の曲率である"),)


def test_read_slide_aux_missing_file_raises(tmp_path: Path) -> None:
    """A missing file raises ``FileNotFoundError``."""
    with pytest.raises(FileNotFoundError):
        read_slide_aux(tmp_path / "missing.slides")
