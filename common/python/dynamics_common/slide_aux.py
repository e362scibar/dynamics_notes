# common/python/dynamics_common/slide_aux.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Read the slide entries that the LaTeX slide macros write during compilation.

Each line of the auxiliary file has the form ``kind:payload``. The kinds are ``title``, ``point``, ``math``,
``figure``, and ``note``. A ``title`` line starts a new slide. The other kinds add a block or a speaker note to the
current slide. The format is line based because TeX cannot escape JSON strings reliably.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

ENTRY_SEPARATOR = ":"
KIND_TITLE = "title"
KIND_NOTE = "note"


class BlockKind(Enum):
    """Kind of a content block on a slide."""

    POINT = "point"
    MATH = "math"
    FIGURE = "figure"


@dataclass(frozen=True)
class Block:
    """One content block on a slide.

    Attributes:
        kind: Kind of the block.
        text: Payload. Plain text for points, LaTeX for math, and a path relative to the topic directory for figures.
    """

    kind: BlockKind
    text: str


@dataclass(frozen=True)
class Slide:
    """One slide read from the auxiliary file.

    Attributes:
        title: Slide title.
        blocks: Content blocks in document order.
        notes: Speaker-note paragraphs in document order.
    """

    title: str
    blocks: tuple[Block, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class _SlideBuilder:
    """Mutable accumulator used while parsing the entries of one slide."""

    title: str
    blocks: list[Block] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def freeze(self) -> Slide:
        """Return an immutable :class:`Slide` built from the accumulated entries.

        Returns:
            The slide with its blocks and notes as tuples.
        """
        return Slide(title=self.title, blocks=tuple(self.blocks), notes=tuple(self.notes))


def _split_entry(line: str, line_number: int) -> tuple[str, str]:
    """Split one entry line into its kind and payload.

    Args:
        line: One non-empty line of the auxiliary file.
        line_number: One-based line number, used in error messages.

    Returns:
        A tuple of the kind and the payload.

    Raises:
        ValueError: If the line has no separator.
    """
    kind, separator, payload = line.partition(ENTRY_SEPARATOR)
    if not separator:
        raise ValueError(f"Slide entry without separator: line={line_number} text={line!r}")
    return kind.strip(), payload.strip()


def _add_entry(current: _SlideBuilder | None, kind: str, payload: str, line_number: int) -> None:
    """Add a non-title entry to the current slide.

    Args:
        current: Slide being built, or ``None`` if no title has been read yet.
        kind: Entry kind other than ``title``.
        payload: Entry payload.
        line_number: One-based line number, used in error messages.

    Raises:
        ValueError: If the kind is unknown or the entry appears before the first title.
    """
    if current is None:
        raise ValueError(f"Slide entry before the first title: line={line_number} kind={kind!r}")
    if kind == KIND_NOTE:
        current.notes.append(payload)
        return
    try:
        block_kind = BlockKind(kind)
    except ValueError as exc:
        raise ValueError(f"Unknown slide entry kind: line={line_number} kind={kind!r}") from exc
    current.blocks.append(Block(kind=block_kind, text=payload))


def parse_slide_aux(lines: Iterable[str]) -> list[Slide]:
    """Parse the lines of a slide auxiliary file.

    Args:
        lines: Lines of the file, with or without line terminators. Blank lines are skipped.

    Returns:
        The slides in document order.

    Raises:
        ValueError: If a line is malformed, a title is blank, or an entry has an unknown kind or appears before the
            first title.
    """
    slides: list[Slide] = []
    current: _SlideBuilder | None = None
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\r\n")
        if not line.strip():
            continue
        kind, payload = _split_entry(line, line_number)
        if kind == KIND_TITLE:
            if not payload:
                raise ValueError(f"Blank slide title: line={line_number}")
            if current is not None:
                slides.append(current.freeze())
            current = _SlideBuilder(title=payload)
            continue
        _add_entry(current, kind, payload, line_number)
    if current is not None:
        slides.append(current.freeze())
    return slides


def read_slide_aux(path: Path) -> list[Slide]:
    """Read and parse a slide auxiliary file.

    Args:
        path: Path to the ``main-<lang>.slides`` file written by the LaTeX build.

    Returns:
        The slides in document order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is malformed (see :func:`parse_slide_aux`).
    """
    with path.open(encoding="utf-8") as handle:
        return parse_slide_aux(handle)
