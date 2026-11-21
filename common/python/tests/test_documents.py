# common/python/tests/test_documents.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Tests for :mod:`dynamics_common.documents`."""

from pathlib import Path

import pytest
from dynamics_common import (
    Document,
    check_repository,
    find_latex_citations,
    find_notebook_citations,
    format_number,
    load_documents,
    next_number,
    number_for_topic,
    render_bibtex,
)

REGISTRY_TOML = """
[documents."2026-001"]
slug = "alpha_topic"
title = "Alpha title"
status = "active"

[documents."2026-002"]
slug = "beta_topic"
title = "Beta title"
status = "withdrawn"
note = "Merged into 2026-001."

[documents."2027-001"]
slug = "gamma_topic"
title = "Gamma title"
status = "active"
author = "Doe, Jane"
"""


def _write(path: Path, content: str) -> Path:
    """Write a text file, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _write_registry(tmp_path: Path, content: str = REGISTRY_TOML) -> Path:
    """Write a registry file into a temporary directory."""
    return _write(tmp_path / "documents.toml", content)


def test_load_documents_reads_fields_sorted_by_number(tmp_path: Path) -> None:
    """Documents are returned sorted by year and serial number with their optional fields."""
    documents = load_documents(_write_registry(tmp_path))
    assert [document.number for document in documents] == ["2026-001", "2026-002", "2027-001"]
    assert documents[0] == Document(number="2026-001", slug="alpha_topic", title="Alpha title")
    assert documents[1].status == "withdrawn"
    assert documents[1].note == "Merged into 2026-001."
    assert documents[2].author == "Doe, Jane"


@pytest.mark.parametrize(
    "content, message",
    [
        ('[documents."26-1"]\nslug = "a"\ntitle = "T"\nstatus = "active"\n', "YYYY-NNN"),
        ('[documents."2026-001"]\nslug = "a"\ntitle = "T"\nstatus = "draft"\n', "invalid status"),
        ('[documents."2026-001"]\ntitle = "T"\nstatus = "active"\n', "non-empty 'slug'"),
        ('[documents."2026-001"]\nslug = "Bad-Slug"\ntitle = "T"\nstatus = "active"\n', "invalid slug"),
        ('[documents."2026-001"]\nslug = "a"\nstatus = "active"\n', "non-empty 'title'"),
        (
            '[documents."2026-001"]\nslug = "a"\ntitle = "T"\nstatus = "active"\n'
            '[documents."2026-002"]\nslug = "a"\ntitle = "U"\nstatus = "active"\n',
            "slug more than once",
        ),
        ("[other]\nx = 1\n", r"\[documents\] table"),
    ],
    ids=["bad_number", "bad_status", "missing_slug", "bad_slug", "missing_title", "duplicate_slug", "no_table"],
)
def test_load_documents_rejects_malformed_registry(tmp_path: Path, content: str, message: str) -> None:
    """A malformed registry raises a ValueError that names the problem."""
    with pytest.raises(ValueError, match=message):
        load_documents(_write_registry(tmp_path, content))


def test_withdrawn_document_may_omit_slug(tmp_path: Path) -> None:
    """A withdrawn document needs no directory, so it needs no slug."""
    content = '[documents."2026-001"]\ntitle = "T"\nstatus = "withdrawn"\n'
    assert load_documents(_write_registry(tmp_path, content))[0].slug == ""


@pytest.mark.parametrize(
    "year, expected",
    [(2026, "2026-003"), (2027, "2027-002"), (2028, "2028-001")],
)
def test_next_number_counts_per_year_and_includes_withdrawn(tmp_path: Path, year: int, expected: str) -> None:
    """The serial number restarts every year and withdrawn numbers are not reused."""
    assert next_number(load_documents(_write_registry(tmp_path)), year) == expected


def test_next_number_fails_when_the_year_is_full() -> None:
    """A year holds 999 documents."""
    full = [Document(number=format_number(2026, 999), slug="x", title="T")]
    with pytest.raises(ValueError, match="out of range"):
        next_number(full, 2026)


@pytest.mark.parametrize(
    "year, serial, expected", [(2026, 1, "2026-001"), (2026, 42, "2026-042"), (2030, 999, "2030-999")]
)
def test_format_number(year: int, serial: int, expected: str) -> None:
    """Numbers are zero padded to three digits."""
    assert format_number(year, serial) == expected


def test_number_for_topic(tmp_path: Path) -> None:
    """Only active documents resolve from their slug."""
    documents = load_documents(_write_registry(tmp_path))
    assert number_for_topic("alpha_topic", documents) == "2026-001"
    with pytest.raises(KeyError):
        number_for_topic("beta_topic", documents)
    with pytest.raises(KeyError):
        number_for_topic("unknown", documents)


def test_render_bibtex_contains_one_report_per_document(tmp_path: Path) -> None:
    """Each document becomes a @report entry keyed by its number."""
    text = render_bibtex(load_documents(_write_registry(tmp_path)))
    assert text.count("@report{") == 3
    assert "@report{2026-001," in text
    assert "  author      = {Doe, Jane}," in text
    assert "  number      = {2027-001}," in text
    assert "  note        = {Merged into 2026-001.}," in text
    assert text.endswith("\n")


def test_find_latex_citations_reads_keys_and_skips_comments() -> None:
    """Document numbers are found in \\cite commands, including optional arguments and key lists."""
    text = "\n".join(
        [
            r"See \cite{2026-001} and \cite[Sec.~2]{lee2019,2027-001}.",
            r"% \cite{2026-002} is commented out.",
            r"No number in \cite{wiedemann2015}.",
        ]
    )
    assert find_latex_citations(text) == [(1, "2026-001"), (1, "2027-001")]


def test_find_notebook_citations_reads_numbers_in_brackets() -> None:
    """Notebook citations are written as [number] or [number, detail]."""
    text = "See [2026-001] and [2027-001, Sec. 3].\nNot a citation: 2026-002 or [lee2019]."
    assert find_notebook_citations(text) == [(1, "2026-001"), (1, "2027-001")]


def _make_repository(tmp_path: Path, latex_number: str = "2026-001", notebook_note: str = "Memo 2026-001") -> tuple:
    """Build a consistent miniature repository and return its three paths."""
    registry = _write_registry(
        tmp_path,
        '[documents."2026-001"]\nslug = "alpha_topic"\ntitle = "Alpha title"\nstatus = "active"\n',
    )
    topics = tmp_path / "topics"
    _write(topics / "alpha_topic" / "main.tex", f"\\setmemonumber{{{latex_number}}}\n")
    _write(topics / "alpha_topic" / "alpha_topic.py", f"# {notebook_note}\n")
    bib = _write(tmp_path / "documents.bib", render_bibtex(load_documents(registry)))
    return registry, bib, topics


def test_check_repository_accepts_a_consistent_repository(tmp_path: Path) -> None:
    """No problem is reported when topics, notebooks, and the BibTeX database agree with the registry."""
    registry, bib, topics = _make_repository(tmp_path)
    assert check_repository(registry, bib, topics) == []


def test_check_repository_detects_a_wrong_latex_number(tmp_path: Path) -> None:
    """The number in main.tex must equal the registered number."""
    registry, bib, topics = _make_repository(tmp_path, latex_number="2026-009")
    problems = check_repository(registry, bib, topics)
    assert len(problems) == 1
    assert "setmemonumber" in problems[0]


def test_check_repository_detects_a_notebook_without_number(tmp_path: Path) -> None:
    """The notebook must mention its memo number."""
    registry, bib, topics = _make_repository(tmp_path, notebook_note="no number here")
    problems = check_repository(registry, bib, topics)
    assert len(problems) == 1
    assert "Memo 2026-001" in problems[0]


def test_check_repository_detects_an_unregistered_topic(tmp_path: Path) -> None:
    """Every topic directory with a main.tex needs an active document number."""
    registry, bib, topics = _make_repository(tmp_path)
    _write(topics / "orphan_topic" / "main.tex", "\\setmemonumber{2026-009}\n")
    problems = check_repository(registry, bib, topics)
    assert problems == ["Topic 'orphan_topic' has no active document number in documents.toml."]


def test_check_repository_detects_an_outdated_bibtex_database(tmp_path: Path) -> None:
    """documents.bib must equal the output of render_bibtex."""
    registry, bib, topics = _make_repository(tmp_path)
    bib.write_text("outdated\n", encoding="utf-8")
    problems = check_repository(registry, bib, topics)
    assert len(problems) == 1
    assert "out of date" in problems[0]


@pytest.mark.parametrize("cite, message", [(r"\cite{2026-009}", "unregistered"), (r"\cite{2026-002}", "withdrawn")])
def test_check_repository_detects_bad_citations(tmp_path: Path, cite: str, message: str) -> None:
    """Citing an unregistered or withdrawn document is a problem."""
    registry = _write_registry(
        tmp_path,
        '[documents."2026-001"]\nslug = "alpha_topic"\ntitle = "A"\nstatus = "active"\n'
        '[documents."2026-002"]\ntitle = "B"\nstatus = "withdrawn"\n',
    )
    topics = tmp_path / "topics"
    _write(topics / "alpha_topic" / "main.tex", f"\\setmemonumber{{2026-001}}\n{cite}\n")
    _write(topics / "alpha_topic" / "alpha_topic.py", "# Memo 2026-001\n")
    bib = _write(tmp_path / "documents.bib", render_bibtex(load_documents(registry)))
    problems = check_repository(registry, bib, topics)
    assert len(problems) == 1
    assert message in problems[0]
