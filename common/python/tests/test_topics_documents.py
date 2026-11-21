# common/python/tests/test_topics_documents.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Check the topics of this repository against the document registry ``common/documents.toml``."""

from dynamics_common import check_repository, load_documents


def test_registry_is_not_empty() -> None:
    """The check is not silently empty."""
    assert load_documents()


def test_repository_agrees_with_document_registry() -> None:
    """Topics, LaTeX sources, notebooks, citations, and documents.bib agree with the registry."""
    problems = check_repository()
    assert not problems, "\n".join(problems)
