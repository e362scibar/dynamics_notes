# common/python/tests/test_topics_glossary.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Check every topic's LaTeX source against the Japanese glossary in ``common/glossary.toml``."""

from pathlib import Path

import pytest
from dynamics_common import load_glossary, scan_latex_file
from dynamics_common.glossary import TOPICS_DIR

TOPIC_LATEX_FILES = sorted(TOPICS_DIR.glob("*/main.tex"))


def test_topics_directory_has_latex_sources() -> None:
    """The scan is not silently empty."""
    assert TOPIC_LATEX_FILES


@pytest.mark.parametrize("path", TOPIC_LATEX_FILES, ids=lambda path: path.parent.name)
def test_topic_uses_preferred_terms(path: Path) -> None:
    """No topic uses a forbidden Japanese form."""
    violations = scan_latex_file(path, load_glossary())
    messages = [
        f"{violation.path}:{violation.line_number}: forbidden '{violation.forbidden}' "
        f"for term '{violation.term_key}': {violation.line}"
        for violation in violations
    ]
    assert not messages, "\n".join(messages)
