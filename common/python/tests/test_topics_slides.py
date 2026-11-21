# common/python/tests/test_topics_slides.py
# Copyright (C) 2026 Hirokazu Maesaka (RIKEN SPring-8 Center)
# SPDX-License-Identifier: GPL-3.0-or-later
# Licensed under GPL-3.0-or-later; see LICENSE. Provided WITHOUT ANY WARRANTY.

"""Check that every section of every topic document has slides."""

from pathlib import Path

import pytest
from dynamics_common import scan_latex_file_for_slides
from dynamics_common.glossary import TOPICS_DIR

TOPIC_LATEX_FILES = sorted(TOPICS_DIR.glob("*/main.tex"))


@pytest.mark.parametrize("path", TOPIC_LATEX_FILES, ids=lambda path: path.parent.name)
def test_every_section_has_slides(path: Path) -> None:
    """Each numbered section has a slide or an explicit '% no slide: <reason>' note."""
    messages = scan_latex_file_for_slides(path)
    assert not messages, "\n".join(messages)
