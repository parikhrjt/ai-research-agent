"""Markdown document parser with front-matter stripping."""

import re
from pathlib import Path

_FRONT_MATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def parse_markdown(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8", errors="replace")
    return _FRONT_MATTER.sub("", content).strip()
