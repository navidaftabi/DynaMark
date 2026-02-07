from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .utils import jsonify


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_json(path, orient="records", lines=True)


def ensure_empty_dir(path: Path) -> None:
    """Create directory; if exists, leave as-is (caller may remove)."""
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
