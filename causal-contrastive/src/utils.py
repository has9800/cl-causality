"""Utilities for logging and helpers."""

import json
from pathlib import Path


def ensure_dir(path: str | Path):
    """Create a directory recursively if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data, path: str | Path):
    """Save a JSON object with indentation."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2), encoding="utf-8")
