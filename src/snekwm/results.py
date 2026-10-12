"""Run directory: every experiment saves its config, metrics and log to disk."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class RunDir:
    """``results/<run>/`` with a ``train.log`` logger and JSON writers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger(f"snekwm.{self.path.as_posix()}")
        self.log.setLevel(logging.INFO)
        self.log.handlers.clear()
        for handler in (
            logging.FileHandler(self.path / "train.log", mode="w"),
            logging.StreamHandler(),
        ):
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            self.log.addHandler(handler)

    def write_json(self, name: str, obj: Any) -> None:
        """Write ``obj`` (dataclasses allowed) to ``<run>/<name>``."""
        if is_dataclass(obj):
            obj = asdict(obj)
        self.write_text(name, json.dumps(obj, indent=2))

    def write_text(self, name: str, text: str) -> None:
        """Write UTF-8 text to ``<run>/<name>`` (Windows defaults to cp1252)."""
        (self.path / name).write_text(text, encoding="utf-8")

    def close(self) -> None:
        """Release log file handles (needed on Windows before deleting)."""
        for handler in self.log.handlers:
            handler.close()
        self.log.handlers.clear()
