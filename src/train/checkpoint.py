from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class CheckpointManager:
    out_dir: Path
    latest_name: str = "ckpt_latest.pt"
    best_name: str = "ckpt_best.pt"

    def __post_init__(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.out_dir / self.latest_name

    @property
    def best_path(self) -> Path:
        return self.out_dir / self.best_name

    def save_latest(self, payload: Dict[str, Any]) -> None:
        torch.save(payload, self.latest_path)

    def save_best(self, payload: Dict[str, Any]) -> None:
        torch.save(payload, self.best_path)

    def load(self, which: str = "latest", map_location: str = "cpu") -> Dict[str, Any]:
        if which == "latest":
            p = self.latest_path
        elif which == "best":
            p = self.best_path
        else:
            raise ValueError("which must be 'latest' or 'best'")

        if not p.exists():
            raise FileNotFoundError(str(p))
        return torch.load(p, map_location=map_location)
