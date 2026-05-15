from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputPaths:
    root: Path

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def ensure(self) -> "OutputPaths":
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        return self


def default_outputs_root() -> Path:
    return Path(__file__).resolve().parent / "outputs"

