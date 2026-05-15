from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts.episodic_memory_dual import ALL_CHARS


def _canon_details(text: str) -> str:
    allowed = set(ALL_CHARS) - set("[]")
    text = (text or "").replace("[", " ").replace("]", " ")
    text = text.replace("\n", " ").replace("\t", " ")
    text = "".join(ch if ch in allowed else " " for ch in text)
    return " ".join(text.split()).strip()


@dataclass
class DetailsConfig:
    seed: int = 123
    max_len: int = 100


def build_or_load_details(
    ids: List[str],
    cfg: DetailsConfig,
    *,
    out_json: str | Path,
    force: bool = False,
) -> Dict[str, str]:
    out_json = Path(out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if out_json.exists() and not force:
        return json.loads(out_json.read_text(encoding="utf-8"))

    rng = np.random.default_rng(int(cfg.seed))
    alphabet = [ch for ch in ALL_CHARS if ch not in "[]"]
    used = set()
    out: Dict[str, str] = {}
    for rid in ids:
        while True:
            L = int(rng.integers(1, int(cfg.max_len) + 1))
            s = "".join(rng.choice(alphabet, size=L, replace=True).tolist())
            s = _canon_details(s)[: int(cfg.max_len)]
            if s and s not in used:
                used.add(s)
                out[str(rid)] = s
                break
    out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
