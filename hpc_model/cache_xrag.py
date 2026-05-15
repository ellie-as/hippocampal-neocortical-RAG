from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_imports() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    nar = (root / "full_model" / "narratives").resolve()
    if str(nar) not in sys.path:
        sys.path.insert(0, str(nar))


def _to_numpy(x: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
    except Exception:
        pass
    return np.asarray(x, dtype=np.float32)


def _raykov_query(story: str, excerpt_chars: int) -> str:
    return f"{story[0:int(excerpt_chars)]}... What happened next? (Be concise.)"


@dataclass
class XragCacheConfig:
    n: int = 100
    seed: int = 123
    category: str = "typical"
    stories_csv: str = "data/stories_train.csv"
    query_excerpt_chars: int = 80
    retriever_query_clip_chars: int = 80
    retriever_query_max_tokens: int = 64
    use_mps: bool = False


def build_or_load_xrag_cache(
    cfg: XragCacheConfig,
    *,
    out_npz: str | Path,
    out_meta_json: str | Path,
    force: bool = False,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Cache:
      - doc xRAG embeddings for each full story (the stored hippocampal keys)
      - query xRAG embeddings for Raykov-style queries (80-char excerpt by default)
    """
    out_npz = Path(out_npz).resolve()
    out_meta_json = Path(out_meta_json).resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)

    if out_npz.exists() and out_meta_json.exists() and not force:
        meta = json.loads(out_meta_json.read_text(encoding="utf-8"))
        meta["loaded_existing"] = True
        return out_npz, out_meta_json, meta

    _ensure_imports()
    from utils import XRAG, get_device, prepare_roc_sets, set_seed  # type: ignore

    set_seed(int(cfg.seed))
    device = get_device(bool(cfg.use_mps))

    sets = prepare_roc_sets(n_typical=int(cfg.n), n_variants=0, rng_seed=int(cfg.seed), stories_csv=str(cfg.stories_csv))
    stories = list(sets[str(cfg.category)])[: int(cfg.n)]
    if len(stories) != int(cfg.n):
        raise RuntimeError(f"Expected {int(cfg.n)} stories, got {len(stories)}")

    class Cfg:
        llm_name = "Hannibal046/xrag-7b"
        retriever_name = "Salesforce/SFR-Embedding-Mistral"
        retriever_batch_size = 16
        retriever_max_length = 256
        docs_per_datastore = 0
        retriever_query_clip_chars = int(cfg.retriever_query_clip_chars)

    xrag = XRAG(Cfg(), device)
    t0 = time.time()
    try:
        xrag.load()
    except Exception:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        xrag.load()
    load_seconds = time.time() - t0

    # Doc embeddings (stored keys)
    _datastore, _raw_doc_emb, xrag_doc = xrag._prepare_datastore(stories)  # CPU [N, D]

    # Query embeddings (Raykov prompt format, but retriever side uses a char clip via cfg)
    ids: List[str] = []
    query_texts: List[str] = []
    prompt_texts: List[str] = []
    q_embs: List[np.ndarray] = []
    for i, story in enumerate(stories):
        rid = f"{cfg.category}_{i:04d}"
        q = _raykov_query(story, int(cfg.query_excerpt_chars))
        prompt, q_emb = xrag._prepare_prompt(q)
        ids.append(rid)
        query_texts.append(q)
        prompt_texts.append(prompt)
        q_embs.append(_to_numpy(q_emb).reshape(-1))

    doc_mat = _to_numpy(xrag_doc)  # (N, D)
    query_mat = np.stack(q_embs, axis=0).astype(np.float32)  # (N, D)
    if doc_mat.shape[0] != int(cfg.n) or query_mat.shape[0] != int(cfg.n):
        raise RuntimeError("Unexpected embedding shapes.")

    # Save arrays + minimal strings; keep full stories in meta json to avoid huge npz.
    np.savez(
        out_npz,
        ids=np.array(ids, dtype=object),
        doc_emb=doc_mat.astype(np.float32),
        query_emb=query_mat.astype(np.float32),
    )

    meta: Dict[str, Any] = {
        "loaded_existing": False,
        "n": int(cfg.n),
        "seed": int(cfg.seed),
        "category": str(cfg.category),
        "stories_csv": str(Path(cfg.stories_csv).resolve()),
        "device": str(device),
        "xrag_load_seconds": float(load_seconds),
        "query_excerpt_chars": int(cfg.query_excerpt_chars),
        "retriever_query_clip_chars": int(cfg.retriever_query_clip_chars),
        "retriever_query_max_tokens": int(cfg.retriever_query_max_tokens),
        "raykov_query_format": "{story[:excerpt_chars]}... What happened next? (Be concise.)",
        "note": "Query embedding uses XRAG._prepare_prompt; retriever side clips by cfg.retriever_query_clip_chars.",
        "queries": {ids[i]: query_texts[i] for i in range(len(ids))},
        "prompts": {ids[i]: prompt_texts[i] for i in range(len(ids))},
        "stories": {ids[i]: stories[i] for i in range(len(ids))},
        "d": int(doc_mat.shape[1]),
        "built_at_unix": time.time(),
    }
    out_meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_npz, out_meta_json, meta
