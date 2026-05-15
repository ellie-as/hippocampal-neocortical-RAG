from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from scripts.episodic_memory_dual import EpisodicDualParams, EpisodicMemoryDual


def _canon(text: str) -> str:
    # Match the hippocampus' representational sanitization (see EpisodicMemoryDual._encode_details).
    from scripts.episodic_memory_dual import ALL_CHARS

    allowed = set(ALL_CHARS) - set("[]")
    text = (text or "").replace("[", " ").replace("]", " ")
    text = text.replace("\n", " ").replace("\t", " ")
    text = "".join(ch if ch in allowed else " " for ch in text)
    return " ".join(text.split()).strip()


def _to_float32(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32)


def _dist2_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """
    Squared L2 distance matrix: (n, d) and (n, d) -> (n, n)
    """
    Q = _to_float32(queries)
    D = _to_float32(docs)
    q2 = (Q * Q).sum(axis=1, keepdims=True)  # (n, 1)
    d2 = (D * D).sum(axis=1, keepdims=True).T  # (1, n)
    cross = Q @ D.T  # (n, n)
    return (q2 + d2 - 2.0 * cross).astype(np.float32)


@dataclass
class EvalConfig:
    decay_rate: float = 0.9
    beta: float = 120.0
    n_auto_iters: int = 1
    n_hetero_iters: int = 1
    denoise_after_hetero: bool = False
    n_post_denoise_iters: int = 2
    hard_hetero: bool = False
    max_chars: int = 120
    hpc_dim: int | None = 512
    proj_seed: int = 0
    details_seed: int | None = None
    details_max_len: int | None = None


def _orthonormal_row_projection(d_in: int, d_out: int, *, seed: int) -> np.ndarray:
    """
    Return an orthonormal row projection P with shape (d_out, d_in) so that:
      x_out = P @ x_in
    and rows of P are orthonormal.
    """
    if d_out > d_in:
        raise ValueError(f"d_out ({d_out}) must be <= d_in ({d_in})")
    rng = np.random.default_rng(int(seed))
    A = rng.normal(size=(d_in, d_out)).astype(np.float32)  # tall
    Q, _ = np.linalg.qr(A, mode="reduced")  # (d_in, d_out) orthonormal columns
    return Q.T.astype(np.float32)  # (d_out, d_in) orthonormal rows


def _maybe_project(X: np.ndarray, *, d_out: Optional[int], seed: int) -> np.ndarray:
    X = _to_float32(X)
    if d_out is None:
        return X
    d_in = int(X.shape[1])
    d_out = int(d_out)
    if d_out == d_in:
        return X
    P = _orthonormal_row_projection(d_in, d_out, seed=int(seed))
    return (X @ P.T).astype(np.float32)


def build_hippocampus(
    *,
    doc_emb: np.ndarray,
    details_by_id: Dict[str, str],
    ids: List[str],
    cfg: EvalConfig,
) -> EpisodicMemoryDual:
    doc_emb = _maybe_project(doc_emb, d_out=cfg.hpc_dim, seed=int(cfg.proj_seed))
    d = int(doc_emb.shape[1])
    params = EpisodicDualParams(
        decay_rate=float(cfg.decay_rate),
        beta_auto=float(cfg.beta),
        beta_hetero=float(cfg.beta),
        n_auto_iters=int(cfg.n_auto_iters),
        n_hetero_iters=int(cfg.n_hetero_iters),
        hard_hetero=bool(cfg.hard_hetero),
        denoise_after_hetero=bool(cfg.denoise_after_hetero),
        n_post_denoise_iters=int(cfg.n_post_denoise_iters),
        proj_seed=0,
    )
    mem = EpisodicMemoryDual(d, params=params)
    for i, rid in enumerate(ids):
        mem.add_episode(doc_emb[i], details_by_id[str(rid)])
    mem.finalize(build_auto=True, build_hetero=True)
    mem._hetero_mhn.beta = float(cfg.beta)
    return mem


def select_episode_auto_mhn(
    mem: EpisodicMemoryDual,
    queries: np.ndarray,
    *,
    cfg: EvalConfig,
) -> np.ndarray:
    """Select episodes by autoassociative MHN retrieval from query xRAG vectors."""
    if mem._auto_mhn is None or mem._auto_mhn.patterns is None:
        raise RuntimeError("Autoassociative MHN not built.")

    Q = _maybe_project(queries, d_out=cfg.hpc_dim, seed=int(cfg.proj_seed))
    Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    V = Q.T.astype(np.float32)  # (d, n_queries)

    patterns = mem._auto_mhn.patterns.astype(np.float32, copy=False)  # (d, n_patterns)
    beta = float(mem._auto_mhn.beta)
    for _ in range(int(cfg.n_auto_iters)):
        sims = (patterns.T @ V).astype(np.float32)  # (n_patterns, n_queries)
        logits = beta * sims
        logits = logits - logits.max(axis=0, keepdims=True)
        W = np.exp(logits).astype(np.float32)
        W = W / (W.sum(axis=0, keepdims=True) + 1e-12)
        V = (patterns @ W).astype(np.float32)
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

    nearest_pattern = np.argmax(patterns.T @ V, axis=0).astype(np.int64)
    episode_for_pattern = np.empty(len(mem._patterns), dtype=np.int64)
    for episode_id, (start, end) in enumerate(mem._episode_ranges):
        episode_for_pattern[int(start):int(end)] = int(episode_id)
    return episode_for_pattern[nearest_pattern]


def decode_from_episode_key(mem: EpisodicMemoryDual, episode_id: int, *, max_chars: int) -> str:
    start = mem._episode_starts[int(episode_id)]
    v = mem._patterns[int(start)].copy()  # stored xRAG key (normalized)

    v = mem._advance(v)
    if bool(mem.params.denoise_after_hetero):
        v = mem._denoise(v, n_iters=int(mem.params.n_post_denoise_iters))

    out: List[str] = []
    for _ in range(int(max_chars) + 5):
        ch = mem._decode_char(v)
        if ch == "]":
            break
        if ch != "[":
            out.append(ch)
        v = mem._advance(v)
        if bool(mem.params.denoise_after_hetero):
            v = mem._denoise(v, n_iters=int(mem.params.n_post_denoise_iters))
    return "".join(out)


def decode_from_episode_keys_batch(
    mem: EpisodicMemoryDual,
    episode_ids: np.ndarray,
    *,
    max_chars: int,
) -> List[str]:
    """
    Vectorized hetero stepping + character decoding for many episodes at once.

    This is substantially faster than looping `mem._advance` per episode
    because it uses GEMM (keys.T @ V) for all states in one call.
    """
    if mem._hetero_mhn is None or mem._hetero_mhn.keys is None or mem._hetero_mhn.values is None:
        raise RuntimeError("Hetero MHN not built.")

    episode_ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    n = int(episode_ids.shape[0])
    if n == 0:
        return []

    # Initial state: exact stored xRAG keys for each selected episode.
    starts = np.asarray([mem._episode_starts[int(eid)] for eid in episode_ids], dtype=np.int64)
    V = np.stack([mem._patterns[int(s)].copy() for s in starts], axis=1).astype(np.float32)  # (d, n)

    keys = mem._hetero_mhn.keys.astype(np.float32, copy=False)  # (d, m)
    values = mem._hetero_mhn.values.astype(np.float32, copy=False)  # (d, m)
    beta = float(mem._hetero_mhn.beta)

    # We'll collect decoded characters per timestep (including '[' and ']' tokens).
    # Max steps includes a small buffer so we can see the end token.
    max_steps = int(max_chars) + 5
    char_ids = np.full((max_steps, n), fill_value=-1, dtype=np.int32)

    from scripts.episodic_memory_dual import ALL_CHARS

    end_id = int(ALL_CHARS.index("]"))
    start_id = int(ALL_CHARS.index("["))

    # Step to first detail token (xRAG -> '['), then decode, then iterate.
    for t in range(max_steps):
        if bool(mem.params.hard_hetero):
            # Hard mode can't be fully vectorized with the current hashed implementation.
            # Fall back to per-column advancement, which is still fast in hard mode.
            for j in range(n):
                V[:, j] = mem._advance(V[:, j])
        else:
            # Apply the hetero step(s) in batch.
            for _ in range(int(mem.params.n_hetero_iters)):
                sims = (keys.T @ V).astype(np.float32)  # (m, n)
                logits = beta * sims
                logits = logits - logits.max(axis=0, keepdims=True)
                W = np.exp(logits).astype(np.float32)
                W = W / (W.sum(axis=0, keepdims=True) + 1e-12)
                V = (values @ W).astype(np.float32)
                V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)

        # Decode current character(s).
        logits_chars = (mem.R.T @ V).astype(np.float32)  # (|chars|, n)
        char_ids[t] = logits_chars.argmax(axis=0).astype(np.int32)

        # Early exit if everyone has reached ']'.
        if (char_ids[t] == end_id).all():
            break

    # Convert to strings per episode.
    outs: List[str] = []
    for j in range(n):
        out_chars: List[str] = []
        for t in range(max_steps):
            cid = int(char_ids[t, j])
            if cid < 0:
                break
            if cid == end_id:
                break
            if cid != start_id:
                out_chars.append(ALL_CHARS[cid])
        outs.append("".join(out_chars))
    return outs


def evaluate(
    *,
    ids: List[str],
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    details_by_id: Dict[str, str],
    cfg: EvalConfig,
) -> Dict[str, Any]:
    mem = build_hippocampus(doc_emb=doc_emb, details_by_id=details_by_id, ids=ids, cfg=cfg)
    sel = select_episode_auto_mhn(mem, query_emb, cfg=cfg)

    correct_sel = int((sel == np.arange(len(ids))).sum())
    if bool(cfg.denoise_after_hetero):
        recalled_all = [decode_from_episode_key(mem, int(sel[i]), max_chars=int(cfg.max_chars)) for i in range(len(ids))]
    else:
        recalled_all = decode_from_episode_keys_batch(mem, sel, max_chars=int(cfg.max_chars))
    correct_recall = 0
    mismatches: List[Dict[str, Any]] = []

    for i, rid in enumerate(ids):
        recalled = recalled_all[int(i)]
        ok = _canon(recalled) == _canon(details_by_id[str(rid)])
        if ok:
            correct_recall += 1
        elif len(mismatches) < 10:
            mismatches.append({"id": str(rid), "selected_id": str(ids[int(sel[i])]), "stored": details_by_id[str(rid)], "recalled": recalled})

    return {
        "n": int(len(ids)),
        "cfg": {
            "decay_rate": float(cfg.decay_rate),
            "beta": float(cfg.beta),
            "selection_method": "auto_mhn",
            "n_auto_iters": int(cfg.n_auto_iters),
            "hard_hetero": bool(cfg.hard_hetero),
            "denoise_after_hetero": bool(cfg.denoise_after_hetero),
            "n_hetero_iters": int(cfg.n_hetero_iters),
            "max_chars": int(cfg.max_chars),
            "hpc_dim": None if cfg.hpc_dim is None else int(cfg.hpc_dim),
            "proj_seed": int(cfg.proj_seed),
            "details_seed": None if cfg.details_seed is None else int(cfg.details_seed),
            "details_max_len": None if cfg.details_max_len is None else int(cfg.details_max_len),
        },
        "selection_accuracy": {"count": correct_sel, "rate": correct_sel / max(1, len(ids))},
        "recall_accuracy_canon": {"count": int(correct_recall), "rate": correct_recall / max(1, len(ids))},
        "mismatches": mismatches,
    }


def sweep_decay(
    *,
    ids: List[str],
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    details_by_id: Dict[str, str],
    decay_values: List[float],
    beta: float,
    n_auto_iters: int,
    max_chars: int,
    hard_hetero: bool,
    hpc_dim: int | None = 512,
    proj_seed: int = 0,
    details_seed: int | None = None,
    details_max_len: int | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for d in decay_values:
        res = evaluate(
            ids=ids,
            doc_emb=doc_emb,
            query_emb=query_emb,
            details_by_id=details_by_id,
            cfg=EvalConfig(
                decay_rate=float(d),
                beta=float(beta),
                n_auto_iters=int(n_auto_iters),
                max_chars=int(max_chars),
                hard_hetero=bool(hard_hetero),
                hpc_dim=hpc_dim,
                proj_seed=int(proj_seed),
                details_seed=details_seed,
                details_max_len=details_max_len,
            ),
        )
        rows.append(
            {
                "decay_rate": float(d),
                "selection_rate": float(res["selection_accuracy"]["rate"]),
                "recall_rate": float(res["recall_accuracy_canon"]["rate"]),
            }
        )
    return {
        "cfg": {
            "beta": float(beta),
            "selection_method": "auto_mhn",
            "n_auto_iters": int(n_auto_iters),
            "max_chars": int(max_chars),
            "hard_hetero": bool(hard_hetero),
            "hpc_dim": None if hpc_dim is None else int(hpc_dim),
            "proj_seed": int(proj_seed),
            "details_seed": None if details_seed is None else int(details_seed),
            "details_max_len": None if details_max_len is None else int(details_max_len),
        },
        "rows": rows,
    }


def sweep_beta(
    *,
    ids: List[str],
    doc_emb: np.ndarray,
    query_emb: np.ndarray,
    details_by_id: Dict[str, str],
    beta_values: List[float],
    decay_rate: float,
    n_auto_iters: int,
    max_chars: int,
    hard_hetero: bool,
    hpc_dim: int | None = 512,
    proj_seed: int = 0,
    details_seed: int | None = None,
    details_max_len: int | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for b in beta_values:
        res = evaluate(
            ids=ids,
            doc_emb=doc_emb,
            query_emb=query_emb,
            details_by_id=details_by_id,
            cfg=EvalConfig(
                decay_rate=float(decay_rate),
                beta=float(b),
                n_auto_iters=int(n_auto_iters),
                max_chars=int(max_chars),
                hard_hetero=bool(hard_hetero),
                hpc_dim=hpc_dim,
                proj_seed=int(proj_seed),
                details_seed=details_seed,
                details_max_len=details_max_len,
            ),
        )
        rows.append(
            {
                "beta": float(b),
                "selection_rate": float(res["selection_accuracy"]["rate"]),
                "recall_rate": float(res["recall_accuracy_canon"]["rate"]),
            }
        )
    return {
        "cfg": {
            "decay_rate": float(decay_rate),
            "selection_method": "auto_mhn",
            "n_auto_iters": int(n_auto_iters),
            "max_chars": int(max_chars),
            "hard_hetero": bool(hard_hetero),
            "hpc_dim": None if hpc_dim is None else int(hpc_dim),
            "proj_seed": int(proj_seed),
            "details_seed": None if details_seed is None else int(details_seed),
            "details_max_len": None if details_max_len is None else int(details_max_len),
        },
        "rows": rows,
    }
