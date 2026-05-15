from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .cache_details import DetailsConfig, build_or_load_details
from .cache_xrag import XragCacheConfig, build_or_load_xrag_cache
from .model_eval import _maybe_project, evaluate, sweep_beta, sweep_decay, EvalConfig
from .paths import OutputPaths, default_outputs_root
from .plotting import plot_three_panel


def _load_npz(npz_path: Path) -> Dict[str, Any]:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_matches(path: Path, expected: Dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        got = _load_json(path)
    except Exception:
        return False
    cfg = got.get("cfg")
    if not isinstance(cfg, dict):
        return False
    for k, v in expected.items():
        if cfg.get(k) != v:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Final hippocampus (dual-MHN) cached pipeline on 100 typical Raykov stimuli.")
    ap.add_argument("--out_root", type=str, default=str(default_outputs_root()))
    ap.add_argument("--force_cache", action="store_true", help="Rebuild xRAG embedding cache + details.")
    ap.add_argument("--force_results", action="store_true", help="Recompute sweeps even if cached JSON exists.")

    # Stimuli / query
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stories_csv", type=str, default="data/stories_train.csv")
    ap.add_argument("--query_excerpt_chars", type=int, default=80)
    ap.add_argument("--retriever_query_clip_chars", type=int, default=80)
    ap.add_argument("--use_mps", action="store_true")

    # Details / decode
    ap.add_argument("--details_max_len", type=int, default=40)
    ap.add_argument("--max_chars", type=int, default=60)
    ap.add_argument("--hpc_dim", type=int, default=512, help="Project xRAG embeddings to this dim for hippocampus decoding.")
    ap.add_argument("--proj_seed", type=int, default=0)

    # Baseline params
    ap.add_argument("--baseline_decay", type=float, default=0.9)
    ap.add_argument("--baseline_beta", type=float, default=400.0)
    ap.add_argument("--decay_sweep_beta", type=float, default=200.0, help="Fixed beta for the decay sweep (choose lower to see structure).")
    ap.add_argument("--n_auto_iters", type=int, default=1)
    ap.add_argument("--hard_hetero", action="store_true", help="Use hard hetero transitions (default: soft).")

    args = ap.parse_args()

    out = OutputPaths(Path(args.out_root).resolve()).ensure()
    figures_dir = Path(__file__).resolve().parent.parent / "figures"

    cache_npz = out.cache_dir / "embeddings_doc_query.npz"
    cache_meta = out.cache_dir / "embeddings_doc_query.meta.json"
    details_json = out.cache_dir / f"details_by_id.seed{int(args.seed)}.maxlen{int(args.details_max_len)}.json"
    baseline_json = out.results_dir / "baseline.json"
    sweep_decay_json = out.results_dir / "sweep_decay.json"
    sweep_beta_json = out.results_dir / "sweep_beta.json"

    # Remove legacy plots we no longer generate.
    for legacy in [
        "query_vs_doc_margin.png",
        "query_vs_doc_pca_pairs.png",
        "query_vs_doc_heatmap.png",
        "recall_vs_decay.png",
        "recall_vs_beta.png",
    ]:
        try:
            (out.plots_dir / legacy).unlink(missing_ok=True)
        except Exception:
            pass

    # ---- 1) Cache xRAG embeddings (doc + query) ----
    xcfg = XragCacheConfig(
        n=int(args.n),
        seed=int(args.seed),
        category="typical",
        stories_csv=str(args.stories_csv),
        query_excerpt_chars=int(args.query_excerpt_chars),
        retriever_query_clip_chars=int(args.retriever_query_clip_chars),
        use_mps=bool(args.use_mps),
    )
    build_or_load_xrag_cache(xcfg, out_npz=cache_npz, out_meta_json=cache_meta, force=bool(args.force_cache))
    arr = _load_npz(cache_npz)
    ids = [str(x) for x in arr["ids"].tolist()]
    doc_emb = np.asarray(arr["doc_emb"], dtype=np.float32)
    query_emb = np.asarray(arr["query_emb"], dtype=np.float32)

    # ---- 2) Cache details (random strings) ----
    build_or_load_details(ids, DetailsConfig(seed=int(args.seed), max_len=int(args.details_max_len)), out_json=details_json, force=bool(args.force_cache))
    details_by_id = json.loads(Path(details_json).read_text(encoding="utf-8"))

    # ---- 3) Baseline evaluation (80-char query; report selection + recall) ----
    expected_base_cfg = {
        "decay_rate": float(args.baseline_decay),
        "beta": float(args.baseline_beta),
        "selection_method": "auto_mhn",
        "n_auto_iters": int(args.n_auto_iters),
        "hard_hetero": bool(args.hard_hetero),
        "denoise_after_hetero": False,
        "n_hetero_iters": 1,
        "max_chars": int(args.max_chars),
        "hpc_dim": int(args.hpc_dim),
        "proj_seed": int(args.proj_seed),
        "details_seed": int(args.seed),
        "details_max_len": int(args.details_max_len),
    }
    if bool(args.force_results) or (not _json_matches(baseline_json, expected_base_cfg)):
        base = evaluate(
            ids=ids,
            doc_emb=doc_emb,
            query_emb=query_emb,
            details_by_id=details_by_id,
            cfg=EvalConfig(
                decay_rate=float(args.baseline_decay),
                beta=float(args.baseline_beta),
                n_auto_iters=int(args.n_auto_iters),
                hard_hetero=bool(args.hard_hetero),
                max_chars=int(args.max_chars),
                hpc_dim=int(args.hpc_dim),
                proj_seed=int(args.proj_seed),
                details_seed=int(args.seed),
                details_max_len=int(args.details_max_len),
            ),
        )
        baseline_json.write_text(json.dumps(base, indent=2), encoding="utf-8")

    # ---- 4) Sweep decay ----
    expected_decay_cfg = {
        "beta": float(args.decay_sweep_beta),
        "selection_method": "auto_mhn",
        "n_auto_iters": int(args.n_auto_iters),
        "max_chars": int(args.max_chars),
        "hard_hetero": bool(args.hard_hetero),
        "hpc_dim": int(args.hpc_dim),
        "proj_seed": int(args.proj_seed),
        "details_seed": int(args.seed),
        "details_max_len": int(args.details_max_len),
    }
    if bool(args.force_results) or (not _json_matches(sweep_decay_json, expected_decay_cfg)):
        # Include a wider range to show where recall becomes unstable.
        decay_vals = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]
        sd = sweep_decay(
            ids=ids,
            doc_emb=doc_emb,
            query_emb=query_emb,
            details_by_id=details_by_id,
            decay_values=decay_vals,
            beta=float(args.decay_sweep_beta),
            n_auto_iters=int(args.n_auto_iters),
            max_chars=int(args.max_chars),
            hard_hetero=bool(args.hard_hetero),
            hpc_dim=int(args.hpc_dim),
            proj_seed=int(args.proj_seed),
            details_seed=int(args.seed),
            details_max_len=int(args.details_max_len),
        )
        sweep_decay_json.write_text(json.dumps(sd, indent=2), encoding="utf-8")
    else:
        sd = _load_json(sweep_decay_json)

    # ---- 5) Sweep beta (same beta for auto + hetero) ----
    expected_beta_cfg = {
        "decay_rate": float(args.baseline_decay),
        "selection_method": "auto_mhn",
        "n_auto_iters": int(args.n_auto_iters),
        "max_chars": int(args.max_chars),
        "hard_hetero": bool(args.hard_hetero),
        "hpc_dim": int(args.hpc_dim),
        "proj_seed": int(args.proj_seed),
        "details_seed": int(args.seed),
        "details_max_len": int(args.details_max_len),
    }
    if bool(args.force_results) or (not _json_matches(sweep_beta_json, expected_beta_cfg)):
        beta_vals = [5, 10, 20, 30, 50, 80, 100, 120, 150, 200, 300, 400, 600]
        sb = sweep_beta(
            ids=ids,
            doc_emb=doc_emb,
            query_emb=query_emb,
            details_by_id=details_by_id,
            beta_values=[float(b) for b in beta_vals],
            decay_rate=float(args.baseline_decay),
            n_auto_iters=int(args.n_auto_iters),
            max_chars=int(args.max_chars),
            hard_hetero=bool(args.hard_hetero),
            hpc_dim=int(args.hpc_dim),
            proj_seed=int(args.proj_seed),
            details_seed=int(args.seed),
            details_max_len=int(args.details_max_len),
        )
        sweep_beta_json.write_text(json.dumps(sb, indent=2), encoding="utf-8")
    else:
        sb = _load_json(sweep_beta_json)

    plot_query_emb = _maybe_project(query_emb, d_out=int(args.hpc_dim), seed=int(args.proj_seed))
    plot_doc_emb = _maybe_project(doc_emb, d_out=int(args.hpc_dim), seed=int(args.proj_seed))
    plot_three_panel(
        plot_query_emb,
        plot_doc_emb,
        decay_rows=sd["rows"],
        beta_rows=sb["rows"],
        out_path=figures_dir / "Figure S2.pdf",
    )

    # ---- Print summary ----
    base = json.loads(baseline_json.read_text(encoding="utf-8"))
    print("[Baseline]")
    print(json.dumps({"selection": base["selection_accuracy"], "recall": base["recall_accuracy_canon"]}, indent=2))
    print(f"✓ Cache: {cache_npz}")
    print(f"✓ Plots: {out.plots_dir}")


if __name__ == "__main__":
    main()
