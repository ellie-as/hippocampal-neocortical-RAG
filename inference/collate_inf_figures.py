"""
Collate inference figures into manuscript-numbered publication-quality PDFs.

Figure 6 (inference behaviour):
  a) Aggregated inference plot (accuracy by hops, spatial vs family)
  b) Accuracy by grid size
  c) Fraction valid vs number of transitions (temp & validity)
  d) Mean max distance vs temperature

Figure 7 (internal representations):
  Spatial task:
    a) Baseline GPT-2: combined PCA (top) + single PCA (bottom)
    b) Our model: combined PCA (top) + single PCA (bottom)
  Family task:
    c) Baseline GPT-2: combined PCA (top) + single PCA (bottom)
    d) Our model: combined PCA (top) + single PCA (bottom)
  Bottom row:
    e) Baseline GPT-2: box plot
    f) Our model: box plot
    g) Pearson correlation vs layer index
    h) Pearson correlation vs prompt length

All data is computed fresh (for visual consistency) but cached to JSON
files under the configured cache directory so subsequent runs are fast.
"""

from __future__ import annotations

import json
import os
import random
import string
import sys
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(INFERENCE_DIR))
sys.path.insert(0, str(REPO_ROOT))

from run_config import load_config

_CONFIG = load_config()
CACHE_DIR = _CONFIG["cache_dir"]
FIGURES_DIR = _CONFIG["figures_dir"]
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Global rcParams for publication figures
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,   # TrueType fonts in PDF
    "ps.fonttype": 42,
})

# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------
SEED = _CONFIG["seed"]

# ---------------------------------------------------------------------------
# L2-normalise representation vectors before PCA?  (easy toggle)
# ---------------------------------------------------------------------------
L2_NORMALIZE_FOR_PCA = True

# Use only entity occurrences from the second half of each walk?
SECOND_HALF_ONLY = False

# Include flanking space tokens when extracting entity representations?
USE_FLANKING = True

# Spatial representation caches generated before this version used independently
# sampled walks for the GPT-2 baseline and the trained model.
SPATIAL_REPS_CACHE_VERSION = 2
FIGURE_S3_CACHE_VERSION = 1
FIGURE_S3_MATCH_FIGURE7B_LAYER = 12
FIGURE_S3_SCATTER_ALPHA = 0.3
FIGURE_S3_SCATTER_SIZE = 2
FIGURE_S3_LEGEND_MARKER_SIZE = 20


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


set_seed()

# ---------------------------------------------------------------------------
# Model / tokenizer helpers
# ---------------------------------------------------------------------------
SPATIAL_MODEL_DIR = _CONFIG["spatial_model_dir"]
FAMILY_MODEL_DIR = _CONFIG["family_model_dir"]


def configure(config_path: str | Path | None = None) -> None:
    """Apply a runtime config to module-level paths used by data generators."""
    global _CONFIG, CACHE_DIR, FIGURES_DIR, SPATIAL_MODEL_DIR, FAMILY_MODEL_DIR, SEED

    _CONFIG = load_config(config_path)
    CACHE_DIR = _CONFIG["cache_dir"]
    FIGURES_DIR = _CONFIG["figures_dir"]
    SPATIAL_MODEL_DIR = _CONFIG["spatial_model_dir"]
    FAMILY_MODEL_DIR = _CONFIG["family_model_dir"]
    SEED = _CONFIG["seed"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    print(f"Using inference config: {_CONFIG['config_path']}")
    print(f"  spatial_model_dir: {SPATIAL_MODEL_DIR}")
    print(f"  family_model_dir:  {FAMILY_MODEL_DIR}")
    print(f"  cache_dir:         {CACHE_DIR}")
    print(f"  figures_dir:       {FIGURES_DIR}")


def _has_weights(model_dir: Path) -> bool:
    return (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()


def _entity_tokenizer_dir() -> Path | None:
    path = INFERENCE_DIR / "tokenizers" / "gpt2-medium_2letter_entities"
    if (path / "tokenizer.json").exists() or (path / "vocab.json").exists():
        return path
    return None


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_cache(name: str) -> dict | None:
    path = CACHE_DIR / f"{name}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            print(f"  [cache] Loaded {name}.json")
            return data
        except Exception:
            pass
    return None


def _save_cache(name: str, data: dict) -> None:
    path = CACHE_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2, default=_json_default))
    print(f"  [cache] Saved {name}.json")


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _cache_has_vectors(data) -> bool:
    if isinstance(data, dict):
        if "vector" in data:
            return True
        return any(_cache_has_vectors(v) for v in data.values())
    if isinstance(data, list):
        return any(_cache_has_vectors(v) for v in data)
    return False


# ============================================================================
# FIGURE 5 DATA GENERATORS
# ============================================================================

def _generate_name() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=2))


def _load_model(model_dir: Path):
    """Load a GPT model, preferring rag_inference.GPT (tokenizer-aware) with fallback.

    The models ship with their custom tokenizer (vocab 50445) in the model
    directory itself, so we load the tokenizer from there (tokenizer_name=None).
    """
    try:
        from rag_composition import GPT
        return GPT(
            base_model=str(model_dir),
            base_model_name="gpt2",
            tokenizer_name=None,
        )
    except Exception:
        from graph_sequence_model import GPT
        return GPT(base_model=str(model_dir), base_model_name="gpt2")


def _test_loop_greedy_fast(
    model,
    loop_templates: list[str | tuple[str, list[int]]],
    n_iters: int = 100,
) -> tuple[float, dict[str, float]]:
    """Lightweight version of test_loop_greedy with configurable iteration count."""
    accuracy_scores: list[int] = []
    results_dict: dict[str, float] = {}

    for item in loop_templates:
        if isinstance(item, tuple):
            template, entity_indices = item
            n_unique = max(entity_indices) + 1
        else:
            template = item
            entity_indices = None

        template_accuracy: list[int] = []

        for _ in range(n_iters):
            if entity_indices is not None:
                names = [_generate_name() for _ in range(n_unique)]
                fill_args = [names[i] for i in entity_indices]
                filled_template = template.format(*fill_args)
                true_final_item = fill_args[-1]
            else:
                names = [_generate_name() for _ in range(template.count("{}") - 1)]
                names += [names[0]]
                filled_template = template.format(*names)
                true_final_item = names[-1]

            input_len = len(filled_template.split())
            prediction = model.continue_input(filled_template[0:-3], max_new_tokens=5, do_sample=False)
            predicted_items = prediction.strip().split()[0:input_len]
            predicted_final_item = predicted_items[-1] if predicted_items else None
            is_correct = int(predicted_final_item == true_final_item)
            template_accuracy.append(is_correct)

        accuracy_scores.extend(template_accuracy)
        results_dict[template] = sum(template_accuracy) / len(template_accuracy)

    overall_avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    return overall_avg_accuracy, results_dict


def _get_aggregated_inference_data(smoke: bool = False) -> dict | None:
    """Run the inference loop templates and return {task: {pattern: accuracy}}."""
    cached = _load_cache("aggregated_inf")
    if cached is not None:
        return cached

    n_iters = 10 if smoke else 100
    data = {}

    # Spatial
    if _has_weights(SPATIAL_MODEL_DIR):
        model = _load_model(SPATIAL_MODEL_DIR)
        loop_templates_spatial = [
            "{} EAST {} WEST {}",
            "{} WEST {} EAST {}",
            "{} NORTH {} SOUTH {}",
            "{} SOUTH {} NORTH {}",
            "{} EAST {} SOUTH {} WEST {} NORTH {}",
            "{} SOUTH {} WEST {} NORTH {} EAST {}",
            "{} WEST {} NORTH {} EAST {} SOUTH {}",
            "{} NORTH {} EAST {} SOUTH {} WEST {}",
            "{} EAST {} EAST {} NORTH {} WEST {} WEST {} SOUTH {}",
            "{} NORTH {} NORTH {} WEST {} SOUTH {} SOUTH {} EAST {}",
            # 8-hop: outer perimeter of 3×3 grid
            "{} EAST {} EAST {} SOUTH {} SOUTH {} WEST {} WEST {} NORTH {} NORTH {}",
            "{} SOUTH {} SOUTH {} WEST {} WEST {} NORTH {} NORTH {} EAST {} EAST {}",
        ]
        if smoke:
            loop_templates_spatial = loop_templates_spatial[:4]  # reduce templates in smoke mode
        _, spatial_results = _test_loop_greedy_fast(model, loop_templates_spatial, n_iters=n_iters)
        data["Spatial"] = spatial_results
    else:
        print("  [aggregated_inf] Spatial model not found")

    # Family
    if _has_weights(FAMILY_MODEL_DIR):
        model = _load_model(FAMILY_MODEL_DIR)
        loop_templates_family: list[str | tuple[str, list[int]]] = [
            "{} CHILD_OF {} PARENT_OF {}",
            "{} PARENT_OF {} CHILD_OF {}",
            "{} GRANDCHILD_OF {} GRANDPARENT_OF {}",
            "{} GRANDPARENT_OF {} GRANDCHILD_OF {}",
            ("{} CHILD_OF {} CHILD_OF {} PARENT_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
            ("{} PARENT_OF {} PARENT_OF {} CHILD_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
            ("{} CHILD_OF {} SPOUSE_OF {} SPOUSE_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
            ("{} PARENT_OF {} SPOUSE_OF {} SPOUSE_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
            "{} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {} GRANDPARENT_OF {} SIBLING_OF {}",
            "{} GRANDPARENT_OF {} SIBLING_OF {} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {}",
        ]
        if smoke:
            loop_templates_family = loop_templates_family[:4]  # reduce templates in smoke mode
        _, family_results = _test_loop_greedy_fast(model, loop_templates_family, n_iters=n_iters)
        data["Family tree"] = family_results
    else:
        print("  [aggregated_inf] Family model not found")

    if data:
        _save_cache("aggregated_inf", data)
    return data if data else None


def _test_loop_sampled_fast(
    model,
    loop_templates: list[str],
    n_iters: int = 50,
) -> tuple[float, dict[str, float]]:
    """Lightweight version of test_loop_sampled with configurable iteration count."""
    accuracy_scores: list[int] = []
    results_dict: dict[str, float] = {}

    for template in loop_templates:
        template_accuracy: list[int] = []

        for _ in range(n_iters):
            names = [_generate_name() for _ in range(template.count("{}") - 1)]
            names += [names[0]]
            filled_template = template.format(*names)

            true_final_item = names[-1]
            input_len = len(filled_template.split())

            prediction = model.continue_input(
                filled_template[0:-3],
                max_new_tokens=5,
                do_sample=True,
                temperature=1.0,
                num_beams=5,
            )
            predicted_items = prediction.strip().split()[0:input_len]
            predicted_final_item = predicted_items[-1] if predicted_items else None
            is_correct = int(predicted_final_item == true_final_item)
            template_accuracy.append(is_correct)

        accuracy_scores.extend(template_accuracy)
        results_dict[template] = sum(template_accuracy) / len(template_accuracy)

    overall_avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    return overall_avg_accuracy, results_dict


def _get_grid_generalisation_data(smoke: bool = False) -> dict | None:
    """Run grid generalisation tests and return {grid_size: (mean_acc, sem)}."""
    cached = _load_cache("grid_generalisation")
    if cached is not None:
        return cached

    if not _has_weights(SPATIAL_MODEL_DIR):
        print("  [grid_gen] Spatial model not found")
        return None

    from graph_sequence_model import generate_loop_templates

    model = _load_model(SPATIAL_MODEL_DIR)

    n_iters = 5 if smoke else 50
    max_grid = 3 if smoke else 5
    loop_templates_dict = generate_loop_templates(min_n=1, max_n=max_grid)

    results = {}
    for n, templates in loop_templates_dict.items():
        if smoke:
            templates = templates[:2]  # only 2 templates per grid size in smoke mode
        accuracies = []
        for template in templates:
            accuracy, _ = _test_loop_sampled_fast(model, [template], n_iters=n_iters)
            accuracies.append(accuracy)
        mean_acc = float(np.mean(accuracies))
        sem_acc = float(np.std(accuracies, ddof=1) / np.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0
        results[str(n)] = [mean_acc, sem_acc]

    _save_cache("grid_generalisation", results)
    return results


def _get_imagination_data(smoke: bool = False) -> dict | None:
    """Run imagination experiment and return structured data for temp/validity + distance plots."""
    cached = _load_cache("imagination")
    if cached is not None:
        return cached

    if not _has_weights(SPATIAL_MODEL_DIR):
        print("  [imagination] Spatial model not found")
        return None

    from graph_sequence_model import track_coordinates

    model = _load_model(SPATIAL_MODEL_DIR)

    n_imagined = 10 if smoke else 50
    imagined_for_temps: dict[float, list[str]] = {}
    for temp in [0, 0.5, 1.0, 1.5, 2.0]:
        imagined: list[str] = []
        for _ in range(n_imagined):
            if temp == 0:
                prediction = model.continue_input(_generate_name(), do_sample=False, max_new_tokens=50)
            else:
                prediction = model.continue_input(
                    _generate_name(), do_sample=True, max_new_tokens=50, temperature=temp
                )
            imagined.append(prediction)
        imagined_for_temps[temp] = imagined

    # Temp & validity data
    lengths = [1, 2, 3, 4, 5, 6]
    validity_data: dict[str, list[float]] = {}
    for temp in [0.5, 1.0, 1.5, 2.0]:
        fractions = []
        for length in lengths:
            valid_count = 0
            for path in imagined_for_temps[temp]:
                shortened = " ".join(path.split()[:2 * length + 1])
                if track_coordinates([shortened]):
                    valid_count += 1
            fractions.append(valid_count / len(imagined_for_temps[temp]))
        validity_data[str(float(temp))] = fractions

    # Distance data
    direction_offsets = {"NORTH": (0, 1), "SOUTH": (0, -1), "EAST": (1, 0), "WEST": (-1, 0)}

    def path_to_max_distance(path: str) -> int:
        x, y = 0, 0
        max_d = 0
        for step in path.split():
            if step in direction_offsets:
                dx, dy = direction_offsets[step]
                x += dx
                y += dy
                max_d = max(max_d, abs(x) + abs(y))
        return max_d

    distance_data: dict[str, dict] = {}
    for temp in [0, 0.5, 1.0, 1.5, 2.0]:
        dists = [path_to_max_distance(p) for p in imagined_for_temps[temp]]
        distance_data[str(float(temp))] = {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "all": dists,
        }

    result = {
        "lengths": lengths,
        "validity": validity_data,
        "distances": distance_data,
    }
    _save_cache("imagination", result)
    return result


# ============================================================================
# FIGURE 6 DATA GENERATORS (representation geometry)
# ============================================================================

def _get_spatial_reps_data(smoke: bool = False) -> dict | None:
    """Gather spatial representation data for both our model and GPT-2 baseline."""
    cached = _load_cache("spatial_reps")
    if cached is not None:
        if cached.get("_cache_version") != SPATIAL_REPS_CACHE_VERSION:
            print("  [cache] Ignoring spatial_reps.json from an older sampling scheme")
        else:
            if _cache_has_vectors(cached):
                cached = _compact_rep_cache(cached)
                _save_cache("spatial_reps", cached)
            return cached

    if not _has_weights(SPATIAL_MODEL_DIR):
        print("  [spatial_reps] Spatial model not found")
        return None

    from plot_reps import (
        GPTWrapper,
        build_grid_graph,
        generate_random_walk,
        average_locations_via_substring,
        calc_pearson_correlation,
        gather_boxplot_data,
        merge_dist_map,
    )

    if smoke:
        n_pca, n_corr = 5, 3
        ctx_lens = [10, 30]
        layers = [6, 12, 18]
    else:
        n_pca, n_corr = 900, 300
        ctx_lens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        layers = list(range(24))

    # Precompute identical graph labels, grid positions, and prompts for both
    # models so GPT-2 and the trained model are evaluated on matched samples.
    print("  [spatial_reps] Precomputing shared spatial prompts...")
    pca_runs = []
    for run in range(n_pca):
        G, node_names, grid_pos = build_grid_graph()
        pca_runs.append((generate_random_walk(G, 50), node_names, grid_pos))

    G_single, names_single, grid_pos_single = build_grid_graph()
    single_run = (
        generate_random_walk(G_single, 100000),
        names_single,
        grid_pos_single,
        list(G_single.edges()),
    )

    layer_runs = []
    for _ in range(n_corr):
        G_l, names_l, gp_l = build_grid_graph()
        layer_runs.append((generate_random_walk(G_l, 50), names_l, gp_l))

    ctx_runs = []
    for _ in range(n_corr):
        G_c, names_c, gp_c = build_grid_graph()
        prompts_by_len = {int(L): generate_random_walk(G_c, L) for L in ctx_lens}
        ctx_runs.append((prompts_by_len, names_c, gp_c))

    models_to_test = ["gpt2-medium", str(SPATIAL_MODEL_DIR)]
    layer_idx = 12
    result: dict = {
        "_cache_version": SPATIAL_REPS_CACHE_VERSION,
        "_shared_sampling": {
            "n_pca": n_pca,
            "n_corr": n_corr,
            "ctx_lens": ctx_lens,
            "layers": layers,
            "walk_length": 50,
            "single_walk_length": 100000,
        },
    }

    for model_name in models_to_test:
        key = "our_model" if "outputs_graph" in model_name else "gpt2_medium"
        print(f"  [spatial_reps] Processing {key}...")
        wrapper = GPTWrapper(model_name)
        s3_points_by_layer: dict[int, list[dict]] | None = {li: [] for li in layers} if key == "our_model" else None

        # --- Combined PCA data ---
        all_points = []
        global_dist: dict[int, list[float]] = {}
        for run, (prompt, node_names, grid_pos) in enumerate(pca_runs):
            if (run + 1) % 50 == 0:
                print(f"    PCA run {run + 1}/{n_pca}")
            hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
            loc_repr = average_locations_via_substring(prompt, off, hs, node_names,
                                                       flanking=USE_FLANKING,
                                                       second_half_only=SECOND_HALF_ONLY)
            for n in node_names:
                if n in loc_repr:
                    all_points.append({
                        "vector": loc_repr[n].tolist(),
                        "grid_position": list(grid_pos[n]),
                    })
            merge_dist_map(global_dist, gather_boxplot_data(node_names, loc_repr, grid_pos))

        # --- Single PCA example ---
        prompt_single, names_single, grid_pos_single, single_edges_raw = single_run
        hs_s, off_s = wrapper.get_hidden_states_with_offsets(prompt_single, layer_idx)
        loc_repr_single = average_locations_via_substring(prompt_single, off_s, hs_s, names_single,
                                                          flanking=USE_FLANKING,
                                                          second_half_only=SECOND_HALF_ONLY)
        single_points = []
        single_edges = []
        for n in names_single:
            if n in loc_repr_single:
                single_points.append({
                    "name": n,
                    "vector": loc_repr_single[n].tolist(),
                    "grid_position": list(grid_pos_single.get(n, (0, 0))),
                })
        for u, v in single_edges_raw:
            if u in loc_repr_single and v in loc_repr_single:
                single_edges.append([u, v])

        # --- Correlation vs layer (single forward pass per prompt) ---
        corrs_by_layer: dict[int, list[float]] = {li: [] for li in layers}
        for ri, (prompt_l, names_l, gp_l) in enumerate(layer_runs):
            if (ri + 1) % 25 == 0:
                print(f"    corr-vs-layer run {ri + 1}/{n_corr}")
            all_hs, off_l = wrapper.get_all_hidden_states_with_offsets(prompt_l)
            for li in layers:
                loc_l = average_locations_via_substring(prompt_l, off_l, all_hs[li], names_l,
                                                       flanking=USE_FLANKING,
                                                       second_half_only=SECOND_HALF_ONLY)
                if s3_points_by_layer is not None:
                    for node_name in names_l:
                        if node_name not in loc_l:
                            continue
                        gx, gy = gp_l[node_name]
                        s3_points_by_layer[li].append({
                            "vector": loc_l[node_name],
                            "run_idx": int(ri),
                            "grid_position": [int(gx), int(gy)],
                        })
                c = calc_pearson_correlation(names_l, loc_l, gp_l)
                if not np.isnan(c):
                    corrs_by_layer[li].append(c)

        corr_by_layer: dict[str, float] = {}
        for li in layers:
            corr_by_layer[str(li)] = float(np.mean(corrs_by_layer[li])) if corrs_by_layer[li] else float("nan")
            print(f"    [{key}] Layer {li}: r={corr_by_layer[str(li)]:.3f}")

        # --- Correlation vs context length ---
        corr_by_ctx: dict[str, float] = {}
        for L in ctx_lens:
            corrs = []
            for prompts_by_len, names_c, gp_c in ctx_runs:
                prompt_c = prompts_by_len[int(L)]
                hs_c, off_c = wrapper.get_hidden_states_with_offsets(prompt_c, layer_idx)
                loc_c = average_locations_via_substring(prompt_c, off_c, hs_c, names_c,
                                                       flanking=USE_FLANKING,
                                                       second_half_only=SECOND_HALF_ONLY)
                c = calc_pearson_correlation(names_c, loc_c, gp_c)
                if not np.isnan(c):
                    corrs.append(c)
            corr_by_ctx[str(L)] = float(np.mean(corrs)) if corrs else float("nan")

        # Serialise boxplot data
        boxplot_data = {str(k): v for k, v in global_dist.items()}

        result[key] = {
            "all_points": _compact_points_with_pca(all_points),
            "boxplot": boxplot_data,
            "single_points": _compact_points_with_pca(single_points),
            "single_edges": single_edges,
            "corr_by_layer": corr_by_layer,
            "corr_by_ctx": corr_by_ctx,
            "ctx_lens": ctx_lens,
            "layers": layers,
        }

        if s3_points_by_layer is not None:
            result["figure_s3_layer_pca"] = _build_figure_s3_cache(s3_points_by_layer)

    if "figure_s3_layer_pca" in result:
        _save_cache("figure_s3_spatial_pca", result["figure_s3_layer_pca"])
    _save_cache("spatial_reps", result)
    return result

def _get_family_reps_data(smoke: bool = False) -> dict | None:
    """Gather family-tree representation data for both our model and GPT-2 baseline."""
    cached = _load_cache("family_reps")
    if cached is not None:
        if _cache_has_vectors(cached):
            cached = _compact_rep_cache(cached)
            _save_cache("family_reps", cached)
        return cached

    if not _has_weights(FAMILY_MODEL_DIR):
        print("  [family_reps] Family model not found")
        return None

    from plot_family_reps import (
        GPTWrapper,
        build_family_tree,
        generate_random_walk,
        average_locations_via_substring,
        gather_boxplot_data,
        merge_dist_map,
        GENERATION_LABELS,
    )

    if smoke:
        n_pca = 5
    else:
        n_pca = 900

    models_to_test = ["gpt2-medium", str(FAMILY_MODEL_DIR)]
    layer_idx = 12
    result: dict = {}

    for model_name in models_to_test:
        key = "our_model" if "outputs_tree" in model_name else "gpt2_medium"
        print(f"  [family_reps] Processing {key}...")
        wrapper = GPTWrapper(model_name)

        # --- Combined PCA data ---
        all_points = []
        global_dist: dict[int, list[float]] = {}
        for run in range(n_pca):
            if (run + 1) % 50 == 0:
                print(f"    PCA run {run + 1}/{n_pca}")
            G, node_names, gen_map = build_family_tree()
            prompt = generate_random_walk(G, 50)
            hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
            loc_repr = average_locations_via_substring(prompt, off, hs, node_names,
                                                       flanking=USE_FLANKING,
                                                       second_half_only=SECOND_HALF_ONLY)
            for n in node_names:
                if n in loc_repr:
                    all_points.append({
                        "vector": loc_repr[n].tolist(),
                        "generation": gen_map[n],
                    })
            merge_dist_map(global_dist, gather_boxplot_data(node_names, loc_repr, gen_map))

        # --- Single PCA example ---
        G_single, names_single, gen_map_single = build_family_tree()
        prompt_single = generate_random_walk(G_single, 100000)
        hs_s, off_s = wrapper.get_hidden_states_with_offsets(prompt_single, layer_idx)
        loc_repr_single = average_locations_via_substring(prompt_single, off_s, hs_s, names_single,
                                                          flanking=USE_FLANKING,
                                                          second_half_only=SECOND_HALF_ONLY)
        single_points = []
        single_edges = []
        for n in names_single:
            if n in loc_repr_single:
                single_points.append({
                    "name": n,
                    "vector": loc_repr_single[n].tolist(),
                    "generation": gen_map_single[n],
                })
        for u, v in G_single.edges():
            if u in loc_repr_single and v in loc_repr_single:
                single_edges.append([u, v])

        boxplot_data = {str(k): v for k, v in global_dist.items()}

        result[key] = {
            "all_points": _compact_points_with_pca(all_points),
            "boxplot": boxplot_data,
            "single_points": _compact_points_with_pca(single_points),
            "single_edges": single_edges,
        }

    result["generation_labels"] = {str(k): v for k, v in GENERATION_LABELS.items()}
    _save_cache("family_reps", result)
    return result


# ============================================================================
# PLOTTING HELPERS
# ============================================================================

COLORS = {"spatial": "red", "family": "blue", "gpt2": "blue", "ours": "red"}


def _pca_2d(vectors: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA
    if len(vectors) < 3:
        return np.zeros((len(vectors), 2))
    X = vectors.copy()
    if L2_NORMALIZE_FOR_PCA:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    X = X - X.mean(axis=0, keepdims=True)
    n_components = min(2, X.shape[0], X.shape[1])
    result = PCA(n_components=n_components, random_state=42).fit_transform(X)
    if result.shape[1] < 2:
        result = np.column_stack([result, np.zeros(len(result))])
    return result


def _coords_from_points(points: list[dict]) -> np.ndarray:
    """Return stored PCA coordinates, with vector-based old-cache fallback."""
    if not points:
        return np.zeros((0, 2))
    first = points[0]
    if "pc1" in first and "pc2" in first:
        return np.asarray([[float(p["pc1"]), float(p["pc2"])] for p in points], dtype=float)
    if "PC1" in first and "PC2" in first:
        return np.asarray([[float(p["PC1"]), float(p["PC2"])] for p in points], dtype=float)
    if "vector" in first:
        return _pca_2d(np.asarray([p["vector"] for p in points], dtype=float))
    raise KeyError("Representation points must contain pc1/pc2 coordinates or legacy vector values.")


def _compact_points_with_pca(points: list[dict]) -> list[dict]:
    """Replace full representation vectors with the PCA coordinates used for plotting."""
    if not points:
        return []
    if all("pc1" in p and "pc2" in p and "vector" not in p for p in points):
        return points
    coords_2d = _coords_from_points(points)
    compact = []
    for point, coord in zip(points, coords_2d):
        row = {
            key: value
            for key, value in point.items()
            if key not in {"vector", "x2d", "y2d", "PC1", "PC2", "pc1", "pc2"}
        }
        row["pc1"] = float(coord[0])
        row["pc2"] = float(coord[1])
        compact.append(row)
    return compact


def _compact_rep_cache(data: dict) -> dict:
    """Compact spatial/family representation caches in place."""
    for value in data.values():
        if not isinstance(value, dict):
            continue
        if "all_points" in value:
            value["all_points"] = _compact_points_with_pca(value["all_points"])
        if "single_points" in value:
            value["single_points"] = _compact_points_with_pca(value["single_points"])
    return data


def _build_figure_s3_cache(points_by_layer: dict[int, list[dict]]) -> dict:
    """Build compact per-layer PCA coordinates for Supplementary Figure S3."""
    rows: list[dict] = []
    for layer_idx in sorted(points_by_layer):
        points = points_by_layer[layer_idx]
        if not points:
            continue
        compact = _compact_points_with_pca(points)
        for point in compact:
            gx, gy = point["grid_position"]
            rows.append({
                "Layer index": int(layer_idx),
                "PC1": float(point["pc1"]),
                "PC2": float(point["pc2"]),
                "Grid x": int(gx),
                "Grid y": int(gy),
                "Run index": int(point["run_idx"]),
            })
    return {
        "_cache_version": FIGURE_S3_CACHE_VERSION,
        "description": "Supplementary Figure S3 spatial PCA coordinates; full hidden-state vectors are not stored.",
        "points": rows,
    }


def _figure7b_combined_rows_for_s3(spatial_data: dict | None) -> list[dict]:
    """Return Figure 7b combined-PCA coordinates in Figure S3 row format."""
    if not isinstance(spatial_data, dict):
        return []
    our_model = spatial_data.get("our_model")
    if not isinstance(our_model, dict):
        return []
    points = our_model.get("all_points", [])
    if not points:
        return []

    coords_2d = _coords_from_points(points)
    grid_positions = sorted({
        tuple(point["grid_position"])
        for point in points
        if "grid_position" in point
    })
    points_per_run = max(1, len(grid_positions))

    rows = []
    for idx, (point, coord) in enumerate(zip(points, coords_2d)):
        if "grid_position" not in point:
            continue
        gx, gy = point["grid_position"]
        rows.append({
            "Layer index": FIGURE_S3_MATCH_FIGURE7B_LAYER,
            "PC1": float(coord[0]),
            "PC2": float(coord[1]),
            "Grid x": int(gx),
            "Grid y": int(gy),
            "Run index": int(point.get("run_idx", idx // points_per_run)),
        })
    return rows


def _match_figure_s3_layer_to_figure7b(cached: dict | None, spatial_data: dict | None) -> dict | None:
    """Use the Figure 7b combined-PCA coordinates for the matching S3 layer."""
    if not isinstance(cached, dict):
        return cached
    fig7b_rows = _figure7b_combined_rows_for_s3(spatial_data)
    if not fig7b_rows:
        return cached

    old_points = cached.get("points", [])
    old_by_layer: dict[int, list[dict]] = {}
    for row in old_points:
        layer = int(row["Layer index"])
        if layer == FIGURE_S3_MATCH_FIGURE7B_LAYER:
            continue
        old_by_layer.setdefault(layer, []).append(row)

    layers = sorted(set(old_by_layer) | {FIGURE_S3_MATCH_FIGURE7B_LAYER})
    points = []
    for layer in layers:
        if layer == FIGURE_S3_MATCH_FIGURE7B_LAYER:
            points.extend(fig7b_rows)
        else:
            points.extend(old_by_layer.get(layer, []))

    updated = dict(cached)
    updated["points"] = points
    updated["description"] = (
        "Supplementary Figure S3 spatial PCA coordinates; full hidden-state vectors "
        f"are not stored. Layer {FIGURE_S3_MATCH_FIGURE7B_LAYER} reuses the Figure "
        "7b combined-PCA coordinates for rendering consistency."
    )
    _save_cache("figure_s3_spatial_pca", updated)
    return updated


def _compute_figure_s3_spatial_pca_data(smoke: bool = False) -> dict | None:
    """Compute compact Supplementary Figure S3 coordinates without saving vectors."""
    if not _has_weights(SPATIAL_MODEL_DIR):
        print("  [figure_s3] Spatial model not found; cannot compute Figure S3 source data")
        return None

    from plot_reps import (
        GPTWrapper,
        build_grid_graph,
        generate_random_walk,
        average_locations_via_substring,
    )

    if smoke:
        n_runs = 5
        layers = [6, 12, 18]
    else:
        n_runs = 300
        layers = list(range(24))

    print("  [figure_s3] Computing spatial all-layer PCA source data...")
    wrapper = GPTWrapper(str(SPATIAL_MODEL_DIR))
    points_by_layer: dict[int, list[dict]] = {li: [] for li in layers}

    for run_idx in range(n_runs):
        if (run_idx + 1) % 25 == 0:
            print(f"    Figure S3 run {run_idx + 1}/{n_runs}")
        G, node_names, grid_pos = build_grid_graph()
        prompt = generate_random_walk(G, 50)
        all_hs, offsets = wrapper.get_all_hidden_states_with_offsets(prompt)
        for layer_idx in layers:
            loc_repr = average_locations_via_substring(
                prompt,
                offsets,
                all_hs[layer_idx],
                node_names,
                flanking=USE_FLANKING,
                second_half_only=SECOND_HALF_ONLY,
            )
            for node_name in node_names:
                if node_name not in loc_repr:
                    continue
                gx, gy = grid_pos[node_name]
                points_by_layer[layer_idx].append({
                    "vector": loc_repr[node_name],
                    "run_idx": int(run_idx),
                    "grid_position": [int(gx), int(gy)],
                })

    data = _build_figure_s3_cache(points_by_layer)
    _save_cache("figure_s3_spatial_pca", data)
    return data


# ============================================================================
# FIGURE 5 PLOTS
# ============================================================================

def plot_aggregated_inf(ax, data: dict) -> None:
    """Grouped bar chart: accuracy by number of hops for spatial + family tasks."""
    def get_hops(key: str) -> int:
        return len(key.split()) // 2

    combined = data
    averages: dict[int, dict[str, list]] = {}

    for task, results in combined.items():
        for pattern, acc in results.items():
            hops = get_hops(pattern)
            averages.setdefault(hops, {}).setdefault(task, []).append(acc)

    hops_sorted = [h for h in sorted(averages.keys()) if h in (2, 4, 6)]
    tasks = [t for t in ["Spatial", "Family tree"] if t in combined]
    task_colors = {"Spatial": "red", "Family tree": "blue"}

    x = np.arange(len(hops_sorted))
    bar_width = 0.35
    offset = bar_width / 2

    for j, task in enumerate(tasks):
        means = []
        stds = []
        for h in hops_sorted:
            vals = averages[h].get(task, [0])
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals)))
        positions = x - offset + j * bar_width
        ax.bar(
            positions, means, bar_width, yerr=stds,
            label=task, color=task_colors.get(task, "gray"),
            alpha=0.4, capsize=3, edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in hops_sorted])
    ax.set_xlabel("Number of transitions")
    ax.set_ylabel("Average accuracy")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_ylim(0.8, 1.01)
    ax._source_data_rows = [
        {
            "Number of transitions": int(h),
            "Task": task,
            "Average accuracy": float(np.mean(averages[h].get(task, [0]))),
            "SD": float(np.std(averages[h].get(task, [0]))),
        }
        for h in hops_sorted
        for task in tasks
    ]


def plot_grid_generalisation(ax, data: dict) -> None:
    """Line plot: accuracy vs grid size."""
    ns = sorted(int(k) for k in data.keys())
    means = [data[str(n)][0] for n in ns]
    sems = [data[str(n)][1] for n in ns]

    xs = [n + 1 for n in ns]
    ax.errorbar(
        xs, means, yerr=sems,
        fmt="o-", capsize=5, color="blue", linewidth=1.5, markersize=4,
    )
    ax.set_xlabel("Grid size")
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    ax._source_data_rows = [
        {
            "Grid size": int(x),
            "Average accuracy": float(mean),
            "SEM": float(error),
        }
        for x, mean, error in zip(xs, means, sems)
    ]


def plot_temp_validity(ax, data: dict) -> None:
    """Line plot: fraction valid vs number of transitions for different temps."""
    lengths = data["lengths"]
    validity = data["validity"]
    cmap = plt.get_cmap("magma")
    temps = sorted(float(t) for t in validity.keys())
    colors = cmap(np.linspace(0.15, 0.75, len(temps)))

    for idx, temp in enumerate(temps):
        fracs = validity[str(float(temp))]
        ax.plot(lengths, fracs, marker="o", label=f"{temp}", color=colors[idx],
                linewidth=1.5, markersize=4)

    ax.set_xlabel("Number of transitions")
    ax.set_ylabel("Fraction valid")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min(lengths) - 0.3, max(lengths) + 0.3)
    ax.legend(title="Temp.", fontsize=8, title_fontsize=9)
    ax._source_data_rows = [
        {
            "Number of transitions": int(length),
            "Temperature": float(temp),
            "Fraction valid": float(frac),
        }
        for temp in temps
        for length, frac in zip(lengths, validity[str(float(temp))])
    ]


def plot_mean_max_distance(ax, data: dict) -> None:
    """Bar plot: mean max distance from origin vs temperature."""
    distances = data["distances"]
    temps = sorted(float(t) for t in distances.keys())
    means = [distances[str(float(t))]["mean"] for t in temps]
    stds = [distances[str(float(t))]["std"] for t in temps]
    all_dists = [distances[str(float(t))]["all"] for t in temps]

    ax.bar(
        temps, means, yerr=stds, width=0.4, capsize=2,
        color="blue", alpha=0.4, edgecolor="none",
    )
    for i, temp in enumerate(temps):
        x_jitter = np.full(len(all_dists[i]), temp)
        ax.scatter(x_jitter, all_dists[i], color="blue", alpha=0.2, s=8, zorder=3)

    all_flat = [d for dists in all_dists for d in dists]
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean max distance")
    ax.set_ylim(0, max(all_flat) * 1.15 if all_flat else 10)
    ax.set_xlim(min(temps) - 0.3, max(temps) + 0.3)


def plot_rag_summary(ax, all_results: dict) -> None:
    """Paired bar chart for manuscript Figure 6k."""
    summary_conds = [
        ("NC", "NC only"),
        ("HPC", "HPC only"),
        ("Mem-1", "RAG\n(single)"),
        ("RAG-2L", "RAG\n(multi)"),
    ]
    task_labels = {"Spatial": "Spatial", "Family": "Family tree"}
    task_colors = {"Spatial": "red", "Family": "blue"}

    task_agg: dict[str, dict[str, list[int]]] = {}
    for group_name, conds in all_results.items():
        if "4-hop" in group_name:
            continue
        task = group_name.split()[0]
        task_agg.setdefault(task, {cn: [] for cn, _ in summary_conds})
        for cn, _ in summary_conds:
            task_agg[task][cn].extend(conds.get(cn, []))

    tasks = [t for t in ["Spatial", "Family"] if t in task_agg]
    x = np.arange(len(summary_conds))
    bar_width = 0.35
    offset = bar_width / 2

    for j, task in enumerate(tasks):
        means, sems = [], []
        for cn, _ in summary_conds:
            vals = task_agg[task][cn]
            means.append(float(np.mean(vals)) if vals else 0.0)
            sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)

        ax.bar(
            x - offset + j * bar_width,
            means,
            bar_width,
            yerr=sems,
            capsize=3,
            label=task_labels.get(task, task),
            color=task_colors.get(task, "gray"),
            alpha=0.4,
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in summary_conds])
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.12)
    if tasks:
        ax.legend(loc="upper left", fontsize=7)
    ax._source_data_rows = [
        {
            "Condition": label.replace("\n", " "),
            "Task": task_labels.get(task, task),
            "Average accuracy": float(np.mean(task_agg[task][cache_name])) if task_agg[task][cache_name] else 0.0,
            "SEM": float(np.std(task_agg[task][cache_name]) / np.sqrt(len(task_agg[task][cache_name])))
            if len(task_agg[task][cache_name]) > 1 else 0.0,
            "n": len(task_agg[task][cache_name]),
        }
        for cache_name, label in summary_conds
        for task in tasks
    ]


def _get_rag_summary_data() -> dict | None:
    path = CACHE_DIR / "rag_composition.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


# ============================================================================
# FIGURE 6 PLOTS
# ============================================================================

def plot_spatial_boxplot(ax, boxplot_data: dict, title: str = "") -> None:
    """Box plot: representation distance vs grid distance."""
    keys = sorted(int(k) for k in boxplot_data.keys())
    data_lists = [boxplot_data[str(k)] for k in keys]
    if not data_lists:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    lw = 0.6
    ax.boxplot(
        data_lists, showfliers=False, showmeans=False,
        boxprops=dict(linewidth=lw),
        whiskerprops=dict(linewidth=lw),
        capprops=dict(linewidth=lw),
        medianprops=dict(linewidth=lw),
    )
    ax.set_xticks(range(1, len(keys) + 1))
    ax.set_xticklabels(keys)
    ax.set_xlabel("Distance in grid")
    ax.set_ylabel("Distance between rep.s")
    ax.set_ylim(bottom=0)
    if title:
        ax.set_title(title, fontsize=9)


def plot_combined_pca_spatial(ax, points: list[dict], show_legend: bool = True) -> None:
    """Combined PCA scatter: all runs, coloured by grid position."""
    if not points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    coords_2d = _coords_from_points(points)

    grid_positions = sorted(set(tuple(p["grid_position"]) for p in points))
    cmap = plt.get_cmap("tab10")
    pos_to_color = {pos: cmap(i % 10) for i, pos in enumerate(grid_positions)}

    for i, p in enumerate(points):
        pos = tuple(p["grid_position"])
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], color=pos_to_color[pos], alpha=0.3, s=8)

    if show_legend:
        for pos in grid_positions:
            ax.scatter([], [], color=pos_to_color[pos], label=f"{pos}", s=20)
        ax.legend(fontsize=7, ncol=3, loc="upper left", markerscale=1.0, handletextpad=0.1,
                  columnspacing=0.3, borderpad=0.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)


def plot_single_pca_spatial(ax, single_points: list[dict], single_edges: list) -> None:
    """Single PCA: one graph's representations with edge lines, coloured by grid position."""
    if not single_points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    coords_2d = _coords_from_points(single_points)
    name_to_idx = {p["name"]: i for i, p in enumerate(single_points)}

    grid_positions = sorted(set(tuple(p["grid_position"]) for p in single_points))
    cmap = plt.get_cmap("tab10")
    pos_to_color = {pos: cmap(i % 10) for i, pos in enumerate(grid_positions)}

    for i, p in enumerate(single_points):
        pos = tuple(p["grid_position"])
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], color=pos_to_color[pos], s=20, zorder=3)

    for u, v in single_edges:
        if u in name_to_idx and v in name_to_idx:
            i, j = name_to_idx[u], name_to_idx[v]
            ax.plot(
                [coords_2d[i, 0], coords_2d[j, 0]],
                [coords_2d[i, 1], coords_2d[j, 1]],
                "k--", alpha=0.3, linewidth=0.5,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)


def plot_combined_pca_family(ax, points: list[dict], generation_labels: dict | None = None,
                             show_legend: bool = True) -> None:
    """Combined PCA scatter for family tree: coloured by generation."""
    if not points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    coords_2d = _coords_from_points(points)

    gens = sorted(set(p["generation"] for p in points))
    cmap = plt.get_cmap("tab10")
    gen_to_color = {g: cmap(i % 10) for i, g in enumerate(gens)}

    if generation_labels is None:
        generation_labels = {0: "Grandparent", 1: "Parent", 2: "Child"}

    for i, p in enumerate(points):
        g = p["generation"]
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], color=gen_to_color[g], alpha=0.3, s=8)

    if show_legend:
        for g in gens:
            label = generation_labels.get(g, generation_labels.get(str(g), f"Gen {g}"))
            ax.scatter([], [], color=gen_to_color[g], label=label, s=20)
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)


def plot_single_pca_family(ax, single_points: list[dict], single_edges: list,
                           generation_labels: dict | None = None,
                           show_legend: bool = True) -> None:
    """Single PCA: one family tree's representations with edge lines."""
    if not single_points:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    coords_2d = _coords_from_points(single_points)
    name_to_idx = {p["name"]: i for i, p in enumerate(single_points)}

    if generation_labels is None:
        generation_labels = {0: "Grandparent", 1: "Parent", 2: "Child"}

    gens = sorted(set(p["generation"] for p in single_points))
    cmap = plt.get_cmap("tab10")
    gen_to_color = {g: cmap(i % 10) for i, g in enumerate(gens)}

    for i, p in enumerate(single_points):
        g = p["generation"]
        ax.scatter(coords_2d[i, 0], coords_2d[i, 1], color=gen_to_color[g], s=20, zorder=3)

    if show_legend:
        for g in gens:
            label = generation_labels.get(g, generation_labels.get(str(g), f"Gen {g}"))
            ax.scatter([], [], color=gen_to_color[g], label=label, s=20)
        ax.legend(fontsize=7)

    for u, v in single_edges:
        if u in name_to_idx and v in name_to_idx:
            i, j = name_to_idx[u], name_to_idx[v]
            ax.plot(
                [coords_2d[i, 0], coords_2d[j, 0]],
                [coords_2d[i, 1], coords_2d[j, 1]],
                "k--", alpha=0.3, linewidth=0.5,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)


def plot_corr_vs_layer(ax, spatial_data: dict) -> None:
    """Line plot: Pearson correlation vs layer for both models."""
    labels = {"our_model": "Our model", "gpt2_medium": "Pre-trained GPT-2"}
    colors = {"our_model": "red", "gpt2_medium": "blue"}

    all_vals: list[float] = []
    all_layers: list[int] = []
    for key in ["gpt2_medium", "our_model"]:
        if key not in spatial_data:
            continue
        corr = spatial_data[key]["corr_by_layer"]
        layer_indices = sorted(int(k) for k in corr.keys())
        values = [corr[str(l)] for l in layer_indices]
        all_vals.extend(values)
        all_layers.extend(layer_indices)
        ax.plot(
            layer_indices, values, "o--",
            label=labels[key], color=colors[key],
            linewidth=1.5, markersize=3,
        )

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Pearson correlation")
    if all_vals:
        ymin = min(0, min(all_vals) - 0.05)
        ymax = min(1.0, max(all_vals) + 0.05)
        ax.set_ylim(ymin, ymax)
    if all_layers:
        ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)
    ax.legend(fontsize=8)


def plot_corr_vs_context(ax, spatial_data: dict) -> None:
    """Line plot: Pearson correlation vs prompt length for both models."""
    labels = {"our_model": "Our model", "gpt2_medium": "Pre-trained GPT-2"}
    colors = {"our_model": "red", "gpt2_medium": "blue"}

    all_vals: list[float] = []
    all_ctx: list[int] = []
    for key in ["gpt2_medium", "our_model"]:
        if key not in spatial_data:
            continue
        corr = spatial_data[key]["corr_by_ctx"]
        ctx_lens = sorted(int(k) for k in corr.keys())
        values = [corr[str(l)] for l in ctx_lens]
        all_vals.extend(values)
        all_ctx.extend(ctx_lens)
        ax.plot(
            ctx_lens, values, "o--",
            label=labels[key], color=colors[key],
            linewidth=1.5, markersize=3,
        )

    ax.set_xlabel("Context length")
    ax.set_ylabel("Pearson correlation")
    if all_vals:
        ymin = min(0, min(all_vals) - 0.05)
        ymax = min(1.0, max(all_vals) + 0.05)
        ax.set_ylim(ymin, ymax)
    if all_ctx:
        ax.set_xlim(min(all_ctx) - 0.5, max(all_ctx) + 0.5)
    ax.legend(fontsize=8)


def _write_source_csv(filename: str, rows: list[dict]) -> None:
    if not rows:
        return
    out_dir = REPO_ROOT / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with (out_dir / filename).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _figure_s3_cache_from(spatial_data: dict | None, smoke: bool = False) -> dict | None:
    if isinstance(spatial_data, dict):
        cached = spatial_data.get("figure_s3_layer_pca")
        if isinstance(cached, dict) and cached.get("_cache_version") == FIGURE_S3_CACHE_VERSION:
            return _match_figure_s3_layer_to_figure7b(cached, spatial_data)

    cached = _load_cache("figure_s3_spatial_pca")
    if isinstance(cached, dict) and cached.get("_cache_version") == FIGURE_S3_CACHE_VERSION:
        return _match_figure_s3_layer_to_figure7b(cached, spatial_data)
    return _match_figure_s3_layer_to_figure7b(
        _compute_figure_s3_spatial_pca_data(smoke=smoke),
        spatial_data,
    )


def _figure_s3_rows(cached: dict) -> list[dict]:
    rows = cached.get("points", [])
    if not rows:
        return []
    fields = ["Layer index", "PC1", "PC2", "Grid x", "Grid y", "Run index"]
    return [{field: row[field] for field in fields} for row in rows]


def _save_figure_s3_pdf(cached: dict) -> None:
    rows = _figure_s3_rows(cached)
    if not rows:
        return

    layers = sorted({int(row["Layer index"]) for row in rows})
    grid_positions = sorted({(int(row["Grid x"]), int(row["Grid y"])) for row in rows})
    cmap = plt.get_cmap("tab10")
    pos_to_color = {pos: cmap(i % 10) for i, pos in enumerate(grid_positions)}

    n_cols = 6 if len(layers) >= 12 else max(1, min(3, len(layers)))
    n_rows = int(np.ceil(len(layers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.35, n_rows * 1.35), squeeze=False)

    by_layer: dict[int, list[dict]] = {layer: [] for layer in layers}
    for row in rows:
        by_layer[int(row["Layer index"])].append(row)

    for ax, layer in zip(axes.ravel(), layers):
        layer_rows = by_layer[layer]
        if layer_rows:
            ax.scatter(
                [float(row["PC1"]) for row in layer_rows],
                [float(row["PC2"]) for row in layer_rows],
                color=[
                    pos_to_color[(int(row["Grid x"]), int(row["Grid y"]))]
                    for row in layer_rows
                ],
                alpha=FIGURE_S3_SCATTER_ALPHA,
                s=FIGURE_S3_SCATTER_SIZE,
            )
        ax.set_title(f"Layer {layer}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)

    for ax in axes.ravel()[len(layers):]:
        ax.axis("off")

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=np.sqrt(FIGURE_S3_LEGEND_MARKER_SIZE),
            color=pos_to_color[pos],
            label=str(pos),
        )
        for pos in grid_positions
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(grid_positions), fontsize=5, frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.08, wspace=0.08, hspace=0.22)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "Figure S3.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _export_figure_s3_source_data(spatial_data: dict | None, smoke: bool = False) -> None:
    cached = _figure_s3_cache_from(spatial_data, smoke=smoke)
    if not cached:
        print("  [source-data] Figure S3 PCA coordinates not found in cache; skipping Figure_S3 export.")
        return

    rows = _figure_s3_rows(cached)
    if not rows:
        print("  [source-data] Figure S3 PCA coordinate cache is empty; skipping Figure_S3 export.")
        return

    out_dir = REPO_ROOT / "source_data"
    for stale in out_dir.glob("Figure_S3*.csv"):
        stale.unlink()
    _write_source_csv("Figure_S3_spatial_pca_all_layers.csv", rows)
    _save_figure_s3_pdf(cached)


def _export_figure7_source_data(spatial_data: dict | None, family_data: dict | None) -> None:
    out_dir = REPO_ROOT / "source_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("Figure_7*.csv"):
        stale.unlink()

    if spatial_data:
        for panel, key, label in [("a", "gpt2_medium", "GPT-2"), ("b", "our_model", "Our model")]:
            if key not in spatial_data:
                continue
            rows = []
            points = spatial_data[key]["all_points"]
            if points:
                coords = _coords_from_points(points)
                for point, coord in zip(points, coords):
                    gx, gy = point["grid_position"]
                    rows.append({
                        "PCA plot": "Combined",
                        "Model": label,
                        "series": f"grid position ({gx}, {gy})",
                        "PC1": float(coord[0]),
                        "PC2": float(coord[1]),
                        "Grid x": gx,
                        "Grid y": gy,
                    })
            single = spatial_data[key]["single_points"]
            if single:
                coords = _coords_from_points(single)
                for point, coord in zip(single, coords):
                    gx, gy = point["grid_position"]
                    rows.append({
                        "PCA plot": "Single graph",
                        "Model": label,
                        "series": f"grid position ({gx}, {gy})",
                        "Node name": point.get("name", ""),
                        "PC1": float(coord[0]),
                        "PC2": float(coord[1]),
                        "Grid x": gx,
                        "Grid y": gy,
                    })
            _write_source_csv(f"Figure_7{panel}_spatial_{key}_pca.csv", rows)

        for panel, key, label in [("e", "gpt2_medium", "GPT-2"), ("f", "our_model", "Our model")]:
            if key not in spatial_data:
                continue
            rows = []
            for grid_distance, values in spatial_data[key]["boxplot"].items():
                for value in values:
                    rows.append({
                        "Model": label,
                        "Grid distance": int(grid_distance),
                        "Representation distance": float(value),
                    })
            _write_source_csv(f"Figure_7{panel}_spatial_{key}_boxplot.csv", rows)

        rows = []
        for key, label in [("gpt2_medium", "GPT-2"), ("our_model", "Our model")]:
            if key not in spatial_data:
                continue
            for layer, corr in sorted(spatial_data[key]["corr_by_layer"].items(), key=lambda item: int(item[0])):
                rows.append({
                    "Model": label,
                    "Layer index": int(layer),
                    "Pearson correlation": float(corr),
                })
        _write_source_csv("Figure_7g_correlation_vs_layer.csv", rows)

        rows = []
        for key, label in [("gpt2_medium", "GPT-2"), ("our_model", "Our model")]:
            if key not in spatial_data:
                continue
            for context_length, corr in sorted(spatial_data[key]["corr_by_ctx"].items(), key=lambda item: int(item[0])):
                rows.append({
                    "Model": label,
                    "Context length": int(context_length),
                    "Pearson correlation": float(corr),
                })
        _write_source_csv("Figure_7h_correlation_vs_context_length.csv", rows)

    if family_data:
        gen_labels = family_data.get("generation_labels", {})
        for panel, key, label in [("c", "gpt2_medium", "GPT-2"), ("d", "our_model", "Our model")]:
            if key not in family_data:
                continue
            rows = []
            points = family_data[key]["all_points"]
            if points:
                coords = _coords_from_points(points)
                for point, coord in zip(points, coords):
                    generation = point["generation"]
                    rows.append({
                        "PCA plot": "Combined",
                        "Model": label,
                        "series": gen_labels.get(str(generation), str(generation)),
                        "PC1": float(coord[0]),
                        "PC2": float(coord[1]),
                        "Generation": generation,
                    })
            single = family_data[key]["single_points"]
            if single:
                coords = _coords_from_points(single)
                for point, coord in zip(single, coords):
                    generation = point["generation"]
                    rows.append({
                        "PCA plot": "Single tree",
                        "Model": label,
                        "series": gen_labels.get(str(generation), str(generation)),
                        "Node name": point.get("name", ""),
                        "PC1": float(coord[0]),
                        "PC2": float(coord[1]),
                        "Generation": generation,
                    })
            _write_source_csv(f"Figure_7{panel}_family_{key}_pca.csv", rows)


# ============================================================================
# MAIN: BUILD FIGURES
# ============================================================================

def build_figure5(smoke: bool = False) -> None:
    """Build manuscript Figure 6: inference behaviour results."""
    print("\n" + "=" * 60)
    print("  Building Figure 6 (inference behaviour)")
    print("=" * 60 + "\n")

    agg_data = _get_aggregated_inference_data(smoke=smoke)
    grid_data = _get_grid_generalisation_data(smoke=smoke)
    imag_data = _get_imagination_data(smoke=smoke)
    rag_data = _get_rag_summary_data()

    side = 1.8  # each subplot is a square of this size
    fig, axes = plt.subplots(1, 4, figsize=(4 * side + 2.0, side))
    plt.subplots_adjust(wspace=0.65)

    for ax in axes:
        ax.set_box_aspect(1)

    # g) Inference by transition count
    if agg_data:
        plot_aggregated_inf(axes[0], agg_data)
    else:
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].set_title("g)", fontsize=10, loc="center")

    # h) Grid generalisation
    if grid_data:
        plot_grid_generalisation(axes[1], grid_data)
    else:
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_title("h)", fontsize=10, loc="center")

    # i) Valid imagined trajectories
    if imag_data:
        plot_temp_validity(axes[2], imag_data)
    else:
        axes[2].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[2].transAxes)
    axes[2].set_title("i)", fontsize=10, loc="center")

    # k) RAG composition
    if rag_data:
        plot_rag_summary(axes[3], rag_data)
    else:
        axes[3].text(
            0.5, 0.5,
            "Run rag_composition.py\nfor panel k",
            ha="center", va="center", transform=axes[3].transAxes,
        )
        axes[3].set_xticks([])
        axes[3].set_yticks([])
    axes[3].set_title("k)", fontsize=10, loc="center")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "Figure 6.pdf"
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_figure6(smoke: bool = False) -> None:
    """Build manuscript Figure 7: internal representations.

    Layout (3 visual rows, 4 columns):
      Row 1:  a-top  b-top  c-top  d-top   (combined PCA: spatial GPT-2, spatial ours, family GPT-2, family ours)
      Row 2:  a-bot  b-bot  c-bot  d-bot   (single PCA)
      Row 3:  e      f      g      h        (boxplots & line plots, all square)
    """
    print("\n" + "=" * 60)
    print("  Building Figure 7 (internal representations)")
    print("=" * 60 + "\n")

    spatial_data = _get_spatial_reps_data(smoke=smoke)
    family_data = _get_family_reps_data(smoke=smoke)

    gen_labels = None
    if family_data and "generation_labels" in family_data:
        gen_labels = family_data["generation_labels"]

    side = 2.5  # size of each square cell
    fig = plt.figure(figsize=(4 * side + 1.2, 3 * side + 0.2))
    # Two vertical blocks: PCA rows (rows 0-1, tight) and bottom row (row 2)
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.08)
    gs_pca = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_outer[0], hspace=0.02, wspace=0.40)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.40)
    # Alias for easy access: gs[row, col] style
    class _GS:
        """Proxy so existing gs[row, col] references keep working."""
        def __getitem__(self, key):
            r, c = key
            if r < 2:
                return gs_pca[r, c]
            return gs_bot[0, c]
    gs = _GS()

    # ---- Rows 1-2: PCA panels (c, d, g, h) --------------------------------
    # Each panel is two squares stacked vertically (combined PCA on top,
    # single PCA on bottom), matching the dimensions of the bottom-row squares.

    # c) Spatial GPT-2 PCA
    ax_c_top = fig.add_subplot(gs[0, 0])
    ax_c_bot = fig.add_subplot(gs[1, 0])
    ax_c_top.set_box_aspect(1)
    ax_c_bot.set_box_aspect(1)
    if spatial_data and "gpt2_medium" in spatial_data:
        plot_combined_pca_spatial(ax_c_top, spatial_data["gpt2_medium"]["all_points"])
        plot_single_pca_spatial(ax_c_bot, spatial_data["gpt2_medium"]["single_points"],
                                spatial_data["gpt2_medium"]["single_edges"])
    ax_c_top.set_title("a) GPT-2 spatial PCA", fontsize=9, loc="center")

    # d) Spatial our-model PCA
    ax_d_top = fig.add_subplot(gs[0, 1])
    ax_d_bot = fig.add_subplot(gs[1, 1])
    ax_d_top.set_box_aspect(1)
    ax_d_bot.set_box_aspect(1)
    if spatial_data and "our_model" in spatial_data:
        plot_combined_pca_spatial(ax_d_top, spatial_data["our_model"]["all_points"],
                                  show_legend=False)
        plot_single_pca_spatial(ax_d_bot, spatial_data["our_model"]["single_points"],
                                spatial_data["our_model"]["single_edges"])
    ax_d_top.set_title("b) Our model spatial PCA", fontsize=9, loc="center")

    # g) Family GPT-2 PCA
    ax_g_top = fig.add_subplot(gs[0, 2])
    ax_g_bot = fig.add_subplot(gs[1, 2])
    ax_g_top.set_box_aspect(1)
    ax_g_bot.set_box_aspect(1)
    if family_data and "gpt2_medium" in family_data:
        plot_combined_pca_family(ax_g_top, family_data["gpt2_medium"]["all_points"], gen_labels,
                                 show_legend=True)
        plot_single_pca_family(ax_g_bot, family_data["gpt2_medium"]["single_points"],
                               family_data["gpt2_medium"]["single_edges"], gen_labels,
                               show_legend=False)
    ax_g_top.set_title("c) GPT-2 family PCA", fontsize=9, loc="center")

    # h) Family our-model PCA
    ax_h_top = fig.add_subplot(gs[0, 3])
    ax_h_bot = fig.add_subplot(gs[1, 3])
    ax_h_top.set_box_aspect(1)
    ax_h_bot.set_box_aspect(1)
    if family_data and "our_model" in family_data:
        plot_combined_pca_family(ax_h_top, family_data["our_model"]["all_points"], gen_labels,
                                 show_legend=False)
        plot_single_pca_family(ax_h_bot, family_data["our_model"]["single_points"],
                               family_data["our_model"]["single_edges"], gen_labels,
                               show_legend=False)
    ax_h_top.set_title("d) Our model family PCA", fontsize=9, loc="center")

    # ---- Row 3: boxplots & line plots (e, f, g, h) – all square -----------

    # e) GPT-2 boxplot  (swapped: GPT-2 now in col 0 to match PCA above)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.set_box_aspect(1)
    if spatial_data and "gpt2_medium" in spatial_data:
        plot_spatial_boxplot(ax_e, spatial_data["gpt2_medium"]["boxplot"])
    else:
        ax_e.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_e.transAxes)
    ax_e.set_title("e) GPT-2", fontsize=9, loc="center")

    # f) Our model boxplot
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.set_box_aspect(1)
    if spatial_data and "our_model" in spatial_data:
        plot_spatial_boxplot(ax_f, spatial_data["our_model"]["boxplot"])
    else:
        ax_f.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_f.transAxes)
    ax_f.set_title("f) Our model", fontsize=9, loc="center")

    # g) Corr vs layer
    ax_g = fig.add_subplot(gs[2, 2])
    ax_g.set_box_aspect(1)
    if spatial_data:
        plot_corr_vs_layer(ax_g, spatial_data)
    else:
        ax_g.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_g.transAxes)
    ax_g.set_title("g)", fontsize=9, loc="center")

    # h) Corr vs context length
    ax_h = fig.add_subplot(gs[2, 3])
    ax_h.set_box_aspect(1)
    if spatial_data:
        plot_corr_vs_context(ax_h, spatial_data)
    else:
        ax_h.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_h.transAxes)
    ax_h.set_title("h)", fontsize=9, loc="center")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "Figure 7.pdf"
    _export_figure7_source_data(spatial_data, family_data)
    _export_figure_s3_source_data(spatial_data, smoke=smoke)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Collate inference figures into manuscript Figure 6.pdf and Figure 7.pdf")
    parser.add_argument("--smoke", action="store_true", help="Use smaller parameters for a quick run.")
    parser.add_argument("--skip-fig5", action="store_true", help="Skip behavioural inference figure.")
    parser.add_argument("--skip-fig6", action="store_true", help="Skip internal representations figure.")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cached data before running.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference_config.json with trained model and output paths.",
    )
    args = parser.parse_args(argv)

    configure(args.config)
    os.chdir(INFERENCE_DIR)

    if args.clear_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(exist_ok=True)
        print("Cleared cache.")

    if not args.skip_fig5:
        build_figure5(smoke=args.smoke)

    if not args.skip_fig6:
        build_figure6(smoke=args.smoke)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
