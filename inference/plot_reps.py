"""
Spatial representation analysis.

Creates 3x3 grid graphs in code (so grid positions are known), samples
random walks from them, and feeds the walks to the model to check whether
latent representations capture the 2-D grid structure.
"""

import random
import string
import time

import numpy as np
import torch
import networkx as nx
import logging
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(321)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Grid graph construction  (collision-free names)
# ---------------------------------------------------------------------------

def _generate_unique_names(n):
    names = set()
    while len(names) < n:
        names.add("".join(random.choices(string.ascii_lowercase, k=2)))
    return list(names)


def build_grid_graph():
    """Build a 3x3 grid graph with 9 unique 2-letter names.

    Returns (G, node_names, grid_positions)
    where grid_positions maps name -> (row, col).
    """
    nodes = _generate_unique_names(9)
    G = nx.DiGraph()

    east_pairs = [(nodes[0], nodes[1]), (nodes[1], nodes[2]),
                  (nodes[3], nodes[4]), (nodes[4], nodes[5]),
                  (nodes[6], nodes[7]), (nodes[7], nodes[8])]
    south_pairs = [(nodes[0], nodes[3]), (nodes[3], nodes[6]),
                   (nodes[1], nodes[4]), (nodes[4], nodes[7]),
                   (nodes[2], nodes[5]), (nodes[5], nodes[8])]
    west_pairs = [(v, u) for u, v in east_pairs]
    north_pairs = [(v, u) for u, v in south_pairs]

    for n in nodes:
        G.add_node(n)
    for u, v in east_pairs:
        G.add_edge(u, v, direction="EAST")
    for u, v in west_pairs:
        G.add_edge(u, v, direction="WEST")
    for u, v in south_pairs:
        G.add_edge(u, v, direction="SOUTH")
    for u, v in north_pairs:
        G.add_edge(u, v, direction="NORTH")

    grid_positions = {}
    idx = 0
    for r in range(3):
        for c in range(3):
            grid_positions[nodes[idx]] = (r, c)
            idx += 1

    return G, nodes, grid_positions


# ---------------------------------------------------------------------------
# Random walk generation
# ---------------------------------------------------------------------------

def generate_random_walk(G, walk_length=50):
    """Return a walk string:  node DIR node DIR … node"""
    current = random.choice(list(G.nodes))
    walk = [current]
    for _ in range(walk_length):
        neighbors = list(G.successors(current))
        if not neighbors:
            break
        nxt = random.choice(neighbors)
        walk.append(G.edges[current, nxt]["direction"])
        walk.append(nxt)
        current = nxt
    return " ".join(walk)


# ---------------------------------------------------------------------------
# GPT wrapper & embedding extraction
# ---------------------------------------------------------------------------

def _default_entity_tokenizer_dir():
    """Return the canonical entity tokenizer path if it exists, else None."""
    from pathlib import Path
    path = Path(__file__).resolve().parent / "tokenizers" / "gpt2-medium_2letter_entities"
    if (path / "tokenizer.json").exists() or (path / "vocab.json").exists():
        return str(path)
    return None


class GPTWrapper:
    def __init__(self, model_name="gpt2", tokenizer_name=None):
        logging.info(f"Loading model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()
        emb_size = self.model.get_input_embeddings().weight.shape[0]

        # Resolve tokenizer: explicit > canonical entity tokenizer > model dir
        tok_path = tokenizer_name or model_name
        if tokenizer_name is None and not model_name.startswith("gpt2"):
            canon = _default_entity_tokenizer_dir()
            if canon is not None:
                candidate = GPT2TokenizerFast.from_pretrained(canon)
                if len(candidate) == emb_size:
                    tok_path = canon
                    logging.info(f"Using canonical entity tokenizer from {canon}")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(tok_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        logging.info(f"Model and tokenizer loaded successfully (vocab={len(self.tokenizer)}, emb={emb_size}).")

    def get_hidden_states_with_offsets(self, prompt, layer_idx):
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        # Remove offset_mapping before passing to model (it's not a model input)
        model_inputs = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            out = self.model(**model_inputs)
        hidden_states = out.hidden_states[layer_idx].squeeze(0).detach().cpu().numpy()
        return hidden_states, offsets

    def get_all_hidden_states_with_offsets(self, prompt):
        """Single forward pass returning hidden states for *every* layer.

        Returns ``(all_hidden_states, offsets)`` where
        ``all_hidden_states[i]`` is the numpy array for layer *i*.
        """
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                             return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        model_inputs = {k: v.to(self.device) for k, v in enc.items() if k != "offset_mapping"}
        with torch.no_grad():
            out = self.model(**model_inputs)
        all_hs = [h.squeeze(0).detach().cpu().numpy() for h in out.hidden_states]
        return all_hs, offsets


def substring_positions(haystack, needle):
    result, start = [], 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        result.append([idx, idx + len(needle)])
        start = idx + 1
    return result


def gather_embeddings_for_span(offsets, hidden_states, span, flanking=True):
    """Gather hidden-state vectors for a character span.

    If *flanking* is True (default), the tokens immediately before and after
    the entity token(s) are included in the average.  For the custom entity
    tokenizer these flanking tokens are the space characters that sit between
    the entity and the neighbouring direction token, and they carry rich
    contextual / positional information about the entity's place on the graph.
    """
    s_need, e_need = span
    n_tokens = len(offsets)
    entity_idxs = [i for i, (s, e) in enumerate(offsets)
                   if not (e <= s_need or s >= e_need)]
    if not entity_idxs:
        return None

    if flanking:
        all_idxs = set(entity_idxs)
        min_idx, max_idx = min(entity_idxs), max(entity_idxs)
        if min_idx > 0:
            all_idxs.add(min_idx - 1)
        if max_idx < n_tokens - 1:
            all_idxs.add(max_idx + 1)
        vecs = [hidden_states[i] for i in sorted(all_idxs)]
    else:
        vecs = [hidden_states[i] for i in entity_idxs]

    return np.mean(vecs, axis=0)


def average_locations_via_substring(prompt, offsets, hidden_states, locs,
                                    flanking=True, second_half_only=False):
    """Average entity representations across all occurrences in the prompt.

    When *flanking* is True the representation for each occurrence includes
    the entity token and its immediately adjacent tokens (typically spaces).

    When *second_half_only* is True, only occurrences whose character start
    position is in the second half of the prompt string are used.
    """
    half = len(prompt) // 2 if second_half_only else 0
    loc_means = {}
    for loc in locs:
        pos_list = substring_positions(prompt, loc)
        if not pos_list:
            continue
        if second_half_only:
            pos_list = [(s, e) for s, e in pos_list if s >= half]
            if not pos_list:
                continue
        vecs = [v for s, e in pos_list
                for v in [gather_embeddings_for_span(offsets, hidden_states,
                                                     (s, e), flanking=flanking)]
                if v is not None]
        if vecs:
            loc_means[loc] = np.mean(vecs, axis=0)
    return loc_means


# ---------------------------------------------------------------------------
# Distance & correlation helpers
# ---------------------------------------------------------------------------

def calc_pearson_correlation(node_names, loc_mean_repr, grid_positions):
    rec = [n for n in node_names if n in loc_mean_repr and n in grid_positions]
    if len(rec) < 2:
        return float("nan")
    man_d, rep_d = [], []
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            r1, c1 = grid_positions[rec[i]]
            r2, c2 = grid_positions[rec[j]]
            man_d.append(abs(r1 - r2) + abs(c1 - c2))
            rep_d.append(np.linalg.norm(loc_mean_repr[rec[i]] - loc_mean_repr[rec[j]]))
    if len(man_d) < 2:
        return float("nan")
    r, _ = pearsonr(man_d, rep_d)
    return r


def gather_boxplot_data(node_names, loc_mean_repr, grid_positions):
    rec = [n for n in node_names if n in loc_mean_repr and n in grid_positions]
    dist_map = {}
    for i in range(len(rec)):
        for j in range(i + 1, len(rec)):
            r1, c1 = grid_positions[rec[i]]
            r2, c2 = grid_positions[rec[j]]
            md = abs(r1 - r2) + abs(c1 - c2)
            if md > 0:
                dist_map.setdefault(md, []).append(
                    np.linalg.norm(loc_mean_repr[rec[i]] - loc_mean_repr[rec[j]]))
    return dist_map


def merge_dist_map(gmap, dmap):
    for d, lst in dmap.items():
        gmap.setdefault(d, []).extend(lst)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all_runs_in_one_pca(all_points, all_edges, model_name, reducer="pca"):
    if not all_points:
        return
    X = np.array([p["vector"] for p in all_points])
    X_c = X - X.mean(axis=0, keepdims=True)
    dim_red = PCA(n_components=2) if reducer == "pca" else UMAP(n_components=2)
    X_2d = dim_red.fit_transform(X_c)
    for i, c in enumerate(X_2d):
        all_points[i]["x2d"], all_points[i]["y2d"] = c[0], c[1]

    grid_pos = sorted(set(p["grid_position"] for p in all_points))
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(3, 3))
    for gi, pos in enumerate(grid_pos):
        pts = [p for p in all_points if p["grid_position"] == pos]
        plt.scatter([p["x2d"] for p in pts], [p["y2d"] for p in pts],
                    color=cmap(gi % 10), label=f"{pos}")
    for e in all_edges:
        i_, j_ = e["u_index"], e["v_index"]
        plt.plot([all_points[i_]["x2d"], all_points[j_]["x2d"]],
                 [all_points[i_]["y2d"], all_points[j_]["y2d"]],
                 "--", color=cmap(grid_pos.index(all_points[i_]["grid_position"]) % 10), alpha=0.0)
    if model_name == "gpt2-medium":
        plt.legend()
    plt.savefig(f"{model_name}_combined_{reducer}.png", dpi=200, bbox_inches="tight")
    logging.info(f"Saved {model_name}_combined_{reducer}.png")


def boxplot_of_dist_map(gmap, model_name):
    if not gmap:
        return
    keys = sorted(gmap.keys())
    plt.figure(figsize=(3, 2))
    plt.boxplot([gmap[k] for k in keys], showfliers=False, showmeans=False)
    plt.xticks(range(1, len(keys) + 1), keys)
    plt.xlabel("Distance in grid")
    plt.ylabel("Distance between rep.s")
    plt.savefig(f"{model_name}_boxplot.png", dpi=200, bbox_inches="tight")
    logging.info(f"Saved {model_name}_boxplot.png")


def plot_embeddings_with_graph_edges(mean_repr, G, model_name="gpt2",
                                     reducer="umap", title="Locations Embedding"):
    keys = sorted(mean_repr.keys())
    if len(keys) < 2:
        return
    X = np.array([mean_repr[k] for k in keys])
    X_c = X - X.mean(axis=0, keepdims=True)
    dim_red = PCA(n_components=2) if reducer == "pca" else UMAP(n_components=2)
    X_2d = dim_red.fit_transform(X_c)
    loc_idx = {k: i for i, k in enumerate(keys)}
    plt.figure(figsize=(3, 3))
    plt.scatter(X_2d[:, 0], X_2d[:, 1])
    for i, loc in enumerate(keys):
        plt.annotate(loc, (X_2d[i, 0], X_2d[i, 1]), xytext=(3, 3), textcoords="offset points")
    for u, v in G.edges():
        if u in loc_idx and v in loc_idx:
            i, j = loc_idx[u], loc_idx[v]
            plt.plot([X_2d[i, 0], X_2d[j, 0]], [X_2d[i, 1], X_2d[j, 1]], "k--", alpha=0.3)
    plt.title(title + f" ({reducer.upper()})")
    ts = str(int(time.time()))
    plt.savefig(f"{model_name}_single_{reducer}_{ts}.png", dpi=200, bbox_inches="tight")
    logging.info(f"Saved {model_name}_single_{reducer}_{ts}.png")


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def multi_graph_allinone(model_name="gpt2", layer_idx=10, n_runs=300,
                         walk_length=50, reducer="pca"):
    logging.info(f"multi_graph_allinone: {model_name}, {n_runs} runs")
    wrapper = GPTWrapper(model_name)
    all_points, all_edges, global_dist = [], [], {}
    pt_idx = 0

    for run in range(n_runs):
        logging.info(f"=== RUN {run+1}/{n_runs} ===")
        G, node_names, grid_pos = build_grid_graph()
        prompt = generate_random_walk(G, walk_length)
        hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
        loc_repr = average_locations_via_substring(prompt, off, hs, node_names)

        node_pt = {}
        for n in (n for n in node_names if n in loc_repr):
            all_points.append({"run_idx": run, "node_name": n,
                               "vector": loc_repr[n], "grid_position": grid_pos[n]})
            node_pt[n] = pt_idx; pt_idx += 1
        for u, v in G.edges():
            if u in node_pt and v in node_pt:
                all_edges.append({"u_index": node_pt[u], "v_index": node_pt[v]})
        merge_dist_map(global_dist, gather_boxplot_data(node_names, loc_repr, grid_pos))

    plot_all_runs_in_one_pca(all_points, all_edges, model_name, reducer)
    boxplot_of_dist_map(global_dist, model_name)


def single_graph_with_edge_lines(model_name="gpt2", layer_idx=10,
                                 walk_length=100000, reducer="pca"):
    G, node_names, _ = build_grid_graph()
    prompt = generate_random_walk(G, walk_length)
    wrapper = GPTWrapper(model_name)
    hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
    loc_repr = average_locations_via_substring(prompt, off, hs, node_names)
    plot_embeddings_with_graph_edges(loc_repr, G, model_name, reducer,
                                     title=f"{model_name} L{layer_idx}")


def rolling_mean(vals, w=3):
    return [np.mean(vals[max(0, i - w + 1):i + 1]) for i in range(len(vals))]


def correlation_vs_context_length_multi(models, layer_idx=12, context_lengths=None,
                                        n_runs=100, rolling_window=1):
    if context_lengths is None:
        context_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    labels = {"outputs_graph": "Our model", "gpt2-medium": "Pre-trained GPT-2"}

    # Pre-build graphs so both models see the same data
    runs = [build_grid_graph() for _ in range(n_runs)]

    plt.figure(figsize=(3, 2))
    for model_name in models:
        wrapper = GPTWrapper(model_name)
        avg_corrs = []
        for L in context_lengths:
            corrs = []
            for G, node_names, grid_pos in runs:
                prompt = generate_random_walk(G, L)
                hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
                loc_repr = average_locations_via_substring(prompt, off, hs, node_names)
                c = calc_pearson_correlation(node_names, loc_repr, grid_pos)
                if not np.isnan(c):
                    corrs.append(c)
            m = np.mean(corrs) if corrs else float("nan")
            avg_corrs.append(m)
            logging.info(f"[{model_name}] ctx={L}, r={m:.3f}")
        if rolling_window > 1:
            plt.plot(context_lengths, rolling_mean(avg_corrs, rolling_window),
                     "o-", label=f"{model_name} (rolled)")
        color = "blue" if model_name == "gpt2-medium" else "red"
        plt.plot(context_lengths, avg_corrs, "o--",
                 label=labels.get(model_name, model_name), color=color)

    plt.xlabel("Context length"); plt.ylabel("Pearson correlation")
    plt.title("Correlation vs context length"); plt.legend()
    plt.savefig("comparison_correlation_vs_context_length.png", dpi=200, bbox_inches="tight")


def correlation_vs_layer_multi(models, layer_indices=None, n_runs=100, walk_length=50):
    if layer_indices is None:
        layer_indices = list(range(24))
    labels = {"outputs_graph": "Our model", "gpt2-medium": "Pre-trained GPT-2"}

    # Pre-build graphs + walks so both models see identical data
    runs = []
    for _ in range(n_runs):
        G, names, gp = build_grid_graph()
        prompt = generate_random_walk(G, walk_length)
        runs.append((prompt, G, names, gp))

    plt.figure(figsize=(3, 2))
    for model_name in models:
        wrapper = GPTWrapper(model_name)
        avg = {}
        for layer_idx in layer_indices:
            corrs = []
            logging.info(f"  Layer {layer_idx} for {model_name}...")
            for prompt, G, names, gp in runs:
                hs, off = wrapper.get_hidden_states_with_offsets(prompt, layer_idx)
                loc_repr = average_locations_via_substring(prompt, off, hs, names)
                c = calc_pearson_correlation(names, loc_repr, gp)
                if not np.isnan(c):
                    corrs.append(c)
            avg[layer_idx] = np.mean(corrs) if corrs else float("nan")
            logging.info(f"[{model_name}] L{layer_idx} r={avg[layer_idx]:.3f}")
        color = "blue" if model_name == "gpt2-medium" else "red"
        plt.plot(list(avg.keys()), list(avg.values()), "o--",
                 label=labels.get(model_name, model_name), color=color)

    plt.xlabel("Layer index"); plt.ylabel("Pearson correlation")
    plt.title("Correlation vs layer"); plt.legend()
    plt.savefig("comparison_correlation_vs_layer.png", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    model_to_test = "outputs_graph"

    if args.smoke:
        n_pca, n_single, n_corr = 5, 1, 3
        ctx_lens = [10, 30]
        layers = [6, 12, 18]
    else:
        n_pca, n_single, n_corr = 300, 3, 100
        ctx_lens = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        layers = list(range(24))

    multi_graph_allinone("gpt2-medium", 12, n_pca, 50, "pca")
    for _ in range(n_single):
        single_graph_with_edge_lines("gpt2-medium", 12, 100000, "pca")

    multi_graph_allinone(model_to_test, 12, n_pca, 50, "pca")
    for _ in range(n_single):
        single_graph_with_edge_lines(model_to_test, 12, 100000, "pca")

    models = ["gpt2-medium", model_to_test]
    correlation_vs_context_length_multi(models, 12, ctx_lens, n_corr)
    correlation_vs_layer_multi(models, layers, n_corr, 50)
