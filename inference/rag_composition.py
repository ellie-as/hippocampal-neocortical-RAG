"""
Test whether RAG can compose information from split context paths.

Uses the same closed-walk (loop) templates as ``graph_sequence_model.py``.
Each template is **known to be solvable** when given as a single line
(the ``graph_sequence_model.py`` inference test confirms this).

For each filled template we **split the context path** into two memories
that share a bridge entity, store them as hippocampal traces tagged with a
graph/environment id, retrieve traces from the matching graph, and test
whether the model still predicts the starting entity when the retrieved
memories are on separate lines.

Example  (4-hop spatial loop)
-----------------------------
Full line::

    ab EAST cd SOUTH ef WEST gh NORTH  →  predict ``ab``

RAG split::

    memory-1:  ab EAST cd SOUTH ef
    memory-2:  ef WEST gh
    query:     gh NORTH  →  predict ``ab``

The answer ``ab`` always appears in memory-1 (it is the loop start), so
the task is **solvable in principle** whenever memory-1 is in context.

Conditions
~~~~~~~~~~
* **Full**  – complete sequence on one line  (known-good baseline)
* **RAG**   – memory-1 + memory-2 + query on separate lines
* **Mem-1** – memory-1 + query only
* **Mem-2** – memory-2 + query only
* **NC**    – query only  (no context)

Key comparisons:

* **Full vs RAG**: does splitting across lines hurt?
* **RAG vs Mem-1 / Mem-2**: does combining two memories help?
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import string
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
FIGURES_DIR = REPO_ROOT / "figures"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(INFERENCE_DIR))

from run_config import load_config

SEED = 42


@dataclass(frozen=True)
class HippocampalTrace:
    """A stored episodic sequence with the environment/family it came from."""

    graph_id: str
    sequence: str


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# GPT wrapper (self-contained; no dependency on rag_inference.py)
# ---------------------------------------------------------------------------


class GPT:
    """Lightweight wrapper around a GPT-2 causal LM for next-token prediction."""

    def __init__(
        self,
        base_model: str | None = None,
        base_model_name: str = "gpt2",
        vocab_size: int = 100,
        tokenizer_name: str | Path | None = None,
    ):
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size

        if self.base_model is not None:
            try:
                from transformers import GPT2LMHeadModel, GPT2TokenizerFast
            except ModuleNotFoundError as e:  # pragma: no cover
                raise ModuleNotFoundError(
                    "Missing dependency `transformers`. "
                    "Install with `pip install -r requirements.txt`."
                ) from e
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            except ImportError:  # pragma: no cover
                self.device = "cpu"
            self.model = GPT2LMHeadModel.from_pretrained(base_model)
            self.model.to(self.device)
            self.model.eval()
            emb_size = self.model.get_input_embeddings().weight.shape[0]

            tokenizer_path = str(tokenizer_name) if tokenizer_name else base_model
            tok = GPT2TokenizerFast.from_pretrained(tokenizer_path)
            if len(tok) != emb_size:
                print(
                    f"  [tokenizer] entity tokenizer ({len(tok)}) != model "
                    f"embeddings ({emb_size}); using model tokenizer from {base_model}"
                )
                tok = GPT2TokenizerFast.from_pretrained(base_model)
            self.tokenizer = tok
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def continue_input(
        self,
        input_sequence: str,
        max_new_tokens: int = 5,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: int = 0,
        do_sample: bool = False,
        temperature: float = 0.7,
        num_beams: int = 1,
    ) -> str:
        input_ids = self.tokenizer.encode(input_sequence, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature,
        )
        sequence = output[0].tolist()
        text = self.tokenizer.decode(sequence)
        return text

    def next_token(self, input_sequence: str, **generate_kwargs) -> str:
        """Return the first predicted *word* (entity) after the input."""
        enc = self.tokenizer.encode(input_sequence, return_tensors="pt").to(self.device)
        out = self.model.generate(enc, max_new_tokens=5, **generate_kwargs)
        new_tokens = out[0][enc.shape[1]:]
        decoded = self.tokenizer.decode(new_tokens).strip()
        return decoded.split()[0] if decoded else ""


# ---------------------------------------------------------------------------
# Templates  (mirrored from graph_sequence_model.py)
# ---------------------------------------------------------------------------
# Each entry: (template_string, entity_indices | None)
# entity_indices=None → simple loop fill:  n_unique = n_slots-1, last=first

SPATIAL_TEMPLATES: list[tuple[str, list[int] | None]] = [
    # 4-hop (closed 1×1 square in each compass rotation)
    ("{} EAST {} SOUTH {} WEST {} NORTH {}", None),
    ("{} SOUTH {} WEST {} NORTH {} EAST {}", None),
    ("{} WEST {} NORTH {} EAST {} SOUTH {}", None),
    ("{} NORTH {} EAST {} SOUTH {} WEST {}", None),
    # 6-hop (closed 2×1 rectangle)
    ("{} EAST {} EAST {} NORTH {} WEST {} WEST {} SOUTH {}", None),
    ("{} NORTH {} NORTH {} WEST {} SOUTH {} SOUTH {} EAST {}", None),
]

FAMILY_TEMPLATES: list[tuple[str, list[int] | None]] = [
    # 4-hop (up-and-back paths; entity indices [0,1,2,1,0])
    ("{} CHILD_OF {} CHILD_OF {} PARENT_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
    ("{} PARENT_OF {} PARENT_OF {} CHILD_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
    ("{} CHILD_OF {} SPOUSE_OF {} SPOUSE_OF {} PARENT_OF {}", [0, 1, 2, 1, 0]),
    ("{} PARENT_OF {} SPOUSE_OF {} SPOUSE_OF {} CHILD_OF {}", [0, 1, 2, 1, 0]),
    # 6-hop (longer chains closing via symmetric relations)
    ("{} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {} GRANDPARENT_OF {} SIBLING_OF {}", None),
    ("{} GRANDPARENT_OF {} SIBLING_OF {} CHILD_OF {} SPOUSE_OF {} CHILD_OF {} SPOUSE_OF {}", None),
]


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def _generate_name() -> str:
    return "".join(random.choices(string.ascii_lowercase, k=2))


def _n_hops(template: str) -> int:
    """Number of relation (hop) tokens in a template."""
    return template.count("{}") - 1


def fill_template(
    template: str,
    entity_indices: list[int] | None = None,
) -> tuple[str, str]:
    """Fill *template* with random 2-letter entities.

    Returns ``(filled_string, target_entity)``.
    """
    if entity_indices is not None:
        n_unique = max(entity_indices) + 1
        names = [_generate_name() for _ in range(n_unique)]
        fill_args = [names[i] for i in entity_indices]
    else:
        n_slots = template.count("{}")
        names = [_generate_name() for _ in range(n_slots - 1)]
        fill_args = names + [names[0]]          # loop: last = first
    filled = template.format(*fill_args)
    target = fill_args[-1]
    return filled, target


def split_for_rag(
    filled: str,
    n_hops: int,
) -> tuple[str, str, str, str]:
    """Split a filled loop into ``(mem1, mem2, query, target)``.

    Tokens of a filled N-hop loop::

        [e0  d1  e1  d2  e2  …  d_N  e_N]
         0   1   2   3   4      2N-1  2N

    * Context  = tokens ``0 … 2N-1``  (everything before the target)
    * Query    = last entity + last direction = tokens ``2(N-1) … 2N-1``
    * The remaining context is split at its midpoint:

      - mem1 = tokens ``0 … bridge``
      - mem2 = tokens ``bridge … 2(N-1)-1``  (one token before the query entity)

    ``mem1`` and ``mem2`` share the **bridge entity**.
    """
    tokens = filled.split()
    # Bridge at the midpoint of the *context* hops (N-1 hops before query)
    bridge_hop = n_hops // 2
    bridge_idx = 2 * bridge_hop                 # token index of bridge entity
    end_ctx = 2 * (n_hops - 1)                  # index of query entity

    mem1 = " ".join(tokens[: bridge_idx + 1])
    mem2 = " ".join(tokens[bridge_idx : end_ctx + 1])
    query = " ".join(tokens[end_ctx : 2 * n_hops])
    target = tokens[2 * n_hops]
    return mem1, mem2, query, target


def retrieve_by_graph_id(
    hippocampus: list[HippocampalTrace],
    graph_id: str,
    *,
    k: int = 2,
) -> list[str]:
    """Retrieve the first ``k`` traces from the same environment/family tree."""
    matches = [trace.sequence for trace in hippocampus if trace.graph_id == graph_id]
    if len(matches) < k:
        raise ValueError(
            f"Need {k} traces for graph_id={graph_id!r}, found {len(matches)}"
        )
    return matches[:k]


# ---------------------------------------------------------------------------
# Walk-like padding helpers
# ---------------------------------------------------------------------------

SPATIAL_INV: dict[str, str] = {
    "NORTH": "SOUTH", "SOUTH": "NORTH",
    "EAST": "WEST", "WEST": "EAST",
}
FAMILY_INV: dict[str, str] = {
    "PARENT_OF": "CHILD_OF", "CHILD_OF": "PARENT_OF",
    "GRANDPARENT_OF": "GRANDCHILD_OF", "GRANDCHILD_OF": "GRANDPARENT_OF",
    "SPOUSE_OF": "SPOUSE_OF", "SIBLING_OF": "SIBLING_OF",
}

INV_MAPS: dict[str, dict[str, str]] = {
    "spatial": SPATIAL_INV,
    "family": FAMILY_INV,
}


def _reverse_fragment(fragment: str, inv_map: dict[str, str]) -> str:
    """Reverse a path fragment: ``'A EAST B SOUTH C'`` → ``'C NORTH B WEST A'``."""
    tokens = fragment.split()
    rev: list[str] = []
    for i in range(len(tokens) - 1, 0, -2):
        rev.append(tokens[i])                     # entity
        rev.append(inv_map[tokens[i - 1]])         # inverse relation
    rev.append(tokens[0])                          # first entity
    return " ".join(rev)


def _pad_fragment(fragment: str, inv_map: dict[str, str]) -> str:
    """Prepend the reverse traversal to create a longer walk-like sequence.

    ``'A EAST B SOUTH C'`` → ``'C NORTH B WEST A EAST B SOUTH C'``

    This is a valid back-and-forth walk that the model has seen
    patterns of during training.
    """
    reverse = _reverse_fragment(fragment, inv_map)
    frag_tokens = fragment.split()
    # Skip first token of fragment (same as last of reverse) to avoid duplication
    return reverse + " " + " ".join(frag_tokens[1:])


def _merge_mem2_query(mem2: str, query: str) -> str:
    """Merge mem2 and query into one line (they share the bridge entity).

    ``mem2='C WEST D'``, ``query='D NORTH'`` → ``'C WEST D NORTH'``
    """
    q_tokens = query.split()
    return mem2 + " " + " ".join(q_tokens[1:])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

COND_NAMES = ["Full", "RAG", "RAG-2L", "RAG-walk", "HPC", "Mem-1", "Mem-2", "NC"]


def _evaluate_trial(
    model,
    filled: str,
    mem1: str,
    mem2: str,
    query: str,
    target: str,
    inv_map: dict[str, str],
) -> dict[str, int]:
    """Run one trial across all conditions; return 0/1 per condition."""
    tokens = filled.split()
    full_prompt = " ".join(tokens[:-1])          # single-line context

    # Merged mem2+query (1 newline instead of 2)
    mem2_query = _merge_mem2_query(mem2, query)

    # Padded fragments (walk-like, longer)
    pad_mem1 = _pad_fragment(mem1, inv_map)
    pad_mem2_query = _pad_fragment(mem2, inv_map)
    # Append the query direction to the padded mem2
    q_tail = " ".join(query.split()[1:])         # just the direction(s)
    pad_mem2_query = pad_mem2_query + " " + q_tail

    prompts = {
        "Full":     full_prompt,
        "RAG":      f"{mem1}\n{mem2}\n{query}",
        "RAG-2L":   f"{mem1}\n{mem2_query}",
        "RAG-walk": f"{pad_mem1}\n{pad_mem2_query}",
        "Mem-1":    f"{mem1}\n{query}",
        "Mem-2":    f"{mem2}\n{query}",
        "NC":       query,
    }

    results: dict[str, int] = {}
    for cond, prompt in prompts.items():
        pred = model.next_token(prompt, do_sample=False)
        results[cond] = int(pred == target)

    # HPC only: randomly guess an entity from the two stored memories
    mem1_entities = mem1.split()[::2]             # entities at even positions
    mem2_entities = mem2.split()[::2]
    pool = list(set(mem1_entities + mem2_entities))
    results["HPC"] = int(random.choice(pool) == target)

    return results


def run_template_group(
    model,
    templates: list[tuple[str, list[int] | None]],
    task_name: str,
    task_key: str,
    *,
    n_per_template: int = 100,
    verbose: bool = True,
) -> dict[str, dict[str, list[int]]]:
    """Evaluate all templates in a group, aggregated by hop-count.

    Returns ``{"Spatial 4-hop": {"Full": [...], "RAG": [...], …}, …}``.
    """
    inv_map = INV_MAPS[task_key]
    results_by_hops: dict[int, dict[str, list[int]]] = {}
    hippocampus: list[HippocampalTrace] = []

    for tmpl_idx, (template, entity_indices) in enumerate(templates):
        nhops = _n_hops(template)
        if nhops < 4:
            continue                              # 2-hop can't split meaningfully

        if nhops not in results_by_hops:
            results_by_hops[nhops] = {cn: [] for cn in COND_NAMES}
        conds = results_by_hops[nhops]

        if verbose:
            print(f"  Template {tmpl_idx}: {template}")
            print(f"    ({nhops}-hop, {n_per_template} trials)")

        for trial in range(n_per_template):
            filled, target = fill_template(template, entity_indices)
            mem1, mem2, query, target2 = split_for_rag(filled, nhops)
            assert target == target2
            graph_id = f"{task_key}:{tmpl_idx}:{trial}"
            hippocampus.extend([
                HippocampalTrace(graph_id, mem1),
                HippocampalTrace(graph_id, mem2),
            ])
            retrieved_mem1, retrieved_mem2 = retrieve_by_graph_id(
                hippocampus, graph_id, k=2,
            )

            res = _evaluate_trial(
                model, filled, retrieved_mem1, retrieved_mem2, query, target, inv_map,
            )
            for cn in COND_NAMES:
                conds[cn].append(res[cn])

            # Show a couple of examples early on
            if verbose and trial < 2 and tmpl_idx < 2:
                toks = filled.split()
                pad_m1 = _pad_fragment(retrieved_mem1, inv_map)
                m2q = _merge_mem2_query(retrieved_mem2, query)
                pad_m2q_tokens = _pad_fragment(retrieved_mem2, inv_map)
                q_tail = " ".join(query.split()[1:])
                pad_m2q = pad_m2q_tokens + " " + q_tail
                print(f"    trial {trial}:")
                print(f"      Full :     {' '.join(toks[:-1])}  →  {target!r}")
                print(f"      Graph ID:  {graph_id}")
                print(f"      Mem-1:     {retrieved_mem1}")
                print(f"      Mem-2:     {retrieved_mem2}")
                print(f"      Query:     {query}")
                print(f"      RAG-2L:    {retrieved_mem1}  |  {m2q}")
                print(f"      RAG-walk:  {pad_m1}  |  {pad_m2q}")
                print(f"      {res}")
                print()

        if verbose:
            # Running totals for this template
            n = len(conds["Full"])
            print(f"    → running ({n} trials):  "
                  + "  ".join(f"{cn}={np.mean(conds[cn]):.0%}" for cn in COND_NAMES))
            print()

    out: dict[str, dict[str, list[int]]] = {}
    for nhops, conds in sorted(results_by_hops.items()):
        out[f"{task_name} {nhops}-hop"] = conds
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(
    all_results: dict[str, dict[str, list[int]]],
    filename: str | Path | None = None,
) -> None:
    filename = Path(filename) if filename is not None else FIGURES_DIR / "rag_composition.pdf"
    """Bar chart comparing conditions, one panel per (task, hop-count)."""
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    groups = list(all_results.keys())
    n_groups = len(groups)
    n_conds = len(COND_NAMES)

    cond_colors: dict[str, str] = {
        "Full": "#2ca02c",
        "RAG": "#ff7f0e",
        "RAG-2L": "#d4a017",
        "RAG-walk": "#17becf",
        "HPC": "#1f77b4",
        "Mem-1": "#9467bd",
        "Mem-2": "#e377c2",
        "NC": "#d62728",
    }

    fig, axes = plt.subplots(
        1, n_groups, figsize=(3.8 * n_groups, 3.5), squeeze=False,
    )

    for ax_idx, group in enumerate(groups):
        ax = axes[0, ax_idx]
        conds = all_results[group]

        means = [
            float(np.mean(conds[cn])) if conds[cn] else 0.0
            for cn in COND_NAMES
        ]
        sems = [
            float(np.std(conds[cn]) / np.sqrt(len(conds[cn])))
            if len(conds[cn]) > 1 else 0.0
            for cn in COND_NAMES
        ]

        x = np.arange(n_conds)
        colors = [cond_colors[cn] for cn in COND_NAMES]
        ax.bar(
            x, means, 0.62, yerr=sems, capsize=3,
            color=colors, alpha=0.82, edgecolor="none",
        )

        for xi, m in zip(x, means):
            ax.text(xi, m + 0.025, f"{m:.0%}", ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(COND_NAMES, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.15)
        ax.set_title(group, fontsize=10)

    plt.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


def plot_task_diagram(filename: str | Path | None = None) -> None:
    """Visualise RAG composition tasks in the style of Figure 5 a/b/d/e.

    Each panel shows:
      • top: full graph (3×3 grid / family tree) with path edges coloured
        by memory assignment and non-path edges in light grey
      • bottom: chain diagram (entity → relation → entity → …)
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.lines import Line2D
    import matplotlib.gridspec as gridspec

    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    MEM1 = "#1f77b4"          # blue
    MEM2 = "#ff7f0e"          # orange
    QCOL = "#2ca02c"          # green
    PATH_FC = "#dcadad"       # light red/pink  (path nodes)
    CTX_FC = "#c8d8e8"        # light blue       (non-path nodes)
    GRAY = "#d0d0d0"          # grid edges

    NODE_SZ = 18
    NODE_SZ_CHAIN = 13
    SHRINK_G = 10             # arrow shrink for graph
    SHRINK_C = 7              # arrow shrink for chain

    # -- helpers ----------------------------------------------------------

    def _node(ax, x, y, name, fc=PATH_FC, sz=NODE_SZ):
        ax.plot(x, y, "o", markersize=sz, color=fc,
                markeredgecolor="black", markeredgewidth=1.0, zorder=3)
        ax.text(x, y, name, ha="center", va="center",
                fontsize=6.5, fontweight="bold", zorder=4)

    def _arrow(ax, x1, y1, x2, y2, color, ls="-", lw=1.8, shrink=SHRINK_G):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=12,
            color=color, lw=lw, linestyle=ls,
            shrinkA=shrink, shrinkB=shrink, zorder=2,
        )
        ax.add_patch(a)

    def _gray_line(ax, x1, y1, x2, y2):
        ax.plot([x1, x2], [y1, y2], color=GRAY, lw=0.8, zorder=1)

    # =====================================================================
    fig = plt.figure(figsize=(7.5, 6.0))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           height_ratios=[2.8, 1], hspace=0.35, wspace=0.35)

    # =================================================================
    # SPATIAL GRAPH  (top-left)
    # =================================================================
    ax_sg = fig.add_subplot(gs[0, 0])

    # All 9 grid nodes
    sp = {}
    for y, row in [(2, ["mn", "op", "qr"]),
                   (1, ["kl", "ij", "gh"]),
                   (0, ["ab", "cd", "ef"])]:
        for x, nm in enumerate(row):
            sp[nm] = (x, y)

    path_sp = {"ab", "cd", "ef", "gh", "ij", "kl"}

    # Gray lines for all grid adjacencies
    done = set()
    for n1, (x1, y1) in sp.items():
        for n2, (x2, y2) in sp.items():
            key = tuple(sorted([n1, n2]))
            if key not in done and abs(x1 - x2) + abs(y1 - y2) == 1:
                _gray_line(ax_sg, x1, y1, x2, y2)
                done.add(key)

    # Coloured path arrows
    for edges, col, ls in [
        ([("ab", "cd"), ("cd", "ef"), ("ef", "gh")], MEM1, "-"),
        ([("gh", "ij"), ("ij", "kl")], MEM2, "-"),
        ([("kl", "ab")], QCOL, "--"),
    ]:
        for src, dst in edges:
            _arrow(ax_sg, *sp[src], *sp[dst], col, ls)

    # Nodes (path = pink, context = blue)
    for nm, (x, y) in sp.items():
        _node(ax_sg, x, y, nm, PATH_FC if nm in path_sp else CTX_FC)

    ax_sg.set_xlim(-0.5, 2.5)
    ax_sg.set_ylim(-0.5, 2.5)
    ax_sg.set_aspect("equal")
    ax_sg.axis("off")
    ax_sg.set_title("Spatial RAG composition", fontsize=10)

    # =================================================================
    # SPATIAL CHAIN  (bottom-left)
    # =================================================================
    ax_sc = fig.add_subplot(gs[1, 0])

    s_ents = ["ab", "cd", "ef", "gh", "ij", "kl", "?"]
    s_rels = ["EAST", "EAST", "NORTH", "WEST", "WEST", "SOUTH"]
    s_cols = [MEM1, MEM1, MEM1, MEM2, MEM2, QCOL]
    s_lss = ["-", "-", "-", "-", "-", "--"]

    dx = 1.0
    for i, ent in enumerate(s_ents):
        x = i * dx
        _node(ax_sc, x, 0, ent, PATH_FC, NODE_SZ_CHAIN)
        if i < len(s_rels):
            _arrow(ax_sc, x, 0, x + dx, 0, s_cols[i], s_lss[i],
                   lw=1.4, shrink=SHRINK_C)
            ax_sc.text(x + dx / 2, 0.28, s_rels[i], fontsize=5.5,
                       ha="center", color=s_cols[i], fontweight="bold")

    ax_sc.text((len(s_ents) - 1) * dx / 2, -0.45,
               "Correct output: ab", fontsize=7, ha="center",
               style="italic")

    # Prompt text
    ax_sc.text((len(s_ents) - 1) * dx / 2, -0.85,
               "'ab EAST cd EAST ef NORTH gh\\n"
               "gh WEST ij WEST kl SOUTH'",
               fontsize=5.5, ha="center", family="monospace",
               bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc"))

    ax_sc.set_xlim(-0.5, (len(s_ents) - 1) * dx + 0.5)
    ax_sc.set_ylim(-1.3, 0.55)
    ax_sc.set_aspect("equal")
    ax_sc.axis("off")

    # =================================================================
    # FAMILY GRAPH  (top-right)
    # =================================================================
    ax_fg = fig.add_subplot(gs[0, 1])

    # Positions: 6 path nodes + 1 context node ("sib"), spread wider
    fm = {
        "n3": (0.8, 2),  "n4": (3.2, 2),            # gen 0
        "n2": (0, 1),    "n1": (2.5, 1), "sib": (4.5, 1),  # gen 1
        "n5": (0.8, 0),  "n0": (2.5, 0),             # gen 2
    }
    path_fm = {"n0", "n1", "n2", "n3", "n4", "n5"}

    # Gray context edges — only non-crossing structural links
    for src, dst in [("n3", "sib"), ("n4", "sib"),
                     ("n1", "n0"), ("n2", "n5")]:
        _gray_line(ax_fg, *fm[src], *fm[dst])

    # Coloured path arrows
    for edges, col, ls in [
        ([("n0", "n1"), ("n1", "n2"), ("n2", "n3")], MEM1, "-"),
        ([("n3", "n4"), ("n4", "n5")], MEM2, "-"),
        ([("n5", "n0")], QCOL, "--"),
    ]:
        for src, dst in edges:
            _arrow(ax_fg, *fm[src], *fm[dst], col, ls)

    # Nodes
    for nm, (x, y) in fm.items():
        _node(ax_fg, x, y, nm, PATH_FC if nm in path_fm else CTX_FC)

    # Generation annotations
    for gy, lbl in [(2, "Gen 0"), (1, "Gen 1"), (0, "Gen 2")]:
        ax_fg.text(-0.9, gy, lbl, fontsize=6, ha="right", va="center",
                   color="gray", style="italic")

    ax_fg.set_xlim(-1.1, 5.3)
    ax_fg.set_ylim(-0.5, 2.5)
    ax_fg.set_aspect("equal")
    ax_fg.axis("off")
    ax_fg.set_title("Family tree RAG composition", fontsize=10)

    # =================================================================
    # FAMILY CHAIN  (bottom-right)
    # =================================================================
    ax_fc = fig.add_subplot(gs[1, 1])

    f_ents = ["n0", "n1", "n2", "n3", "n4", "n5", "?"]
    f_rels = ["CHILD_OF", "SPOUSE_OF", "CHILD_OF",
              "SPOUSE_OF", "GP_OF", "SIBLING_OF"]
    f_cols = [MEM1, MEM1, MEM1, MEM2, MEM2, QCOL]
    f_lss = ["-", "-", "-", "-", "-", "--"]

    fdx = 1.15
    for i, ent in enumerate(f_ents):
        x = i * fdx
        _node(ax_fc, x, 0, ent, PATH_FC, NODE_SZ_CHAIN)
        if i < len(f_rels):
            _arrow(ax_fc, x, 0, x + fdx, 0, f_cols[i], f_lss[i],
                   lw=1.4, shrink=SHRINK_C)
            ax_fc.text(x + fdx / 2, 0.28, f_rels[i], fontsize=4.5,
                       ha="center", color=f_cols[i], fontweight="bold")

    ax_fc.text((len(f_ents) - 1) * fdx / 2, -0.45,
               "Correct output: n0", fontsize=7, ha="center",
               style="italic")

    ax_fc.text((len(f_ents) - 1) * fdx / 2, -0.85,
               "'n0 CHILD_OF n1 SPOUSE_OF n2 CHILD_OF n3\\n"
               "n3 SPOUSE_OF n4 GP_OF n5 SIBLING_OF'",
               fontsize=5, ha="center", family="monospace",
               bbox=dict(boxstyle="round,pad=0.3", fc="#f7f7f7", ec="#cccccc"))

    ax_fc.set_xlim(-0.5, (len(f_ents) - 1) * fdx + 0.5)
    ax_fc.set_ylim(-1.3, 0.55)
    ax_fc.set_aspect("equal")
    ax_fc.axis("off")

    # =================================================================
    # Shared legend
    # =================================================================
    legend_elements = [
        Line2D([0], [0], color=MEM1, lw=2, label="Memory 1"),
        Line2D([0], [0], color=MEM2, lw=2, label="Memory 2"),
        Line2D([0], [0], color=QCOL, lw=2, linestyle="--",
               label="Query → ?"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    filename = Path(filename) if filename is not None else FIGURES_DIR / "rag_composition_diagram.pdf"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


def plot_paired_summary(
    all_results: dict[str, dict[str, list[int]]],
    filename: str | Path | None = None,
    panel_label: str | None = "k)",
) -> None:
    """Paired bar chart: Spatial vs Family, aggregated across hop counts.

    X-axis conditions (mapped from internal names):
        NC only      ← "NC"
        HPC only     ← "HPC"
        RAG (single) ← "Mem-1"
        RAG (multi)  ← "RAG-2L"
    """
    filename = Path(filename) if filename is not None else FIGURES_DIR / "rag_composition_summary.pdf"

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Map internal condition names → display labels
    SUMMARY_CONDS = [
        ("NC", "NC only"),
        ("HPC", "HPC only"),
        ("Mem-1", "RAG\n(single)"),
        ("RAG-2L", "RAG\n(multi)"),
    ]

    task_labels = {"Spatial": "Spatial", "Family": "Family tree"}
    task_colors = {"Spatial": "red", "Family": "blue"}

    # Aggregate across hop-counts per task, excluding 4-hop
    # (4-hop family templates are self-sufficient with a single memory)
    task_agg: dict[str, dict[str, list[int]]] = {}  # task → cond → flat list
    for group_name, conds in all_results.items():
        if "4-hop" in group_name:
            continue  # skip 4-hop (single memory is self-sufficient)
        task = group_name.split()[0]  # "Spatial" or "Family"
        if task not in task_agg:
            task_agg[task] = {cn: [] for cn, _ in SUMMARY_CONDS}
        for cn, _ in SUMMARY_CONDS:
            task_agg[task][cn].extend(conds.get(cn, []))

    tasks = [t for t in ["Spatial", "Family"] if t in task_agg]
    n_conds = len(SUMMARY_CONDS)

    fig, ax = plt.subplots(figsize=(2.7, 2))
    x = np.arange(n_conds)
    bar_width = 0.35
    offset = bar_width / 2

    for j, task in enumerate(tasks):
        means, sems = [], []
        for cn, _ in SUMMARY_CONDS:
            vals = task_agg[task][cn]
            m = float(np.mean(vals)) if vals else 0.0
            s = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            means.append(m)
            sems.append(s)

        positions = x - offset + j * bar_width
        ax.bar(
            positions, means, bar_width,
            yerr=sems, capsize=3,
            label=task_labels.get(task, task),
            color=task_colors.get(task, "gray"),
            alpha=0.4, edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in SUMMARY_CONDS])
    ax.set_ylabel("Average accuracy")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=7)
    if panel_label:
        ax.set_title(panel_label, fontsize=10, loc="center")

    plt.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
    print(f"Saved → {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    import os

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to inference_config.json with trained model and output paths.",
    )
    pre_args, remaining = pre_parser.parse_known_args()
    run_config = load_config(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Test RAG composition with split context paths",
        parents=[pre_parser],
    )
    parser.add_argument("--n-per-template", type=int, default=100,
                        help="Trials per template (default 100)")
    parser.add_argument("--tasks", type=str, default="spatial,family",
                        help="Comma-separated tasks to run")
    parser.add_argument(
        "--spatial-model", type=str,
        default=str(run_config["spatial_model_dir"]),
    )
    parser.add_argument(
        "--family-model", type=str,
        default=str(run_config["family_model_dir"]),
    )
    parser.add_argument("--seed", type=int, default=int(run_config.get("seed", 42)))
    parser.add_argument("--clear-cache", action="store_true",
                        help="Ignore cached results and re-run evaluation.")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip evaluation; just re-plot from cached results.")
    args = parser.parse_args(remaining)
    args.config = pre_args.config
    run_config = load_config(args.config)

    os.chdir(INFERENCE_DIR)
    set_seed(args.seed)

    import json

    CACHE_DIR = run_config["cache_dir"]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "rag_composition.json"
    print(f"Using inference config: {run_config['config_path']}")
    print(f"  spatial_model_dir: {args.spatial_model}")
    print(f"  family_model_dir:  {args.family_model}")
    print(f"  cache_dir:         {CACHE_DIR}")

    # ------------------------------------------------------------------
    # Load or compute results
    # ------------------------------------------------------------------
    if args.plot_only and cache_file.exists():
        all_results = json.loads(cache_file.read_text())
        print(f"Loaded cached results from {cache_file}")
    elif not args.clear_cache and cache_file.exists():
        all_results = json.loads(cache_file.read_text())
        print(f"Loaded cached results from {cache_file}")
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]
        all_results: dict[str, dict[str, list[int]]] = {}

        for task in tasks:
            if task == "spatial":
                model_dir = args.spatial_model
                templates = SPATIAL_TEMPLATES
            elif task == "family":
                model_dir = args.family_model
                templates = FAMILY_TEMPLATES
            else:
                print(f"Unknown task {task!r} — skipping")
                continue

            md = Path(model_dir)
            if not (md / "model.safetensors").exists() and \
               not (md / "pytorch_model.bin").exists():
                print(f"Model not found at {model_dir} — skipping {task}")
                continue

            set_seed(args.seed)
            print(f"\n{'=' * 60}")
            print(f"  {task.upper()} — RAG split-context composition")
            print(f"  Model: {model_dir}")
            print(f"{'=' * 60}\n")

            model = GPT(base_model=model_dir, base_model_name="gpt2")
            task_results = run_template_group(
                model,
                templates,
                task.capitalize(),
                task,
                n_per_template=args.n_per_template,
            )
            all_results.update(task_results)

        # Save to cache
        cache_file.write_text(json.dumps(all_results, indent=2))
        print(f"\nCached results → {cache_file}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    for group, conds in all_results.items():
        print(f"\n  {group}:")
        for cn in COND_NAMES:
            sc = conds.get(cn, [])
            acc = np.mean(sc) if sc else 0.0
            print(f"    {cn:8s}: {acc:.1%}  ({sum(sc)}/{len(sc)})")

        # Highlight key comparisons
        rag = float(np.mean(conds["RAG"]))
        m1 = float(np.mean(conds["Mem-1"]))
        m2 = float(np.mean(conds["Mem-2"]))
        best_single = max(m1, m2)
        delta = rag - best_single
        print(f"    RAG − best-single = {delta:+.1%}"
              f"{'  ✓ composition' if delta > 0.02 else ''}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
