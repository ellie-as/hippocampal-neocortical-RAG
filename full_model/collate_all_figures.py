"""
Collate all model simulation plots into a single figure.
Regenerates all plots from underlying data for visual consistency.
"""

from pathlib import Path
import copy
import inspect
import pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
from scipy.stats import sem, ttest_ind, ttest_rel, ttest_1samp
from scipy.spatial.distance import cosine as cos_dist
from sentence_transformers import SentenceTransformer
from wordfreq import zipf_frequency
from scipy.stats import gaussian_kde
import re
from collections import Counter
import sys

# ============================================================================
# PATHS & CONFIG
# ============================================================================
BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
sys.path.insert(0, str(REPO_ROOT))
DATA_DIR = BASE / "output" / "data"
FIGURES_DIR = REPO_ROOT / "figures"
LLM_RATINGS_DIR = BASE / "LLM ratings"
from scripts.source_data import export_figure_source_data

# Color scheme (matches HIPPOCORPUS notebook) - for word freq plots
COLOR_ORIGINAL = "tomato"
COLOR_ENCODED = "#4C78A8"
COLOR_CONSOLIDATED = "#F58518"

# Different colors for part a) similarity/memory plot
COLOR_SIMILARITY = "#9467BD"  # purple
COLOR_MEMORY = "#2CA02C"  # green

# Different colors for the LLM-vs-forgetting plot (part f): three purple shades.
COLOR_LLM_1 = "#3F007D"  # deep purple
COLOR_LLM_2 = "#6A51A3"  # medium purple
COLOR_LLM_3 = "#9E9AC8"  # light purple

# Q&A table config
QA_TABLE_NUM_QUESTIONS = 10
QA_TABLE_MAX_EPOCHS = None  # set to an int to truncate columns (keeps epoch 0..N)

# Slightly increase font sizes in the final collated figure.
FONT_SCALE = 1.22

def _fs(points: float) -> float:
    return float(points) * FONT_SCALE

_CAP_LABEL = {
    "original": "Original",
    "recalled": "Recalled",
    "retold": "Retold",
    "encoded": "Encoded",
    "consolidated": "Consolidated",
}

def _cap(s: str) -> str:
    return _CAP_LABEL.get(str(s), str(s))

def _maybe_float(value):
    return None if pd.isna(value) else float(value)

def _scaled_rcparams() -> dict:
    def _to_points(v) -> float:
        return float(FontProperties(size=v).get_size_in_points())
    keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ]
    return {k: _to_points(plt.rcParams[k]) * FONT_SCALE for k in keys}

# ============================================================================
# DATA LOADING
# ============================================================================
def load_pickle(name):
    path = DATA_DIR / name
    if path.exists():
        return pickle.load(open(path, "rb"))
    return None


def _disable_all_grids(fig):
    for ax in fig.get_axes():
        ax.grid(False, which="both")


def save_figure3_pdf(enc, con, forg, embedder, output=None):
    """Save the narrative encoding/consolidation/forgetting row as Figure 3."""
    fig = plt.figure(figsize=(16, 2.56))
    gs = gridspec.GridSpec(
        1, 5, figure=fig,
        wspace=0.38,
        width_ratios=[1.55, 0.01, 0.8, 0.8, 0.8],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    plot_similarity_vs_memsize(ax1, copy.deepcopy(enc), embedder)

    ax2 = fig.add_subplot(gs[0, 2])
    plot_recall_vs_consolidation(ax2, copy.deepcopy(enc), copy.deepcopy(con), embedder)

    ax3 = fig.add_subplot(gs[0, 3])
    plot_forgetting(ax3, copy.deepcopy(enc), copy.deepcopy(forg), embedder)

    ax4 = fig.add_subplot(gs[0, 4])
    plot_semantic_memory_consolidation(ax4)

    ax1.set_title('a) ' + ax1.get_title(), fontsize=_fs(10))
    ax2.set_title('b) ' + ax2.get_title(), fontsize=_fs(10))
    ax3.set_title('c) ' + ax3.get_title(), fontsize=_fs(10))
    ax4.set_title('d) ' + ax4.get_title(), fontsize=_fs(10))

    _disable_all_grids(fig)
    out_path = Path(output) if output else FIGURES_DIR / "Figure 3.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_figure_source_data(fig, "Figure 3")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_figure4_pdf(output=None):
    """Save the memory-content rows as Figure 4 with labels restarted at a."""
    fig = plt.figure(figsize=(16, 8.6))
    gs = gridspec.GridSpec(
        3, 6, figure=fig,
        height_ratios=[1, 1, 1],
        hspace=0.64,
        wspace=0.4,
    )

    gs_row1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=gs[0, :],
        wspace=0.35,
        width_ratios=[1, 1, 1, 1.2],
    )
    ax_llm = [fig.add_subplot(gs_row1[0, i]) for i in range(3)]
    plot_llm_attributes(ax_llm, version="model")

    ax_forg = fig.add_subplot(gs_row1[0, 3])
    plot_llm_vs_forgetting(ax_forg)

    ax_nfrd = [fig.add_subplot(gs[1, i]) for i in range(3)]
    plot_llm_attributes(ax_nfrd, version="nfrd")
    ax_nfrd[0].set_title("Concreteness", fontsize=_fs(9))
    ax_nfrd[1].set_title("Richness", fontsize=_fs(9))
    ax_nfrd[2].set_title("Specificity", fontsize=_fs(9))

    ax_hc = [fig.add_subplot(gs[1, i + 3]) for i in range(3)]
    plot_llm_attributes(ax_hc, version="hippocorpus")
    ax_hc[0].set_title("Concreteness", fontsize=_fs(9))
    ax_hc[1].set_title("Richness", fontsize=_fs(9))
    ax_hc[2].set_title("Specificity", fontsize=_fs(9))

    gs_row3 = gridspec.GridSpecFromSubplotSpec(
        1, 4,
        subplot_spec=gs[2, :],
        wspace=0.35,
        width_ratios=[0.9, 0.6, 1.25, 1.25],
    )
    ax_wf1 = fig.add_subplot(gs_row3[0, 0])
    plot_word_frequency_bars(ax_wf1, version="model")

    ax_wf2 = fig.add_subplot(gs_row3[0, 1])
    plot_word_frequency_bars(ax_wf2, version="nfrd")

    shared_ymin = min(ax_wf1.get_ylim()[0], ax_wf2.get_ylim()[0])
    shared_ymax = max(ax_wf1.get_ylim()[1], ax_wf2.get_ylim()[1])
    for ax in [ax_wf1, ax_wf2]:
        ax.set_ylim(shared_ymin, shared_ymax)

    ax_wf3 = fig.add_subplot(gs_row3[0, 2])
    plot_word_frequency_density(ax_wf3, version="model")

    ax_wf4 = fig.add_subplot(gs_row3[0, 3])
    plot_word_frequency_density(ax_wf4, version="nfrd")

    for ax in ax_llm:
        ax.set_title(ax.get_title(), fontsize=_fs(9))
    ax_forg.set_title('b) ' + ax_forg.get_title(), fontsize=_fs(10))

    fig.canvas.draw()

    bbox0 = ax_llm[0].get_position()
    bbox2 = ax_llm[2].get_position()
    fig.text((bbox0.x0 + bbox2.x1) / 2, bbox0.y1 + 0.025,
             'a)  Memory attributes: model data', fontsize=_fs(10), ha='center')

    bbox_nfrd0 = ax_nfrd[0].get_position()
    bbox_nfrd2 = ax_nfrd[2].get_position()
    fig.text((bbox_nfrd0.x0 + bbox_nfrd2.x1) / 2, bbox_nfrd0.y1 + 0.025,
             'c)  Memory attributes: NFRD', fontsize=_fs(10), ha='center')

    bbox_hc0 = ax_hc[0].get_position()
    bbox_hc2 = ax_hc[2].get_position()
    fig.text((bbox_hc0.x0 + bbox_hc2.x1) / 2, bbox_hc0.y1 + 0.025,
             'd)  Memory attributes: HIPPOCORPUS', fontsize=_fs(10), ha='center')

    ax_wf1.set_title('e) ' + ax_wf1.get_title(), fontsize=_fs(10))
    ax_wf2.set_title('f) ' + ax_wf2.get_title(), fontsize=_fs(10))
    ax_wf3.set_title('g) ' + ax_wf3.get_title(), fontsize=_fs(10))
    ax_wf4.set_title('h) ' + ax_wf4.get_title(), fontsize=_fs(10))

    _disable_all_grids(fig)
    out_path = Path(output) if output else FIGURES_DIR / "Figure 4.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_figure_source_data(fig, "Figure 4")
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

# ---------------------------------------------------------------------------
# Static-data cache (NFRD / HIPPOCORPUS — independent of LoRA config)
# ---------------------------------------------------------------------------
_STATIC_CACHE: dict = {}

def _load_static_cache(path: str | Path) -> bool:
    global _STATIC_CACHE
    p = Path(path)
    if p.exists():
        with open(p, "rb") as fh:
            _STATIC_CACHE = pickle.load(fh)
        print(f"[StaticCache] Loaded from {p}")
        return True
    return False

def _save_static_cache(path: str | Path) -> None:
    if not _STATIC_CACHE:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as fh:
        pickle.dump(_STATIC_CACHE, fh)
    print(f"[StaticCache] Saved to {p}")

def _get_nfrd_llm_df() -> pd.DataFrame | None:
    if "nfrd_llm_df" in _STATIC_CACHE:
        return _STATIC_CACHE["nfrd_llm_df"]
    p = LLM_RATINGS_DIR / "nfrd_llm_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    _STATIC_CACHE["nfrd_llm_df"] = df
    return df

def _get_hippocorpus_llm_df() -> pd.DataFrame | None:
    if "hippocorpus_llm_df" in _STATIC_CACHE:
        return _STATIC_CACHE["hippocorpus_llm_df"]
    p = LLM_RATINGS_DIR / "story_llm_ratings.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    _STATIC_CACHE["hippocorpus_llm_df"] = df
    return df

def _get_hippocorpus_stories_df() -> pd.DataFrame | None:
    if "hippocorpus_stories_df" in _STATIC_CACHE:
        return _STATIC_CACHE["hippocorpus_stories_df"]
    hc_path = BASE / "data" / "hippoCorpusV2.csv"
    if not hc_path.exists():
        return None
    df = pd.read_csv(hc_path)
    _STATIC_CACHE["hippocorpus_stories_df"] = df
    return df

def load_nfrd_raw_texts():
    """Load NFRD original and recall texts with story pairing."""
    if "nfrd_texts" in _STATIC_CACHE:
        return _STATIC_CACHE["nfrd_texts"]

    possible_roots = [
        BASE.parent / "data" / "Naturalistic-Free-Recall-Dataset",
        Path("/Users/eleanorspens/PycharmProjects/Naturalistic-Free-Recall-Dataset"),
    ]
    
    for root in possible_roots:
        story_dir = root / "story_transcript"
        recall_dir = root / "recall_transcripts"
        if story_dir.exists() and recall_dir.exists():
            originals, recalls, recall_story_idx = [], [], []
            for idx, story in enumerate(["baseball", "eyespy", "oregontrail", "pieman"]):
                orig_file = story_dir / f"{story}_transcript.txt"
                if orig_file.exists():
                    originals.append(orig_file.read_text(errors='ignore'))
                
                recall_subdir = recall_dir / story
                if recall_subdir.exists():
                    for f in recall_subdir.glob("*.txt"):
                        recalls.append(f.read_text(errors='ignore'))
                        recall_story_idx.append(idx)
            
            if originals and recalls:
                result = {"original": originals, "recall": recalls, "recall_story_idx": recall_story_idx}
                _STATIC_CACHE["nfrd_texts"] = result
                return result
    
    result = {"original": [], "recall": [], "recall_story_idx": []}
    _STATIC_CACHE["nfrd_texts"] = result
    return result

# ============================================================================
# WORD FREQUENCY HELPERS
# ============================================================================
def get_word_counts(text):
    words = re.findall(r"[a-z]+", text.lower())
    return dict(Counter(words))

def avg_zipf_frequency(counts):
    if not counts:
        return 0.0
    total = sum(counts.values())
    weighted = sum(zipf_frequency(w, 'en') * c for w, c in counts.items())
    return weighted / total if total > 0 else 0.0

def strip_first_sentence(text):
    text = text.strip()
    for sep in ['. ', '! ', '? ']:
        if sep in text:
            idx = text.index(sep)
            return text[idx + len(sep):]
    return text

# ============================================================================
# PLOT FUNCTIONS
# ============================================================================
def plot_similarity_vs_memsize(ax, enc, embedder):
    """Plot similarity vs memory size."""
    if enc is None:
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    recalled = enc["recalled_stories"]
    mem_sizes = enc["memory_sizes"]
    originals = recalled["full"]
    
    cos_sim = lambda a, b: 1 - cos_dist(a, b)
    n_lvls = [0, 1, 3]
    
    sims_lvl = {n: [] for n in n_lvls}
    sims_img = []
    
    for i, orig in enumerate(originals):
        emb_o = embedder.encode(orig)
        sims_img.append(cos_sim(emb_o, embedder.encode(recalled["imagined"][i])))
        for n in n_lvls:
            sims_lvl[n].append(cos_sim(emb_o, embedder.encode(recalled[n][i])))
    
    labels = ["Imagined", "Gist", "+1 detail", "+3 details", "Full"]
    mean_sim = [np.mean(sims_img)] + [np.mean(sims_lvl[n]) for n in n_lvls] + [1.0]
    err_sim = [sem(sims_img)] + [sem(sims_lvl[n]) for n in n_lvls] + [0.0]
    mean_mem = [0] + [np.mean(mem_sizes[n]) for n in n_lvls] + [np.mean([len(o.split()) for o in originals])]
    err_mem = [0] + [sem(mem_sizes[n]) for n in n_lvls] + [0]
    
    x = np.arange(len(labels))
    ax2 = ax.twinx()
    
    # Panel a) color scheme: two shades of red.
    color_similarity = "#B22222"  # firebrick
    color_memory = "#F08080"      # light coral
    ax.bar(x - 0.2, mean_sim, 0.4, yerr=err_sim, color=color_similarity, alpha=0.9, capsize=3, label="Similarity")
    ax2.bar(x + 0.2, mean_mem, 0.4, yerr=err_mem, color=color_memory, alpha=0.9, capsize=3, label="Memory size")
    
    ax.set_ylabel("Similarity", color=color_similarity)
    ax.tick_params(axis='y', colors=color_similarity)
    ax2.set_ylabel("Memory (tokens)", color=color_memory)
    ax2.tick_params(axis='y', colors=color_memory)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=_fs(8))
    ax.set_title("Similarity vs. memory size", fontsize=_fs(10))
    ax.set_ylim(0, 1.1)
    ax._source_data_rows = [
        {
            "Memory condition": label,
            "Similarity": float(sim),
            "Similarity SEM": float(sim_err),
            "Memory (tokens)": float(mem),
            "Memory (tokens) SEM": float(mem_err),
        }
        for label, sim, sim_err, mem, mem_err in zip(labels, mean_sim, err_sim, mean_mem, err_mem)
    ]
    ax._source_data_title = "Similarity vs memory size"
    ax2._skip_source_data = True


def plot_recall_vs_consolidation(ax, enc, con, embedder):
    """Plot recall performance across consolidation epochs."""
    if enc is None or con is None:
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    # Convert distances to similarities
    con["epoch_dist_orig"] = [[1 - v for v in d] for d in con["epoch_dist_orig"]]
    con["epoch_dist_enc"] = [[1 - v for v in d] for d in con["epoch_dist_enc"]]
    
    cos_sim = lambda a, b: 1 - cos_dist(a, b)
    originals = enc["recalled_stories"]["full"]
    encoded = enc["recalled_stories"][0]
    imag = enc["recalled_stories"]["imagined"]
    
    sim_o0 = [cos_sim(embedder.encode(i), embedder.encode(o)) for i, o in zip(imag, originals)]
    sim_e0 = [cos_sim(embedder.encode(i), embedder.encode(g)) for i, g in zip(imag, encoded)]
    
    mu_o = [np.mean(sim_o0)] + [np.mean(d) for d in con["epoch_dist_orig"]]
    se_o = [sem(sim_o0)] + [sem(d) for d in con["epoch_dist_orig"]]
    mu_e = [np.mean(sim_e0)] + [np.mean(d) for d in con["epoch_dist_enc"]]
    se_e = [sem(sim_e0)] + [sem(d) for d in con["epoch_dist_enc"]]
    
    epochs = list(range(len(mu_o)))
    ax.errorbar(epochs, mu_o, yerr=se_o, marker="o", capsize=3, color=COLOR_ORIGINAL, label="To original")
    ax.errorbar(epochs, mu_e, yerr=se_e, marker="s", capsize=3, color=COLOR_ENCODED, label="To encoded")
    
    ax.set_xlabel("Epoch", fontsize=_fs(9))
    ax.set_ylabel("Cosine similarity", fontsize=_fs(9))
    ax.set_title("Recall vs. consolidation", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7), loc='lower right')
    ax.set_ylim(0.63, 0.9)
    ax.grid(True, alpha=0.3)
    ax._source_data_rows = [
        {
            "Epoch": int(epoch),
            "Comparison": comparison,
            "Cosine similarity": float(mean),
            "SEM": float(error),
        }
        for comparison, means, errors in [
            ("To original", mu_o, se_o),
            ("To encoded", mu_e, se_e),
        ]
        for epoch, mean, error in zip(epochs, means, errors)
    ]


def plot_forgetting(ax, enc, forg, embedder):
    """Plot forgetting curves."""
    if enc is None or forg is None or "recalls" not in forg:
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    cos_sim = lambda a, b: 1 - cos_dist(a, b)
    originals = enc["recalled_stories"]["full"]
    encoded = enc["recalled_stories"][0]
    recalls = forg["recalls"]
    
    n_episodes = len(recalls)
    sims_orig = [[] for _ in range(n_episodes)]
    sims_enc = [[] for _ in range(n_episodes)]
    
    for i, orig in enumerate(originals):
        emb_o = embedder.encode(orig)
        emb_e = embedder.encode(encoded[i])
        for ep in range(n_episodes):
            if i < len(recalls[ep]):
                emb_r = embedder.encode(recalls[ep][i])
                sims_orig[ep].append(cos_sim(emb_o, emb_r))
                sims_enc[ep].append(cos_sim(emb_e, emb_r))
    
    mu_o = [np.mean(s) for s in sims_orig]
    se_o = [sem(s) if len(s) > 1 else 0 for s in sims_orig]
    mu_e = [np.mean(s) for s in sims_enc]
    se_e = [sem(s) if len(s) > 1 else 0 for s in sims_enc]
    
    episodes = list(range(n_episodes))
    ax.errorbar(episodes, mu_o, yerr=se_o, marker="o", capsize=3, color=COLOR_ORIGINAL, label="To original")
    ax.errorbar(episodes, mu_e, yerr=se_e, marker="s", capsize=3, color=COLOR_ENCODED, label="To encoded")
    
    tick_candidates = [0, 2, 4, 6, 8]
    ticks = [t for t in tick_candidates if t < n_episodes]
    if ticks:
        ax.set_xticks(ticks)
    
    ax.set_xlabel("Stage", fontsize=_fs(9))
    ax.set_ylabel("Cosine similarity", fontsize=_fs(9))
    ax.set_title("Recall vs. forgetting", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7), loc='lower left')
    ax.set_ylim(0.63, 0.9)
    ax.grid(True, alpha=0.3)
    ax._source_data_rows = [
        {
            "Stage": int(stage),
            "Comparison": comparison,
            "Cosine similarity": float(mean),
            "SEM": float(error),
        }
        for comparison, means, errors in [
            ("To original", mu_o, se_o),
            ("To encoded", mu_e, se_e),
        ]
        for stage, mean, error in zip(episodes, means, errors)
    ]


def plot_semantic_memory_consolidation(ax):
    """Plot semantic memory accuracy over consolidation epochs."""
    con = load_pickle("consolidation_recall.pkl")
    
    if con is None or "semantic_accuracy" not in con:
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    sem_acc = con["semantic_accuracy"]
    epochs = list(range(len(sem_acc)))
    
    ax.plot(epochs, sem_acc, marker='o', color=COLOR_SIMILARITY, linewidth=2, markersize=5)
    ax.axhline(1/3, color='gray', linestyle='--', alpha=0.5, label='Chance')
    
    ax.set_xlabel("Epoch", fontsize=_fs(9))
    ax.set_ylabel("Accuracy", fontsize=_fs(9))
    ax.set_title("Semantic memory", fontsize=_fs(10))
    ax.set_ylim(0.25, 0.85)
    ax.legend(fontsize=_fs(7), loc='lower right')
    ax.grid(True, alpha=0.3)
    ax._source_data_rows = [
        {"Epoch": int(epoch), "series": "Semantic memory", "Accuracy": float(acc)}
        for epoch, acc in zip(epochs, sem_acc)
    ] + [
        {"Epoch": int(epoch), "series": "Chance", "Accuracy": float(1 / 3)}
        for epoch in epochs
    ]


def export_semantic_qa_table_csv(con: dict) -> None:
    """
    Export a CSV of semantic Q&A examples across consolidation epochs.

    Reads: consolidation_recall.pkl["semantic_results"] (epoch 0 baseline is index 0).
    Writes:
      - output/data/semantic_qa_table.csv
    """
    if not con or "semantic_results" not in con:
        print("[QATable] consolidation semantic_results not found; skipping Q&A CSV export.")
        return

    sem_results = list(con.get("semantic_results") or [])
    if not sem_results or not sem_results[0].get("per_question"):
        print("[QATable] No per-question semantic results; skipping Q&A CSV export.")
        return

    semantic_questions = list(con.get("semantic_questions") or [])
    semantic_stories = list(con.get("semantic_stories") or [])

    # Determine epoch labels (epoch 0 baseline + logged epochs)
    n_epochs = len(sem_results) - 1
    if QA_TABLE_MAX_EPOCHS is not None:
        n_epochs = min(n_epochs, int(QA_TABLE_MAX_EPOCHS))
        sem_results = sem_results[: n_epochs + 1]

    per_q0 = sem_results[0]["per_question"]
    n_questions = min(QA_TABLE_NUM_QUESTIONS, len(per_q0))
    q_indices = list(range(n_questions))

    epoch_cols = [f"Epoch {i}" for i in range(len(sem_results))]

    # Build CSV rows
    csv_rows = []
    for qi in q_indices:
        pq0 = per_q0[qi]
        question = (pq0.get("question", "") or "").strip()

        # Prefer the original question dict (contains context); fall back to story-derived context if needed.
        qdict = semantic_questions[qi] if qi < len(semantic_questions) else {}
        context = (qdict.get("context") or "").strip()
        if not context and qi < len(semantic_stories):
            context = semantic_stories[qi].split(".")[0].strip() + "."

        correct = (qdict.get("correct") or pq0.get("correct_answer") or "").strip()
        wrongs = qdict.get("wrong") or pq0.get("wrong_answers") or []
        wrong1 = (wrongs[0] if len(wrongs) > 0 else "").strip()
        wrong2 = (wrongs[1] if len(wrongs) > 1 else "").strip()

        prompt = (
            f"<s>[INST] Remember the story in which '{context}' {question} "
            f"Answer with a word or short phrase. [/INST]"
        )

        csv_row = {
            "question_idx": qi,
            "story_idx": int(pq0.get("story_idx", qi)),
            "question": question,
            "context": context,
            "prompt": prompt,
            "correct_answer": correct,
            "wrong_answer_1": wrong1,
            "wrong_answer_2": wrong2,
        }

        for ep_i, r in enumerate(sem_results):
            pq = (r.get("per_question") or [])
            ans_raw = (pq[qi].get("model_answer", "") if qi < len(pq) else "") or ""
            csv_row[f"Epoch {ep_i}"] = ans_raw.strip()
        csv_rows.append(csv_row)

    # Save CSV for easy reuse (paper/table exports)
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    cols = [
        "question_idx",
        "story_idx",
        "question",
        "context",
        "prompt",
        "correct_answer",
        "wrong_answer_1",
        "wrong_answer_2",
    ] + epoch_cols
    df_csv = pd.DataFrame(csv_rows)
    for c in cols:
        if c not in df_csv.columns:
            df_csv[c] = ""
    df_csv = df_csv[cols]
    df_csv.to_csv(DATA_DIR / "semantic_qa_table.csv", index=False)
    print(f"[QATable] Saved: {DATA_DIR / 'semantic_qa_table.csv'}")


def plot_llm_attributes(axes, version="model"):
    """Plot LLM attributes bar charts."""
    if version == "model":
        ratings_path = DATA_DIR / "story_llm_ratings_simulated.csv"
        groups = ["original", "encoded", "consolidated"]
        colors = [COLOR_ORIGINAL, COLOR_ENCODED, COLOR_CONSOLIDATED]
        col_name = "version"
        df = pd.read_csv(ratings_path) if ratings_path.exists() else None
    elif version == "hippocorpus":
        groups = ["recalled", "retold"]
        colors = [COLOR_ENCODED, COLOR_CONSOLIDATED]
        col_name = "memType"
        df = _get_hippocorpus_llm_df()
    elif version == "nfrd":
        groups = ["original", "recalled"]
        colors = [COLOR_ORIGINAL, COLOR_ENCODED]
        col_name = "type"
        df = _get_nfrd_llm_df()
    
    if df is None:
        for ax in axes:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    metrics = [
        ("concrete_vs_abstract", "Concreteness"),
        ("rich_vs_poor_details", "Richness"),
        ("specific_vs_general", "Specificity"),
    ]
    panel_by_version = {"model": "a", "nfrd": "c", "hippocorpus": "d"}
    title_by_version = {
        "model": "Memory attributes model",
        "nfrd": "Memory attributes NFRD",
        "hippocorpus": "Memory attributes HIPPOCORPUS",
    }
    
    for ax, (col, title) in zip(axes, metrics):
        means = df.groupby(col_name)[col].mean().reindex(groups)
        sems = df.groupby(col_name)[col].sem().reindex(groups)
        
        ax.bar(range(len(groups)), means, yerr=sems, color=colors, capsize=3, alpha=0.9)
        ax.set_title(title, fontsize=_fs(9))
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([_cap(g) for g in groups], rotation=20, fontsize=_fs(8))
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Significance tests - all pairwise comparisons for 3 groups
        # Use paired t-test for model version (same stories at different stages)
        if len(groups) == 3:
            comparisons = [(0, 1), (1, 2), (0, 2)]  # orig-enc, enc-cons, orig-cons
            y_positions = [0.92, 0.96, 1.00]
            for (i1, i2), y in zip(comparisons, y_positions):
                if version == "model" and "item_id" in df.columns:
                    # Paired t-test: match by item_id
                    df1 = df[df[col_name] == groups[i1]][[col, "item_id"]].dropna()
                    df2 = df[df[col_name] == groups[i2]][[col, "item_id"]].dropna()
                    merged = df1.merge(df2, on="item_id", suffixes=("_1", "_2"))
                    if len(merged) > 1:
                        _, p = ttest_rel(merged[f"{col}_1"], merged[f"{col}_2"])
                    else:
                        p = 1.0
                else:
                    g1v = df[df[col_name] == groups[i1]][col].dropna()
                    g2v = df[df[col_name] == groups[i2]][col].dropna()
                    _, p = ttest_ind(g1v, g2v, equal_var=False)
                sym = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
                ax.plot([i1, i2], [y, y], color='black', lw=0.8)
                ax.text((i1 + i2) / 2, y + 0.005, sym, ha='center', fontsize=_fs(6), color='black')
        elif len(groups) == 2:
            if version == "nfrd" and {"story_name", col_name}.issubset(df.columns):
                # NFRD recalls are matched to one of four source stories. Test
                # whether recall-level changes from the matched original story
                # differ from zero, instead of treating originals and recalls
                # as independent groups.
                originals = (
                    df[df[col_name] == groups[0]]
                    .dropna(subset=["story_name", col])
                    .drop_duplicates("story_name")
                    .set_index("story_name")[col]
                )
                recalled = df[df[col_name] == groups[1]].dropna(subset=["story_name", col]).copy()
                recalled["_matched_original"] = recalled["story_name"].map(originals)
                changes = (recalled[col] - recalled["_matched_original"]).dropna()
                _, p = ttest_1samp(changes, 0) if len(changes) > 1 else (np.nan, 1.0)
            else:
                g1v = df[df[col_name] == groups[0]][col].dropna()
                g2v = df[df[col_name] == groups[1]][col].dropna()
                _, p = ttest_ind(g1v, g2v, equal_var=False)
            sym = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            y = 0.95
            ax.plot([0, 1], [y, y], color='black', lw=0.8)
            ax.text(0.5, y + 0.01, sym, ha='center', fontsize=_fs(7), color='black')

        ax._source_data_panel = panel_by_version.get(version)
        ax._source_data_title = f"{title_by_version.get(version, version)} {title}"
        ax._source_data_rows = [
            {
                "Memory stage": _cap(group),
                "Score": float(means.loc[group]),
                "SEM": _maybe_float(sems.loc[group]),
            }
            for group in groups
            if group in means.index and not pd.isna(means.loc[group])
        ]
    
    axes[0].set_ylabel("Score", fontsize=_fs(9))


def plot_llm_vs_forgetting(ax):
    """Plot LLM attributes vs forgetting."""
    forg_path = DATA_DIR / "forgetting_llm_ratings.csv"
    if not forg_path.exists():
        ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
        return
    
    df = pd.read_csv(forg_path)
    metrics = ["concrete_vs_abstract", "rich_vs_poor_details", "specific_vs_general"]
    colors = [COLOR_LLM_1, COLOR_LLM_2, COLOR_LLM_3]
    
    for metric, color in zip(metrics, colors):
        grp = df.groupby("episode")[metric]
        means = grp.mean()
        sems = grp.sem()
        ax.errorbar(means.index, means.values, yerr=sems.values, marker="o", 
                   color=color, capsize=3, label=metric.split("_")[0].title())
    
    ax.set_xlabel("Stage", fontsize=_fs(9))
    ax.set_ylabel("Score", fontsize=_fs(9))
    ax.set_title("Memory attributes and forgetting", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7), loc="lower center")
    ax.grid(True, alpha=0.3)
    metric_labels = {
        "concrete_vs_abstract": "Concreteness",
        "rich_vs_poor_details": "Richness",
        "specific_vs_general": "Specificity",
    }
    rows = []
    for metric in metrics:
        grp = df.groupby("episode")[metric]
        means = grp.mean()
        sems = grp.sem()
        for stage in means.index:
            rows.append(
                {
                    "Stage": int(stage),
                    "Metric": metric_labels[metric],
                    "Score": float(means.loc[stage]),
                    "SEM": _maybe_float(sems.loc[stage]),
                }
            )
    ax._source_data_panel = "b"
    ax._source_data_title = "Memory attributes and forgetting"
    ax._source_data_rows = rows


def plot_word_frequency_bars(ax, version="model"):
    """Plot word frequency bar charts."""
    # Match default Matplotlib bar width used in rows above.
    bar_width = 0.8
    if version == "model":
        enc = load_pickle("recalled_stories.pkl")
        con = load_pickle("consolidation_recall.pkl")
        
        if enc is None:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
            return
        
        originals = enc["recalled_stories"]["full"]
        recalled = [strip_first_sentence(t) for t in enc["recalled_stories"][0]]
        
        if con:
            consolidated_raw = con["epoch_recalls"][-1]
            recalled_lens = [len(t.split()) for t in recalled]
            consolidated = []
            for t, max_len in zip(consolidated_raw, recalled_lens):
                t = strip_first_sentence(t)
                words = t.split()[:max_len]
                consolidated.append(" ".join(words))
        else:
            consolidated = recalled
        
        groups = ["original", "encoded", "consolidated"]
        texts = [originals, recalled, consolidated]
        colors = [COLOR_ORIGINAL, COLOR_ENCODED, COLOR_CONSOLIDATED]
        
    elif version == "nfrd":
        nfrd = load_nfrd_raw_texts()
        if not nfrd["original"] or not nfrd["recall"]:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
            return
        
        groups = ["original", "recalled"]
        texts = [nfrd["original"], nfrd["recall"]]
        colors = [COLOR_ORIGINAL, COLOR_ENCODED]
        nfrd_paired = True
        orig_freqs_by_story = [avg_zipf_frequency(get_word_counts(t)) for t in nfrd["original"] if t.strip()]
        recall_story_idx = nfrd.get("recall_story_idx", [])
    
    elif version == "hippocorpus":
        df = _get_hippocorpus_stories_df()
        if df is None:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
            return
        
        recalled_texts = df[df["memType"] == "recalled"]["story"].dropna().tolist()
        retold_texts = df[df["memType"] == "retold"]["story"].dropna().tolist()
        
        groups = ["recalled", "retold"]
        texts = [recalled_texts, retold_texts]
        colors = [COLOR_ENCODED, COLOR_CONSOLIDATED]
        nfrd_paired = False
    
    if version not in ["nfrd"]:
        nfrd_paired = False
    
    freqs = []
    for text_list in texts:
        f = [avg_zipf_frequency(get_word_counts(t)) for t in text_list if t.strip()]
        freqs.append(f)
    
    means = [np.mean(f) for f in freqs]
    sems = [sem(f) if len(f) > 1 else 0 for f in freqs]
    
    ax.bar(range(len(groups)), means, width=bar_width, yerr=sems, color=colors, capsize=3, alpha=0.9)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_cap(g) for g in groups], rotation=20, fontsize=_fs(8))
    ax.set_ylabel("Zipf frequency", fontsize=_fs(9))
    ax.set_title("Model" if version == "model" else "NFRD", fontsize=_fs(10))
    ax._source_data_panel = "e" if version == "model" else "f"
    ax._source_data_title = "Word frequency bars " + ("Model" if version == "model" else "NFRD")
    ax._source_data_rows = [
        {
            "Memory stage": _cap(group),
            "Zipf frequency": float(mean),
            "SEM": float(error),
        }
        for group, mean, error in zip(groups, means, sems)
    ]
    
    # Zoom y-axis to accommodate significance bars
    y_min = min(np.array(means) - np.array(sems)) - 0.1
    y_max = max(np.array(means) + np.array(sems)) + 0.25
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Significance tests
    bar_top = max(np.array(means) + np.array(sems))
    if len(groups) == 3:
        comparisons = [(0, 1), (1, 2), (0, 2)]
        y_positions = [bar_top + 0.03, bar_top + 0.08, bar_top + 0.13]
        for (i1, i2), y in zip(comparisons, y_positions):
            _, p = ttest_ind(freqs[i1], freqs[i2], equal_var=False)
            sym = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            ax.plot([i1, i2], [y, y], color='black', lw=0.8)
            ax.text((i1 + i2) / 2, y + 0.005, sym, ha='center', fontsize=_fs(6), color='black')
    elif len(groups) == 2:
        # For NFRD, use one-sample t-test on differences (each recall vs its original story)
        if version == "nfrd" and nfrd_paired and recall_story_idx:
            diffs = []
            for i, recall_freq in enumerate(freqs[1]):
                if i < len(recall_story_idx):
                    story_idx = recall_story_idx[i]
                    if story_idx < len(orig_freqs_by_story):
                        diffs.append(recall_freq - orig_freqs_by_story[story_idx])
            if diffs:
                _, p = ttest_1samp(diffs, 0)
            else:
                p = 1.0
        else:
            _, p = ttest_ind(freqs[0], freqs[1], equal_var=False)
        sym = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        y = bar_top + 0.03
        ax.plot([0, 1], [y, y], color='black', lw=0.8)
        ax.text(0.5, y + 0.005, sym, ha='center', fontsize=_fs(7), color='black')


def plot_word_frequency_density(ax, version="model"):
    """Plot word frequency density (KDE), token-weighted via word counts."""
    def kde_1d(values: np.ndarray, weights: np.ndarray, *, bw: float) -> gaussian_kde:
        sig = inspect.signature(gaussian_kde)
        if "weights" in sig.parameters:
            return gaussian_kde(values, weights=weights, bw_method=bw)

        # Fallback for older SciPy without weights support: sample approximately.
        cap = 200_000
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return gaussian_kde(np.array([0.0]))
        if weights.sum() <= cap and np.all(weights == np.floor(weights)):
            expanded = np.repeat(values, weights.astype(int))
        else:
            rng = np.random.default_rng(0)
            p = weights / max(weights.sum(), 1.0)
            expanded = rng.choice(values, size=cap, replace=True, p=p)
        return gaussian_kde(expanded, bw_method=bw)

    def counts_from_texts(texts: list[str]) -> Counter[str]:
        c = Counter()
        for t in texts:
            if t and t.strip():
                c.update(re.findall(r"[a-z]+", t.lower()))
        return c

    def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if values.size == 0:
            return float("nan")
        if weights.size != values.size:
            raise ValueError("weights must be same length as values")
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0, 1]")
        order = np.argsort(values)
        v = values[order]
        w = weights[order]
        cw = np.cumsum(w)
        total = float(cw[-1])
        if total <= 0:
            return float("nan")
        target = q * total
        idx = int(np.searchsorted(cw, target, side="left"))
        idx = max(0, min(idx, len(v) - 1))
        return float(v[idx])

    if version == "model":
        enc = load_pickle("recalled_stories.pkl")
        con = load_pickle("consolidation_recall.pkl")
        
        if enc is None:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
            return
        
        originals = enc["recalled_stories"]["full"]
        recalled = [strip_first_sentence(t) for t in enc["recalled_stories"][0]]
        
        if con:
            consolidated_raw = con["epoch_recalls"][-1]
            recalled_lens = [len(t.split()) for t in recalled]
            consolidated = []
            for t, max_len in zip(consolidated_raw, recalled_lens):
                t = strip_first_sentence(t)
                words = t.split()[:max_len]
                consolidated.append(" ".join(words))
        else:
            consolidated = recalled
        
        series = {"original": originals, "encoded": recalled, "consolidated": consolidated}
        colors = [COLOR_ORIGINAL, COLOR_ENCODED, COLOR_CONSOLIDATED]
        
    elif version == "nfrd":
        nfrd = load_nfrd_raw_texts()
        if not nfrd["original"] or not nfrd["recall"]:
            ax.text(0.5, 0.5, "Data not found", ha='center', va='center', transform=ax.transAxes)
            return
        series = {"original": nfrd["original"], "recalled": nfrd["recall"]}
        colors = [COLOR_ORIGINAL, COLOR_ENCODED]

    per_group = {}
    for label, texts in series.items():
        wc = counts_from_texts(texts)
        if not wc:
            continue
        words = list(wc.keys())
        weights = np.array([wc[w] for w in words], dtype=float)
        values = np.array([zipf_frequency(w, "en") for w in words], dtype=float)
        per_group[label] = (values, weights)

    if not per_group:
        ax.text(0.5, 0.5, "No tokens", ha='center', va='center', transform=ax.transAxes)
        return

    all_values = np.concatenate([v for (v, _w) in per_group.values() if len(v)])
    # Force a consistent Zipf x-range so the density plots always run to 10,
    # even if the observed values top out around ~8.
    x_min, x_max = 0.0, 10.0
    # Bandwidth controls smoothing; lower = less smoothing (sharper curves).
    x_grid = np.linspace(x_min, x_max, 600)

    keys = list(series.keys())
    y_by_label = {}
    mean_by_label = {}
    quartiles_by_label: dict[str, tuple[float, float, float]] = {}
    for label in keys:
        if label not in per_group:
            continue
        values, weights = per_group[label]
        color = colors[keys.index(label)]
        kde = kde_1d(values, weights, bw=0.25)
        y = kde(x_grid)
        y_by_label[label] = y
        ax.plot(x_grid, y, label=_cap(label), color=color, lw=2)
        ax.fill_between(x_grid, 0, y, color=color, alpha=0.15)
        mean = float(np.average(values, weights=weights))
        mean_by_label[label] = mean
        ax.axvline(mean, color=color, ls='--', alpha=0.85, lw=1.3)
        q1 = weighted_quantile(values, weights, 0.25)
        q2 = weighted_quantile(values, weights, 0.50)
        q3 = weighted_quantile(values, weights, 0.75)
        quartiles_by_label[label] = (q1, q2, q3)

    if y_by_label:
        max_y = max(float(np.max(v)) for v in y_by_label.values() if len(v))
        if max_y > 0:
            ax.set_ylim(0, max_y * 1.08)

    # Show quartiles (IQR bar + median tick) along the bottom inside the axes.
    # This avoids cluttering the density area with extra vertical lines.
    trans = ax.get_xaxis_transform()  # x in data coords, y in axes fraction
    for i, label in enumerate(keys):
        if label not in quartiles_by_label:
            continue
        q1, q2, q3 = quartiles_by_label[label]
        color = colors[keys.index(label)]
        y0 = 0.03 + 0.035 * i
        ax.plot([q1, q3], [y0, y0], transform=trans, color=color, lw=2.0, solid_capstyle="butt")
        ax.plot([q2, q2], [y0 - 0.010, y0 + 0.010], transform=trans, color=color, lw=2.0)
    
    ax.set_xlabel("Zipf frequency", fontsize=_fs(9))
    ax.set_ylabel("Density", fontsize=_fs(9))
    ax.set_title("Model" if version == "model" else "NFRD", fontsize=_fs(10))
    ax.legend(fontsize=_fs(7))
    ax.set_xlim(x_min, x_max)
    ax.grid(False)
    ax._source_data_panel = "g" if version == "model" else "h"
    ax._source_data_title = "Word frequency density " + ("Model" if version == "model" else "NFRD")
    ax._source_data_rows = [
        {
            "series": _cap(label),
            "Zipf frequency": float(x_value),
            "Density": float(y_value),
        }
        for label in keys
        if label in y_by_label
        for x_value, y_value in zip(x_grid, y_by_label[label])
    ]


# ============================================================================
# MAIN
# ============================================================================
def main(*, data_dir: str | None = None, figures_dir: str | None = None,
         static_cache: str | None = None):
    global DATA_DIR, FIGURES_DIR
    if data_dir is not None:
        DATA_DIR = Path(data_dir)
    if figures_dir is not None:
        FIGURES_DIR = Path(figures_dir)

    if static_cache:
        _load_static_cache(static_cache)

    plt.rcParams.update(_scaled_rcparams())
    print(f"Loading data from {DATA_DIR} ...")
    enc = load_pickle("recalled_stories.pkl")
    con = load_pickle("consolidation_recall.pkl")
    forg = load_pickle("forgetting_multi.pkl")

    # Also export a semantic Q&A CSV if available.
    export_semantic_qa_table_csv(con)
    
    print("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Saving figures to {FIGURES_DIR} ...")
    save_figure3_pdf(enc, con, forg, embedder)
    save_figure4_pdf()

    if static_cache:
        _save_static_cache(static_cache)


if __name__ == "__main__":
    import argparse as _ap

    _parser = _ap.ArgumentParser(description="Collate all model simulation plots.")
    _parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Override data directory (default: full_model/output/data)",
    )
    _parser.add_argument(
        "--figures_dir", type=str, default=None,
        help="Override output figure directory (default: ../figures)",
    )
    _parser.add_argument(
        "--static_cache", type=str, default=None,
        help="Path to cache NFRD/HIPPOCORPUS data (avoids recomputation across sweep configs)",
    )
    _args = _parser.parse_args()
    main(data_dir=_args.data_dir, figures_dir=_args.figures_dir, static_cache=_args.static_cache)
