import json, pickle, random, time, pprint
from pathlib import Path
from typing import Any, Dict, List
from math import ceil, sqrt
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import sem, ttest_rel
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine as cos_dist
from transformers import AutoTokenizer
import openai
from scipy.stats import sem
import numpy as np
from pathlib import Path


plt.rcParams.update({
    "figure.figsize"   : (6, 3),   # ← one size fits all
    "font.size"        : 12,
    "axes.titlesize"   : 14,
    "axes.labelsize"   : 12,
    "xtick.labelsize"  : 12,
    "ytick.labelsize"  : 12,
    "legend.fontsize"  : 12,
})


SYSTEM_PROMPT = """Your task is score text on three metrics: how concrete (vs abstract) it is, how rich in detail it is, and how specific (vs general) it is.

Return ONLY a JSON dictionary with 3 keys, each a float 0-1:

{
  "concrete_vs_abstract": 0-1,
  "rich_vs_poor_details": 0-1,
  "specific_vs_general":  0-1
}

A higher score corresponds to more concrete, richer in detail, or more specific text."""


def plot_consolidation(con: Dict, outdir: Path) -> None:
    con["epoch_dist_orig"] = [[1 - v for v in d] for d in con["epoch_dist_orig"]]
    con["epoch_dist_enc"]  = [[1 - v for v in d] for d in con["epoch_dist_enc"]]
    
    epochs = np.arange(1, len(con["epoch_dist_orig"]) + 1)
    mu_o   = [np.mean(d) for d in con["epoch_dist_orig"]]
    se_o   = [sem(d)     for d in con["epoch_dist_orig"]]
    mu_e   = [np.mean(d) for d in con["epoch_dist_enc"]]
    se_e   = [sem(d)     for d in con["epoch_dist_enc"]]

    plt.figure(figsize=(6, 3))
    plt.errorbar(epochs[0:8], mu_o[0:8], yerr=se_o[0:8], marker="o", label="Original", capsize=5)
    plt.errorbar(epochs[0:8], mu_e[0:8], yerr=se_e[0:8], marker="s", label="Encoded", capsize=5)
    plt.xlabel("Epoch"); plt.ylabel("Cosine similarity")
    plt.title("b)   Recall performance vs. consolidation", pad=12)
    plt.ylim(0.68, 0.9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "recall_drift_posthoc.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_similarity_memory(enc: Dict, originals: List[str],
                           tokenizer_name: str, outdir: Path) -> None:
    recalled   = enc["recalled_stories"];  mem_sizes = enc["memory_sizes"]
    embedder   = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer  = AutoTokenizer.from_pretrained(tokenizer_name)

    labels = ["Imagined", "Gist", "+ 1 detail", "+ 3 details", "Full"]
    n_lvls = [0, 1, 3];   width = .4
    cos_sim = lambda a, b: 1 - cos_dist(a, b)

    sims_lvl = {n: [] for n in n_lvls}; sims_img = []; full_lens = []
    for i, orig in enumerate(originals):
        emb_o = embedder.encode(orig)
        img_version = recalled["imagined"][i]
        img_version = img_version[img_version.index('[/INST]') + 7:]
        sims_img.append(cos_sim(emb_o, embedder.encode(img_version)))
        full_lens.append(len(tokenizer(orig)["input_ids"]))
        for n in n_lvls:
            sims_lvl[n].append(cos_sim(emb_o, embedder.encode(recalled[n][i])))

    mean_sim = [np.mean(sims_img)] + [np.mean(sims_lvl[n]) for n in n_lvls] + [1.0]
    err_sim  = [sem(sims_img)]     + [sem(sims_lvl[n])     for n in n_lvls] + [0.0]
    mean_mem = [0] + [np.mean(mem_sizes[n]) for n in n_lvls] + [np.mean(full_lens)]
    err_mem  = [0] + [sem(mem_sizes[n])     for n in n_lvls] + [sem(full_lens)]

    # grab the default color cycle
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    sim_color = cycle[0]   # blue
    mem_color = cycle[1]   # orange

    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(6, 3))

    ax1.bar(x - width/2, mean_sim, width, yerr=err_sim,
            color=sim_color, alpha=.9, capsize=5)
    ax1.set_ylabel("Similarity", color=sim_color)
    ax1.tick_params(axis='y', colors=sim_color)
    ax1.set_ylim(0.6, 1.02)

    ax2 = ax1.twinx()
    ax2.bar(x + width/2, mean_mem, width, yerr=err_mem,
            color=mem_color, alpha=.9, capsize=5)
    ax2.set_ylabel("Memory size (tokens)", color=mem_color)
    ax2.tick_params(axis='y', colors=mem_color)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15)
    ax1.set_title("a)   Similarity vs. memory size", pad=12)

    fig.tight_layout()
    fig.savefig(outdir / "similarity_vs_memsize.png", dpi=300, bbox_inches="tight")
    plt.close()
    

def plot_forgetting(forg: Dict, outdir: Path) -> None:
    """
    Draws two forgetting curves:

        • blue  – mean cosine-distance of recalls to the ORIGINAL stories
        • orange – mean cosine-distance of the same recalls to the ENCODED (0-detail) stories

    Everything needed is recovered from the existing pickles:
        output/data/recalled_stories.pkl
        output/data/forgetting[_multi].pkl
    """

    if "distances" not in forg or "recalls" not in forg:
        print("[plot_forgetting]  key 'distances' or 'recalls' missing – skipping.")
        return

    data_dir   = outdir.parent / "data"              # output/data
    enc_path   = data_dir / "recalled_stories.pkl"
    try:
        enc = pickle.load(open(enc_path, "rb"))
    except FileNotFoundError:
        print(f"[plot_forgetting] cannot find {enc_path} – skipping plot.")
        return

    encoded_stories = enc["recalled_stories"][0]     # detail-level 0 (gist)

    recalls = forg["recalls"]
    if recalls and isinstance(recalls[0], str):
        recalls = [recalls]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    cos = lambda a, b: 1 - cos_dist(a, b)   # convert to similarity

    dist_enc = []
    for episode in recalls:
        dists = []
        for rec, enc_story in zip(episode, encoded_stories):
            d = cos(embedder.encode(rec), embedder.encode(enc_story))
            dists.append(d)
        dist_enc.append(dists)

    dist_orig = [[1 - d for d in ep] for ep in forg["distances"]]

    means_o   = [np.mean(ep) for ep in dist_orig]
    sems_o    = [sem(ep)     for ep in dist_orig]
    means_e   = [np.mean(ep) for ep in dist_enc]
    sems_e    = [sem(ep)     for ep in dist_enc]

    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    x     = range(len(means_o))

    plt.figure(figsize=(6, 3))
    plt.errorbar(x[0:], means_o[0:], yerr=sems_o[0:],
                 marker="o", capsize=5, color=cycle[0],
                 label="Original")
    plt.errorbar(x[0:], means_e[0:], yerr=sems_e[0:],
                 marker="s", capsize=5, color=cycle[1],
                 label="Encoded")

    plt.xlabel("Delay (stages)")
    plt.ylabel("Cosine similarity")
    plt.title("c)   Simulating forgetting", pad=12)
    plt.legend()
    plt.ylim(0.68, 0.9)
    plt.tight_layout()
    plt.savefig(outdir / "forgetting_curve_posthoc.png",
                dpi=300, bbox_inches="tight")
    plt.close()


def llm_scores(text: str, client, model="gpt-4o-mini") -> dict:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": text[:16_000]}]
    resp = client.chat.completions.create(
        model=model, temperature=0.0,
        messages=msgs, response_format={"type": "json_object"})
    return json.loads(resp.choices[0].message.content)


def build_llm_csv(originals: List[str], enc: Dict, con: Dict | None,
                  forg: Dict | None, data_dir: Path) -> tuple[pd.DataFrame | None,
                                                               pd.DataFrame | None]:

    api_key = "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"
    client  = openai.OpenAI(api_key=api_key)

    # by-version
    file_ver = data_dir / "story_llm_ratings_simulated.csv"
    df_ver   = pd.read_csv(file_ver) if file_ver.exists() else None
    if df_ver is None and con is not None:
        rec_enc   = enc["recalled_stories"][0]
        rec_imag  = enc["recalled_stories"]["imagined"]
        rec_cons  = con["epoch_recalls"][-1]
        records   = []
        for i, orig in enumerate(originals):
            variants = [("original", orig),
                        ("encoded",  rec_enc[i][:len(orig)]),
                        ("consolidated", rec_cons[i][:len(orig)]),
                        ("imagined", rec_imag[i][:len(orig)])]
            for v, txt in variants:
                s = llm_scores(txt, client) | {"item_id": i, "version": v}
                records.append(s)
        df_ver = pd.DataFrame(records); df_ver.to_csv(file_ver, index=False)

    # forgetting
    file_forg = data_dir / "forgetting_llm_ratings.csv"
    df_forg   = pd.read_csv(file_forg) if file_forg.exists() else None
    if df_forg is None and forg is not None:
        recalls = forg.get("recalls")
        if recalls and isinstance(recalls[0], str):
            recalls = [recalls]
        records = []
        for epi, episode in enumerate(recalls or []):
            for i, txt in enumerate(episode):
                s = llm_scores(txt, client) | {"episode": epi, "item_id": i}
                records.append(s)
        if records:
            df_forg = pd.DataFrame(records); df_forg.to_csv(file_forg, index=False)

    return df_ver, df_forg


def plot_llm_ratings(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [("concrete_vs_abstract", "Concreteness"),
               ("rich_vs_poor_details", "Richness"),
               ("specific_vs_general",  "Specificity")]
    groups  = ["original", "encoded", "consolidated", "imagined"]
    colors  = ["#4C78A8", "#F58518", "#54A24B", "tomato"]

    fig, axes = plt.subplots(1, 3, figsize=(6, 3), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        means = df.groupby("version")[col].mean().reindex(groups)
        sems  = df.groupby("version")[col].sem().reindex(groups)
        ax.bar(groups, means, yerr=sems, color=colors, capsize=4)
        ax.set_title(title, pad=12); ax.set_xticklabels(groups, rotation=20)
    fig.tight_layout()
    fig.savefig(outdir / "llm_attribute_bars.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_llm_ratings_forgetting(df: pd.DataFrame, outdir: Path) -> None:
    metrics = [("concrete_vs_abstract", "Concreteness"),
               ("rich_vs_poor_details", "Richness"),
               ("specific_vs_general",  "Specificity")]
    colors  = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 3))
    for (col, title), ax, c in zip(metrics, axes, colors):
        grp = df.groupby("episode")[col]
        means = grp.mean()
        sems = grp.sem()
        ax.errorbar(means.index, means.values, yerr=sems.values,
                    marker="o", color=c, capsize=3)
        ax.set_xlabel("Delay (stages)"); ax.set_title(title, pad=12)
        ax.set_xticks(grp.mean().index[0:])
    axes[0].set_ylabel("Score")
    fig.suptitle('d)   Memory attributes and forgetting', fontsize=14)
    fig.tight_layout()
    fig.savefig(outdir / "llm_attribute_vs_forgetting.png", dpi=300, bbox_inches="tight")
    plt.close()


def make_mosaic(plot_dir: Path) -> None:
    files_subset = [
        'similarity_vs_memsize.png',          # top-left  → wider
        'recall_drift_posthoc.png',           # top-right → narrower
        'forgetting_curve_posthoc.png',       # bottom-left → narrower
        'llm_attribute_vs_forgetting.png'     # bottom-right → wider
    ]
    pngs = [plot_dir / fname for fname in files_subset if (plot_dir / fname).exists()]
    if len(pngs) < 4:
        print("[make_mosaic] Not all plots found – skipping.")
        return

    fig = plt.figure(figsize=(12, 6))  # Adjust total figure size
    gs  = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], wspace=0.01, hspace=0.01)

    # Top-left (wider)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(plt.imread(pngs[0]))
    ax1.axis("off")

    # Top-right (narrower)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(plt.imread(pngs[1]))
    ax2.axis("off")

    # Bottom-left (narrower)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.imshow(plt.imread(pngs[2]))
    ax3.axis("off")

    # Bottom-right (wider)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(plt.imread(pngs[3]))
    ax4.axis("off")

    fig.savefig(plot_dir / "all_plots_grid.png", dpi=300, bbox_inches="tight")
    plt.close()


def main(outdir="./output"):
    data_dir = Path(outdir) / "data"
    plot_dir = Path(outdir) / "plots";  plot_dir.mkdir(parents=True, exist_ok=True)

    enc  = pickle.load(open(data_dir / "recalled_stories.pkl", "rb"))
    originals = list(enc["recalled_stories"]["full"])

    con  = pickle.load(open(data_dir / "consolidation_recall.pkl", "rb")) \
           if (data_dir / "consolidation_recall.pkl").exists() else None

    forg_pkl = data_dir / "forgetting_multi.pkl"
    forg = pickle.load(open(forg_pkl, "rb")) if forg_pkl.exists() else None

    plot_similarity_memory(enc, originals,
                           tokenizer_name="mistralai/Mistral-7B-Instruct-v0.2",
                           outdir=plot_dir)
    if con is not None:
        plot_consolidation(con, plot_dir)
    if forg is not None:
        plot_forgetting(forg, plot_dir)

    df_ver, df_forg = build_llm_csv(originals, enc, con, forg, data_dir)
    if df_ver is not None:
        plot_llm_ratings(df_ver, plot_dir)
    if df_forg is not None:
        plot_llm_ratings_forgetting(df_forg, plot_dir)

    make_mosaic(plot_dir)
    print("Plots written to:", plot_dir.resolve())

if __name__ == "__main__":
    main()
