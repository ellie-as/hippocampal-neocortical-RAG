#!/usr/bin/env python3
r"""
Analyze checkpoint samples for PCA behavior. Run from full_model/narratives.
Usage: python analyze_ckpt_pca.py --ckpt_dir ~/Downloads/latest\ ckpts
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_config as CFG
from utils import load_bartlett, load_topic_corpus_wiki, recall_prefix


def _first_sentence(txt: str) -> str:
    return recall_prefix()


def _trim(texts, trim_to):
    if trim_to is None:
        return texts
    if isinstance(texts, str):
        return texts[:trim_to] if trim_to else texts
    return [t[:trim_to] for t in texts]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=str(Path.home() / "Downloads" / "latest ckpts"))
    ap.add_argument("--epoch", type=int, default=None, help="Single epoch (omit with --all)")
    ap.add_argument("--temp", type=float, default=None, help="Single temp (omit with --all)")
    ap.add_argument("--all", action="store_true", help="Sweep all epoch/temp combos, print Bartlett→Recalled for each")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir).expanduser()

    bartlett = load_bartlett()
    first_sent = _first_sentence(bartlett)
    trim_to = len(bartlett)  # match plot_config "bartlett"

    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from scipy.spatial.distance import cosine

    emb = SentenceTransformer(getattr(CFG, "embedding_model", "all-MiniLM-L6-v2"))
    topics_cfg = getattr(CFG, "topics", ["Universe", "Politics", "Health", "Sport", "Nature"])

    # Load all topic data and discover epoch_temp keys
    data_by_topic = {}
    all_keys = set()
    for topic in topics_cfg:
        f = ckpt_dir / f"{topic}_checkpoint_samples.json"
        if not f.exists():
            cands = list(ckpt_dir.glob(f"{topic}_checkpoint_samples*.json"))
            f = cands[0] if cands else None
        if f is None or not f.exists():
            continue
        data = json.loads(f.read_text())
        data_by_topic[topic] = data
        for k in data:
            if "_" in k:
                try:
                    ep, tp = k.split("_", 1)
                    int(ep)
                    float(tp)
                    all_keys.add(k)
                except ValueError:
                    pass

    if not data_by_topic or not all_keys:
        print("No checkpoint data found.")
        return

    # Sort keys by (epoch, temp)
    def sort_key(k):
        a, b = k.split("_", 1)
        return (int(a), float(b))
    sorted_keys = sorted(all_keys, key=sort_key)

    # Load background once
    topics_with_data = sorted(data_by_topic.keys())
    print("Loading Wikipedia background...")
    bg_texts = load_topic_corpus_wiki(
        topics_with_data, seed=42,
        articles_per_topic=getattr(CFG, "articles_per_topic", 1000),
        chars_per_article=getattr(CFG, "chars_per_article", 2000),
    )

    def embed_list(texts):
        return emb.encode(_trim(texts, trim_to), show_progress_bar=False)

    bart_emb = embed_list([bartlett])[0]
    bg_means_by_topic = {}
    for topic in topics_with_data:
        texts = bg_texts.get(topic, [])
        if texts:
            embs = embed_list(texts)
            bg_means_by_topic[topic] = np.mean(embs, axis=0)
        else:
            bg_means_by_topic[topic] = np.zeros_like(bart_emb)

    # ---- ALL mode: print Bartlett→Recalled for every epoch/temp/topic ----
    if args.all:
        print(f"\nBartlett→Recalled cosine distance (lower=closer to topic centroid)\n")
        for key in sorted_keys:
            ep, tp = key.split("_", 1)
            print(f"--- epoch={ep} temp={tp} ---")
            for topic in topics_with_data:
                texts = [t for t in data_by_topic[topic].get(key, []) if isinstance(t, str) and t.strip()]
                if not texts:
                    continue
                recalled = [f"{first_sent} {t}".strip() if first_sent else t for t in texts[:100]]
                rec_embs = embed_list(recalled)
                rec_mean = np.mean(rec_embs, axis=0)
                ctr = bg_means_by_topic[topic]
                bart_dist = cosine(bart_emb, ctr)
                rec_dist = cosine(rec_mean, ctr)
                mark = "✓" if rec_dist < bart_dist else "✗"
                print(f"  {topic}: Bartlett {bart_dist:.2f} → Recalled {rec_dist:.2f} {mark}")
            print()
        return

    # ---- Single epoch/temp mode (original detailed analysis) ----
    epoch = args.epoch if args.epoch is not None else 5
    temp = args.temp if args.temp is not None else 0.5
    cache_key = f"{epoch}_{float(temp)}"

    samples_by_topic = {}
    for topic in topics_with_data:
        texts = [t for t in data_by_topic[topic].get(cache_key, []) if isinstance(t, str) and t.strip()]
        if texts:
            samples_by_topic[topic] = texts

    if not samples_by_topic:
        print(f"No data for {cache_key}")
        return

    print(f"Bartlett length: {len(bartlett)} chars, trim to {trim_to}")
    print(f"[{cache_key}] {sum(len(v) for v in samples_by_topic.values())} total samples\n")

    all_bg_embs = []
    bg_means = []
    group_sizes = []
    for topic in sorted(samples_by_topic.keys()):
        texts = bg_texts.get(topic, [])
        if texts:
            embs = embed_list(texts)
            all_bg_embs.append(embs)
            bg_means.append(np.mean(embs, axis=0))
            group_sizes.append(len(texts))
        else:
            group_sizes.append(0)
            bg_means.append(np.zeros_like(bart_emb))

    all_bg_points = np.vstack(all_bg_embs) if all_bg_embs else np.zeros((0, bart_emb.shape[0]))
    bg_means_arr = np.vstack(bg_means)

    topics_sorted = sorted(samples_by_topic.keys())
    rec_means = []
    rec_means_raw = []
    for topic in topics_sorted:
        texts = samples_by_topic[topic]
        recalled_full = [f"{first_sent} {t}".strip() if first_sent else t for t in texts[:100]]
        embs_full = embed_list(recalled_full)
        embs_raw = embed_list(texts[:100])
        rec_means.append(np.mean(embs_full, axis=0))
        rec_means_raw.append(np.mean(embs_raw, axis=0))

    rec_means_arr = np.vstack(rec_means)
    rec_means_raw_arr = np.vstack(rec_means_raw)

    print("--- Cosine distance to topic centroid (lower = closer) ---")
    print(f"{'Topic':<10} {'Bartlett':>10} {'Recalled(full)':>14} {'Recalled(raw)':>14}  Arrow dir?")
    for i, topic in enumerate(topics_sorted):
        ctr = bg_means_arr[i]
        bart_dist = cosine(bart_emb, ctr)
        rec_dist = cosine(rec_means_arr[i], ctr)
        rec_raw_dist = cosine(rec_means_raw_arr[i], ctr)
        closer = "✓ closer" if rec_dist < bart_dist else "✗ FARTHER"
        print(f"{topic:<10} {bart_dist:>10.4f} {rec_dist:>14.4f} {rec_raw_dist:>14.4f}  {closer}")

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_bg_points)
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_} sum={sum(pca.explained_variance_ratio_):.4f}")

    bg_2d = pca.transform(bg_means_arr)
    rec_2d = pca.transform(rec_means_arr)
    bart_2d = pca.transform(bart_emb.reshape(1, -1))[0]

    print("\n--- 2D positions (Bartlett origin, arrows to recalled) ---")
    print(f"Bartlett 2D: ({bart_2d[0]:.3f}, {bart_2d[1]:.3f})")
    for i, topic in enumerate(topics_sorted):
        cx, cy = bg_2d[i]
        rx, ry = rec_2d[i]
        dx, dy = rx - bart_2d[0], ry - bart_2d[1]
        # Arrow from Bartlett to recalled; centroid at (cx, cy)
        # For "toward topic", we expect (rx, ry) to be in direction of (cx, cy) from Bartlett
        to_centroid = np.array([cx - bart_2d[0], cy - bart_2d[1]])
        to_recalled = np.array([dx, dy])
        dot = np.dot(to_centroid, to_recalled)
        align = "aligned" if dot > 0 else "OPPOSITE"
        print(f"  {topic}: centroid=({cx:.3f},{cy:.3f}) recalled=({rx:.3f},{ry:.3f}) "
              f"arrow=({dx:.3f},{dy:.3f}) {align}")

    # Key diagnostic: recall cluster tightness vs centroid spread
    rec_centroid_2d = np.mean(rec_2d, axis=0)
    rec_std = np.std(rec_2d, axis=0)
    centroid_spread = np.std(bg_2d, axis=0)
    print("\n--- DIAGNOSIS: Recall cluster vs centroid spread ---")
    print(f"Recalled points centroid 2D: ({rec_centroid_2d[0]:.4f}, {rec_centroid_2d[1]:.4f})")
    print(f"Recalled std across topics:  ({rec_std[0]:.4f}, {rec_std[1]:.4f})")
    print(f"Topic centroids std:         ({centroid_spread[0]:.4f}, {centroid_spread[1]:.4f})")
    if np.max(rec_std) < 0.5 * np.max(centroid_spread):
        print(">>> WARNING: Recalled points cluster much tighter than topic centroids!")
        print(">>> In 2D PCA, arrows appear similar because recalls embed to similar region.")
        print(">>> Cosine-to-centroid (bar chart) is more reliable than 2D arrow direction.")

    # Check: are recalled points between Bartlett and centroid, or past centroid?
    print("\n--- Position relative to centroid ---")
    for i, topic in enumerate(topics_sorted):
        bc = np.linalg.norm(bg_2d[i] - bart_2d)
        rc = np.linalg.norm(rec_2d[i] - bg_2d[i])
        br = np.linalg.norm(rec_2d[i] - bart_2d)
        # If recalled is "between" Bartlett and centroid: br < bc and rc < bc
        between = br < bc and rc < bc
        past = br > bc
        print(f"  {topic}: d(Bartlett,centroid)={bc:.3f} d(recalled,centroid)={rc:.3f} "
              f"d(Bartlett,recalled)={br:.3f}  {'between' if between else 'past' if past else 'mixed'}")

    # Sample text inspection
    print("\n--- Sample recalled text (first 200 chars) per topic ---")
    for topic in topics_sorted[:2]:  # first 2 only
        t = samples_by_topic[topic][0]
        full = f"{first_sent} {t}".strip()
        print(f"[{topic}] raw len={len(t)} full len={len(full)}")
        print(f"  Full (trimmed): {full[:200]}...")
        print()


if __name__ == "__main__":
    main()
