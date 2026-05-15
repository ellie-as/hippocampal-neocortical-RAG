"""
Configuration for collate_figures.py.

Edit values here to change which epoch/temperature conditions are shown in each
panel of the combined narrative figure, without touching plotting code.
"""

# ---------------------------------------------------------------------------
# Results directory (where bartlett_twostage.py wrote per-topic outputs)
# ---------------------------------------------------------------------------
results_dir = "output_twostage"

# Topics (order determines plot order)
topics = ["Universe", "Politics", "Health", "Sport", "Nature"]

# ---------------------------------------------------------------------------
# Panels a) PCA  &  c) Cosine-distance bar chart
# These share the same samples: recalled stories at a given epoch/temp.
# ---------------------------------------------------------------------------
pca_epoch = 5
pca_temp = 0.5
pca_num_samples = 100        # samples per topic (for PCA means + cosine stats)
# Optional: path to an external checkpoint_samples directory for PCA/cosine panels.
# When set, uses ALL texts from the specified ep-temp key instead of generation caches.
pca_ckpt_dir = None #"/Users/eleanorspens/Downloads/checkpoint_samples"

# Projection strategy for panel a).
# Options:
#   "pca_all"               - fit PCA on [background points + centroids + recalled means + Bartlett]
#   "pca_background"        - fit PCA on background points only, then project others
#   "pca_centroids_bartlett"- fit PCA on [topic centroids + Bartlett], then project all points
#   "pca_centroids_recalled"- fit PCA on [topic centroids + recalled means], then project all points
#   "umap_all"              - fit UMAP on all plotted points
#   "tsne_all"              - fit t-SNE on all plotted points
# For your issue, try:
#   1) "pca_centroids_bartlett"
#   2) "umap_all"
pca_projection = "pca_background"

# UMAP/t-SNE options (used only when projection is umap_all/tsne_all)
pca_umap_n_neighbors = 30
pca_umap_min_dist = 0.1
pca_tsne_perplexity = 30

# ---------------------------------------------------------------------------
# Panel d) Fraction of new words vs temperature
# Sweeps temperature at a fixed epoch.  Each point on the x-axis is a
# different temperature; the y-axis is the fraction of words in the recalled
# text that are *not* in the original Bartlett story.
# ---------------------------------------------------------------------------
newwords_vs_temp_epoch = 10  # which epoch's recalls to use

# ---------------------------------------------------------------------------
# Panel e) Fraction of new words vs epoch
# Sweeps epoch at a fixed temperature.  Each point on the x-axis is a
# training epoch; the y-axis is the fraction of novel words.
# ---------------------------------------------------------------------------
newwords_vs_epoch_temp = 0.5  # which temperature's recalls to use

# ---------------------------------------------------------------------------
# Panel f) Cosine distance to original: encoded vs consolidated
# Shows two bars:
#   "Encoded"      – cosine distance between the xRAG-encoded gist of the
#                    Bartlett story and the original (before consolidation).
#   "Consolidated" – cosine distance between the LoRA-consolidated recall
#                    and the original (after consolidation training).
# Uses artifacts from bartlett_encoding_vs_consolidation.py:
#   {enc_vs_con_dir}/statistics.json   (pre-computed distances)
#   {enc_vs_con_dir}/encoded_samples.json
#   {enc_vs_con_dir}/consolidated_samples.json
# ---------------------------------------------------------------------------
enc_vs_con_dir = "bartlett_encoding_vs_consolidation"

# ---------------------------------------------------------------------------
# Panel g) Word clouds
# Higher temperatures produce more semantic intrusions (more interesting clouds).
# ---------------------------------------------------------------------------
wordcloud_epoch = 5
wordcloud_temp = 1.0
wordcloud_num_samples = 10000   # samples per topic for word clouds

# --- Word cloud data source ---
# When True, aggregate ALL epoch-temp recalls from the _ckpt_cache (or
# wordcloud_ckpt_dir) instead of sampling from a single epoch/temp.
wordcloud_all_ckpts = False
# Optional: path to an external checkpoint_samples directory.  When None,
# uses {results_dir}/_ckpt_cache.
wordcloud_ckpt_dir = None #"/Users/eleanorspens/Downloads/checkpoint_samples"

# --- Word cloud style options ---
# Weighting:
#   "raw"              – raw word frequency (bigger word = appears more often)
#   "tfidf_topics"     – TF-IDF wrt the 5 categories: IDF = how rare across topics
#   "tfidf_english"    – TF-IDF wrt English language: IDF from wordfreq library
#   "topic_contrast"   – log-ratio: how over-represented a word is in this topic
#                        compared to the other topics (highlights topic-specific words)
#   "contrastive_bg"   – keep only recall words that exist in the topic's Wikipedia
#                        background AND are over-represented vs other topics' backgrounds.
#                        Best for showing genuine topic-related semantic intrusions.
#   "noun_tfidf_contrast_bg_english" – noun-only per-topic TF-IDF over recalls,
#                        contrasted across topics, filtered to background vocabulary
#                        and common English words.
wordcloud_weighting = "noun_tfidf_contrast_bg_english"

# POS filtering (requires spaCy with an English model, e.g. en_core_web_sm):
#   "all"    – keep all content words (default)
#   "nouns"  – keep only nouns (NOUN, PROPN)
#   "nouns_adjs" – keep nouns and adjectives (NOUN, PROPN, ADJ)
wordcloud_pos_filter = "all"

# Cross-topic exclusion:
#   If True, words that appear in *every* topic's recalls are excluded from
#   all word clouds.  This removes generic words that don't discriminate
#   between topics ("story", "said", etc.) and highlights topic-specific
#   semantic intrusions.
wordcloud_exclude_shared = False

# Minimum word length (characters) to include in the cloud.
wordcloud_min_word_len = 1

# When True, exclude words not in the English dictionary (wordfreq).
# Requires the wordfreq package. Set False to show all tokens.
wordcloud_english_only = True

# Optional per-topic frequency filter (applied after tokenization/POS filtering).
# These thresholds are counts within each topic's sampled recalls.
# Set either to None to disable that bound.
#
# Examples:
#   wordcloud_min_freq = 3     # drop words that appear 1–2 times
#   wordcloud_max_freq = 50    # drop extremely common words that dominate
wordcloud_min_freq = 50
wordcloud_max_freq = 5000

# Optional topic boost: when True, boost weights of words that are common in the
# topic training corpus for that topic (Wikipedia topic articles; see utils.load_topic_corpus_wiki).
# This can help surface topic-related words even if they are not the top raw intrusions.
wordcloud_topic_boost = False
wordcloud_topic_boost_strength = 1.0  # 0 disables effect; higher => stronger boost

# ---------------------------------------------------------------------------
# Background data (Wikipedia articles for topic centroids + PCA clouds)
# ---------------------------------------------------------------------------
articles_per_topic = 1000
chars_per_article = 5000
use_tfidf_filter = True

# ---------------------------------------------------------------------------
# Embedding model (SBERT)
# Used for PCA, cosine-distance bars, and encoding-vs-consolidation panel.
# ---------------------------------------------------------------------------
embedding_model = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Embedding truncation
# Before computing SBERT embeddings, all texts (background articles, recalled
# stories, and the Bartlett story itself) are truncated to this many characters
# so that cosine distances are computed on a level playing field.
# Set to "bartlett" (the default) to auto-truncate to len(Bartlett story),
# or an integer for a fixed character limit, or None to disable truncation.
# ---------------------------------------------------------------------------
embed_trim_chars = "bartlett"

# ---------------------------------------------------------------------------
# Cosine aggregation method
# How to summarise multiple recalled-sample embeddings into a single cosine
# distance from the topic centroid (used in panel c and check_recall scripts).
#   "mean_of_distances"  – compute cosine distance per sample, then average.
#   "distance_of_mean"   – average the embeddings first, then compute one
#                          cosine distance from that mean embedding.
# "distance_of_mean" is less sensitive to outlier samples; "mean_of_distances"
# is the more standard estimator and pairs naturally with SEM error bars.
# ---------------------------------------------------------------------------
cosine_aggregation = "mean_of_distances"

# ---------------------------------------------------------------------------
# Generation settings
# ---------------------------------------------------------------------------
max_new_tokens = 500

# When True, enforce generated recall continuations to be at least as long
# (in tokenizer tokens) as the Bartlett story text.
# This is applied in collate_figures sampling paths used by the combined plot.
enforce_mean_length = True
