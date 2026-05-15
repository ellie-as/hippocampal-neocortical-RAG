from __future__ import annotations
import gc
import json
import math
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# ---- Bartlett story ---------------------------------------------------------
# Canonical location: full model/data/bartlett.txt
# All scripts should use BARTLETT_TXT (the absolute path) or load_bartlett()
# rather than embedding the story inline.
_NARRATIVES_DIR = Path(__file__).resolve().parent          # full model/narratives
_FULLMODEL_DIR = _NARRATIVES_DIR.parent                    # full model
BARTLETT_TXT = str(_FULLMODEL_DIR / "data" / "bartlett.txt")
BARTLETT_PROMPT_CUE = "One night two men from Egulac..."
EXCLUDE_OFFENSIVE_WORDS = {
    # a few wikipedia-specific words to remove from word clouds
    "revisions", "comment", "original", "answer", "essays",
    # a few offensive words to remove from word clouds
    "rape", "intercourse", "abuse", "whites",
}

BARTLETT_FALLBACK = """One night two young men from Egulac went down to the river to hunt seals and while they were there it became foggy and calm. Then they heard war-cries, and they thought: "Maybe this is a war-party". They escaped to the shore, and hid behind a log. Now canoes came up, and they heard the noise of paddles, and saw one canoe coming up to them. There were five men in the canoe, and they said:
"What do you think? We wish to take you along. We are going up the river to make war on the people."
One of the young men said,"I have no arrows."
"Arrows are in the canoe," they said.
"I will not go along. I might be killed. My relatives do not know where I have gone. But you," he said, turning to the other, "may go with them."
So one of the young men went, but the other returned home.
And the warriors went on up the river to a town on the other side of Kalama. The people came down to the water and they began to fight, and many were killed. But presently the young man heard one of the warriors say, "Quick, let us go home: that man has been hit." Now he thought: "Oh, they are ghosts." He did not feel sick, but they said he had been shot.
So the canoes went back to Egulac and the young man went ashore to his house and made a fire. And he told everybody and said: "Behold I accompanied the ghosts, and we went to fight. Many of our fellows were killed, and many of those who attacked us were killed. They said I was hit, and I did not feel sick."
He told it all, and then he became quiet. When the sun rose he fell down. Something black came out of his mouth. His face became contorted. The people jumped up and cried.
He was dead."""

# ---- spaCy (for phrase splitting) ------------------------------------------
try:
    import spacy  # type: ignore
    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        _nlp = None
except Exception:
    spacy = None  # type: ignore
    _nlp = None

import sys
_XRAG_ROOT = (Path(__file__).resolve().parent.parent / "xRAG")
if _XRAG_ROOT.exists():
    sys.path.insert(0, str(_XRAG_ROOT))
else:
    # Keep backward compatibility if repo layout changes.
    sys.path.insert(0, str(Path("../xRAG").resolve()))
from src.language_modeling.lm_utils import XRAG_TOKEN   # single source of truth
from src.model import SFR, XMistralForCausalLM

# Import global LoRA configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_config import ACTIVE as LORA_CONFIG

from sentence_transformers import SentenceTransformer


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def get_device(use_mps: bool = False) -> torch.device:
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def first_sentence(text: str) -> str:
    """First sentence (fallback: ~12 tokens)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    if parts and len(parts[0].split()) >= 3:
        return parts[0].strip().rstrip(".!?")
    return " ".join(text.strip().split()[:12])


def recall_prefix() -> str:
    """Short Bartlett cue used for Stage 2 training, recall, and recalled-text embeddings."""
    return BARTLETT_PROMPT_CUE


def load_bartlett(path: str | None = None) -> str:
    """Load the Bartlett story from *path* (default: ``full model/data/bartlett.txt``).

    Falls back to the built-in ``BARTLETT_FALLBACK`` string when the file is
    missing so that training/plotting can still run.
    """
    if path is None:
        path = BARTLETT_TXT
    p = Path(path)
    if not p.exists():
        return BARTLETT_FALLBACK
    txt = p.read_text(encoding="utf-8").strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt

def load_topic_corpus_imdb(
    topics: List[str],
    seed: int,
    articles_per_topic: int = 500,
    chars_per_article: int = 500,
    docs_per_datastore: Optional[int] = 1000,
) -> Dict[str, List[str]]:
    """
    Load background docs from HF 'adrienheymans/imdb-movie-genres' (same as your working script).
    """
    print("Loading IMDB genres dataset …")
    ds = load_dataset("adrienheymans/imdb-movie-genres", split="train")

    has_genre_str = "genre" in ds.features and str(ds.features["genre"].dtype).startswith("string")
    has_genres_list = "genres" in ds.features

    if not has_genre_str and not has_genres_list:
        raise RuntimeError(
            "Expected a 'genre' (string) or 'genres' (list) column in 'adrienheymans/imdb-movie-genres'."
        )

    def extract_genres(example) -> List[str]:
        if has_genre_str and isinstance(example.get("genre"), str) and example["genre"].strip():
            return [example["genre"].strip()]
        if has_genres_list and isinstance(example.get("genres"), (list, tuple)):
            return [str(g).strip() for g in example["genres"] if str(g).strip()]
        return []

    from collections import defaultdict
    genre_to_descs: Dict[str, List[str]] = defaultdict(list)
    for ex in ds:
        desc = str(ex.get("text") or "").strip()
        if not desc:
            continue
        for g in extract_genres(ex):
            genre_to_descs[g].append(desc)

    available_genres = set(genre_to_descs.keys())
    print(f"IMDB Genres found: {sorted(available_genres)}")

    def _norm(s: str) -> str:
        return s.strip().lower()

    rng = random.Random(seed)
    topic_texts: Dict[str, List[str]] = {}

    for topic in topics:
        matches = [g for g in available_genres if _norm(g) == _norm(topic)]
        if not matches:
            matches = [g for g in available_genres if _norm(topic) in _norm(g) or _norm(g) in _norm(topic)]

        if not matches:
            print(f"⚠︎ IMDB: No descriptions found for requested genre '{topic}'.")
            topic_texts[topic] = []
            continue

        pool: List[str] = []
        for g in matches:
            pool.extend(genre_to_descs[g])

        rng.shuffle(pool)
        take_n = min(articles_per_topic, len(pool))
        picked = [t[: chars_per_article] for t in pool[:take_n]]
        if docs_per_datastore is not None:
            picked = picked[: docs_per_datastore]

        topic_texts[topic] = picked
        print(f"IMDB genre {topic}: selected {len(picked)} descriptions (matched labels: {matches})")
    return topic_texts


def _tfidf_filter_closest(texts: List[str], n_keep: int) -> List[str]:
    """Keep the *n_keep* texts closest to the mean TF-IDF vector.

    This removes topical outliers (mis-categorised articles) that would
    dilute the category centroid.
    """
    if len(texts) <= n_keep:
        return texts

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_distances

    vec = TfidfVectorizer(max_features=10_000, stop_words="english")
    tfidf = vec.fit_transform(texts)                # sparse (n_docs, vocab)
    centroid = np.asarray(tfidf.mean(axis=0))          # dense (1, vocab)
    dists = cosine_distances(tfidf, centroid).ravel()  # (n_docs,)
    keep_idx = np.argsort(dists)[:n_keep]
    keep_idx_sorted = sorted(keep_idx)               # preserve original order
    return [texts[i] for i in keep_idx_sorted]


def load_topic_corpus_wiki(topics: List[str], seed: int,
                      articles_per_topic: int = 500,
                      chars_per_article: int = 500,
                      docs_per_datastore: Optional[int] = 1000,
                      use_tfidf_filter: bool = True) -> Dict[str, List[str]]:

    print("Loading Wikipedia topics dataset …")
    ds = load_dataset("tarekziade/wikipedia-topics", split="train")

    def not_people(example):
        cats = example.get("categories") or []
        return not any(str(c).lower() == "people" for c in cats)
    ds = ds.filter(not_people)

    topic_aliases = {
        "Universe":   ["Space", "Astronomy", "Cosmology", "Universe"],
        "Sport":      ["Entertainment"],
        "Nature":     ["Nature", "Natural sciences", "Environment"],
        "Health":     ["Health", "Medicine"],
        "Politics":   ["Politics", "Government"],
        "Technology": ["Technology", "Computing", "Engineering"],
    }

    def roughly_matches(cat: str, target: str) -> bool:
        c = cat.lower(); t = target.lower()
        return (t == c) or (t in c) or (t.rstrip("s") in c) or ((t + "s") in c)

    topic_texts: Dict[str, List[str]] = {}
    for topic in topics:
        cand_labels = topic_aliases.get(topic, [topic])

        def has_exact(ex):
            cats = ex.get("categories") or []
            wanted = {cl.lower() for cl in cand_labels}
            return any(str(c).lower() in wanted for c in cats)
        d_exact = ds.filter(has_exact)

        if len(d_exact) == 0:
            def has_rough(ex):
                cats = ex.get("categories") or []
                return any(roughly_matches(str(c), t) for c in cats for t in cand_labels)
            d_sel = ds.filter(has_rough)
        else:
            d_sel = d_exact

        if len(d_sel) == 0:
            print(f"⚠︎ No articles found for topic '{topic}' using aliases {cand_labels}.")
            topic_texts[topic] = []
            continue

        # Oversample candidates only when TF-IDF filtering has room to drop outliers.
        n_candidates = min(
            len(d_sel),
            articles_per_topic * 2 if use_tfidf_filter else articles_per_topic,
        )
        idx = list(range(len(d_sel)))
        random.Random(seed).shuffle(idx)
        idx = idx[:n_candidates]
        candidates = [str(d_sel[i]["text"])[: chars_per_article] for i in idx]

        if use_tfidf_filter:
            # Keep the articles_per_topic docs closest to the mean TF-IDF vector.
            texts = _tfidf_filter_closest(candidates, articles_per_topic)
            selection_desc = "TF-IDF filtered"
        else:
            texts = candidates[:articles_per_topic]
            selection_desc = "seeded sample"

        if docs_per_datastore is not None:
            texts = texts[: docs_per_datastore]
        topic_texts[topic] = texts
        print(f"Topic {topic}: selected {len(texts)} from {n_candidates} candidates "
              f"({selection_desc}, aliases: {cand_labels})")

    return topic_texts

class XRAG:
    """
    Reusable xRAG wrapper:
      - Registers XRAG_TOKEN as a single token and binds it in the model
      - Batched datastore building
      - Plain and retrieval-augmented generation
      - Surprise-phrase extraction (spaCy + perplexity)
    The 'cfg' passed to XRAG can expose:
        llm_name, retriever_name,
        retriever_batch_size, retriever_max_length, docs_per_datastore
    """
    def __init__(self, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.llm = None; self.llm_tok = None
        self.retriever = None; self.ret_tok = None
        self._xrag_token_id: Optional[int] = None

    def _register_xrag_token(self):
        self.llm_tok.add_special_tokens({"additional_special_tokens": [XRAG_TOKEN]})
        self.llm.resize_token_embeddings(len(self.llm_tok))
        xrag_id = self.llm_tok.convert_tokens_to_ids(XRAG_TOKEN)
        if xrag_id == self.llm_tok.unk_token_id:
            raise RuntimeError(f"XRAG_TOKEN {XRAG_TOKEN!r} mapped to UNK — registration failed.")
        self.llm.set_xrag_token_id(xrag_id)
        self._xrag_token_id = xrag_id
        test_prompt = f"Background: {XRAG_TOKEN}\nQuestion: What happened?"
        ids = self.llm_tok(test_prompt, return_tensors="pt").input_ids
        if not (ids == xrag_id).any().item():
            raise RuntimeError("XRAG token is not present as a single id in the prompt. Check tokenizer setup.")

    def load(self):
        if self.llm is not None:
            return
        print("Loading xRAG models …")
        self.llm = XMistralForCausalLM.from_pretrained(
            self.cfg.llm_name, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        self.llm_tok = AutoTokenizer.from_pretrained(
            self.cfg.llm_name, add_eos_token=False, use_fast=False, padding_side="left"
        )
        self._register_xrag_token()

        self.retriever = SFR.from_pretrained(self.cfg.retriever_name, torch_dtype=torch.bfloat16)
        self.retriever = self.retriever.to(self.device).eval()
        self.ret_tok = AutoTokenizer.from_pretrained(self.cfg.retriever_name)
        print("xRAG ready")

    def release(self):
        for attr in ("llm", "llm_tok", "retriever", "ret_tok"):
            if hasattr(self, attr):
                try:
                    obj = getattr(self, attr); del obj
                except Exception:
                    pass
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _run_plain(self, prompt: str, max_new: int) -> str:
        ids = self.llm_tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.llm.generate(
                ids,
                do_sample=False,
                max_new_tokens=max_new,
                min_new_tokens=10,
                pad_token_id=self.llm_tok.pad_token_id or self.llm_tok.eos_token_id,
                no_repeat_ngram_size=0,
            )
        new_tokens = out[0][ids.shape[1]:]
        return self.llm_tok.decode(new_tokens, skip_special_tokens=True).strip()

    def _prepare_datastore(self, docs: List[str]) -> Tuple[Tuple[List[str], torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        if getattr(self.cfg, "docs_per_datastore", None):
            docs = docs[: self.cfg.docs_per_datastore]

        bs = int(getattr(self.cfg, "retriever_batch_size", 16))
        max_len = int(getattr(self.cfg, "retriever_max_length", 256))

        doc_emb_chunks, xrag_emb_chunks = [], []
        self.retriever.eval(); self.llm.eval()

        for i in range(0, len(docs), bs):
            chunk = docs[i: i+bs]
            inp = self.ret_tok(
                chunk,
                max_length=max_len,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                chunk_doc_emb = self.retriever.get_doc_embedding(**inp)    # [B, D]
                chunk_xrag_emb = self.llm.projector(chunk_doc_emb)         # [B, D]
            doc_emb_chunks.append(chunk_doc_emb.detach().to("cpu"))
            xrag_emb_chunks.append(chunk_xrag_emb.detach().to("cpu"))
            del inp, chunk_doc_emb, chunk_xrag_emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        doc_emb = torch.cat(doc_emb_chunks, dim=0)     # CPU [N, D]
        xrag_emb = torch.cat(xrag_emb_chunks, dim=0)   # CPU [N, D]
        return (docs, doc_emb, xrag_emb), doc_emb, xrag_emb

    def _prepare_prompt(self, question: str) -> Tuple[str, torch.Tensor]:
        # Build the retriever-side query text.
        #
        # Historical default: only embed the prefix up to the first '.' to keep prompts short/stable.
        # For experiments, you can instead clip to a fixed number of characters by setting
        # `cfg.retriever_query_clip_chars` (int). If both are absent/None, it falls back to the
        # historical behaviour.
        q_txt = str(question)
        clip_chars = getattr(self.cfg, "retriever_query_clip_chars", None)
        if clip_chars is not None:
            q_txt = q_txt[: int(clip_chars)]
        else:
            q_txt = q_txt.split(".")[0]

        inp = self.ret_tok(q_txt, max_length=180, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q = self.retriever.get_query_embedding(**inp)
            q = self.llm.projector(q)
        tpl = (
            "<s>[INST] Refer to the background document and answer the question.\n\n"
            "Background: {document}\n\nQuestion: {question} [/INST] The answer is: "
        )
        prompt = tpl.format(document=XRAG_TOKEN, question=question)
        if self._xrag_token_id is not None:
            ids = self.llm_tok(prompt, return_tensors="pt").input_ids
            assert (ids == self._xrag_token_id).any().item(), "XRAG token id not found in the prompt."
        return prompt, q

    def _nearest_doc_embed(self, q_emb: torch.Tensor, datastore) -> torch.Tensor:
        docs, raw_cpu, xrag_cpu = datastore
        q_cpu = q_emb.detach().to("cpu").float()
        dist = torch.cdist(q_cpu, xrag_cpu.float(), p=2)
        idx = dist.argmin(dim=1)[0].item()
        return raw_cpu[idx].to(self.device)  # RAW SFR doc embedding

    def _run_xrag(self, prompt: str, raw_doc_emb: torch.Tensor, max_new: int) -> str:
        emb = raw_doc_emb
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)  # [1, D]
        ids = self.llm_tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.llm.generate(
                ids,
                do_sample=False,
                # temperature=0.1,
                max_new_tokens=max_new,
                min_new_tokens=10,
                pad_token_id=self.llm_tok.pad_token_id or self.llm_tok.eos_token_id,
                retrieval_embeds=emb.to(self.device),
                no_repeat_ngram_size=0,
            )
        new_tokens = out[0]
        return self.llm_tok.decode(new_tokens, skip_special_tokens=True).strip()

    def _perplexity(self, text: str) -> float:
        enc = self.llm_tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.llm(**enc, labels=enc.input_ids).loss
        return math.exp(loss.item())

    def surprise_phrases(self, story: str, gist: str, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
        punct_xlat = str.maketrans({c: None for c in string.punctuation if c != "'"})
        phrases: List[str] = []
        if _nlp is None:
            # Fallback (no spaCy model available): split into short clauses using
            # punctuation + basic conjunctions.
            for sent in re.split(r"(?<=[.!?])\\s+", story.strip()):
                if not sent.strip():
                    continue
                chunks = re.split(r"(?:,|;|\\band\\b|\\bbut\\b|\\bor\\b)\\s+", sent, flags=re.IGNORECASE)
                for ch in chunks:
                    ph = ch.translate(punct_xlat).strip()
                    if ph:
                        phrases.append(ph)
        else:
            doc = _nlp(story)
            for sent in doc.sents:
                cur: List[str] = []
                for tok in sent:
                    if tok.dep_ == "cc" or tok.text == ",":
                        if cur:
                            ph = " ".join(cur).translate(punct_xlat).strip()
                            if ph:
                                phrases.append(ph)
                            cur = []
                    else:
                        cur.append(tok.text)
                if cur:
                    ph = " ".join(cur).translate(punct_xlat).strip()
                    if ph:
                        phrases.append(ph)
        scored = []
        for ph in phrases:
            ppl = self._perplexity(f"{gist}\n\n{ph}")
            scored.append((ph, ppl))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored if top_k is None else scored[:top_k]


class Consolidator:
    """
    Reusable LoRA consolidator. The script using this should:
      • build tokenizer
      • call build_model()
      • build dataset via texts_to_ds()
      • (optionally) use RecallLogger for epoch logging
    """
    def __init__(self, cfg: Any, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def build_model(self):
        if getattr(self.cfg, "use_4bit", True):
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                quantization_config=qcfg,
                device_map="auto",
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model, torch_dtype=torch.bfloat16
            ).to(self.device)

        base.config.use_cache = False
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base)

        lora_cfg = LoraConfig(
            r=getattr(self.cfg, "lora_r", LORA_CONFIG.r),
            lora_alpha=getattr(self.cfg, "lora_alpha", LORA_CONFIG.alpha),
            lora_dropout=getattr(self.cfg, "lora_dropout", LORA_CONFIG.dropout),
            target_modules=getattr(self.cfg, "target_modules", LORA_CONFIG.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)

        model.requires_grad_(False)
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True

        if not hasattr(model, "hf_device_map"):
            model.to(self.device)
        model.train()
        return model

    def texts_to_ds(self, texts: List[str], tok: AutoTokenizer, max_len: int = 2048, chatml: bool = True) -> Dataset:
        def _chatml(txt: str) -> str:
            if "." in txt:
                first, rest = txt.split(".", 1)
            elif "\n" in txt:
                first, rest = txt.split("\n", 1)
            else:
                first, rest = txt, txt
            # Match each script's wrapping externally if needed.
            return (f"<s>[INST] {first.strip()}. What happened (in detail)? [/INST] "
                    f"{rest.strip()} </s>")

        wrapped = [(_chatml(t) if chatml else t) for t in texts]

        def _prep(batch):
            enc = tok(batch["text"], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            enc["labels"] = enc["input_ids"].clone()
            return enc

        ds = Dataset.from_dict({"text": wrapped})
        return ds.map(_prep, batched=True, remove_columns=["text"])

    class RecallLogger(TrainerCallback):
        def __init__(self, tok, prompt: str, original: str,
                     bg_texts: List[str], temps: List[float],
                     max_new_tokens: int, out_dir: Path):
            self.tok = tok
            self.prompt = prompt
            self.original = original
            self.temps = temps
            self.max_new_tokens = max_new_tokens
            self.out_dir = out_dir
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.bg_center = self._center(bg_texts)
            self.epoch_logs: List[Dict] = []

        def _center(self, texts: List[str]) -> np.ndarray:
            if not texts:
                return np.zeros((384,), dtype=np.float32)
            embs = self.embedder.encode(texts, show_progress_bar=False)
            return np.asarray(embs).mean(axis=0)

        def _embed(self, txt: str) -> np.ndarray:
            return self.embedder.encode(txt)

        def _cosdist(self, a: np.ndarray, b: np.ndarray) -> float:
            a = a / (np.linalg.norm(a) + 1e-8)
            b = b / (np.linalg.norm(b) + 1e-8)
            return float(1.0 - np.dot(a, b))

        @torch.no_grad()
        def _gen(self, model, do_sample: bool, temperature: float | None = None) -> str:
            ids = self.tok(self.prompt, return_tensors="pt").input_ids.to(model.device)
            out = model.generate(
                ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if (do_sample and temperature is not None and temperature > 0.0) else None,
                no_repeat_ngram_size=0,
                min_new_tokens=20,
                pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
            )
            new_tokens = out[0][ids.shape[1]:]
            return self.tok.decode(new_tokens, skip_special_tokens=True).strip()

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            greedy = self._gen(model, do_sample=False)
            logs = {"epoch": int(state.epoch), "greedy": greedy, "temps": {}, "metrics": {}}

            e_o = self._embed(self.original)
            e_g = self._embed(greedy)
            logs["metrics"]["greedy_cos_to_original"] = self._cosdist(e_g, e_o)
            logs["metrics"]["greedy_cos_to_bg_center"] = self._cosdist(e_g, self.bg_center)

            for t in self.temps:
                if t is None or t <= 0.0:
                    s = greedy; e_s = e_g
                else:
                    s = self._gen(model, do_sample=True, temperature=t)
                    e_s = self._embed(s)
                logs["temps"][str(t)] = s
                logs["metrics"][f"temp_{t}_cos_to_original"] = self._cosdist(e_s, e_o)
                logs["metrics"][f"temp_{t}_cos_to_bg_center"] = self._cosdist(e_s, self.bg_center)

            self.epoch_logs.append(logs)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            with open(self.out_dir / f"epoch_{int(state.epoch):02d}.json", "w") as fh:
                json.dump(logs, fh, indent=2)

        def save_all(self):
            if not self.epoch_logs:
                return
            with open(self.out_dir / "all_generations.json", "w") as fh:
                json.dump(self.epoch_logs, fh, indent=2)


# --------------------------- Raykov/ROC helpers ------------------------------
def get_stories(csv_path: str = "../../data/stories_train.csv") -> List[str]:
    df = pd.read_csv(csv_path)
    df["combined"] = df[[f"sentence{i}" for i in range(1, 6)]].astype(str).agg(" ".join, axis=1)
    return df["combined"].tolist()


def _random_sentence(
    pool: List[str],
    rng: random.Random | None = None,
    min_chars: int = 0,
    exclude: str | None = None,
) -> str:
    chooser = rng if rng is not None else random

    for _ in range(1000):
        s = chooser.choice(pool)
        if exclude is not None and s == exclude:
            continue
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip()]
        if not parts:
            parts = [s.strip()]
        sent = chooser.choice(parts).strip()
        if len(sent) >= min_chars:
            return sent

    candidates: List[str] = []
    for s in pool:
        if exclude is not None and s == exclude:
            continue
        candidates.extend(p.strip() for p in re.split(r"(?<=[.!?])\s+", s) if p.strip())
    if not candidates:
        return ""
    return max(candidates, key=len)


def prepare_roc_sets(n_typical: int, n_variants: int, rng_seed: int,
                     stories_csv: str = "../../data/stories_train.csv",
                     prompt_cue_chars: int = 100) -> Dict[str, List[str]]:

    stories = get_stories(stories_csv)
    rng = random.Random(rng_seed)
    rng.shuffle(stories)

    lengths = [len(s) for s in stories]
    mean_len = int(np.mean(lengths))
    print(f"[Prep] Mean length across ROC Stories: {mean_len}")

    typical = stories[: n_typical]
    lengthened, shortened = [], []
    skipped_updated_prompt_leaks = 0

    for s in stories[n_typical:]:
        if len(lengthened) >= n_variants and len(shortened) >= n_variants:
            break

        L = len(s)
        if L < mean_len:
            delta = mean_len - L
            if delta > 50:
                if L < prompt_cue_chars:
                    skipped_updated_prompt_leaks += 1
                    continue
                tail_len = delta - 1  # account for the separating space below
                filler = _random_sentence(stories, rng=rng, min_chars=tail_len, exclude=s)
                new_s = s + " " + filler[:tail_len]
                lengthened.append(new_s)
        elif L > mean_len:
            delta = L - mean_len
            if delta > 50:
                new_s = s[0:mean_len]
                shortened.append(new_s)

    if skipped_updated_prompt_leaks:
        print(
            "[Prep] Skipped "
            f"{skipped_updated_prompt_leaks} updated candidates shorter than "
            f"the {prompt_cue_chars}-character prompt cue."
        )

    # Append 'The end.' to match original script
    # stories = [s + " The end." for s in stories]
    # shortened = [s + " The end." for s in shortened]
    # lengthened = [s + " The end." for s in lengthened]

    return {
        "typical": stories[: n_typical],
        "incomplete": shortened[: n_variants],
        "updated": lengthened[: n_variants],
        "mean_len": mean_len,
    }


def prompts_from_sets(dsets: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Build prompts to exactly match:
      "<s>[INST] SEED. What happened? [/INST]"
    used in the Raykov pipeline.
    """
    prompts: List[Dict[str, Any]] = []
    for cat in ("typical", "incomplete", "updated"):
        for i, s in enumerate(dsets[cat]):
            seed = first_sentence(s)
            prompts.append({
                "id": f"{cat}_{i:04d}",
                "category": cat,
                "input_text": s,
                "prompt_str": f"<s>[INST] {seed}. What happened? [/INST]",
            })
    return prompts
