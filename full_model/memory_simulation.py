from __future__ import annotations
import gc
import inspect
import json
import math
import os
import pickle
import random
import string
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import sem
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainerCallback, TrainingArguments)
import spacy

# Import global LoRA configuration
from lora_config import ACTIVE as LORA_CONFIG
nlp = spacy.load("en_core_web_sm")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def _data_file(name: str) -> Path:
    """Resolve data files from the current run directory or this script's data directory."""
    cwd_path = Path("data") / name
    if cwd_path.exists():
        return cwd_path
    return DEFAULT_DATA_DIR / name


def make_trainer(*, model, args, train_dataset, tok, eval_dataset=None, callbacks=None):
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks

    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tok
    return Trainer(**trainer_kwargs)

# Sentence embedder for semantic similarity (used in semantic memory test)
_semantic_embedder = None
def get_semantic_embedder():
    global _semantic_embedder
    if _semantic_embedder is None:
        _semantic_embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_embedder



def _load_questions_jsonl(path: Path) -> Dict[str, Dict]:
    """
    Load a JSONL file of semantic questions.

    Each line should be a JSON object containing:
      - story_start (str): a prefix that matches story.startswith(story_start)
      - question (str)
      - correct (str)
      - wrong (list[str]) with >=2 entries
    Optional:
      - context (str): snippet used in the prompt
      - entity_type (str)
    """
    questions: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            story_start = obj.get("story_start")
            if not story_start or not isinstance(story_start, str):
                raise ValueError(f"{path}:{line_no}: missing/invalid story_start")
            if "question" not in obj or "correct" not in obj or "wrong" not in obj:
                raise ValueError(f"{path}:{line_no}: missing required fields (question/correct/wrong)")
            if not isinstance(obj["wrong"], list) or len(obj["wrong"]) < 2:
                raise ValueError(f"{path}:{line_no}: wrong must be a list with >=2 entries")
            questions[story_start] = obj
    return questions


def generate_story_question(story: str, questions_map: Dict[str, Dict] | None = None) -> Dict | None:
    """
    Get a semantic question for a story.
    Returns None if no matching story_start exists.
    """
    if not questions_map:
        return None

    first_sentence = story.split(".")[0].strip() + "."
    
    for story_start, qa in questions_map.items():
        if story.startswith(story_start):
            return {
                "question": qa["question"],
                "correct": qa["correct"],
                "wrong": qa["wrong"],
                "entity_type": qa.get("entity_type", "SEMANTIC_QA"),
                "context": qa.get("context", first_sentence),
            }
    
    return None


def get_stories_with_questions(all_stories: List[str], questions_map: Dict[str, Dict] | None = None) -> List[Tuple[str, Dict]]:
    """
    Filter stories to only those with semantic questions.
    Returns list of (story, question_dict) tuples.
    """
    if not questions_map:
        return []

    result: List[Tuple[str, Dict]] = []
    for story in all_stories:
        q = generate_story_question(story, questions_map=questions_map)
        if q is not None:
            result.append((story, q))
    return result


def _compute_answer_nll(model, tok, prompt: str, answer: str, device) -> float:
    """
    Compute the negative log-likelihood of an answer given a prompt.
    Lower NLL = model thinks answer is more likely.
    """
    # Concatenate prompt and answer
    full_text = prompt + " " + answer
    inputs = tok(full_text, return_tensors="pt").to(device)
    prompt_len = len(tok(prompt, return_tensors="pt").input_ids[0])
    
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        # Get per-token loss
        logits = outputs.logits
        
        # Shift for causal LM: predict token i from token i-1
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        # Compute cross-entropy loss per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Only consider loss on answer tokens (after prompt)
        answer_losses = losses[prompt_len-1:]  # -1 because of shift
        
        # Average NLL over answer tokens
        nll = answer_losses.mean().item()
    
    return nll


def semantic_memory_test(
    model, 
    tok, 
    stories: List[str],
    questions: List[Dict],
    device,
    max_new_tokens: int = 30
) -> Dict:
    """
    Test semantic memory by asking factual questions about stories.
    
    Args:
        model: The language model
        tok: Tokenizer
        stories: List of original stories (for context in prompt)
        questions: List of question dicts from generate_story_question
        device: torch device
        max_new_tokens: Max tokens for answer
    
    Returns:
        Dict with accuracy, per-question results, and model answers
    """
    embedder = get_semantic_embedder()
    
    results = []
    correct_count = 0
    
    model.eval()
    
    for i, (story, q) in enumerate(zip(stories, questions)):
        # Format question prompt - include story context
        context = q.get("context", story.split(".")[0].strip() + ".")
        prompt = f"<s>[INST] Remember the story in which '{context}' {q['question']} Answer with a word or short phrase. [/INST]"
        
        # Generate answer
        inputs = tok(prompt, return_tensors="pt")
        ids = inputs["input_ids"].to(device)
        attn = inputs.get("attention_mask")
        gen_kwargs = {}
        if attn is not None:
            gen_kwargs["attention_mask"] = attn.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                **gen_kwargs,
            )
        
        new_tokens = out[0][ids.shape[1]:]
        model_answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Evaluate using embedding similarity
        emb_answer = embedder.encode(model_answer)
        emb_correct = embedder.encode(q["correct"])
        emb_wrong = [embedder.encode(w) for w in q["wrong"]]
        
        sim_correct = float(np.dot(emb_answer, emb_correct) / 
                           (np.linalg.norm(emb_answer) * np.linalg.norm(emb_correct) + 1e-8))
        sim_wrong = [float(np.dot(emb_answer, ew) / 
                          (np.linalg.norm(emb_answer) * np.linalg.norm(ew) + 1e-8)) 
                    for ew in emb_wrong]
        max_sim_wrong = max(sim_wrong)
        
        is_correct_embedding = sim_correct > max_sim_wrong
        
        # String-based scoring: check if correct answer appears in model answer
        model_answer_lower = model_answer.lower()
        correct_lower = q["correct"].lower()
        wrong_lower = [w.lower() for w in q["wrong"]]
        
        contains_correct = correct_lower in model_answer_lower
        contains_wrong = any(w in model_answer_lower for w in wrong_lower)
        is_correct_string = contains_correct and not contains_wrong
        
        # Perplexity-based scoring: compare NLL of correct vs wrong answers
        nll_correct = _compute_answer_nll(model, tok, prompt, q["correct"], device)
        nll_wrong = [_compute_answer_nll(model, tok, prompt, w, device) for w in q["wrong"]]
        min_nll_wrong = min(nll_wrong)
        is_correct_perplexity = nll_correct < min_nll_wrong
        
        # Use embedding-based as primary, but track all three
        is_correct = is_correct_embedding
        if is_correct:
            correct_count += 1
        
        results.append({
            "story_idx": i,
            "question": q["question"],
            "correct_answer": q["correct"],
            "wrong_answers": q["wrong"],
            "model_answer": model_answer,
            "sim_correct": sim_correct,
            "max_sim_wrong": max_sim_wrong,
            "is_correct": is_correct,  # embedding-based
            "is_correct_embedding": is_correct_embedding,
            "is_correct_string": is_correct_string,
            "contains_correct": contains_correct,
            "contains_wrong": contains_wrong,
            "nll_correct": nll_correct,
            "min_nll_wrong": min_nll_wrong,
            "nll_wrong": nll_wrong,
            "is_correct_perplexity": is_correct_perplexity,
        })
    
    accuracy = correct_count / len(stories) if stories else 0.0
    string_correct_count = sum(1 for r in results if r["is_correct_string"])
    string_accuracy = string_correct_count / len(stories) if stories else 0.0
    ppl_correct_count = sum(1 for r in results if r["is_correct_perplexity"])
    ppl_accuracy = ppl_correct_count / len(stories) if stories else 0.0
    
    return {
        "accuracy": accuracy,  # embedding-based (primary)
        "accuracy_string": string_accuracy,  # string-based
        "accuracy_perplexity": ppl_accuracy,  # perplexity-based
        "correct_count": correct_count,
        "correct_count_string": string_correct_count,
        "correct_count_perplexity": ppl_correct_count,
        "total": len(stories),
        "per_question": results,
    }


sys.path.append("./xRAG")
from src.language_modeling.lm_utils import XRAG_TOKEN, get_retrieval_embeds
from src.model import SFR, XMistralForCausalLM


@dataclass
class Config:

    llm_name: str = "Hannibal046/xrag-7b"
    retriever_name: str = "Salesforce/SFR-Embedding-Mistral"
    consolidation_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    use_mps: bool = False          
    use_8bit: bool = False          
    use_4bit: bool = True
    use_cpu_offload: bool = False

    num_epochs: int = 10
    batch_size: int = 1
    learning_rate: float = 5e-5
    print_steps: int = 10
    max_new_tokens: int = 100
    oversample: int = 1

    # LoRA settings (defaults from global lora_config.py)
    lora_r: int = field(default_factory=lambda: LORA_CONFIG.r)
    lora_alpha: int = field(default_factory=lambda: LORA_CONFIG.alpha)
    lora_dropout: float = field(default_factory=lambda: LORA_CONFIG.dropout)
    target_modules: List[str] = field(default_factory=lambda: LORA_CONFIG.target_modules.copy())

    # number of stories to compress / encode / consolidate
    num_stories: int = 500 #200
    detail_levels: List[int] = field(default_factory=lambda: [0, 1, 3])
    # Which encoded variant to use during consolidation training:
    #  - integer detail level (e.g. "0", "1", "3")
    #  - "full" to consolidate on the verbatim original stories (upper bound)
    consolidation_encoding: str = "0"
    seed: int = 123

    log_recall_every_n_epochs: int = 1

    # number of new stories in each phase of forgetting
    memory_set_size: int = 50
    epochs_per_set: int = 20
    num_forgetting_phases: int = 8 

    output_dir: str = "output"
    semantic_questions_file: str | None = None
    story_order_file: str | None = None

    def __post_init__(self):
        """House‑keeping – make output folders."""

        for sub in ("plots", "models", "data"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)

class MemorySimulator:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._auto_device()
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.semantic_questions: Dict[str, Dict] = {}
        self._maybe_load_external_semantic_questions()

        self.results: Dict[str, Dict] = {
            "encoding": {}, "consolidation": {},
            "forgetting": {}, "hippocorpus_analysis": {}
        }

        self.llm = None; self.llm_tok = None; self.retriever = None; self.ret_tok = None

    def _maybe_load_external_semantic_questions(self):
        candidates: list[Path] = []
        if self.cfg.semantic_questions_file:
            candidates.append(Path(self.cfg.semantic_questions_file))
        # Common convenience locations when running from the repo root or full_model/.
        candidates.append(_data_file("semantic_questions.jsonl"))
        candidates.append(Path(self.cfg.output_dir) / "data" / "semantic_questions.jsonl")

        for p in candidates:
            if not p.exists():
                continue
            loaded = _load_questions_jsonl(p)
            self.semantic_questions = loaded
            print(f"[SemanticQ] Loaded {len(loaded)} questions from {p}")
            return
        print("[SemanticQ] No semantic_questions.jsonl found; semantic accuracy will be empty.")

    def _auto_device(self):
        if self.cfg.use_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _lazy_load_models(self):
        """Load xRAG LLM + retriever only when we really need them."""
        if self.llm is not None:
            return  # already loaded
        print("Loading xRAG models…")

        if self.cfg.use_4bit and self.device.type == "mps":
            self.llm = XMistralForCausalLM.from_pretrained(
                self.cfg.llm_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=False,
            ).to(self.device).eval()
        else:
            self.llm = XMistralForCausalLM.from_pretrained(
                self.cfg.llm_name, torch_dtype=torch.bfloat16
            ).to(self.device).eval()

        self.llm_tok = AutoTokenizer.from_pretrained(
            self.cfg.llm_name, add_eos_token=False, use_fast=False, padding_side="left"
        )
        self.llm.set_xrag_token_id(self.llm_tok.convert_tokens_to_ids(XRAG_TOKEN))

        # ── Retriever ---------------------------------------------------------
        self.retriever = SFR.from_pretrained(self.cfg.retriever_name, torch_dtype=torch.bfloat16)
        self.retriever = self.retriever.to(self.device).eval()
        self.ret_tok = AutoTokenizer.from_pretrained(self.cfg.retriever_name)

        print("xRAG components ready")

    def load_data(self) -> Tuple[List[str], pd.DataFrame]:

        df = pd.read_csv(_data_file("stories_train.csv"))
        df["combined"] = (
            df[[f"sentence{i}" for i in range(1, 6)]]
            .astype(str)
            .agg(" ".join, axis=1)
        )

        story_order_path = Path(self.cfg.story_order_file) if self.cfg.story_order_file else _data_file("selected_story_ids.csv")
        story_order_df = pd.read_csv(story_order_path)
        if "storyid" not in story_order_df.columns:
            raise ValueError(f"{story_order_path} must contain a 'storyid' column")

        ordered_ids = story_order_df["storyid"].dropna().astype(str).tolist()
        # To intentionally generate a new fixed story order, uncomment this line,
        # inspect the resulting run, and save the IDs back to selected_story_ids.csv.
        # ordered_ids = df["storyid"].sample(frac=1, random_state=self.cfg.seed).astype(str).tolist()

        story_by_id = dict(zip(df["storyid"].astype(str), df["combined"]))
        prioritized: List[str] = []
        seen_ids: set[str] = set()
        missing_ids: list[str] = []
        duplicate_ids: list[str] = []

        for story_id in ordered_ids:
            if story_id in seen_ids:
                duplicate_ids.append(story_id)
                continue
            seen_ids.add(story_id)
            story = story_by_id.get(story_id)
            if story is None:
                missing_ids.append(story_id)
                continue
            prioritized.append(story)

        if missing_ids:
            raise ValueError(f"{story_order_path} contains {len(missing_ids)} unknown story IDs")
        if duplicate_ids:
            print(f"[Data] Ignored {len(duplicate_ids)} duplicate story IDs in {story_order_path}")

        remaining = [
            story
            for story_id, story in zip(df["storyid"].astype(str), df["combined"])
            if story_id not in seen_ids
        ]
        if remaining:
            print(f"[Data] Appending {len(remaining)} stories not listed in {story_order_path}")
            prioritized.extend(remaining)

        print(f"[Data] Loaded fixed story order from {story_order_path} ({len(prioritized)} stories)")
    
        # primary slice for encoding / consolidation
        stories_subset = prioritized[: self.cfg.num_stories]
    
        # keep the whole corpus for future forgetting
        self._full_story_pool = prioritized
    
        hippo = pd.read_csv(_data_file("hippoCorpusV2.csv"))
        hippo = hippo[["recAgnPairId", "memType", "story"]].dropna(subset=["story"])
    
        return stories_subset, hippo


    def _release_xrag(self):
        """
        Free *all* xRAG components (LLM, tokenizer and retriever) so the
        consolidation model can load without running out of GPU memory.
        """
        for attr in ("llm", "llm_tok", "retriever", "ret_tok"):
            if getattr(self, attr, None) is not None:
                delattr(self, attr)
        # flush PyTorch / CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _run_plain(self, prompt: str) -> str:
        """
        Generate with **no retrieval embedding** (used for ‘imagined’ and
        ‘full-detail’ prompts)."""
        inputs = self.llm_tok(prompt, return_tensors="pt")
        ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask")
        gen_kwargs = {}
        if attn is not None:
            gen_kwargs["attention_mask"] = attn.to(self.device)
        with torch.no_grad():
            out = self.llm.generate(
                ids,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.llm_tok.pad_token_id,
                **gen_kwargs,
            )
    
        gen_ids = out[0][ids.shape[1]:]          # keep only the continuation
        return self.llm_tok.decode(gen_ids, skip_special_tokens=True).strip()



    def simulate_encoding(self, stories: List[str]) -> Dict:

        out_file = Path(self.cfg.output_dir, "data", "recalled_stories.pkl")

        # If cached file exists, load and return it (but sanity-check it first)
        if out_file.exists():
            print(f"Found cached encoding data at {out_file}, loading it…")
            with open(out_file, "rb") as f:
                cached = pickle.load(f)

            try:
                recalled = cached.get("recalled_stories", {})
                if not isinstance(recalled, dict) or not recalled:
                    raise ValueError("missing recalled_stories")

                if 0 in recalled:
                    sample_level = 0
                else:
                    sample_level = next((k for k in recalled.keys() if isinstance(k, int)), None)
                if sample_level is None:
                    sample_level = "full" if "full" in recalled else next(iter(recalled.keys()))

                cached_texts = recalled.get(sample_level, [])
                if not isinstance(cached_texts, list) or not cached_texts:
                    raise ValueError("empty recalled_stories at sample level")

                def first_sent(txt: str) -> str:
                    return txt.split(".", 1)[0].strip().lower()

                expected = {first_sent(s) for s in stories}
                cached_set = {first_sent(s) for s in cached_texts}
                overlap = len(expected & cached_set) / max(1, len(expected))

                missing_levels = [lvl for lvl in self.cfg.detail_levels if lvl not in recalled]

                if len(cached_texts) != len(stories) or overlap < 0.95 or missing_levels:
                    print(
                        "[Cache] Encoding cache does not match the current story set/config "
                        f"(len={len(cached_texts)} vs {len(stories)}, overlap={overlap:.2f}, missing_levels={missing_levels})."
                    )
                    print("[Cache] Recomputing encoding. (Use --force to clear caches explicitly.)")
                else:
                    self.results["encoding"] = cached
                    self.original_stories = stories
                    return self.results["encoding"]
            except Exception as err:
                print(f"[Cache] Could not validate encoding cache ({err}); recomputing encoding.")

        self._lazy_load_models()
        self.original_stories = stories    

        # single retrieval datastore for all questions
        datastore, *_ = self._prepare_datastore(stories)

        levels        = self.cfg.detail_levels              # [0, 1, 5]
        recalled      = {lvl: [] for lvl in levels}
        mem_sizes     = {lvl: [] for lvl in levels}
        details_used  = {lvl: [] for lvl in levels}

        # extra categories
        recalled["imagined"], mem_sizes["imagined"] = [], []
        recalled["full"],     mem_sizes["full"]     = [], []

        for doc in tqdm(stories, desc="Encoding stories"):
            first_sent = doc.split(".", 1)[0]

            # 1.  IMAGINED (no context)
            prompt = f"<s>[INST] {first_sent}. What happened (in detail)? [/INST]"
            gen    = self._run_plain(prompt)
            recalled["imagined"].append(gen)
            mem_sizes["imagined"].append(0)

            # GIST-ONLY  (detail-level 0)
            q0      = f"{first_sent}. What happened (in detail)?"
            p0, emb = self._prepare_prompt(q0)
            ans0    = self._run_xrag(p0, self._nearest_doc_embed(emb, datastore))
            recalled[0].append(ans0)
            mem_sizes[0].append(1)

            # gather “surprising” phrases for richer variants
            surprising = self._surprise_phrases(doc, ans0)  # [(phrase, ppl), …]

            for n in levels[1:]:
                subset = [ph for ph, _ in surprising[:n]]

                if subset:
                    qn  = (f"{first_sent}. What happened (in detail)? "
                           f"Other details to include: {', '.join(subset)}.")
                    pn, emb_n = self._prepare_prompt(qn)
                    ansn = self._run_xrag(pn,
                                          self._nearest_doc_embed(emb_n, datastore))
                else:
                    ansn = ans0

                recalled[n].append(ansn)

                detail_tok_len = sum(
                    len(self.llm_tok.encode(ph, add_special_tokens=False))
                    for ph in subset
                )

                mem_sizes[n].append(1 + detail_tok_len)     
                details_used[n].append(subset)

            # FULL-DETAIL (verbatim text)
            recalled["full"].append(doc)
            tok_len = len(self.llm_tok(doc, add_special_tokens=False)["input_ids"])
            mem_sizes["full"].append(tok_len)

        self.results["encoding"] = {
            "recalled_stories": recalled,
            "memory_sizes"    : mem_sizes,
            "details"         : details_used,
        }
        out_file = Path(self.cfg.output_dir, "data", "recalled_stories.pkl")
        with open(out_file, "wb") as f:
            pickle.dump(self.results["encoding"], f)

        return self.results["encoding"]

    
    def simulate_consolidation(self, encoded: List[str], originals: List[str]) -> Dict:
        """
        Fine-tune the consolidation model on *encoded* stories.  
        At the end of every epoch (or every N epochs) we log:
            • fresh recall generations
            • cosine distance to ORIGINAL stories
            • cosine distance to ENCODED stories
        """
    
        # def chatml(txt: str) -> str:
        #     first, rest = (txt.split("\n", 1)
        #                    if "\n" in txt else txt.split(".", 1))
        #     return (f"<s>[INST] {first.strip()} What happened (in detail)? [/INST] "
        #             f"{rest.strip()} </s>")

        def chatml(txt: str) -> str:
            if "." in txt:
                first, rest = txt.split(".", 1)
            elif "\n" in txt:
                first, rest = txt.split("\n", 1)
            else:
                print("No newlines or full stops in:", txt)
                first, rest = txt, txt
            return (f"<s>[INST] {first.strip()} What happened (in detail)? [/INST] "
                    f"{rest.strip()} </s>")

    
        train_texts = [chatml(t) for t in encoded] * self.cfg.oversample
        eval_texts  = [chatml(t) for t in originals]
    
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
    
        def _prep(batch):
            enc = tok(batch["text"], return_tensors="pt",
                      padding=True, truncation=True)
            enc["labels"] = enc["input_ids"].clone()
            return enc
    
        train_ds = (Dataset.from_dict({"text": train_texts})
                    .map(_prep, batched=True, remove_columns=["text"]))
        eval_ds  = (Dataset.from_dict({"text": eval_texts})
                    .map(_prep, batched=True, remove_columns=["text"]))
    
        try:
            if self.cfg.use_4bit:
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
                raise ValueError
        except Exception as err:
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model, torch_dtype=torch.bfloat16
            ).to(self.device)
    
        base.gradient_checkpointing_enable()
        base = prepare_model_for_kbit_training(base)
    
        lora_cfg = LoraConfig(
            r              = self.cfg.lora_r,
            lora_alpha     = self.cfg.lora_alpha,
            lora_dropout   = self.cfg.lora_dropout,
            target_modules = self.cfg.target_modules,
            bias           = "none",
            task_type      = "CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
    
        # Generate semantic memory questions for each story (once, before training)
        print("Finding stories with semantic questions for semantic memory test...")
        semantic_test_pairs = get_stories_with_questions(originals, questions_map=self.semantic_questions)
        semantic_stories = [s for s, q in semantic_test_pairs]
        semantic_questions = [q for s, q in semantic_test_pairs]
        print(f"  Found {len(semantic_stories)} stories with semantic questions")
        
        # Test baseline (epoch 0) semantic memory BEFORE any training
        print("Testing baseline semantic memory (epoch 0, before training)...")
        model.eval()
        if not semantic_stories:
            print("  WARNING: 0 semantic-question stories found; semantic accuracy will be uninformative.")
            print("  (Common cause: story selection changed but an old encoding cache was reused. Try --force.)")
            baseline_sem_result = {
                "accuracy": 0.0,
                "accuracy_string": 0.0,
                "accuracy_perplexity": 0.0,
                "correct_count": 0,
                "correct_count_string": 0,
                "correct_count_perplexity": 0,
                "total": 0,
                "per_question": [],
            }
        else:
            baseline_sem_result = semantic_memory_test(
                model, tok, semantic_stories, semantic_questions,
                self.device, max_new_tokens=30
            )
        print(
            "  Baseline semantic accuracy (embed/string/ppl): "
            f"{baseline_sem_result['accuracy']:.1%} / {baseline_sem_result['accuracy_string']:.1%} / "
            f"{baseline_sem_result['accuracy_perplexity']:.1%} "
            f"({len(semantic_stories)} questions)"
        )
        
        class RecallTracker(TrainerCallback):
            def __init__(self, outer, sem_stories, sem_questions):
                self.outer = outer
                self.sem_stories = sem_stories
                self.sem_questions = sem_questions
                self.epoch_recalls, self.dist_orig, self.dist_enc = [], [], []
                self.semantic_accuracy = []  # Track semantic memory per epoch
                self.semantic_results = []   # Detailed results per epoch
    
            def on_epoch_end(self, args, state, control, **_):
                if ((state.epoch + 1) %
                        self.outer.cfg.log_recall_every_n_epochs) != 0:
                    return
    
                rec, d_o, d_e = [], [], []
                for orig, enc in zip(originals, encoded):
                    prompt = (f"<s>[INST] {orig.split('.')[0]}."
                              f" What happened (in detail)? [/INST]")
                    inputs = tok(prompt, return_tensors="pt")
                    ids = inputs["input_ids"].to(self.outer.device)
                    attn = inputs.get("attention_mask")
                    gen_kwargs = {}
                    if attn is not None:
                        gen_kwargs["attention_mask"] = attn.to(self.outer.device)
                    with torch.no_grad():
                        out = model.generate(
                            ids,
                            max_new_tokens=self.outer.cfg.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tok.pad_token_id,
                            **gen_kwargs,
                        )

                    # ids           -> tensor containing the prompt tokens
                    # out[0]        -> tensor containing  prompt  +  generated tokens
                    new_tokens = out[0][ids.shape[1]:]
                    gen        = tok.decode(new_tokens,
                                            skip_special_tokens=True).strip()

                    gen = orig.split('.')[0] + '. ' + gen
                    rec.append(gen)
    
                    v_gen = self.outer.embedder.encode(gen)
                    d_o.append(cos_dist(v_gen, self.outer.embedder.encode(orig)))
                    d_e.append(cos_dist(v_gen, self.outer.embedder.encode(enc)))
    
                self.epoch_recalls.append(rec)
                self.dist_orig.append(d_o)
                self.dist_enc.append(d_e)
                
                # Semantic memory test (only on stories with configured semantic questions)
                print(f"  Testing semantic memory at epoch {int(state.epoch)}...")
                sem_result = semantic_memory_test(
                    model, tok, self.sem_stories, self.sem_questions,
                    self.outer.device, max_new_tokens=30
                )
                self.semantic_accuracy.append(sem_result["accuracy"])
                self.semantic_results.append(sem_result)
                print(
                    "  Semantic accuracy (embed/string/ppl): "
                    f"{sem_result['accuracy']:.1%} / {sem_result['accuracy_string']:.1%} / "
                    f"{sem_result['accuracy_perplexity']:.1%} "
                    f"({len(self.sem_stories)} questions)"
                )
    
        tracker = RecallTracker(self, semantic_stories, semantic_questions)
    
        t_args = TrainingArguments(
            output_dir       = Path(self.cfg.output_dir, "models", "consolidation"),
            seed             = self.cfg.seed,
            num_train_epochs = self.cfg.num_epochs,
            per_device_train_batch_size = self.cfg.batch_size,
            learning_rate    = self.cfg.learning_rate,
            fp16             = False,
            save_strategy    = "epoch",
            logging_steps    = self.cfg.print_steps,
            #evaluation_strategy = "epoch",
            report_to        = [],
            dataloader_pin_memory=False,
        )
        trainer = make_trainer(
            model=model,
            args=t_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tok=tok,
            callbacks=[tracker],
        )
    
        if not hasattr(model, "hf_device_map"):
            model.to(self.device)
    
        print("→ Consolidation fine-tuning…")
        trainer.train()
    
        save_dir = Path(self.cfg.output_dir, "models", "consolidation")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

        # Prepend baseline (epoch 0) to the per-epoch results
        all_semantic_accuracy = [baseline_sem_result["accuracy"]] + tracker.semantic_accuracy
        all_semantic_results = [baseline_sem_result] + tracker.semantic_results
        
        self.results["consolidation"] = {
            "epoch_recalls"   : tracker.epoch_recalls,
            "epoch_dist_orig" : tracker.dist_orig,
            "epoch_dist_enc"  : tracker.dist_enc,
            "semantic_accuracy": all_semantic_accuracy,  # Includes epoch 0
            "semantic_results": all_semantic_results,    # Includes epoch 0
            "semantic_stories": semantic_stories,
            "semantic_questions": semantic_questions,
            "baseline_semantic": baseline_sem_result,    # Also store separately for clarity
        }
        with open(Path(self.cfg.output_dir, "data", "consolidation_recall.pkl"), "wb") as f:
            pickle.dump(self.results["consolidation"], f)
        
        # Plot semantic memory over consolidation (includes epoch 0)
        self._plot_semantic_consolidation(all_semantic_accuracy)
    
        return self.results["consolidation"]
    
    def simulate_chunked_forgetting(self, first_set: List[str]) -> Dict:
        """
        After consolidation, run up to `num_forgetting_phases` forgetting episodes.
        At every episode:
          1.  Sample `memory_set_size` **new** stories (no repeats).
          2.  Fine-tune on that chunk.
          3.  Measure recall of the *original* `first_set`.
        """
    
        from peft import PeftModel
        from datasets import Dataset
    
        # rebuild consolidated model
        adapter_dir = Path(self.cfg.output_dir, "models", "consolidation")
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)

        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
    
        try:                                   
            if self.cfg.use_4bit:
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
                raise ValueError
        except Exception:
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model, torch_dtype=torch.bfloat16
            )
    
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        # Only move to device if not using device_map (4-bit uses device_map="auto")
        if not hasattr(model, "hf_device_map"):
            model = model.to(self.device)
        model.eval()
    
        cos   = lambda a, b: cos_dist(a, b)
        embed = self.embedder.encode
        
        print("Finding stories with semantic questions for forgetting semantic test...")
        semantic_pairs = get_stories_with_questions(first_set, questions_map=self.semantic_questions)
        forgetting_sem_stories = [s for s, q in semantic_pairs]
        forgetting_sem_questions = [q for s, q in semantic_pairs]
        print(f"  Found {len(forgetting_sem_stories)} stories with semantic questions")
    
        def _recall_first():
            gens, dists = [], []
            for s in first_set:
                prompt = f"<s>[INST] {s.split('.')[0]}. What happened (in detail)? [/INST]"
                inputs = tok(prompt, return_tensors="pt")
                ids = inputs["input_ids"].to(self.device)
                attn = inputs.get("attention_mask")
                gen_kwargs = {}
                if attn is not None:
                    gen_kwargs["attention_mask"] = attn.to(self.device)
                with torch.no_grad():
                    out = model.generate(
                        ids,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        **gen_kwargs,
                    )

                new_tokens = out[0][ids.shape[1]:]
                gen = tok.decode(new_tokens, skip_special_tokens=True).strip()

                gen = s.split('.')[0] + '. ' + gen
                gens.append(gen)
                dists.append(cos(embed(gen), embed(s)))
            return gens, dists
        
        def _test_semantic():
            """Test semantic memory on stories with configured semantic questions."""
            result = semantic_memory_test(
                model, tok, forgetting_sem_stories, forgetting_sem_questions,
                self.device, max_new_tokens=30
            )
            return result["accuracy"], result
    
        # helper: fine-tune on one new chunk
        def _finetune(chunk: list[str]) -> float:
            """
            • Adds labels so the model returns a loss.
            • Re-enables gradients only on the LoRA adapter weights.
            • Uses a sane max_length (2048) to avoid tokenizer overflow.
            • Returns the latest reported training loss or NaN if none logged.
            """
            MAX_SEQ_LEN = 2048

            def to_features(batch):
                enc = tok(
                    batch["text"],
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN,
                    return_tensors="pt",
                )
                enc["labels"] = enc["input_ids"].clone()   # full-sequence LM loss
                return enc

            ds = (Dataset.from_dict({"text": chunk})
                  .map(to_features, batched=True, remove_columns=["text"]))

            model.requires_grad_(False)                   # freeze everything
            for n, p in model.named_parameters():
                if "lora_" in n:                          # LoRA adapter tensors
                    p.requires_grad = True
            model.train()                                 # enable training mode

            t_args = TrainingArguments(
                output_dir       = Path(self.cfg.output_dir, "models", "forget_tmp"),
                num_train_epochs = self.cfg.epochs_per_set,
                per_device_train_batch_size = self.cfg.batch_size,
                learning_rate    = self.cfg.learning_rate,
                fp16             = False,
                logging_steps    = self.cfg.print_steps,
                report_to        = [],
                label_names      = ["labels"],
            )

            trainer = Trainer(model=model, args=t_args, train_dataset=ds)
            trainer.train()

            loss_entries = (
                entry["loss"]
                for entry in reversed(trainer.state.log_history)
                if "loss" in entry
            )
            last_loss = next(loss_entries, None)
            return float(last_loss) if last_loss is not None else float("nan")


        pool = [s for s in getattr(self, "_full_story_pool") if s not in first_set]
    
        losses, distances, recalls = [], [], []
        semantic_accuracies, semantic_results = [], []
    
        # baseline (episode 0)
        g0, d0 = _recall_first()
        recalls.append(g0); distances.append(d0)
        
        print("Testing semantic memory at baseline...")
        sem_acc_0, sem_res_0 = _test_semantic()
        semantic_accuracies.append(sem_acc_0)
        semantic_results.append(sem_res_0)
        print(
            "  Semantic accuracy (embed/string/ppl): "
            f"{sem_res_0['accuracy']:.1%} / {sem_res_0['accuracy_string']:.1%} / "
            f"{sem_res_0['accuracy_perplexity']:.1%}"
        )
    
        for epi in range(1, self.cfg.num_forgetting_phases + 1):
    
            if len(pool) < self.cfg.memory_set_size:
                print(f"⚠︎  Pool exhausted after {epi-1} phases."); break
    
            chunk = random.sample(pool, k=self.cfg.memory_set_size)
            for s in chunk: pool.remove(s)          # ensure no repeats
    
            print(f"Episode {epi}: training on {len(chunk)} new stories …")
            losses.append(_finetune(chunk))
    
            g, d = _recall_first()
            recalls.append(g); distances.append(d)
            
            # Test semantic memory after forgetting
            print(f"Testing semantic memory after episode {epi}...")
            sem_acc, sem_res = _test_semantic()
            semantic_accuracies.append(sem_acc)
            semantic_results.append(sem_res)
            print(
                "  Semantic accuracy (embed/string/ppl): "
                f"{sem_res['accuracy']:.1%} / {sem_res['accuracy_string']:.1%} / "
                f"{sem_res['accuracy_perplexity']:.1%}"
            )
    
        out = {
            "losses": losses, 
            "distances": distances, 
            "recalls": recalls,
            "semantic_accuracies": semantic_accuracies,
            "semantic_results": semantic_results,
            "semantic_stories": forgetting_sem_stories,
            "semantic_questions": forgetting_sem_questions,
        }
        data_dir = Path(self.cfg.output_dir, "data"); data_dir.mkdir(exist_ok=True, parents=True)
        pickle.dump(out, open(data_dir / "forgetting_multi.pkl", "wb"))
    
        self.results["forgetting"] = out
        
        # Plot semantic memory over forgetting
        self._plot_semantic_forgetting(semantic_accuracies)
        
        return out

    def _prepare_datastore(self, docs: List[str]):
        inp = self.ret_tok(docs, max_length=500, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            doc_emb = self.retriever.get_doc_embedding(**inp)
            xrag_emb = self.llm.projector(doc_emb)
        return (docs, doc_emb, xrag_emb), doc_emb, xrag_emb

    def _prepare_prompt(self, question: str):
        inp = self.ret_tok(question, max_length=180, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q = self.retriever.get_query_embedding(**inp)
            q = self.llm.projector(q)
        tpl = """[INST] Refer to the background document and answer the question.\n\nBackground: {document}\n\nQuestion: {question} [/INST] The answer is:"""
        return tpl.format(document=XRAG_TOKEN, question=question), q

    def _nearest_doc_embed(self, q_emb, datastore):
        docs, raw, xrag = datastore
        dist = torch.cdist(q_emb.float(), xrag.float(), p=2)
        idx = dist.argmin(dim=1)[0].item()
        return raw[idx]

    def _run_xrag(self, prompt: str, emb):
        inputs = self.llm_tok(prompt, return_tensors="pt")
        ids = inputs["input_ids"].to(self.device)
        attn = inputs.get("attention_mask")
        gen_kwargs = {}
        if attn is not None:
            gen_kwargs["attention_mask"] = attn.to(self.device)
        out = self.llm.generate(
            ids,
            do_sample=False,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.llm_tok.pad_token_id,
            retrieval_embeds=emb.unsqueeze(0),
            **gen_kwargs,
        )
        return self.llm_tok.batch_decode(out, skip_special_tokens=True)[0]

    def _surprise_phrases(self, story: str, gist: str,
                          top_k: int | None = None) -> list[tuple[str, float]]:
        """
        Split `story` into clean, punctuation-free phrases at:  
          • sentence boundaries (spaCy's `doc.sents`)  
          • commas and coordinating conjunctions (`dep_ == "cc"`)
    
        Compute perplexity of each phrase when appended to `gist`
        (higher PPL ⇒ more 'surprising').  
        Return a list [(phrase, ppl), …] sorted *descending* by ppl.
        """
        import string
        punct_xlat = str.maketrans({c: None for c in string.punctuation if c != "'"})
        
        doc = nlp(story)
        phrases: list[str] = []
    
        for sent in doc.sents:                      # sentence-level split
            cur: list[str] = []
            for tok in sent:
                if tok.dep_ == "cc" or tok.text == ",":
                    if cur:
                        ph = " ".join(cur).translate(punct_xlat).strip()
                        if ph: phrases.append(ph)
                        cur = []
                else:
                    cur.append(tok.text)
            if cur:                                 # last chunk in sentence
                ph = " ".join(cur).translate(punct_xlat).strip()
                if ph: phrases.append(ph)
    
        # — perplexity scoring —
        scored = []
        for ph in phrases:
            prompt = f"{gist}\n\n{ph}"
            ppl    = self._perplexity(prompt)
            scored.append((ph, ppl))
    
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored if top_k is None else scored[:top_k]

    def _perplexity(self, text):
        enc = self.llm_tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            loss = self.llm(**enc, labels=enc.input_ids).loss
        return math.exp(loss.item())
    
    def _plot_semantic_consolidation(self, semantic_accuracies: List[float]):
        """Plot semantic memory accuracy over consolidation epochs (including epoch 0 baseline)."""
        if not semantic_accuracies:
            print("[Plot] No semantic accuracy data for consolidation")
            return
        
        # Epochs start from 0 (baseline before training)
        epochs = list(range(len(semantic_accuracies)))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, semantic_accuracies, marker='o', linewidth=2, markersize=8, color='#6a00a8')
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Semantic Memory Accuracy", fontsize=12)
        ax.set_title("Semantic Memory During Consolidation", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Chance (1/3)")
        ax.axvline(0, color="blue", linestyle=":", alpha=0.3, label="Baseline (pre-training)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        out_path = Path(self.cfg.output_dir, "plots", "semantic_memory_consolidation.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Plot] Saved {out_path}")
    
    def _plot_semantic_forgetting(self, semantic_accuracies: List[float]):
        """Plot semantic memory accuracy over forgetting phases."""
        if not semantic_accuracies:
            print("[Plot] No semantic accuracy data for forgetting")
            return
        
        # Phase 0 is baseline (before any forgetting)
        phases = list(range(len(semantic_accuracies)))
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(phases, semantic_accuracies, marker='s', linewidth=2, markersize=8, color='#e16462')
        ax.set_xlabel("Forgetting Phase (0 = baseline)", fontsize=12)
        ax.set_ylabel("Semantic Memory Accuracy", fontsize=12)
        ax.set_title("Semantic Memory During Forgetting", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Chance (1/3)")
        ax.axhline(semantic_accuracies[0], color="blue", linestyle=":", alpha=0.5, label=f"Baseline ({semantic_accuracies[0]:.1%})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        out_path = Path(self.cfg.output_dir, "plots", "semantic_memory_forgetting.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Plot] Saved {out_path}")
    
    def plot_combined_memory(self):
        """
        Plot combined episodic and semantic memory curves.
        Call after both consolidation and forgetting are complete.
        """
        plots_dir = Path(self.cfg.output_dir, "plots")
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data if not in memory
        cons_path = Path(self.cfg.output_dir, "data", "consolidation_recall.pkl")
        forg_path = Path(self.cfg.output_dir, "data", "forgetting_multi.pkl")
        
        if cons_path.exists():
            with open(cons_path, "rb") as f:
                cons_data = pickle.load(f)
        else:
            cons_data = self.results.get("consolidation", {})
        
        if forg_path.exists():
            with open(forg_path, "rb") as f:
                forg_data = pickle.load(f)
        else:
            forg_data = self.results.get("forgetting", {})
        
        # --- Consolidation: Episodic vs Semantic ---
        if cons_data.get("epoch_dist_orig") and cons_data.get("semantic_accuracy"):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            epoch_dists = cons_data["epoch_dist_orig"]
            epochs = list(range(1, len(epoch_dists) + 1))

            sem_acc = list(cons_data["semantic_accuracy"])
            # semantic_accuracy is often stored with a baseline at epoch 0, while
            # epoch_dist_orig starts at epoch 1. Align lengths robustly.
            if len(sem_acc) == len(epochs) + 1:
                sem_acc = sem_acc[1:]

            episodic_means = [np.mean(d) for d in epoch_dists]
            episodic_sems = [sem(d) for d in epoch_dists]
            episodic_sim = [1 - m for m in episodic_means]

            n = min(len(epochs), len(episodic_sim), len(episodic_sems), len(sem_acc))
            if n == 0:
                print("[Plot] WARNING: no consolidation points available for combined plot.")
                return
            if n != len(epochs) or n != len(sem_acc):
                print(
                    f"[Plot] WARNING: consolidation semantic/episodic length mismatch "
                    f"(semantic={len(sem_acc)}, episodic={len(epochs)}). Truncating to {n}."
                )
            epochs = epochs[:n]
            episodic_sim = episodic_sim[:n]
            episodic_sems = episodic_sems[:n]
            sem_acc = sem_acc[:n]

            ax1 = axes[0]
            ax1.errorbar(
                epochs,
                episodic_sim,
                yerr=episodic_sems,
                marker="o",
                linewidth=2,
                capsize=4,
                color="#0d0887",
                label="Episodic (similarity)",
            )
            ax1.plot(epochs, sem_acc, marker="s", linewidth=2, color="#6a00a8", label="Semantic (accuracy)")
            ax1.set_xlabel("Epoch", fontsize=12)
            ax1.set_ylabel("Memory Performance", fontsize=12)
            ax1.set_title("Consolidation: Episodic vs Semantic Memory", fontsize=14)
            ax1.set_ylim(0, 1.05)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # --- Forgetting: Episodic vs Semantic ---
            if forg_data.get("distances") and forg_data.get("semantic_accuracies"):
                dist_by_phase = forg_data["distances"]
                sem_by_phase = list(forg_data["semantic_accuracies"])
                phases = list(range(len(dist_by_phase)))

                forg_episodic_means = [np.mean(d) for d in dist_by_phase]
                forg_episodic_sems = [sem(d) if len(d) > 1 else 0 for d in dist_by_phase]
                forg_episodic_sim = [1 - m for m in forg_episodic_means]

                n2 = min(len(phases), len(forg_episodic_sim), len(forg_episodic_sems), len(sem_by_phase))
                if n2 == 0:
                    print("[Plot] WARNING: no forgetting points available for combined plot.")
                else:
                    if n2 != len(phases) or n2 != len(sem_by_phase):
                        print(
                            f"[Plot] WARNING: forgetting semantic/episodic length mismatch "
                            f"(semantic={len(sem_by_phase)}, episodic={len(phases)}). Truncating to {n2}."
                        )
                    phases = phases[:n2]
                    forg_episodic_sim = forg_episodic_sim[:n2]
                    forg_episodic_sems = forg_episodic_sems[:n2]
                    sem_by_phase = sem_by_phase[:n2]

                    ax2 = axes[1]
                    ax2.errorbar(
                        phases,
                        forg_episodic_sim,
                        yerr=forg_episodic_sems,
                        marker="o",
                        linewidth=2,
                        capsize=4,
                        color="#0d0887",
                        label="Episodic (similarity)",
                    )
                    ax2.plot(phases, sem_by_phase, marker="s", linewidth=2, color="#e16462", label="Semantic (accuracy)")
                    ax2.set_xlabel("Forgetting Phase (0 = baseline)", fontsize=12)
                    ax2.set_ylabel("Memory Performance", fontsize=12)
                    ax2.set_title("Forgetting: Episodic vs Semantic Memory", fontsize=14)
                    ax2.set_ylim(0, 1.05)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)

            fig.tight_layout()
            out_path = plots_dir / "episodic_vs_semantic_combined.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[Plot] Saved {out_path}")

    def run(self):
        """
        Full pipeline:
            1. Encode stories with xRAG
            2. Clean up memory
            3. Simulate consolidation
            4. Simulate forgetting
        """
        stories, hippo = self.load_data()
    
        # 1. Encoding
        enc = self.simulate_encoding(stories)
    
        # 2. Clean up memory
        self._release_xrag()
    
        # 3. Consolidation
        cons_src = str(self.cfg.consolidation_encoding).strip().lower()
        if cons_src == "full":
            encoded_for_consolidation = enc["recalled_stories"]["full"]
            print("[Consolidation] Using 'full' (verbatim) stories as consolidation targets.")
        else:
            try:
                cons_level = int(cons_src)
            except ValueError as err:
                raise ValueError(
                    f"Invalid consolidation_encoding={self.cfg.consolidation_encoding!r}. "
                    "Use an integer detail level (e.g. 0, 1, 3) or 'full'."
                ) from err
            if cons_level not in enc["recalled_stories"]:
                raise KeyError(
                    f"consolidation_encoding level {cons_level} not found in encoding cache. "
                    f"Available levels: {list(enc['recalled_stories'].keys())}"
                )
            encoded_for_consolidation = enc["recalled_stories"][cons_level]
            print(f"[Consolidation] Using {cons_level}-detail encoded stories as consolidation targets.")

        self.simulate_consolidation(encoded_for_consolidation, stories)
    
        # 4. Forgetting
        self.simulate_chunked_forgetting(stories)
        
        # 5. Combined episodic vs semantic plot
        self.plot_combined_memory()
    
        print("Simulation complete – results in", self.cfg.output_dir)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Memory simulation with consolidation and forgetting")
    parser.add_argument("--force", action="store_true", 
                        help="Force retrain: delete all cached data and models before running")
    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output directory for results")
    parser.add_argument("--num_stories", type=int, default=500,
                        help="Number of stories to consolidate")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of consolidation epochs")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed")
    parser.add_argument(
        "--consolidation_encoding",
        type=str,
        default="0",
        help="Which encoded variant to use during consolidation: integer detail level (e.g. 0,1,3) or 'full'.",
    )
    parser.add_argument(
        "--semantic_questions_file",
        type=str,
        default=None,
        help="Optional JSONL file of semantic questions.",
    )
    parser.add_argument(
        "--story_order_file",
        type=str,
        default=None,
        help="Optional CSV with a storyid column defining the fixed story order.",
    )
    args = parser.parse_args()
    
    # If --force, delete cached data and models
    if args.force:
        import shutil
        output_dir = Path(args.output_dir)
        
        # Delete cached encoding data
        cached_encoding = output_dir / "data" / "recalled_stories.pkl"
        if cached_encoding.exists():
            print(f"[--force] Removing cached encoding: {cached_encoding}")
            cached_encoding.unlink()
        
        # Delete cached consolidation results
        for name in ("consolidation_recall.pkl", "consolidation_results.pkl"):
            cached_cons = output_dir / "data" / name
            if cached_cons.exists():
                print(f"[--force] Removing cached consolidation results: {cached_cons}")
                cached_cons.unlink()
        
        # Delete cached forgetting results
        cached_forg = output_dir / "data" / "forgetting_multi.pkl"
        if cached_forg.exists():
            print(f"[--force] Removing cached forgetting results: {cached_forg}")
            cached_forg.unlink()
        
        # Delete saved models
        models_dir = output_dir / "models"
        if models_dir.exists():
            print(f"[--force] Removing saved models: {models_dir}")
            shutil.rmtree(models_dir)
        
        print("[--force] All cached data cleared, starting fresh training")
    
    # Create config with CLI overrides
    cfg = Config(
        output_dir=args.output_dir,
        num_stories=args.num_stories,
        num_epochs=args.num_epochs,
        seed=args.seed,
        consolidation_encoding=args.consolidation_encoding,
        semantic_questions_file=args.semantic_questions_file,
        story_order_file=args.story_order_file,
    )
    
    sim = MemorySimulator(cfg)
    sim.run()


if __name__ == "__main__":
    main()
