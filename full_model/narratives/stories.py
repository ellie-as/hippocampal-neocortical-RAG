from __future__ import annotations
import inspect
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse

import numpy as np
import random
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback

from utils import (
    set_seed, get_device, first_sentence,
    XRAG, Consolidator, prepare_roc_sets, prompts_from_sets
)

# Import global LoRA configuration
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_config import ACTIVE as LORA_CONFIG


@dataclass
class Config:
    # xRAG components
    llm_name: str = "Hannibal046/xrag-7b"
    retriever_name: str = "Salesforce/SFR-Embedding-Mistral"

    # Consolidation backbone
    consolidation_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Device/precision
    use_mps: bool = False
    use_4bit: bool = True

    # Training
    num_epochs: int = 3
    batch_size: int = 1
    learning_rate: float = 5e-5
    print_steps: int = 20
    max_new_tokens: int = 300

    # LoRA settings (defaults from global lora_config.py)
    lora_r: int = field(default_factory=lambda: LORA_CONFIG.r)
    lora_alpha: int = field(default_factory=lambda: LORA_CONFIG.alpha)
    lora_dropout: float = field(default_factory=lambda: LORA_CONFIG.dropout)
    target_modules: List[str] = field(default_factory=lambda: LORA_CONFIG.target_modules.copy())

    # xRAG detail levels we keep (we train on level 0 only)
    detail_levels: List[int] = field(default_factory=lambda: [0, 1, 3])

    # Sampling temps for post-consolidation recall logging (0.0 acts greedy)
    temps: List[float] = field(default_factory=lambda: [0.0])

    # ROC Stories prep (matches your prepare_data)
    n_typical: int = 50
    n_variants: int = 50
    rng_seed: int = 123
    recall_prompt_chars: int = 100

    # Retriever batching (used by XRAG)
    retriever_batch_size: int = 16
    retriever_max_length: int = 500
    docs_per_datastore: int = 1000

    # Paths
    output_dir: str = "output_raykov_xrag_fixed_firstline_50"
    stories_csv: str = "../../data/stories_train.csv"  # kept for parity

    def __post_init__(self):
        for sub in ("plots", "models", "data"):
            Path(self.output_dir, sub).mkdir(parents=True, exist_ok=True)


class RaykovXRAGRunner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.rng_seed)
        self.device = get_device(cfg.use_mps)
        self.xrag = XRAG(cfg, self.device)

    def recall_question(self, story: str) -> str:
        return f"{story[0:self.cfg.recall_prompt_chars]}... What happened next? (Be concise.)"

    @staticmethod
    def plain_recall_prompt(question: str) -> str:
        return f"<s>[INST] {question} [/INST] The answer is: "

    @staticmethod
    def _extract_question(prompt: str) -> str | None:
        if "Question:" not in prompt:
            return None
        question = prompt.split("Question:", 1)[1]
        if "[/INST]" in question:
            question = question.split("[/INST]", 1)[0]
        question = question.strip()
        return question or None

    @staticmethod
    def _story_by_id(sets: Dict[str, Any]) -> Dict[str, str]:
        stories: Dict[str, str] = {}
        for cat in ("typical", "incomplete", "updated"):
            for i, story in enumerate(sets[cat]):
                stories[f"{cat}_{i:04d}"] = story
        return stories

    def _with_plain_prompt(self, row: Dict[str, Any], story: str | None = None) -> Dict[str, Any]:
        row = dict(row)
        current_prompt = row.get("prompt_str", "")
        if "<xRAG>" in current_prompt or "Background:" in current_prompt:
            row.setdefault("xrag_prompt_str", current_prompt)

        question = row.get("question") or self._extract_question(current_prompt)
        if question is None:
            if story is None:
                raise ValueError(f"Cannot recover recall question for row {row.get('id')!r}")
            question = self.recall_question(story)

        row["question"] = question
        row["prompt_str"] = self.plain_recall_prompt(question)
        return row

    def encode_story(self, story: str, datastore) -> Dict[str, Any]:
        self.xrag.load()
        seed = first_sentence(story)

        recalled = {lvl: [] for lvl in self.cfg.detail_levels}
        mem_sizes = {lvl: [] for lvl in self.cfg.detail_levels}
        details_used = {lvl: [] for lvl in self.cfg.detail_levels}
        recalled["imagined"], mem_sizes["imagined"] = [], []
        recalled["full"], mem_sizes["full"] = [], []

        # Level 0 (gist) — xRAG retrieval
        q0 = self.recall_question(story)
        # emb0 is an xRAG embedding
        p0, emb0 = self.xrag._prepare_prompt(q0)
        ans0 = self.xrag._run_xrag(p0, self.xrag._nearest_doc_embed(emb0, datastore), self.cfg.max_new_tokens)
        recalled[0].append(ans0)
        mem_sizes[0].append(1)
        print("Original story:", story)
        print("Prompt:", p0)
        print("Level 0 answer:", ans0)


        # Full (verbatim)
        recalled["full"].append(story)
        mem_sizes["full"].append(len(self.xrag.llm_tok(story, add_special_tokens=False)["input_ids"]))

        return {
            "recalled": recalled,
            "mem_sizes": mem_sizes,
            "details": details_used,
            "pre_xrag_gist": ans0,
            "pre_xrag_prompt": p0,
            "recall_question": q0,
            "post_prompt": self.plain_recall_prompt(q0),
        }
    
    def run(self):
        cfg = self.cfg
        out_dir = Path(cfg.output_dir)
        data_dir = out_dir / "data"
        models_dir = out_dir / "models"
        stories_cache = data_dir / "stories_prepared.pkl"
        pre_cache = data_dir / "generations_pre.json"

        sets = None
        pre_recall_rows: List[Dict[str, Any]] | None = None
        if stories_cache.exists() and pre_cache.exists():
            print("[Resume] Loading cached ROC sets and pre-consolidation generations …")
            with open(stories_cache, "rb") as fh:
                sets = pickle.load(fh)
            with open(pre_cache) as fh:
                cached_rows = json.load(fh)
            expected_rows = sum(len(sets[cat]) for cat in ("typical", "incomplete", "updated"))
            if len(cached_rows) == expected_rows:
                story_by_id = self._story_by_id(sets)
                pre_recall_rows = [
                    self._with_plain_prompt(row, story_by_id.get(row.get("id", "")))
                    for row in cached_rows
                ]
                if pre_recall_rows != cached_rows:
                    print("[Resume] Updating cached pre generations with plain consolidation prompts …")
                    with open(pre_cache, "w") as fh:
                        json.dump(pre_recall_rows, fh, indent=2)
            else:
                print(
                    f"[Resume] Cached pre generations incomplete "
                    f"({len(cached_rows)}/{expected_rows}); rerunning xRAG encoding."
                )

        if pre_recall_rows is None:
            # 1) Prepare ROC sets
            print("[Prep] Preparing ROC Stories sets …")
            sets = prepare_roc_sets(
                cfg.n_typical,
                cfg.n_variants,
                cfg.rng_seed,
                cfg.stories_csv,
                prompt_cue_chars=cfg.recall_prompt_chars,
            )
            with open(stories_cache, "wb") as fh:
                pickle.dump(sets, fh)

            # 2) Build xRAG datastore from all stories
            print("[xRAG] Building datastore from ROC sets …")
            self.xrag.load()
            all_docs: List[str] = []
            for cat in ("typical", "incomplete", "updated"):
                all_docs.extend(sets[cat])
            datastore = self.xrag._prepare_datastore(all_docs)

            # 3) xRAG ENCODING + PRE-consolidation recall (encoded prompt)
            print("[Encode] Running xRAG encoding across all stories …")
            pre_recall_rows = []
            for cat in ("typical", "incomplete", "updated"):
                for i, s in enumerate(tqdm(sets[cat], desc=f"[Encode] {cat}")):
                    encoded = self.encode_story(s, datastore)
                    pre_recall_rows.append({
                        "id": f"{cat}_{i:04d}",
                        "category": cat,
                        "question": encoded["recall_question"],
                        "prompt_str": encoded["post_prompt"],       # prompt used for consolidation/post recall
                        "xrag_prompt_str": encoded["pre_xrag_prompt"],  # exact encoded prompt used pre
                        "generation": encoded["pre_xrag_gist"],     # model's pre answer (target)
                        "note": "Pre-consolidation xRAG recall (gist style)",
                    })

            # Persist pre generations
            with open(pre_cache, "w") as fh:
                json.dump(pre_recall_rows, fh, indent=2)

            # Release xRAG before consolidation
            self.xrag.release()
    
        # 4) Consolidation — train on (plain cue prompt -> pre xRAG answer)
        print("[Train] Building consolidation model …")
        tok = AutoTokenizer.from_pretrained(cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
    
        consolidator = Consolidator(cfg, self.device)
    
        # Concatenate the post-consolidation cue with the pre-consolidation reconstruction.
        train_texts = [
            f"{rec['prompt_str']}{rec['generation'].strip()}</s>"
            for rec in pre_recall_rows
        ]
        
        train_ds = consolidator.texts_to_ds(train_texts, tok, max_len=2048, chatml=False)
    
        t_args = TrainingArguments(
            output_dir=str(models_dir / "ckpts"),
            seed=cfg.rng_seed,
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            fp16=False,
            logging_steps=cfg.print_steps,
            save_strategy="epoch",
            eval_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            label_names=["labels"],
        )
    
        class EpochLogger(TrainerCallback):
            def __init__(self, tok, prompts, max_new, outdir: Path):
                self.tok = tok
                self.prompts = prompts            # list of dicts with 'prompt_str'
                self.max_new = max_new
                self.outdir = outdir
                self.epoch_records: List[Dict[str, Any]] = []
    
            @staticmethod
            def _decode_new(tok, out_ids, in_ids_len: int) -> str:
                new_tokens = out_ids[0][in_ids_len:]
                return tok.decode(new_tokens, skip_special_tokens=True).strip()
    
            def on_epoch_end(self, args, state, control, model=None, **kwargs):
                if int(state.epoch) == cfg.num_epochs:
                    ep = int(state.epoch)
                    print(f"[Recall/Post] Epoch {ep} logging …")
                    rows = []
                    was_training = model.training
                    model.eval()
                    for pr in tqdm(self.prompts, desc="[Recall/Post] generating"):
                        inputs = self.tok(pr["prompt_str"], return_tensors="pt")
                        ids = inputs["input_ids"].to(model.device)
                        attn = inputs.get("attention_mask")
                        gen_kwargs = {}
                        if attn is not None:
                            gen_kwargs["attention_mask"] = attn.to(model.device)
                        with torch.no_grad():
                            out = model.generate(
                                ids, max_new_tokens=self.max_new, do_sample=False,
                                no_repeat_ngram_size=0, pad_token_id=self.tok.pad_token_id,
                                **gen_kwargs,
                            )
                        greedy = self._decode_new(self.tok, out, ids.shape[1])
                        row = dict(pr)
                        row["generations"] = {"0.0": greedy}
                        rows.append(row)
                    if was_training:
                        model.train()
                    with open(self.outdir / f"generations_epoch_{ep:02d}.json", "w") as fh:
                        json.dump(rows, fh, indent=2)
                    self.epoch_records.append({"epoch": ep, "rows": rows})
    
        # Post recall uses the same plain cue prompts as consolidation training.
        prompts_for_post = pre_recall_rows
    
        logger = EpochLogger(tok, prompts_for_post, cfg.max_new_tokens, data_dir)
        model = consolidator.build_model()
        trainer_kwargs = {
            "model": model,
            "args": t_args,
            "train_dataset": train_ds,
            "callbacks": [logger],
        }
        trainer_params = inspect.signature(Trainer.__init__).parameters
        if "processing_class" in trainer_params:
            trainer_kwargs["processing_class"] = tok
        elif "tokenizer" in trainer_params:
            trainer_kwargs["tokenizer"] = tok
        trainer = Trainer(**trainer_kwargs)
        trainer.train()
    
        # Save final LoRA
        model.save_pretrained(models_dir / "final")
        tok.save_pretrained(models_dir / "final")
    
        # 5) Save final post generations (last-epoch dump or fallback)
        if logger.epoch_records:
            last = logger.epoch_records[-1]["rows"]
        else:
            # Fallback: greedy-only one final pass with the same plain prompts
            rows = []
            model.eval()
            for pr in tqdm(prompts_for_post, desc="[Recall/Post] generating (fallback)"):
                inputs = tok(pr["prompt_str"], return_tensors="pt")
                ids = inputs["input_ids"].to(model.device)
                attn = inputs.get("attention_mask")
                gen_kwargs = {}
                if attn is not None:
                    gen_kwargs["attention_mask"] = attn.to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        ids, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                        no_repeat_ngram_size=0, pad_token_id=tok.pad_token_id,
                        **gen_kwargs,
                    )
                new_tokens = out[0][ids.shape[1]:]
                greedy = tok.decode(new_tokens, skip_special_tokens=True).strip()
                row = dict(pr); row["generations"] = {"0.0": greedy}
                rows.append(row)
            last = rows
    
        with open(data_dir / "generations_post.json", "w") as fh:
            json.dump(last, fh, indent=2)
    
        print(f"[Done] Results saved to: {cfg.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="output_raykov_xrag_fixed_firstline_50")
    ap.add_argument("--n_typical", type=int, default=50)
    ap.add_argument("--n_variants", type=int, default=50)
    ap.add_argument("--rng_seed", type=int, default=123)
    ap.add_argument("--recall_prompt_chars", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=300)
    args = ap.parse_args()

    cfg = Config(
        output_dir=args.output_dir,
        n_typical=args.n_typical,
        n_variants=args.n_variants,
        rng_seed=args.rng_seed,
        recall_prompt_chars=args.recall_prompt_chars,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"[Config] n_typical={cfg.n_typical} n_variants={cfg.n_variants}")
    runner = RaykovXRAGRunner(cfg)
    runner.run()
