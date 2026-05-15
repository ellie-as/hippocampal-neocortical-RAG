#!/usr/bin/env python3
"""
Two-stage Bartlett training:
  Stage 1: Train on background topic only
  Stage 2: Load stage 1 model, train on Bartlett story only

This separates the background knowledge consolidation from the episodic memory encoding.

Usage:
    python bartlett_twostage.py --output_dir output_twostage --stage1_epochs 5 --articles_per_topic 1000
"""
from __future__ import annotations
import argparse
import inspect
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel

from utils import (
    set_seed, get_device, Consolidator,
    load_bartlett, load_topic_corpus_wiki,
    BARTLETT_TXT, BARTLETT_FALLBACK, recall_prefix,
)

# Import global LoRA configuration
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_config import ACTIVE as LORA_CONFIG


def make_trainer(*, model, args, train_dataset, tok, callbacks=None):
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tok
    if callbacks is not None:
        trainer_kwargs["callbacks"] = callbacks
    return Trainer(**trainer_kwargs)


@dataclass
class Config:
    # Consolidation model (base for LoRA)
    consolidation_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # Device/precision
    use_mps: bool = False
    use_4bit: bool = True

    # Training - Stage 1 (background)
    stage1_epochs: int = 10
    stage1_batch_size: int = 1
    stage1_learning_rate: float = 5e-5
    
    # Training - Stage 2 (Bartlett)
    stage2_epochs: int = 10
    stage2_batch_size: int = 1
    stage2_learning_rate: float = 5e-5
    bartlett_repeats: int = 1  # Repeat Bartlett story N times per epoch for more gradient updates
    stage2_bg_replay: int = 50  # Number of background docs to replay per epoch in Stage 2
                                 # Prevents catastrophic forgetting of background knowledge
                                 # Set to 0 to disable replay (old behaviour)
    
    print_steps: int = 20
    max_new_tokens: int = 500
    min_new_tokens: int = -1  # -1 = auto (match Bartlett token count)

    # LoRA settings (defaults from global lora_config.py)
    lora_r: int = field(default_factory=lambda: LORA_CONFIG.r)
    lora_alpha: int = field(default_factory=lambda: LORA_CONFIG.alpha)
    lora_dropout: float = field(default_factory=lambda: LORA_CONFIG.dropout)
    target_modules: List[str] = field(default_factory=lambda: LORA_CONFIG.target_modules.copy())

    # Bartlett + topics
    bartlett_path: str = BARTLETT_TXT
    topics: List[str] = field(default_factory=lambda: [
        "Nature", "Politics", "Universe", "Sport", "Health",
    ])
    articles_per_topic: int = 1000
    chars_per_article: int = 1000
    use_tfidf_filter: bool = True

    # Sampling temps for recall logging
    temps: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5])

    # Misc
    seed: int = 42
    output_dir: str = "output_twostage"
    force_stage1: bool = False  # Re-run Stage 1 even if model exists
    force_stage2: bool = False  # Re-run Stage 2 even if outputs exist

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# Bartlett story: loaded from full model/data/bartlett.txt via utils.load_bartlett()
# BARTLETT_FALLBACK is imported from utils for use when the file is missing.


def chatml_format(txt: str, prompt_cue: str | None = None, include_full_text: bool = False) -> str:
    """Convert text to ChatML format for training."""
    if "." in txt:
        first, rest = txt.split(".", 1)
    elif "\n" in txt:
        first, rest = txt.split("\n", 1)
    else:
        first, rest = txt, txt
    cue = prompt_cue if prompt_cue is not None else f"{first.strip()}."
    answer = txt.strip() if include_full_text else rest.strip()
    return (f"<s>[INST] {cue} What happened (in detail)? [/INST] "
            f"{answer} </s>")


class TwoStageRunner:
    """
    Two-stage training:
      Stage 1: Train on background topic only (build semantic knowledge)
      Stage 2: Train on Bartlett story (encode episodic memory)
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = get_device(cfg.use_mps)
    
    def load_bartlett(self) -> str:
        """Load the Bartlett story."""
        try:
            return load_bartlett(self.cfg.bartlett_path)
        except:
            print(f"Could not load {self.cfg.bartlett_path}, using built-in story")
            return BARTLETT_FALLBACK.strip()
    
    def stage1_train_background(
        self, 
        background_docs: List[str], 
        out_dir: Path,
        topic: str
    ) -> Path:
        """
        Stage 1: Train on background documents only.
        Returns path to saved model.
        """
        print(f"\n{'='*60}")
        print(f"STAGE 1: Training on background ({topic})")
        print(f"{'='*60}")
        print(f"Documents: {len(background_docs)}")
        print(f"Epochs: {self.cfg.stage1_epochs}")
        
        # Prepare tokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        tok.padding_side = "right"
        
        # Format training texts
        train_texts = [chatml_format(doc) for doc in background_docs]
        random.shuffle(train_texts)
        
        # Build model with LoRA
        consolidator = Consolidator(self.cfg, self.device)
        model = consolidator.build_model()
        train_ds = consolidator.texts_to_ds(train_texts, tok, max_len=2048, chatml=False)
        
        # Training arguments
        stage1_dir = out_dir / "stage1_background"
        
        # fp16/bf16 only works on CUDA
        use_fp16 = torch.cuda.is_available()
        
        t_args = TrainingArguments(
            output_dir=str(stage1_dir),
            seed=self.cfg.seed,
            num_train_epochs=self.cfg.stage1_epochs,
            per_device_train_batch_size=self.cfg.stage1_batch_size,
            learning_rate=self.cfg.stage1_learning_rate,
            fp16=use_fp16,
            logging_steps=self.cfg.print_steps,
            save_strategy="epoch",
            eval_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            label_names=["labels"],
        )
        
        trainer = make_trainer(model=model, args=t_args, train_dataset=train_ds, tok=tok)
        
        print(f"→ Training on {len(train_texts)} background documents...")
        trainer.train()
        
        # Save stage 1 model
        model_path = stage1_dir / "model"
        model.save_pretrained(model_path)
        tok.save_pretrained(model_path)
        print(f"Stage 1 model saved to: {model_path}")
        
        # Clean up
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return model_path
    
    def stage2_train_bartlett(
        self,
        bartlett: str,
        stage1_model_path: Path,
        background_docs: List[str],
        out_dir: Path,
        topic: str
    ):
        """
        Stage 2: Load stage 1 model, train on Bartlett story only.
        """
        print(f"\n{'='*60}")
        print(f"STAGE 2: Training on Bartlett story")
        print(f"{'='*60}")
        print(f"Epochs: {self.cfg.stage2_epochs}")
        print(f"Loading model from: {stage1_model_path}")
        
        # Load tokenizer
        tok = AutoTokenizer.from_pretrained(stage1_model_path)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        tok.padding_side = "right"
        
        # Load stage 1 model (LoRA on base)
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        
        if self.cfg.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        # Prepare base model for training (important for 4-bit)
        from peft import prepare_model_for_kbit_training
        base_model.config.use_cache = False
        base_model.gradient_checkpointing_enable()
        if self.cfg.use_4bit:
            base_model = prepare_model_for_kbit_training(base_model)
        
        # Load LoRA weights from stage 1 and continue training (no merge)
        model = PeftModel.from_pretrained(base_model, stage1_model_path)
        
        # Enable training on the existing LoRA weights
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        
        model.train()
        model.print_trainable_parameters()
        print("→ Continuing training on Stage 1 LoRA weights (no merge)")
        
        # # Merge stage 1 LoRA into base weights so topic knowledge is permanent,
        # # then add a fresh LoRA adapter for stage 2 (Bartlett encoding).
        # from peft import LoraConfig, get_peft_model
        # stage1_peft = PeftModel.from_pretrained(base_model, stage1_model_path)
        # model = stage1_peft.merge_and_unload()
        # print("→ Merged Stage 1 LoRA into base weights")

        # lora_cfg = LoraConfig(
        #     r=self.cfg.lora_r,
        #     lora_alpha=self.cfg.lora_alpha,
        #     lora_dropout=self.cfg.lora_dropout,
        #     target_modules=self.cfg.target_modules,
        #     bias="none",
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model, lora_cfg)
        
        # Diagnostic: compute LoRA weight norm before training
        def lora_weight_norm():
            total = 0.0
            for name, param in model.named_parameters():
                if "lora_" in name and param.requires_grad:
                    total += param.data.norm().item()
            return total
        
        lora_norm_before = lora_weight_norm()
        print(f"→ LoRA weight norm before Stage 2: {lora_norm_before:.4f}")
        
        # Format Bartlett for training, repeated to get more gradient updates
        # In one-stage with 500 bg docs, Bartlett is seen once per epoch among 501 texts
        # Here we can repeat it to simulate similar exposure
        bartlett_prompt_cue = recall_prefix()
        bartlett_texts = [
            chatml_format(bartlett, prompt_cue=bartlett_prompt_cue, include_full_text=True)
        ] * self.cfg.bartlett_repeats
        print(f"→ Bartlett story repeated {self.cfg.bartlett_repeats} times in training set")

        # Background replay: mix in a random subsample of background docs each epoch
        # to prevent catastrophic forgetting of Stage 1 knowledge
        n_replay = self.cfg.stage2_bg_replay
        if n_replay > 0 and background_docs:
            rng = random.Random(self.cfg.seed)
            replay_pool = [chatml_format(doc) for doc in background_docs]
            replay_sample = rng.sample(replay_pool, min(n_replay, len(replay_pool)))
            train_texts = bartlett_texts + replay_sample
            rng.shuffle(train_texts)
            print(f"→ Background replay: {len(replay_sample)} docs mixed in "
                  f"(total {len(train_texts)} training texts per epoch)")
        else:
            train_texts = bartlett_texts
            print(f"→ No background replay (stage2_bg_replay=0)")
        
        # Create dataset
        from datasets import Dataset
        def tokenize(examples):
            encoded = {"input_ids": [], "attention_mask": [], "labels": []}
            answer_marker = "[/INST]"
            for text in examples["text"]:
                enc = tok(
                    text,
                    truncation=True,
                    max_length=2048,
                    padding="max_length",
                )
                labels = list(enc["input_ids"])

                answer_start_char = text.index(answer_marker) + len(answer_marker)
                prompt_ids = tok(
                    text[:answer_start_char],
                    truncation=True,
                    max_length=2048,
                    padding=False,
                )["input_ids"]
                prompt_len = min(len(prompt_ids), len(labels))

                labels[:prompt_len] = [-100] * prompt_len
                labels = [
                    -100 if mask == 0 else label
                    for mask, label in zip(enc["attention_mask"], labels)
                ]

                encoded["input_ids"].append(enc["input_ids"])
                encoded["attention_mask"].append(enc["attention_mask"])
                encoded["labels"].append(labels)
            return encoded
        ds = Dataset.from_dict({"text": train_texts})
        ds = ds.map(tokenize, batched=True, remove_columns=["text"])
        
        # Setup recall logging
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        original_emb = embedder.encode([bartlett])[0]
        bg_embs = embedder.encode(background_docs[:50])
        bg_center = np.mean(bg_embs, axis=0)
        
        # Get first sentence for prompt
        first_sent = bartlett_prompt_cue
        prompt = f"<s>[INST] {bartlett_prompt_cue} What happened (in detail)? [/INST]"
        
        # Resolve min/max new tokens: -1 means auto-compute from Bartlett token count
        # so that generated recalls are exactly the same length as the original story
        if self.cfg.min_new_tokens == -1:
            _prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            _target_ids = tok(f"{prompt} {bartlett}", add_special_tokens=False)["input_ids"]
            resolved_min_new = max(1, len(_target_ids) - len(_prompt_ids))
            resolved_max_new = resolved_min_new
            print(f"  [length guard] Auto min/max_new_tokens = {resolved_min_new} (from prompt-conditioned Bartlett token count)")
        elif self.cfg.min_new_tokens > 0:
            resolved_min_new = self.cfg.min_new_tokens
            resolved_max_new = self.cfg.max_new_tokens
        else:
            resolved_min_new = 0
            resolved_max_new = self.cfg.max_new_tokens
        
        epoch_logs = []
        
        def log_recall(epoch: int):
            """Log recall at end of epoch."""
            model.eval()
            ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
            gen_kwargs = dict(
                    do_sample=False,
                    max_new_tokens=resolved_max_new,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            if resolved_min_new > 0:
                gen_kwargs["min_new_tokens"] = resolved_min_new
            with torch.no_grad():
                out = model.generate(ids, **gen_kwargs)
            new_tokens = out[0][ids.shape[1]:]
            greedy = tok.decode(new_tokens, skip_special_tokens=True).strip()
            
            greedy_emb = embedder.encode([greedy])[0]
            from scipy.spatial.distance import cosine
            cos_to_orig = float(cosine(greedy_emb, original_emb))
            cos_to_bg = float(cosine(greedy_emb, bg_center))
            
            log = {
                "epoch": epoch,
                "greedy": greedy,
                "cos_to_original": cos_to_orig,
                "cos_to_background": cos_to_bg,
            }
            epoch_logs.append(log)
            print(f"  Epoch {epoch}: cos_to_orig={cos_to_orig:.4f}, cos_to_bg={cos_to_bg:.4f}")
            print(f"  Recall (first 150 chars): {greedy[:150]}...")
            model.train()
        
        # Training
        stage2_dir = out_dir / "stage2_bartlett"
        
        # fp16 only works on CUDA
        use_fp16 = torch.cuda.is_available()
        
        t_args = TrainingArguments(
            output_dir=str(stage2_dir),
            seed=self.cfg.seed,
            num_train_epochs=self.cfg.stage2_epochs,
            per_device_train_batch_size=self.cfg.stage2_batch_size,
            learning_rate=self.cfg.stage2_learning_rate,
            fp16=use_fp16,
            logging_steps=self.cfg.print_steps,
            save_strategy="epoch",
            eval_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            label_names=["labels"],
        )
        
        from transformers import TrainerCallback
        
        class RecallCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                log_recall(int(state.epoch))
        
        trainer = make_trainer(
            model=model,
            args=t_args,
            train_dataset=ds,
            tok=tok,
            callbacks=[RecallCallback()],
        )
        
        print(f"→ Training on Bartlett story...")
        trainer.train()
        
        # Diagnostic: check if LoRA weights changed
        lora_norm_after = lora_weight_norm()
        print(f"→ LoRA weight norm after Stage 2: {lora_norm_after:.4f}")
        print(f"→ LoRA weight change: {abs(lora_norm_after - lora_norm_before):.4f}")
        
        # Save final model (to location expected by plot.py)
        model_path = out_dir / "model" / "final"
        model_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_path)
        tok.save_pretrained(model_path)
        print(f"Stage 2 model saved to: {model_path}")
        
        # Save logs
        logs_path = stage2_dir / "epoch_logs.json"
        with open(logs_path, "w") as f:
            json.dump(epoch_logs, f, indent=2)
        
        # Create summary
        df = pd.DataFrame([
            {"epoch": log["epoch"], 
             "cos_to_original": log["cos_to_original"],
             "cos_to_background": log["cos_to_background"]}
            for log in epoch_logs
        ])
        df.to_csv(stage2_dir / "summary.csv", index=False)
        
        # Plot
        if len(epoch_logs) > 0:
            plt.figure(figsize=(8, 5))
            epochs = [log["epoch"] for log in epoch_logs]
            cos_orig = [log["cos_to_original"] for log in epoch_logs]
            cos_bg = [log["cos_to_background"] for log in epoch_logs]
            
            plt.plot(epochs, cos_orig, 'o-', label="To Original", color='#e16462')
            plt.plot(epochs, cos_bg, 's-', label="To Background", color='#0d0887')
            plt.xlabel("Epoch")
            plt.ylabel("Cosine Distance")
            plt.title(f"Two-Stage Training: {topic}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(stage2_dir / "drift_plot.png", dpi=300)
            plt.close()
        
        # Clean up
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _build_ckpt_cache_and_analysis(self, output_dir: Path, topics: List[str]) -> None:
        """
        Build _ckpt_cache and _analysis from epoch_logs.json so collate_figures.py
        word clouds work without needing legacy plot.py (which does full checkpoint sampling).
        Uses greedy recall text from each epoch only.
        """
        ckpt_cache_dir = output_dir / "_ckpt_cache"
        analysis_dir = output_dir / "_analysis"
        ckpt_cache_dir.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        all_samples_rows = []
        for topic in topics:
            logs_path = output_dir / topic / "stage2_bartlett" / "epoch_logs.json"
            if not logs_path.exists():
                continue
            try:
                logs = json.loads(logs_path.read_text())
            except Exception as e:
                print(f"  [Cache] Could not read {logs_path}: {e}")
                continue
            # Build checkpoint cache: key "epoch_0.0" -> [greedy_text] (greedy = temp 0)
            cached = {}
            for entry in logs:
                ep = entry.get("epoch")
                greedy = entry.get("greedy") or entry.get("greedy_text", "")
                if ep is not None and greedy:
                    cached[f"{ep}_0.0"] = [greedy]
            if cached:
                cache_file = ckpt_cache_dir / f"{topic}_checkpoint_samples.json"
                cache_file.write_text(json.dumps(cached, indent=2))
                print(f"  [Cache] Wrote {cache_file.name} ({len(cached)} epochs)")
            # Last epoch greedy for _analysis fallback
            if logs:
                last = logs[-1]
                text = last.get("greedy") or last.get("greedy_text", "")
                cos_bg = last.get("cos_to_background")
                if text:
                    all_samples_rows.append({
                        "topic": topic,
                        "sample_idx": 0,
                        "temperature": 0.0,
                        "text": text,
                        "cos_to_bg": cos_bg if cos_bg is not None else "",
                    })
        if all_samples_rows:
            df = pd.DataFrame(all_samples_rows)
            out_csv = analysis_dir / "final_temp0.0_all_samples.csv"
            df.to_csv(out_csv, index=False)
            print(f"  [Cache] Wrote {out_csv.name} ({len(df)} topics)")
    
    def run(self):
        """Run two-stage training for all topics."""
        cfg = self.cfg
        
        # Load background documents
        topic_docs = load_topic_corpus_wiki(
            topics=cfg.topics,
            seed=cfg.seed,
            articles_per_topic=cfg.articles_per_topic,
            chars_per_article=cfg.chars_per_article,
            use_tfidf_filter=cfg.use_tfidf_filter,
        )
        
        # Load Bartlett
        bartlett = self.load_bartlett()
        
        for topic in cfg.topics:
            docs = topic_docs.get(topic, [])
            if not docs:
                print(f"Skipping {topic} (no docs)")
                continue
            
            topic_dir = Path(cfg.output_dir) / topic
            topic_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n{'#'*60}")
            print(f"# TOPIC: {topic}")
            print(f"{'#'*60}")
            
            # Check if Stage 1 model already exists
            stage1_dir = topic_dir / "stage1_background"
            stage1_path = stage1_dir / "model"
            
            if (
                not self.cfg.force_stage1
                and stage1_path.exists()
                and (stage1_path / "adapter_config.json").exists()
            ):
                print(f"\n✓ Stage 1 model found at: {stage1_path}")
                print(f"  Skipping Stage 1 training, loading existing model...")
            else:
                # Stage 1: Train on background
                stage1_path = self.stage1_train_background(docs, topic_dir, topic)

            # Optionally force Stage 2 retrain while reusing Stage 1.
            if cfg.force_stage2:
                stage2_dir = topic_dir / "stage2_bartlett"
                final_dir = topic_dir / "model" / "final"
                if stage2_dir.exists():
                    print(f"  [--force_stage2] Removing: {stage2_dir}")
                    shutil.rmtree(stage2_dir, ignore_errors=True)
                if final_dir.exists():
                    print(f"  [--force_stage2] Removing: {final_dir}")
                    shutil.rmtree(final_dir, ignore_errors=True)

            # Stage 2: Train on Bartlett
            self.stage2_train_bartlett(bartlett, stage1_path, docs, topic_dir, topic)
        
        # Build _ckpt_cache and _analysis from epoch_logs so collate_figures word clouds work
        # without needing legacy plot.py
        self._build_ckpt_cache_and_analysis(Path(cfg.output_dir), cfg.topics)
        
        print(f"\n{'='*60}")
        print(f"All done! Results in: {cfg.output_dir}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Two-stage Bartlett training")
    parser.add_argument("--output_dir", type=str, default="output_twostage")
    parser.add_argument("--bartlett_path", type=str, default=BARTLETT_TXT)
    parser.add_argument("--topics", type=str, nargs="+",
                        default=["Nature", "Politics", "Universe", "Sport", "Health"],
                        help="Topics for Stage 1 background training (default: all five)")
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--articles_per_topic", type=int, default=1000,
                        help="Wikipedia articles per topic for Stage 1 (default matches run_all.py)")
    parser.add_argument("--chars_per_article", type=int, default=1000)
    parser.add_argument("--no_tfidf_filter", action="store_true",
                        help="Disable TF-IDF centrality filtering; use the seeded topic sample directly")
    parser.add_argument("--stage1_learning_rate", type=float, default=5e-5)
    parser.add_argument("--stage2_learning_rate", type=float, default=5e-5)
    parser.add_argument("--use_mps", action="store_true")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                        help="Use 4-bit quantization (requires CUDA)")
    parser.add_argument("--bartlett_repeats", type=int, default=1,
                        help="Repeat Bartlett story N times per epoch (default 1 to match original)")
    parser.add_argument("--stage2_bg_replay", type=int, default=50,
                        help="Background docs to replay per epoch in Stage 2 (default 50, 0 to disable)")
    parser.add_argument("--min_new_tokens", type=int, default=-1,
                        help="Minimum generated tokens for recall (-1 = auto from Bartlett, 0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force_stage1", action="store_true",
                        help="Re-run Stage 1 training even if a model already exists")
    parser.add_argument("--force_stage2", action="store_true",
                        help="Re-run Stage 2 training by deleting stage2_bartlett/ and model/final/ for each topic")
    
    args = parser.parse_args()
    
    cfg = Config(
        output_dir=args.output_dir,
        bartlett_path=args.bartlett_path,
        topics=args.topics,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        articles_per_topic=args.articles_per_topic,
        chars_per_article=args.chars_per_article,
        use_tfidf_filter=not args.no_tfidf_filter,
        stage1_learning_rate=args.stage1_learning_rate,
        stage2_learning_rate=args.stage2_learning_rate,
        use_mps=args.use_mps,
        use_4bit=args.use_4bit,
        bartlett_repeats=args.bartlett_repeats,
        stage2_bg_replay=args.stage2_bg_replay,
        min_new_tokens=args.min_new_tokens,
        seed=args.seed,
        force_stage1=args.force_stage1,
        force_stage2=args.force_stage2,
    )
    
    runner = TwoStageRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
