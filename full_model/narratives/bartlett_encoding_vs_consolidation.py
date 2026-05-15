#!/usr/bin/env python3
"""
Bartlett Encoding vs Consolidation Comparison

This script:
  1. Encodes the Bartlett story using xRAG (gist-style reconstruction)
  2. Consolidates the encoded version via LoRA fine-tuning on Mistral
  3. Collects 10 samples for each condition (encoded vs consolidated)
  4. Plots a bar chart: "Encoded" vs "Consolidated" with cosine distance to original

Output: bartlett_encoding_vs_consolidation/

Usage:
    python bartlett_encoding_vs_consolidation.py
    python bartlett_encoding_vs_consolidation.py --n_samples 20 --consolidation_epochs 5
"""
from __future__ import annotations

import argparse
import json
import gc
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import cosine as cos_dist
from scipy.stats import sem
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset

from utils import (
    set_seed, get_device, XRAG, Consolidator,
    load_bartlett, BARTLETT_TXT, recall_prefix,
)

# Import global LoRA configuration
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lora_config import ACTIVE as LORA_CONFIG


@dataclass
class Config:
    # Models (same as memory_simulation.py, bartlett_onestage.py, stories.py)
    llm_name: str = "Hannibal046/xrag-7b"
    retriever_name: str = "Salesforce/SFR-Embedding-Mistral"
    consolidation_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Device/precision
    use_mps: bool = False
    use_4bit: bool = True
    
    # xRAG retriever settings
    retriever_batch_size: int = 16
    retriever_max_length: int = 500
    docs_per_datastore: int = 100
    
    # Consolidation training
    consolidation_epochs: int = 5
    batch_size: int = 1
    learning_rate: float = 5e-5
    max_new_tokens: int = 400
    
    # LoRA settings
    lora_r: int = field(default_factory=lambda: LORA_CONFIG.r)
    lora_alpha: int = field(default_factory=lambda: LORA_CONFIG.alpha)
    lora_dropout: float = field(default_factory=lambda: LORA_CONFIG.dropout)
    target_modules: List[str] = field(default_factory=lambda: LORA_CONFIG.target_modules.copy())
    
    # Sampling for plotted statistics.
    n_samples: int = 10
    sample_temperature: float = 0.5
    
    # Encoding detail level (0 = gist only; 1, 3, ... = gist + n surprise phrases in query, as in memory_simulation)
    encoding_detail_level: int = 0
    
    # Data
    bartlett_path: str = BARTLETT_TXT
    
    # Misc
    seed: int = 42
    output_dir: str = "bartlett_encoding_vs_consolidation"


def chatml_prompt_answer(prompt_cue: str, answer: str) -> str:
    """Format a fixed recall cue and answer for ChatML training."""
    return (f"<s>[INST] {prompt_cue} What happened (in detail)? [/INST] "
            f"{answer.strip()} </s>")


def make_prompt(first_sent: str) -> str:
    """Create recall prompt from the shortened Bartlett cue."""
    return f"<s>[INST] {first_sent} What happened (in detail)? [/INST]"


def prompt_conditioned_target_tokens(tok, prompt: str, target_text: str) -> int:
    """Token count to generate when matching the main Bartlett length guard."""
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(f"{prompt} {target_text}", add_special_tokens=False)["input_ids"]
    return max(1, len(target_ids) - len(prompt_ids))


def make_trainer(*, model, args, train_dataset, tok, callbacks=None) -> Trainer:
    """Build Trainer across Transformers versions with tokenizer API changes."""
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


class EncodingConsolidationExperiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = get_device(cfg.use_mps)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.out_dir = Path(cfg.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        set_seed(self.cfg.seed)
        
        # Load Bartlett story
        try:
            bartlett = load_bartlett(self.cfg.bartlett_path)
        except FileNotFoundError:
            print(f"Bartlett file not found at {self.cfg.bartlett_path}, using built-in story")
            bartlett = BARTLETT_STORY.strip()
        # Shortened cue for recall prompt, matching bartlett_twostage.
        first_sent_raw = recall_prefix()
        first_sent = recall_prefix()
        print(f"Loaded Bartlett story ({len(bartlett)} chars)")
        print(f"Recall prompt seed: {first_sent}")
        
        # Embed original for comparison
        original_emb = self.embedder.encode(bartlett)
        
        # Step 1: Encode using xRAG
        print("\n" + "="*60)
        print("STEP 1: xRAG Encoding")
        print("="*60)
        _, encoded_samples = self._encode_xrag(bartlett, first_sent_raw)
        
        # Step 2: Consolidate via LoRA fine-tuning
        print("\n" + "="*60)
        print("STEP 2: Consolidation via LoRA")
        print("="*60)
        # Each temp-sampled xRAG encoding is consolidated separately so the
        # plotted statistics are matched encoded/consolidated memories.
        consolidated_samples = self._consolidate_samples(encoded_samples, first_sent_raw, bartlett)
        
        # Step 3: Compute distances and plot
        print("\n" + "="*60)
        print("STEP 3: Analysis and Plotting")
        print("="*60)
        self._analyze_and_plot(original_emb, encoded_samples, consolidated_samples, first_sent)
        
        print(f"\nAll outputs saved to: {self.out_dir}")
        
    def _clean_xrag_output(self, output: str) -> str:
        """Strip prompt echoes and 'The answer is:' prefix from xRAG output."""
        if "[/INST]" in output:
            output = output.split("[/INST]")[-1].strip()
        if output.startswith("The answer is:"):
            output = output[len("The answer is:"):].strip()
        return output

    def _generate_xrag(
        self,
        xrag: XRAG,
        prompt: str,
        raw_doc_emb: torch.Tensor,
        *,
        max_new_tokens: int,
        min_new_tokens: int | None = None,
        do_sample: bool,
        temperature: float | None = None,
    ) -> str:
        """Generate from xRAG with explicit length control."""
        emb = raw_doc_emb.unsqueeze(0) if raw_doc_emb.dim() == 1 else raw_doc_emb
        prompt_inputs = xrag.llm_tok(prompt, return_tensors="pt").to(self.device)
        ids = prompt_inputs.input_ids
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=xrag.llm_tok.pad_token_id or xrag.llm_tok.eos_token_id,
            retrieval_embeds=emb.to(self.device),
            no_repeat_ngram_size=0,
        )
        if "attention_mask" in prompt_inputs:
            gen_kwargs["attention_mask"] = prompt_inputs.attention_mask
        if min_new_tokens is not None and min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if do_sample and temperature is not None and temperature > 0:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            out = xrag.llm.generate(ids, **gen_kwargs)
        # XMistral.generate() switches to inputs_embeds when retrieval_embeds
        # are supplied, so its returned sequence is the continuation already.
        # Match the shared XRAG._run_xrag path used by the simple stories code.
        return xrag.llm_tok.decode(out[0], skip_special_tokens=True).strip()

    def _prompt_answer_ds(self, texts: List[str], tok: AutoTokenizer, max_len: int = 2048) -> Dataset:
        """Tokenize fixed prompt-answer examples, masking prompt tokens in labels."""
        answer_marker = "[/INST]"

        def tokenize(examples):
            encoded = {"input_ids": [], "attention_mask": [], "labels": []}
            for text in examples["text"]:
                enc = tok(
                    text,
                    truncation=True,
                    max_length=max_len,
                    padding="max_length",
                )
                labels = list(enc["input_ids"])
                answer_start_char = text.index(answer_marker) + len(answer_marker)
                prompt_ids = tok(
                    text[:answer_start_char],
                    truncation=True,
                    max_length=max_len,
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

        ds = Dataset.from_dict({"text": texts})
        return ds.map(tokenize, batched=True, remove_columns=["text"])

    def _encode_xrag(self, bartlett: str, first_sent: str) -> tuple[str, List[str]]:
        """Encode the Bartlett story using xRAG. Returns (greedy_temp0, all_samples_for_stats)."""
        # Initialize xRAG
        xrag = XRAG(self.cfg, self.device)
        xrag.load()
        
        # Build datastore with just the Bartlett story
        datastore, _, _ = xrag._prepare_datastore([bartlett])
        print("Built xRAG datastore with Bartlett story only")
        
        # Level-0 (gist) query
        question0 = f"{first_sent} What happened (in detail)?"
        prompt0, q_emb0 = xrag._prepare_prompt(question0)
        raw_doc_emb = xrag._nearest_doc_embed(q_emb0, datastore)
        
        # Get level-0 greedy (needed for surprise_phrases when detail_level > 0)
        level0_greedy = xrag._run_xrag(prompt0, raw_doc_emb, self.cfg.max_new_tokens)
        level0_greedy = self._clean_xrag_output(level0_greedy)
        
        # If encoding_detail_level > 0, build query with extra detail phrases (same as memory_simulation)
        if self.cfg.encoding_detail_level > 0:
            surprising = xrag.surprise_phrases(bartlett, level0_greedy, top_k=self.cfg.encoding_detail_level)
            phrases = [ph for ph, _ in surprising]
            if phrases:
                question = (
                    f"{first_sent} What happened (in detail)? "
                    f"Other details to include: {', '.join(phrases)}."
                )
                prompt, q_emb = xrag._prepare_prompt(question)
                raw_doc_emb = xrag._nearest_doc_embed(q_emb, datastore)
                print(f"  Encoding with {self.cfg.encoding_detail_level} detail phrase(s) in query")
            else:
                prompt, raw_doc_emb = prompt0, raw_doc_emb
        else:
            prompt, raw_doc_emb = prompt0, raw_doc_emb
            print("  Encoding at detail level 0 (gist only)")
        
        bartlett_new_tokens = prompt_conditioned_target_tokens(xrag.llm_tok, prompt, bartlett)
        encoded_max_new_tokens = self.cfg.max_new_tokens
        print(
            f"  Length cap: max_new_tokens={encoded_max_new_tokens}; "
            f"no min_new_tokens (Bartlett token count would be {bartlett_new_tokens})"
        )
        
        # 1 greedy (temp=0) sample at chosen detail level
        greedy = self._generate_xrag(
            xrag,
            prompt,
            raw_doc_emb,
            max_new_tokens=encoded_max_new_tokens,
            do_sample=False,
        )
        greedy = self._clean_xrag_output(greedy)
        print(f"  Greedy (temp=0): {greedy[:80]}...")
        
        # Sampled recalls used for panel f statistics.
        n_sampled = self.cfg.n_samples
        sampled = []
        print(f"  Collecting {n_sampled} samples at temp={self.cfg.sample_temperature}...")
        for i in range(n_sampled):
            output = self._generate_xrag(
                xrag,
                prompt,
                raw_doc_emb,
                max_new_tokens=encoded_max_new_tokens,
                do_sample=True,
                temperature=self.cfg.sample_temperature,
            )
            output = self._clean_xrag_output(output)
            sampled.append(output)
            print(f"  Sampled {i+1}/{n_sampled}: {output[:80]}...")
        
        # Release xRAG memory
        xrag.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Save with temp=0 stored separately and encoding_detail_level recorded
        encoded_path = self.out_dir / "encoded_samples.json"
        with open(encoded_path, "w") as f:
            json.dump({
                "greedy_temp0": greedy,
                "sampled": sampled,
                "sample_temperature": self.cfg.sample_temperature,
                "statistics_sample_source": "sampled",
                "encoding_detail_level": self.cfg.encoding_detail_level,
                "length_guard_new_tokens": None,
                "bartlett_token_count_new_tokens": bartlett_new_tokens,
                "generation_max_new_tokens": encoded_max_new_tokens,
                "generation_min_new_tokens": None,
            }, f, indent=2)
        print(f"Saved encoded samples to {encoded_path} (diagnostic greedy + {n_sampled} statistics samples at temp={self.cfg.sample_temperature}, detail_level={self.cfg.encoding_detail_level})")
        
        return greedy, sampled
    
    def _consolidate_samples(self, encoded_samples: List[str], first_sent: str, bartlett: str) -> List[str]:
        """Consolidate each encoded sample separately, returning one recall per sample."""
        consolidated = []
        per_sample = []

        print(f"Consolidating {len(encoded_samples)} encoded samples separately...")
        for sample_idx, encoded_text in enumerate(encoded_samples, start=1):
            print("\n" + "-"*60)
            print(f"Consolidating encoded sample {sample_idx}/{len(encoded_samples)}")
            result = self._consolidate_one(encoded_text, first_sent, bartlett, sample_idx)
            consolidated.append(result["sampled_temp"])
            per_sample.append({
                "sample_index": sample_idx,
                "encoded_sample": encoded_text,
                **result,
            })

        consolidated_path = self.out_dir / "consolidated_samples.json"
        with open(consolidated_path, "w") as f:
            json.dump({
                "sampled": consolidated,
                "sample_temperature": self.cfg.sample_temperature,
                "statistics_sample_source": "one_consolidated_recall_per_encoded_sample",
                "training_prompt": make_prompt(first_sent),
                "per_encoded_sample": per_sample,
                # Keep this scalar key for older plotting fallbacks that expect
                # greedy_temp0 to be a string, not a list.
                "greedy_temp0": per_sample[0]["greedy_temp0"] if per_sample else "",
                "greedy_temp0_by_encoded_sample": [
                    item["greedy_temp0"] for item in per_sample
                ],
                "length_guard_new_tokens": (
                    per_sample[0]["length_guard_new_tokens"] if per_sample else None
                ),
            }, f, indent=2)
        print(
            f"Saved consolidated samples to {consolidated_path} "
            f"({len(consolidated)} matched consolidated recalls at temp={self.cfg.sample_temperature})"
        )

        return consolidated

    def _consolidate_one(self, encoded_text: str, first_sent: str, bartlett: str, sample_idx: int) -> Dict[str, object]:
        """Consolidate one encoded text via LoRA fine-tuning, then collect one sample."""
        # Initialize consolidator
        set_seed(self.cfg.seed + sample_idx)
        consolidator = Consolidator(self.cfg, self.device)
        
        # Build model
        print("Building consolidation model...")
        model = consolidator.build_model()
        
        # Build tokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        
        # Train with the same fixed Bartlett cue used at recall time. The answer
        # is the xRAG-encoded recall, not text split at its first sentence.
        train_text = chatml_prompt_answer(first_sent, encoded_text)
        train_texts = [train_text] * 5  # Repeat for stability
        ds = self._prompt_answer_ds(train_texts, tok, max_len=2048)
        
        # Training arguments
        train_dir = self.out_dir / "consolidation_checkpoints" / f"encoded_{sample_idx:02d}"
        use_fp16 = torch.cuda.is_available()
        
        t_args = TrainingArguments(
            output_dir=str(train_dir),
            seed=self.cfg.seed,
            num_train_epochs=self.cfg.consolidation_epochs,
            per_device_train_batch_size=self.cfg.batch_size,
            learning_rate=self.cfg.learning_rate,
            fp16=use_fp16,
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            label_names=["labels"],
        )
        
        # Simple callback to track progress
        class ProgressCallback(TrainerCallback):
            def on_epoch_end(self, args, state, control, **kwargs):
                print(f"  Completed epoch {int(state.epoch)}/{args.num_train_epochs}")
        
        trainer = make_trainer(
            model=model,
            args=t_args,
            train_dataset=ds,
            tok=tok,
            callbacks=[ProgressCallback()],
        )
        
        print(f"Training for {self.cfg.consolidation_epochs} epochs...")
        trainer.train()
        
        # Collect one consolidated recall for this encoded memory. The greedy
        # output is retained for diagnostics, but panel f statistics use the
        # matched temp-sampled recall.
        prompt = make_prompt(first_sent)
        model.eval()
        target_new_tokens = prompt_conditioned_target_tokens(tok, prompt, bartlett)
        print(f"  Length guard: min/max_new_tokens={target_new_tokens} (Bartlett token count)")
        
        prompt_inputs = tok(prompt, return_tensors="pt").to(model.device)
        ids = prompt_inputs.input_ids
        attention_mask = prompt_inputs.get("attention_mask")
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=target_new_tokens,
                min_new_tokens=target_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_tokens = out[0][ids.shape[1]:]
        greedy = tok.decode(new_tokens, skip_special_tokens=True).strip()
        if greedy.startswith("The answer is:"):
            greedy = greedy[len("The answer is:"):].strip()
        print(f"  Greedy (temp=0): {greedy[:80]}...")
        
        with torch.no_grad():
            out = model.generate(
                ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=self.cfg.sample_temperature,
                max_new_tokens=target_new_tokens,
                min_new_tokens=target_new_tokens,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        new_tokens = out[0][ids.shape[1]:]
        sampled = tok.decode(new_tokens, skip_special_tokens=True).strip()
        if sampled.startswith("The answer is:"):
            sampled = sampled[len("The answer is:"):].strip()
        print(f"  Sampled temp={self.cfg.sample_temperature}: {sampled[:80]}...")
        
        # Clean up
        del model, trainer, consolidator, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "greedy_temp0": greedy,
            "sampled_temp": sampled,
            "sample_temperature": self.cfg.sample_temperature,
            "length_guard_new_tokens": target_new_tokens,
        }
    
    def _analyze_and_plot(
        self,
        original_emb: np.ndarray,
        encoded_samples: List[str],
        consolidated_samples: List[str],
        first_sent: str,
    ):
        """Compute cosine distances and create bar chart.
        Outputs are expected to contain the full recalled story.
        """
        def compute_distances(samples: List[str]) -> List[float]:
            distances = []
            for s in samples:
                emb = self.embedder.encode(s)
                dist = cos_dist(emb, original_emb)
                distances.append(float(dist))
            return distances
        
        encoded_dists = compute_distances(encoded_samples)
        consolidated_dists = compute_distances(consolidated_samples)
        if len(encoded_dists) != len(consolidated_dists):
            raise ValueError(
                f"Encoded/consolidated sample count mismatch: "
                f"{len(encoded_dists)} vs {len(consolidated_dists)}"
            )
        
        # Compute statistics
        encoded_mean = np.mean(encoded_dists)
        encoded_sem = sem(encoded_dists)
        consolidated_mean = np.mean(consolidated_dists)
        consolidated_sem = sem(consolidated_dists)
        
        print(f"\nEncoded: mean={encoded_mean:.4f}, SEM={encoded_sem:.4f}")
        print(f"Consolidated: mean={consolidated_mean:.4f}, SEM={consolidated_sem:.4f}")
        
        # Save statistics
        stats = {
            "n_samples": self.cfg.n_samples,
            "sample_temperature": self.cfg.sample_temperature,
            "statistics_sample_source": "matched_encoded_and_consolidated_samples",
            "paired_n": len(encoded_dists),
            "paired_distances": [
                {
                    "sample_index": i,
                    "encoded_distance": encoded_dist,
                    "consolidated_distance": consolidated_dist,
                }
                for i, (encoded_dist, consolidated_dist) in enumerate(
                    zip(encoded_dists, consolidated_dists), start=1
                )
            ],
            "encoded": {
                "distances": encoded_dists,
                "mean": encoded_mean,
                "sem": encoded_sem,
            },
            "consolidated": {
                "distances": consolidated_dists,
                "mean": consolidated_mean,
                "sem": consolidated_sem,
            },
        }
        stats_path = self.out_dir / "statistics.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")
        
        # Create bar chart
        self._create_bar_chart(encoded_mean, encoded_sem, consolidated_mean, consolidated_sem)
        
    def _create_bar_chart(
        self,
        encoded_mean: float,
        encoded_sem: float,
        consolidated_mean: float,
        consolidated_sem: float,
    ):
        """Create bar chart with Encoded vs Consolidated."""
        # Set up figure
        plt.rcParams.update({
            "font.size": 12,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        })
        
        fig, ax = plt.subplots(figsize=(4, 4))
        
        labels = ["Encoded", "Consolidated"]
        means = [encoded_mean, consolidated_mean]
        sems = [encoded_sem, consolidated_sem]
        colors = ["#4C78A8", "#F58518"]  # Blue for encoded, orange for consolidated
        
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Cosine distance")
        ax.set_title("Bartlett Story Recall:\nEncoded vs Consolidated")
        
        # Set y-axis limits with some padding
        y_max = max(means) + max(sems) + 0.05
        ax.set_ylim(0, min(y_max, 1.0))
        
        ax.grid(True, alpha=0.3, axis="y")
        
        # Caption: all statistics samples are drawn at the configured temp.
        ax.text(0.98, 0.02,
                f"{self.cfg.n_samples} samples at temp={self.cfg.sample_temperature}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="gray")
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.out_dir / "encoding_vs_consolidation_bar.png"
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved bar chart to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Bartlett Encoding vs Consolidation Comparison")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples per condition")
    parser.add_argument("--consolidation_epochs", type=int, default=5, help="Epochs for consolidation training")
    parser.add_argument("--bartlett_path", type=str, default=BARTLETT_TXT, help="Path to Bartlett story")
    parser.add_argument("--output_dir", type=str, default="bartlett_encoding_vs_consolidation", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample_temperature", type=float, default=0.5, help="Temperature for non-greedy samples")
    parser.add_argument("--encoding_detail_level", type=int, default=0,
                        help="Encoding detail level: 0 = gist only; 1, 3, ... = gist + n surprise phrases in query")
    args = parser.parse_args()
    
    cfg = Config(
        n_samples=args.n_samples,
        consolidation_epochs=args.consolidation_epochs,
        bartlett_path=args.bartlett_path,
        output_dir=args.output_dir,
        seed=args.seed,
        sample_temperature=args.sample_temperature,
        encoding_detail_level=args.encoding_detail_level,
    )
    
    experiment = EncodingConsolidationExperiment(cfg)
    experiment.run()


if __name__ == "__main__":
    main()
