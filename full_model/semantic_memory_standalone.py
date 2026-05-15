"""
Standalone Semantic Memory Test

Runs the semantic memory analysis using saved models from memory_simulation.py.
Tests stories that have entries in semantic_questions.jsonl.

Usage:
    python semantic_memory_standalone.py
    python semantic_memory_standalone.py --output_dir output
"""
from __future__ import annotations
import argparse
import json
import gc
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from scipy.stats import sem
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data"


def _data_file(name: str) -> Path:
    """Resolve data files from the current run directory or this script's data directory."""
    direct_path = Path(name)
    if direct_path.exists():
        return direct_path
    cwd_data_path = Path("data") / name
    if cwd_data_path.exists():
        return cwd_data_path
    return DEFAULT_DATA_DIR / name



# ============================================================================
# SEMANTIC MEMORY TEST
# ============================================================================

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
    IDENTICAL to memory_simulation.py
    """
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    results = []
    correct_count = 0
    
    model.eval()
    
    for i, (story, q) in enumerate(zip(stories, questions)):
        # Format question prompt - include story context
        context = q.get("context", story.split(".")[0])
        context = q.get("context", story.split(".")[0].strip() + ".")
        prompt = f"<s>[INST] Remember the story in which '{context}' {q['question']} Answer with a word or short phrase. [/INST]"
        
        # Generate answer
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            out = model.generate(
                ids, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
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


# ============================================================================
# MAIN STANDALONE LOGIC
# ============================================================================

@dataclass
class Config:
    output_dir: str = "output"
    consolidation_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    use_4bit: bool = True
    seed: int = 123
    max_new_tokens: int = 30
    semantic_questions_file: str | None = None


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


class SemanticMemoryAnalyzer:
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = self._auto_device()
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.semantic_questions: Dict[str, Dict] = {}
        self._maybe_load_external_semantic_questions()
    
    def _auto_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _maybe_load_external_semantic_questions(self):
        candidates: list[Path] = []
        if self.cfg.semantic_questions_file:
            candidates.append(Path(self.cfg.semantic_questions_file))
        # Convenience defaults when running from the repo root or full_model/.
        candidates.append(Path("semantic_questions.jsonl"))
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
    
    def load_stories_with_questions(self) -> List[Tuple[str, Dict]]:
        """
        Load only stories that have semantic questions.
        Returns list of (story, question_dict) tuples.
        """
        df = pd.read_csv(_data_file("stories_train.csv"))
        df["combined"] = (
            df[[f"sentence{i}" for i in range(1, 6)]]
            .astype(str)
            .agg(" ".join, axis=1)
        )
        all_stories = df["combined"].tolist()
        
        # Only include stories with configured semantic questions.
        result = []
        for story in all_stories:
            for story_start, qa in self.semantic_questions.items():
                if story.startswith(story_start):
                    first_sentence = story.split(".")[0].strip() + "."
                    question = {
                        "question": qa["question"],
                        "correct": qa["correct"],
                        "wrong": qa["wrong"],
                        "entity_type": qa.get("entity_type", "SEMANTIC_QA"),
                        "context": qa.get("context", first_sentence),
                    }
                    result.append((story, question))
                    break
        
        print(f"[Data] Found {len(result)} stories with semantic questions")
        return result
    
    def load_base_model(self):
        """Load the base model WITHOUT any LoRA adapter (for baseline testing)."""
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        
        if self.cfg.use_4bit:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                quantization_config=qcfg,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        
        model.eval()
        return model, tok
    
    def load_model(self, adapter_dir: Path):
        """Load the consolidated model with LoRA adapter."""
        tok = AutoTokenizer.from_pretrained(self.cfg.consolidation_model)
        if tok.pad_token is None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        
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
            base = AutoModelForCausalLM.from_pretrained(
                self.cfg.consolidation_model,
                torch_dtype=torch.bfloat16,
            ).to(self.device)
        
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        if not hasattr(model, "hf_device_map"):
            model = model.to(self.device)
        model.eval()
        
        return model, tok
    
    def find_checkpoints(self, model_dir: Path) -> List[Path]:
        """Find all checkpoint directories."""
        checkpoints = []
        for p in model_dir.iterdir():
            if p.is_dir() and p.name.startswith("checkpoint-"):
                checkpoints.append(p)
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return checkpoints
    
    def run(self):
        """Run semantic memory analysis on saved models."""
        output_dir = Path(self.cfg.output_dir)
        model_dir = output_dir / "models" / "consolidation"
        
        if not model_dir.exists():
            print(f"ERROR: No saved model found at {model_dir}")
            print("Run memory_simulation.py first to train the model.")
            return
        
        # Load only stories with configured semantic questions.
        print("Loading stories with semantic questions...")
        story_question_pairs = self.load_stories_with_questions()
        stories = [s for s, q in story_question_pairs]
        questions = [q for s, q in story_question_pairs]
        print(f"Testing {len(stories)} stories with semantic questions")
        
        # Find checkpoints for per-epoch analysis
        checkpoints = self.find_checkpoints(model_dir)
        print(f"Found {len(checkpoints)} checkpoints")
        
        results = {
            "questions": questions,
            "baseline": None,
            "per_epoch": [],
            "final": None,
        }
        
        # Test baseline (epoch 0) - base model without any fine-tuning
        print("\n[Epoch 0] Loading base model (no fine-tuning)...")
        base_model, tok = self.load_base_model()
        
        print("[Epoch 0] Testing baseline semantic memory...")
        baseline_result = semantic_memory_test(
            base_model, tok, stories, questions,
            self.device, self.cfg.max_new_tokens
        )
        baseline_result["epoch"] = 0
        results["baseline"] = baseline_result
        results["per_epoch"].append(baseline_result)
        print(f"[Epoch 0] Baseline accuracy: {baseline_result['accuracy']:.1%}")
        
        # Cleanup base model
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Test each checkpoint
        if checkpoints:
            for ckpt in checkpoints:
                epoch = int(ckpt.name.split("-")[1])
                print(f"\n[Epoch {epoch}] Loading model from {ckpt.name}...")
                
                model, tok = self.load_model(ckpt)
                
                print(f"[Epoch {epoch}] Testing semantic memory...")
                sem_result = semantic_memory_test(
                    model, tok, stories, questions,
                    self.device, self.cfg.max_new_tokens
                )
                sem_result["epoch"] = epoch
                results["per_epoch"].append(sem_result)
                print(f"[Epoch {epoch}] Accuracy: {sem_result['accuracy']:.1%}")
                
                # Cleanup
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        # Test final model
        print(f"\n[Final] Loading final model...")
        model, tok = self.load_model(model_dir)
        
        print(f"[Final] Testing semantic memory...")
        final_result = semantic_memory_test(
            model, tok, stories, questions,
            self.device, self.cfg.max_new_tokens
        )
        results["final"] = final_result
        print(f"[Final] Accuracy: {final_result['accuracy']:.1%}")
        
        # Save results
        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True, parents=True)
        
        with open(data_dir / "semantic_memory_standalone.pkl", "wb") as f:
            pickle.dump(results, f)
        print(f"\nResults saved to {data_dir / 'semantic_memory_standalone.pkl'}")
        
        # Generate plots
        self._plot_results(results, output_dir / "plots")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _plot_results(self, results: Dict, plots_dir: Path):
        """Generate plots for semantic memory analysis."""
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        if results["per_epoch"]:
            epochs = [r["epoch"] for r in results["per_epoch"]]
            accuracies = [r["accuracy"] for r in results["per_epoch"]]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, accuracies, marker='o', linewidth=2, markersize=8, color='#6a00a8')
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel("Semantic Memory Accuracy", fontsize=12)
            ax.set_title("Semantic Memory During Consolidation", fontsize=14)
            ax.set_ylim(0, 1.05)
            ax.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Chance (1/3)")
            if 0 in epochs:
                ax.axvline(0, color="blue", linestyle=":", alpha=0.3, label="Baseline (pre-training)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            out_path = plots_dir / "semantic_memory_standalone.png"
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            print(f"[Plot] Saved {out_path}")
    
    def _print_summary(self, results: Dict):
        """Print a summary of results."""
        print("\n" + "="*60)
        print("SEMANTIC MEMORY SUMMARY")
        print("="*60)
        
        if results["per_epoch"]:
            # Embedding-based accuracy
            print("\nEmbedding-based accuracy by epoch:")
            for r in results["per_epoch"]:
                print(f"  Epoch {r['epoch']}: {r['accuracy']:.1%} ({r['correct_count']}/{r['total']})")
            
            # String-based accuracy
            print("\nString-match accuracy by epoch:")
            for r in results["per_epoch"]:
                str_count = r.get("correct_count_string", sum(1 for q in r["per_question"] if q.get("is_correct_string", False)))
                str_acc = str_count / r["total"] if r["total"] else 0
                print(f"  Epoch {r['epoch']}: {str_acc:.1%} ({str_count}/{r['total']})")
            
            # Perplexity-based accuracy
            print("\nPerplexity-based accuracy by epoch:")
            for r in results["per_epoch"]:
                ppl_count = r.get("correct_count_perplexity", sum(1 for q in r["per_question"] if q.get("is_correct_perplexity", False)))
                ppl_acc = ppl_count / r["total"] if r["total"] else 0
                print(f"  Epoch {r['epoch']}: {ppl_acc:.1%} ({ppl_count}/{r['total']})")
        
        if results["final"]:
            print(f"\nFinal model:")
            print(f"  Embedding: {results['final']['accuracy']:.1%}")
            print(f"  String:    {results['final'].get('accuracy_string', 0):.1%}")
            print(f"  Perplexity: {results['final'].get('accuracy_perplexity', 0):.1%}")
        
        # Show some example Q&A
        if results["final"] and results["final"]["per_question"]:
            print("\nExample Q&A (first 5):")
            for r in results["final"]["per_question"][:5]:
                emb = "✓" if r["is_correct_embedding"] else "✗"
                strg = "✓" if r.get("is_correct_string", False) else "✗"
                ppl = "✓" if r.get("is_correct_perplexity", False) else "✗"
                print(f"  [Emb:{emb} Str:{strg} Ppl:{ppl}] Q: {r['question'][:50]}...")
                print(f"      Model: '{r['model_answer'][:40]}' | Correct: '{r['correct_answer']}'")


def main():
    parser = argparse.ArgumentParser(description="Standalone semantic memory analysis (semantic questions JSONL optional)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--semantic_questions_file", type=str, default=None, help="Optional semantic_questions.jsonl file.")
    
    args = parser.parse_args()
    
    cfg = Config(
        output_dir=args.output_dir,
        seed=args.seed,
        use_4bit=not args.no_4bit,
        max_new_tokens=args.max_new_tokens,
        semantic_questions_file=args.semantic_questions_file,
    )
    
    analyzer = SemanticMemoryAnalyzer(cfg)
    analyzer.run()


if __name__ == "__main__":
    main()
