from __future__ import annotations
import gc
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
nlp = spacy.load("en_core_web_sm")

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

    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", 
                                                               "gate_proj", "up_proj", "down_proj",])

    # number of stories to compress / encode / consolidate
    num_stories: int = 500 #200
    detail_levels: List[int] = field(default_factory=lambda: [0, 1, 3])
    seed: int = 123

    log_recall_every_n_epochs: int = 1

    # number of new stories in each phase of forgetting
    memory_set_size: int = 50
    epochs_per_set: int = 20
    num_forgetting_phases: int = 8 

    output_dir: str = "output"

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

        self.results: Dict[str, Dict] = {
            "encoding": {}, "consolidation": {},
            "forgetting": {}, "hippocorpus_analysis": {}
        }

        self.llm = None; self.llm_tok = None; self.retriever = None; self.ret_tok = None

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

        df = pd.read_csv("stories_train.csv")
        df["combined"] = (
            df[[f"sentence{i}" for i in range(1, 6)]]
            .astype(str)
            .agg(" ".join, axis=1)
        )
    
        all_stories = df["combined"].tolist()
        random.shuffle(all_stories)
    
        # primary slice for encoding / consolidation
        stories_subset = all_stories[: self.cfg.num_stories]
    
        # keep the whole corpus for future forgetting
        self._full_story_pool = all_stories
    
        hippo = pd.read_csv("hippoCorpusV2.csv")
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
        ids = self.llm_tok(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            out = self.llm.generate(
                ids,
                do_sample=False,
                max_new_tokens=self.cfg.max_new_tokens,
                pad_token_id=self.llm_tok.pad_token_id,
            )
    
        gen_ids = out[0][ids.shape[1]:]          # keep only the continuation
        return self.llm_tok.decode(gen_ids, skip_special_tokens=True).strip()



    def simulate_encoding(self, stories: List[str]) -> Dict:

        out_file = Path(self.cfg.output_dir, "data", "recalled_stories.pkl")

        # If cached file exists, load and return it
        if out_file.exists():
            print(f"Found cached encoding data at {out_file}, loading it…")
            with open(out_file, "rb") as f:
                self.results["encoding"] = pickle.load(f)
            return self.results["encoding"]

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
    
        class RecallTracker(TrainerCallback):
            def __init__(self, outer):
                self.outer = outer
                self.epoch_recalls, self.dist_orig, self.dist_enc = [], [], []
    
            def on_epoch_end(self, args, state, control, **_):
                if ((state.epoch + 1) %
                        self.outer.cfg.log_recall_every_n_epochs) != 0:
                    return
    
                rec, d_o, d_e = [], [], []
                for orig, enc in zip(originals, encoded):
                    prompt = (f"<s>[INST] {orig.split('.')[0]}."
                              f" What happened (in detail)? [/INST]")
                    ids = tok(prompt, return_tensors="pt").input_ids.to(self.outer.device)
                    with torch.no_grad():
                        out = model.generate(ids,
                                             max_new_tokens=self.outer.cfg.max_new_tokens,
                                             do_sample=False)

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
    
        tracker = RecallTracker(self)
    
        t_args = TrainingArguments(
            output_dir       = Path(self.cfg.output_dir, "models", "consolidation"),
            seed             = self.cfg.seed,
            num_train_epochs = self.cfg.num_epochs,
            per_device_train_batch_size = self.cfg.batch_size,
            learning_rate    = self.cfg.learning_rate,
            fp16             = True,
            save_strategy    = "epoch",
            logging_steps    = self.cfg.print_steps,
            evaluation_strategy = "epoch",
            report_to        = [],
            dataloader_pin_memory=False,
        )
        trainer = Trainer(model=model, args=t_args,
                          train_dataset=train_ds, eval_dataset=eval_ds,
                          tokenizer=tok, callbacks=[tracker])
    
        if not hasattr(model, "hf_device_map"):
            model.to(self.device)
    
        print("→ Consolidation fine-tuning…")
        trainer.train()
    
        save_dir = Path(self.cfg.output_dir, "models", "consolidation")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)

        self.results["consolidation"] = {
            "epoch_recalls"   : tracker.epoch_recalls,
            "epoch_dist_orig" : tracker.dist_orig,
            "epoch_dist_enc"  : tracker.dist_enc,
        }
        with open(Path(self.cfg.output_dir, "data", "consolidation_recall.pkl"), "wb") as f:
            pickle.dump(self.results["consolidation"], f)
    
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
    
        model = PeftModel.from_pretrained(base, str(adapter_dir)).to(self.device).eval()
    
        cos   = lambda a, b: cos_dist(a, b)
        embed = self.embedder.encode
    
        def _recall_first():
            gens, dists = [], []
            for s in first_set:
                prompt = f"<s>[INST] {s.split('.')[0]}. What happened (in detail)? [/INST]"
                ids = tok(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=self.cfg.max_new_tokens,
                                         do_sample=False)

                # ids           -> tensor containing the prompt tokens
                # out[0]        -> tensor containing  prompt  +  generated tokens
                new_tokens = out[0][ids.shape[1]:]
                gen        = tok.decode(new_tokens,
                                        skip_special_tokens=True).strip()

                gen = s.split('.')[0] + '. ' + gen
                gens.append(gen)
                dists.append(cos(embed(gen), embed(s)))
            return gens, dists
    
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
                output_dir       = Path(cfg.output_dir, "models", "forget_tmp"),
                num_train_epochs = cfg.epochs_per_set,
                per_device_train_batch_size = cfg.batch_size,
                learning_rate    = cfg.learning_rate,
                fp16             = True,
                logging_steps    = cfg.print_steps,
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
    
        # baseline (episode 0)
        g0, d0 = _recall_first()
        recalls.append(g0); distances.append(d0)
    
        for epi in range(1, self.cfg.num_forgetting_phases + 1):
    
            if len(pool) < self.cfg.memory_set_size:
                print(f"⚠︎  Pool exhausted after {epi-1} phases."); break
    
            chunk = random.sample(pool, k=self.cfg.memory_set_size)
            for s in chunk: pool.remove(s)          # ensure no repeats
    
            print(f"Episode {epi}: training on {len(chunk)} new stories …")
            losses.append(_finetune(chunk))
    
            g, d = _recall_first()
            recalls.append(g); distances.append(d)
    
        out = {"losses": losses, "distances": distances, "recalls": recalls}
        data_dir = Path(self.cfg.output_dir, "data"); data_dir.mkdir(exist_ok=True, parents=True)
        pickle.dump(out, open(data_dir / "forgetting_multi.pkl", "wb"))
    
        self.results["forgetting"] = out
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
        ids = self.llm_tok(prompt, return_tensors="pt").input_ids.to(self.device)
        out = self.llm.generate(ids, do_sample=False, max_new_tokens=self.cfg.max_new_tokens, pad_token_id=self.llm_tok.pad_token_id, retrieval_embeds=emb.unsqueeze(0))
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
        self.simulate_consolidation(
            enc["recalled_stories"][0],   # use 0-detail variant
            stories
        )
    
        # 4. Forgetting
        self.simulate_chunked_forgetting(stories)
    
        print("Simulation complete – results in", self.cfg.output_dir)


if __name__ == "__main__":
    cfg = Config()
    sim = MemorySimulator(cfg)
    sim.run()
