# ────────────────────────────────────────────────────────────────────────────
# File: gpt2_with_projector.py
# Description: GPT-2 language model augmented with a projector that maps
#              retriever embeddings into the model’s hidden space.
#              Fix #1 applied – correct boolean-mask indexing for xRAG tokens.
# ────────────────────────────────────────────────────────────────────────────

import re
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class GPT2WithProjectorConfig(GPT2Config):
    """Adds projector-related parameters to the regular GPT-2 config."""

    def __init__(
        self,
        projector_type: str = "mlp2x_gelu",
        retriever_hidden_size: int = 128,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size


# ---------------------------------------------------------------------------
# Projector module
# ---------------------------------------------------------------------------

class Projector(nn.Module):
    """Lightweight MLP that maps a retriever vector → GPT-2 embedding space."""

    def __init__(self, config: GPT2WithProjectorConfig):
        super().__init__()
        projector_type = config.projector_type
        mlp_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)

        if mlp_match:  # MLP depth specified, e.g. "mlp2x_gelu"
            depth = int(mlp_match.group(1))
            layers = []
            for i in range(depth):
                if i > 0:
                    layers.append(nn.GELU())
                in_size = (config.retriever_hidden_size if i == 0 else config.n_embd)
                linear = nn.Linear(in_size, config.n_embd)
                nn.init.xavier_uniform_(linear.weight, gain=0.1)
                nn.init.zeros_(linear.bias)
                layers.append(linear)
            self.projector = nn.Sequential(*layers)
        else:  # simple linear
            self.projector = nn.Linear(config.retriever_hidden_size, config.n_embd)
            nn.init.xavier_uniform_(self.projector.weight, gain=0.1)
            nn.init.zeros_(self.projector.bias)

        self.norm = nn.LayerNorm(config.n_embd, eps=1e-6)

    # ---------------------------------------------------------------------
    # Forward pass with defensive checks – catches NaN/Inf early.
    # ---------------------------------------------------------------------
    def forward(self, context_embedding: torch.Tensor) -> torch.Tensor:
        orig_dtype = context_embedding.dtype

        # Replace invalid numbers *before* they propagate.
        if torch.isnan(context_embedding).any() or torch.isinf(context_embedding).any():
            context_embedding = torch.where(
                torch.isnan(context_embedding) | torch.isinf(context_embedding),
                torch.zeros_like(context_embedding),
                context_embedding,
            )

        # vec = torch.nn.functional.normalize(context_embedding, dim=-1)
        # out = self.projector(vec.float())
        
        out = self.projector(context_embedding.float())
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.where(
                torch.isnan(out) | torch.isinf(out),
                torch.randn_like(out) * 1e-2,
                out,
            )

        out = self.norm(out)
        if torch.isnan(out).any() or torch.isinf(out).any():
            out = torch.where(
                torch.isnan(out) | torch.isinf(out),
                torch.randn_like(out) * 1e-2,
                out,
            )
        return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# GPT-2 wrapper that understands xRAG retrieval vectors
# ---------------------------------------------------------------------------

class GPT2WithProjector(GPT2LMHeadModel):
    def __init__(self, config: GPT2WithProjectorConfig):
        super().__init__(config)
        if getattr(config, "retriever_hidden_size", 0) > 0:
            self.projector = Projector(config)
            self.retriever_hidden_size = config.retriever_hidden_size
        self.post_init()
        self.xrag_token_id: Optional[int] = None  # set externally

    # ---------------------------------------------------------------------
    # Helper
    # ---------------------------------------------------------------------
    def set_xrag_token_id(self, token_id: int) -> None:
        self.xrag_token_id = token_id

    # ---------------------------------------------------------------------
    # Prepare combined input embeddings (text + retrieval)
    #   Fix #1: use boolean mask rather than torch.nonzero indexing.
    # ---------------------------------------------------------------------
    def prepare_inputs_embeds(self, input_ids, retrieval_embeds):
        """
        Returns a fresh tensor whose value depends on
        (a) frozen token embeddings  and
        (b) projector(retrieval_embeds).

        No in-place writes → gradients flow back to the projector.
        """
        # 1) frozen token embeddings  ---------------------------------------
        base_embeds = self.transformer.wte(input_ids)  # (B, T, d); requires_grad = False

        # no retrieval?  just return them
        if retrieval_embeds is None:
            return base_embeds

        # 2) flatten and project retrieval embeddings  ----------------------
        retrieval_embeds = retrieval_embeds.view(-1, self.retriever_hidden_size)
        mask = (input_ids == self.xrag_token_id)          # (B, T) bool

        # Handle count mismatch exactly as before
        num_xrag = mask.sum().item()
        if num_xrag == 0:
            return base_embeds                            # nothing to replace

        if num_xrag != retrieval_embeds.size(0):
            min_n = min(num_xrag, retrieval_embeds.size(0))
            mask_flat = mask.view(-1).nonzero().squeeze()[:min_n]
            mask = torch.zeros_like(mask, dtype=torch.bool).view(-1)
            mask[mask_flat] = True
            mask = mask.view(*input_ids.shape)
            retrieval_embeds = retrieval_embeds[:min_n]

        projected = self.projector(retrieval_embeds)      # (num_xrag, d)

        inputs_embeds = base_embeds.clone()
        inputs_embeds[mask] = projected        # clone() now carries grad

        return inputs_embeds


    # ---------------------------------------------------------------------
    # Forward – mostly the same, but calls prepare_inputs_embeds.
    # ---------------------------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        inputs_embeds = kwargs.pop("inputs_embeds", None)
        at_generation_start = inputs_embeds is not None

        if not at_generation_start and retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            input_ids = None  # we now pass embeddings instead of ids

        outputs = super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

        if torch.isnan(outputs.logits).any() or torch.isinf(outputs.logits).any():
            outputs.logits = torch.where(
                torch.isnan(outputs.logits) | torch.isinf(outputs.logits),
                torch.zeros_like(outputs.logits),
                outputs.logits,
            )
        return outputs

    # ---------------------------------------------------------------------
    # Generation wrapper – unchanged except for mask fix indirectly.
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        retrieval_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported for generate")

        if retrieval_embeds is not None:
            inputs_embeds = self.prepare_inputs_embeds(input_ids, retrieval_embeds)
            return super().generate(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
        else:
            return super().generate(
                attention_mask=attention_mask,
                input_ids=input_ids,
                **kwargs,
            )

