"""
Episodic memory using the dual MHN architecture.

This is a simpler alternative to the EDEN-based episodic memory.
It stores episodes as (xRAG key, details sequence) and retrieves
using the two-MHN design:

    1. Autoassociative MHN: finds the correct episode and denoises state
    2. Heteroassociative MHN: steps through the details sequence

Episode structure stored:
    [xRAG_0, detail_0_0, detail_0_1, ..., xRAG_1, detail_1_0, detail_1_1, ...]
    
Transitions stored:
    xRAG_i → detail_i_0, detail_i_0 → detail_i_1, ..., detail_i_T → detail_i_T (self-loop at end)
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

try:
    from dual_mhn import DualMHN, DualMHNParams, AutoassociativeMHN, HeteroassociativeMHN
except ModuleNotFoundError:
    from scripts.dual_mhn import DualMHN, DualMHNParams, AutoassociativeMHN, HeteroassociativeMHN

try:
    from hpc_utils import convert_string_pattern
except ModuleNotFoundError:
    from scripts.hpc_utils import convert_string_pattern


ALL_CHARS = string.ascii_letters + string.digits + string.punctuation + " []"


@dataclass
class EpisodicDualParams:
    """Parameters for dual-MHN episodic memory."""
    # Decaying context encoding
    decay_rate: float = 0.9
    # MHN parameters
    beta_auto: float = 50.0
    beta_hetero: float = 50.0
    n_auto_iters: int = 3
    n_hetero_iters: int = 1
    # Use hard (argmax) attention for heteroassociative transitions.
    # This avoids mixture states when keys are highly correlated (common with decaying context).
    hard_hetero: bool = True
    denoise_after_hetero: bool = True
    n_post_denoise_iters: int = 2
    # Random projection seed
    proj_seed: int = 0


class EpisodicMemoryDual:
    """
    Episodic memory using dual MHN architecture.
    
    Store episodes as (xRAG, details_string) pairs.
    Retrieve details from a noisy xRAG query.
    """
    
    def __init__(self, d: int, *, params: EpisodicDualParams | None = None):
        self.d = d
        self.params = params or EpisodicDualParams()
        
        # Random projection from char space to xRAG space
        rng = np.random.default_rng(self.params.proj_seed)
        A = rng.normal(size=(d, len(ALL_CHARS))).astype(np.float32)
        Q, _ = np.linalg.qr(A, mode="reduced")
        self.R = Q.astype(np.float32)  # (d, |chars|) orthonormal columns
        
        # Storage
        self._patterns: List[np.ndarray] = []       # all timestep patterns (d,)
        self._transitions: List[Tuple[int, int]] = []  # (from_idx, to_idx) pairs
        self._episode_starts: List[int] = []        # index of each episode's xRAG
        self._episode_ranges: List[Tuple[int, int]] = []  # (start, end) for each episode
        
        # Built networks (populated by finalize())
        self._auto_mhn: AutoassociativeMHN | None = None
        self._hetero_mhn: HeteroassociativeMHN | None = None
        self._finalized = False
    
    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        v = v.astype(np.float32).reshape(-1)
        return v / (np.linalg.norm(v) + 1e-12)
    
    def _encode_details(self, details: str, xrag_context: np.ndarray) -> List[np.ndarray]:
        """
        Encode details string into a sequence of (d,) vectors.
        
        IMPORTANT: Each vector is mixed with the xRAG context so that patterns
        become episode-specific. Without this, '[' at the start of episode 0
        would be identical to '[' at the start of episode 1.
        
        Args:
            details: string to encode
            xrag_context: the episode's xRAG vector (d,) to mix in
        """
        # Sanitize
        details = details.replace("[", " ").replace("]", " ")
        details = details.replace("\n", " ").replace("\t", " ")
        allowed = set(ALL_CHARS) - set("[]")
        details = "".join(ch if ch in allowed else " " for ch in details)
        details = " ".join(details.split())
        
        # Wrap with start/end tokens
        seq = "[" + details + "]"
        pats = convert_string_pattern(seq, decay_rate=self.params.decay_rate)  # (T, |chars|, 1)
        
        # Project xRAG to orthogonal subspace (so it doesn't bias character decoding)
        xrag_orth = xrag_context - self.R @ (self.R.T @ xrag_context)
        xrag_orth = xrag_orth / (np.linalg.norm(xrag_orth) + 1e-12)
        
        # Mix ratio: how much xRAG context vs character info
        context_weight = 0.3  # 30% xRAG context, 70% character trace
        
        vecs = []
        for t in range(pats.shape[0]):
            p = pats[t].reshape(-1)  # (|chars|,)
            char_vec = self.R @ p  # (d,)
            char_vec = char_vec / (np.linalg.norm(char_vec) + 1e-12)
            # Mix character with xRAG context
            mixed = (1 - context_weight) * char_vec + context_weight * xrag_orth
            vecs.append(self._norm(mixed))
        return vecs
    
    def add_episode(self, xrag: Any, details: str) -> None:
        """
        Add an episode: xRAG key vector + details string.
        
        Args:
            xrag: numpy array or torch tensor of shape (d,) or (1, d)
            details: string to store as the episode's content
        """
        if self._finalized:
            raise RuntimeError("Cannot add episodes after finalize().")
        
        # Convert xRAG to numpy
        try:
            import torch
            if isinstance(xrag, torch.Tensor):
                xrag = xrag.detach().cpu().float().numpy()
        except ImportError:
            pass
        xrag = np.asarray(xrag, dtype=np.float32).reshape(-1)
        if xrag.shape[0] != self.d:
            raise ValueError(f"Expected xRAG dim {self.d}, got {xrag.shape[0]}")
        xrag = self._norm(xrag)
        
        # Encode details (with xRAG context mixed in for episode-specificity)
        detail_vecs = self._encode_details(details, xrag)
        
        # Record episode
        start_idx = len(self._patterns)
        self._episode_starts.append(start_idx)
        
        # Add xRAG as first pattern
        self._patterns.append(xrag)
        
        # Add detail vectors
        for v in detail_vecs:
            self._patterns.append(v)
        
        end_idx = len(self._patterns)
        self._episode_ranges.append((start_idx, end_idx))
        
        # Add transitions: xRAG → detail_0, detail_0 → detail_1, ..., detail_T → detail_T
        for i in range(start_idx, end_idx - 1):
            self._transitions.append((i, i + 1))
        # Self-loop at end
        self._transitions.append((end_idx - 1, end_idx - 1))
    
    def finalize(self, *, build_auto: bool = True, build_hetero: bool = True) -> None:
        """Build the MHN networks from stored patterns and transitions."""
        if not self._patterns:
            raise RuntimeError("No episodes stored.")
        
        patterns = np.stack(self._patterns, axis=1)  # (d, n_patterns)
        if bool(build_auto):
            # Build autoassociative MHN (stores all patterns)
            self._auto_mhn = AutoassociativeMHN(self.d, beta=self.params.beta_auto)
            self._auto_mhn.store(patterns)
        
        if bool(build_hetero):
            # Build heteroassociative MHN (stores transitions)
            keys = []
            values = []
            for from_idx, to_idx in self._transitions:
                keys.append(self._patterns[from_idx])
                values.append(self._patterns[to_idx])
            keys = np.stack(keys, axis=1)    # (d, n_transitions)
            values = np.stack(values, axis=1)
            
            self._hetero_mhn = HeteroassociativeMHN(self.d, beta=self.params.beta_hetero)
            self._hetero_mhn.store(keys, values)
        
        self._finalized = True
    
    def _denoise(self, v: np.ndarray, n_iters: int | None = None) -> np.ndarray:
        """Denoise v using autoassociative MHN."""
        n = n_iters if n_iters is not None else self.params.n_auto_iters
        return self._auto_mhn.retrieve(v, n_iters=n)
    
    def _advance(self, v: np.ndarray, n_iters: int | None = None) -> np.ndarray:
        """Advance v to next timestep using heteroassociative MHN."""
        n = n_iters if n_iters is not None else self.params.n_hetero_iters
        return self._hetero_mhn.retrieve(v, n_iters=n, hard=bool(self.params.hard_hetero))
    
    def _find_nearest_pattern_idx(self, v: np.ndarray) -> int:
        """Find the index of the stored pattern most similar to v."""
        patterns = np.stack(self._patterns, axis=1)
        v = self._norm(v)
        sims = patterns.T @ v
        return int(np.argmax(sims))
    
    def _decode_char(self, v: np.ndarray) -> str:
        """Decode a character from state vector v."""
        v = v.reshape(-1)
        logits = self.R.T @ v  # (|chars|,)
        return ALL_CHARS[int(np.argmax(logits))]
    
    def retrieve_episode(
        self,
        query_xrag: Any,
        *,
        max_chars: int = 500,
        snap_to_episode_xrag: bool = True,
    ) -> Tuple[np.ndarray, str]:
        """
        Retrieve episode from a (possibly noisy) xRAG query.
        
        Args:
            query_xrag: query vector (d,) or (1, d)
            max_chars: maximum characters to decode
            
        Returns:
            x_hat: reconstructed xRAG (d,)
            details: decoded details string
        """
        if not self._finalized:
            raise RuntimeError("Call finalize() before retrieval.")
        
        # Convert query
        try:
            import torch
            if isinstance(query_xrag, torch.Tensor):
                query_xrag = query_xrag.detach().cpu().float().numpy()
        except ImportError:
            pass
        q = np.asarray(query_xrag, dtype=np.float32).reshape(-1)
        q = self._norm(q)
        
        # Step 1: Denoise to find the correct episode (episode selection)
        v = self._denoise(q)
        nearest_idx = self._find_nearest_pattern_idx(v)
        
        # Find which episode we landed on
        episode_id = None
        for i, (start, end) in enumerate(self._episode_ranges):
            if start <= nearest_idx < end:
                episode_id = i
                break
        
        if episode_id is None:
            # Fallback: use the xRAG with highest similarity
            xrag_sims = [float(q @ self._patterns[s]) for s in self._episode_starts]
            episode_id = int(np.argmax(xrag_sims))
        
        ep_start, _ = self._episode_ranges[episode_id]
        x_hat = self._patterns[ep_start].copy()  # the episode's xRAG
        if bool(snap_to_episode_xrag):
            # Start hetero-associative stepping from the episode's xRAG, not from whichever within-episode
            # pattern the autoassociative network happened to converge to (could be a detail token).
            v = x_hat.copy()
        
        # Step 2: Advance to first detail (xRAG → '[')
        v = self._advance(v)
        if self.params.denoise_after_hetero:
            v = self._denoise(v, n_iters=self.params.n_post_denoise_iters)
        
        # Step 3: Decode details by alternating advance/denoise
        out = []
        for _ in range(max_chars + 5):
            ch = self._decode_char(v)
            
            if ch == "]":  # End token
                break
            if ch != "[":  # Skip start token
                out.append(ch)
            
            # Advance to next timestep
            v = self._advance(v)
            if self.params.denoise_after_hetero:
                v = self._denoise(v, n_iters=self.params.n_post_denoise_iters)
        
        return x_hat, "".join(out)
    
    def compute_energy(self, v: np.ndarray) -> float:
        """Compute energy of state v in the autoassociative MHN."""
        if not self._finalized:
            raise RuntimeError("Call finalize() first.")
        v = self._norm(v)
        patterns = np.stack(self._patterns, axis=1)
        sims = patterns.T @ v
        logits = self.params.beta_auto * sims
        max_logit = logits.max()
        return float(-max_logit - np.log(np.sum(np.exp(logits - max_logit)) + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing EpisodicMemoryDual...")
    
    d = 256
    params = EpisodicDualParams(
        beta_auto=50.0,   # Higher for sharper episode selection
        beta_hetero=100.0,  # Higher for sharper transitions
        denoise_after_hetero=False,  # Don't denoise after transition
        n_auto_iters=5,
    )
    mem = EpisodicMemoryDual(d, params=params)
    
    # Create synthetic xRAG vectors
    rng = np.random.default_rng(42)
    episodes = [
        {"xrag": rng.normal(size=(d,)), "text": "The cat sat on the mat."},
        {"xrag": rng.normal(size=(d,)), "text": "Memory is reconstructive."},
        {"xrag": rng.normal(size=(d,)), "text": "Hippocampus stores episodes."},
    ]
    
    for ep in episodes:
        xrag = ep["xrag"] / np.linalg.norm(ep["xrag"])
        mem.add_episode(xrag, ep["text"])
    
    mem.finalize()
    
    # Test retrieval with noisy query
    print("\n" + "="*50)
    print("Testing retrieval with 20% noise:")
    for i, ep in enumerate(episodes):
        xrag = ep["xrag"] / np.linalg.norm(ep["xrag"])
        noise = 0.2 * rng.normal(size=(d,))
        query = xrag + noise
        query = query / np.linalg.norm(query)
        
        x_hat, txt = mem.retrieve_episode(query, max_chars=100)
        cos = float(x_hat @ xrag)
        match = "✓" if txt.strip() == ep["text"] else "✗"
        
        print(f"\nEpisode {i}: {match}")
        print(f"  Stored:    '{ep['text']}'")
        print(f"  Retrieved: '{txt}'")
        print(f"  xRAG cos:  {cos:.3f}")
