"""
Dual Modern Hopfield Network (MHN) for sequence memory.

Architecture:
    - Shared state vector v
    - Autoassociative MHN: denoises v onto nearest stored timestep pattern
    - Heteroassociative MHN: advances v from timestep n to timestep n+1
    - Controller: alternates auto (denoise) → hetero (advance) → repeat

This is a simplified alternative to EDEN that's easier to understand and tune.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np

import hashlib


@dataclass
class DualMHNParams:
    """Parameters for the dual MHN model."""
    # Softmax inverse temperature for autoassociative retrieval
    beta_auto: float = 10.0
    # Softmax inverse temperature for heteroassociative retrieval
    beta_hetero: float = 10.0
    # Number of autoassociative iterations per step (denoising)
    n_auto_iters: int = 3
    # Number of heteroassociative iterations per step (usually 1)
    n_hetero_iters: int = 1
    # Whether to re-denoise after hetero step
    denoise_after_hetero: bool = True
    # Iterations for post-hetero denoising
    n_post_denoise_iters: int = 2


class AutoassociativeMHN:
    """
    Modern Hopfield Network for autoassociative retrieval.
    
    Stores a set of patterns. Given query v, retrieves a weighted combination
    that denoises v toward the nearest stored pattern.
    """
    
    def __init__(self, d: int, beta: float = 10.0):
        self.d = d
        self.beta = beta
        self.patterns: np.ndarray | None = None  # (d, n_patterns)
    
    def store(self, patterns: np.ndarray) -> None:
        """
        Store patterns for retrieval.
        
        Args:
            patterns: (d, n) array where each column is a pattern
        """
        if patterns.ndim == 1:
            patterns = patterns.reshape(-1, 1)
        assert patterns.shape[0] == self.d
        patterns = patterns.astype(np.float32)
        norms = np.linalg.norm(patterns, axis=0, keepdims=True)
        self.patterns = patterns / (norms + 1e-12)
    
    def retrieve(self, v: np.ndarray, n_iters: int = 1, *, hard: bool = False) -> np.ndarray:
        """
        Run autoassociative retrieval to denoise v.
        
        Args:
            v: (d,) or (d, 1) query vector
            n_iters: number of retrieval iterations
            
        Returns:
            v_out: (d,) denoised vector
        """
        if self.patterns is None:
            raise RuntimeError("No patterns stored. Call store() first.")
        
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        
        for _ in range(n_iters):
            # Compute similarities: (n,)
            sims = self.patterns.T @ v  # (n,)
            if hard:
                weights = np.zeros_like(sims, dtype=np.float32)
                weights[int(np.argmax(sims))] = 1.0
            else:
                # Softmax attention weights
                logits = self.beta * sims
                logits = logits - logits.max()  # numerical stability
                weights = np.exp(logits)
                weights = weights / (weights.sum() + 1e-12)
            # Weighted sum of patterns
            v = self.patterns @ weights  # (d,)
            # Normalize
            v = v / (np.linalg.norm(v) + 1e-12)
        
        return v
    
    def attention_weights(self, v: np.ndarray) -> np.ndarray:
        """Return the softmax attention weights over stored patterns."""
        if self.patterns is None:
            raise RuntimeError("No patterns stored.")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        sims = self.patterns.T @ v
        logits = self.beta * sims
        logits = logits - logits.max()
        weights = np.exp(logits)
        return weights / (weights.sum() + 1e-12)


class HeteroassociativeMHN:
    """
    Modern Hopfield Network for heteroassociative retrieval.
    
    Stores key-value pairs (transitions). Given query v, retrieves a weighted
    combination of values based on similarity to keys.
    """
    
    def __init__(self, d: int, beta: float = 10.0):
        self.d = d
        self.beta = beta
        self.keys: np.ndarray | None = None    # (d, n_transitions)
        self.values: np.ndarray | None = None  # (d, n_transitions)
        self._key_hash_to_index: dict[int, int] | None = None

    @staticmethod
    def _hash_vec(v: np.ndarray) -> int:
        h = hashlib.blake2b(v.tobytes(), digest_size=8).digest()
        return int.from_bytes(h, byteorder="little", signed=False)
    
    def store(self, keys: np.ndarray, values: np.ndarray) -> None:
        """
        Store key-value transition pairs.
        
        Args:
            keys: (d, n) array where column i is the key (source pattern)
            values: (d, n) array where column i is the value (target pattern)
        """
        if keys.ndim == 1:
            keys = keys.reshape(-1, 1)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        assert keys.shape == values.shape
        assert keys.shape[0] == self.d
        keys = keys.astype(np.float32)
        values = values.astype(np.float32)
        k_norms = np.linalg.norm(keys, axis=0, keepdims=True)
        v_norms = np.linalg.norm(values, axis=0, keepdims=True)
        self.keys = keys / (k_norms + 1e-12)
        self.values = values / (v_norms + 1e-12)
        # Build an exact-lookup cache for hard retrieval: if v exactly matches a stored key,
        # we can retrieve its associated value without a full similarity scan.
        self._key_hash_to_index = {self._hash_vec(self.keys[:, i]): int(i) for i in range(self.keys.shape[1])}
    
    def retrieve(self, v: np.ndarray, n_iters: int = 1, *, hard: bool = False) -> np.ndarray:
        """
        Run heteroassociative retrieval to advance v to next timestep.
        
        Args:
            v: (d,) or (d, 1) query vector
            n_iters: number of retrieval iterations
            
        Returns:
            v_out: (d,) advanced vector
        """
        if self.keys is None or self.values is None:
            raise RuntimeError("No transitions stored. Call store() first.")
        
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        
        for _ in range(n_iters):
            # Compute similarities to keys: (n,)
            if hard:
                if self._key_hash_to_index is not None:
                    hit = self._key_hash_to_index.get(self._hash_vec(v))
                else:
                    hit = None
                if hit is None:
                    sims = self.keys.T @ v  # (n,)
                    hit = int(np.argmax(sims))
                # In hard mode, jump directly to the selected value (avoid matmul + renorm drift).
                v = self.values[:, int(hit)].copy()
                continue
            else:
                sims = self.keys.T @ v  # (n,)
                # Softmax attention weights
                logits = self.beta * sims
                logits = logits - logits.max()
                weights = np.exp(logits)
                weights = weights / (weights.sum() + 1e-12)
            # Weighted sum of values
            v = self.values @ weights  # (d,)
            # Normalize
            v = v / (np.linalg.norm(v) + 1e-12)
        
        return v
    
    def attention_weights(self, v: np.ndarray) -> np.ndarray:
        """Return the softmax attention weights over stored transitions."""
        if self.keys is None:
            raise RuntimeError("No transitions stored.")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        sims = self.keys.T @ v
        logits = self.beta * sims
        logits = logits - logits.max()
        weights = np.exp(logits)
        return weights / (weights.sum() + 1e-12)


class DualMHN:
    """
    Dual MHN controller for sequence memory.
    
    Stores a sequence of patterns and provides:
        - store_sequence(): store timestep patterns and transitions
        - replay(): generate sequential replay from a starting state
    """
    
    def __init__(self, d: int, params: DualMHNParams | None = None):
        self.d = d
        self.params = params or DualMHNParams()
        
        self.auto_mhn = AutoassociativeMHN(d, beta=self.params.beta_auto)
        self.hetero_mhn = HeteroassociativeMHN(d, beta=self.params.beta_hetero)
        
        self._patterns: np.ndarray | None = None
        self._n_patterns: int = 0
    
    def store_sequence(self, patterns: np.ndarray) -> None:
        """
        Store a sequence of patterns.
        
        Args:
            patterns: (d, T) array where column t is the pattern at timestep t
        """
        if patterns.ndim == 1:
            patterns = patterns.reshape(-1, 1)
        assert patterns.shape[0] == self.d
        
        patterns = patterns.astype(np.float32)
        # Normalize each pattern
        norms = np.linalg.norm(patterns, axis=0, keepdims=True)
        patterns = patterns / (norms + 1e-12)
        
        self._patterns = patterns
        self._n_patterns = patterns.shape[1]
        
        # Store all patterns in autoassociative MHN
        self.auto_mhn.store(patterns)
        
        # Store transitions in heteroassociative MHN
        # Key: pattern at t, Value: pattern at t+1
        if self._n_patterns > 1:
            keys = patterns[:, :-1]    # patterns 0 to T-2
            values = patterns[:, 1:]   # patterns 1 to T-1
            self.hetero_mhn.store(keys, values)
    
    def denoise(self, v: np.ndarray, n_iters: int | None = None, *, hard: bool = False) -> np.ndarray:
        """Denoise v using the autoassociative MHN."""
        n = n_iters if n_iters is not None else self.params.n_auto_iters
        return self.auto_mhn.retrieve(v, n_iters=n, hard=hard)
    
    def advance(self, v: np.ndarray, n_iters: int | None = None, *, hard: bool = False) -> np.ndarray:
        """Advance v to next timestep using the heteroassociative MHN."""
        n = n_iters if n_iters is not None else self.params.n_hetero_iters
        return self.hetero_mhn.retrieve(v, n_iters=n, hard=hard)
    
    def step(self, v: np.ndarray) -> np.ndarray:
        """
        One step of the controller: denoise → advance → (optional) denoise.
        
        Args:
            v: current state vector
            
        Returns:
            v_next: state after one controller step
        """
        # Denoise
        v = self.denoise(v)
        # Advance
        v = self.advance(v)
        # Optional post-denoise
        if self.params.denoise_after_hetero:
            v = self.denoise(v, n_iters=self.params.n_post_denoise_iters)
        return v
    
    def replay(self, v0: np.ndarray, n_steps: int, 
               return_trajectory: bool = False) -> np.ndarray | Tuple[np.ndarray, List[np.ndarray]]:
        """
        Replay sequence starting from v0.
        
        Args:
            v0: starting state (noisy or exact)
            n_steps: number of steps to replay
            return_trajectory: if True, also return list of intermediate states
            
        Returns:
            v_final: final state after n_steps
            trajectory: (optional) list of states [v0_denoised, v1, v2, ...]
        """
        v = np.asarray(v0, dtype=np.float32).reshape(-1)
        v = v / (np.linalg.norm(v) + 1e-12)
        
        # Initial denoise
        v = self.denoise(v)
        
        trajectory = [v.copy()] if return_trajectory else None
        
        for _ in range(n_steps):
            v = self.step(v)
            if return_trajectory:
                trajectory.append(v.copy())
        
        if return_trajectory:
            return v, trajectory
        return v
    
    def get_patterns(self) -> np.ndarray | None:
        """Return the stored patterns (d, T)."""
        return self._patterns
    
    def find_nearest_pattern_idx(self, v: np.ndarray) -> int:
        """Return the index of the stored pattern most similar to v."""
        if self._patterns is None:
            raise RuntimeError("No patterns stored.")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = self._patterns.T @ v
        return int(np.argmax(sims))
    
    def compute_energy(self, v: np.ndarray) -> float:
        """
        Compute Hopfield-style energy for the autoassociative MHN.
        
        E = -log(sum_i exp(beta * v^T xi_i))
        
        Lower energy = closer to a stored pattern.
        """
        if self._patterns is None:
            raise RuntimeError("No patterns stored.")
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        v = v / (np.linalg.norm(v) + 1e-12)
        sims = self._patterns.T @ v  # (n,)
        logits = self.params.beta_auto * sims
        # log-sum-exp for numerical stability
        max_logit = logits.max()
        energy = -max_logit - np.log(np.sum(np.exp(logits - max_logit)) + 1e-12)
        return float(energy)


# ─────────────────────────────────────────────────────────────────────────────
# Demo / test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a simple sequence of patterns
    d = 64
    T = 10
    rng = np.random.default_rng(42)
    
    # Random orthogonal-ish patterns
    patterns = rng.normal(size=(d, T)).astype(np.float32)
    patterns = patterns / np.linalg.norm(patterns, axis=0, keepdims=True)
    
    # Build dual MHN
    params = DualMHNParams(beta_auto=10.0, beta_hetero=10.0, n_auto_iters=3)
    mhn = DualMHN(d, params)
    mhn.store_sequence(patterns)
    
    # Start from noisy version of pattern 0
    v0 = patterns[:, 0] + 0.3 * rng.normal(size=(d,)).astype(np.float32)
    v0 = v0 / np.linalg.norm(v0)
    
    # Replay
    v_final, trajectory = mhn.replay(v0, n_steps=T-1, return_trajectory=True)
    
    # Check which pattern each trajectory state is closest to
    print("Replay trajectory (closest pattern index):")
    for i, v in enumerate(trajectory):
        idx = mhn.find_nearest_pattern_idx(v)
        cos = float(v @ patterns[:, idx])
        print(f"  Step {i}: closest to pattern {idx}, cos={cos:.3f}")
    
    # Plot energy over replay
    energies = [mhn.compute_energy(v) for v in trajectory]
    plt.figure(figsize=(5, 3))
    plt.plot(energies, marker='o')
    plt.xlabel("Replay step")
    plt.ylabel("Energy")
    plt.title("Dual MHN: Energy during sequence replay")
    plt.tight_layout()
    plt.savefig("/tmp/dual_mhn_energy.png", dpi=150)
    print("\nSaved /tmp/dual_mhn_energy.png")
