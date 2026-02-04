"""
Internal signal analysis for grokking detection.

This module implements the KEY CONTRIBUTION of the research:
- Representation rank tracking via SVD
- Attention entropy computation

These internal signals are hypothesized to change during the grokking transition,
potentially providing early indicators before test accuracy improves.
"""
import torch
import numpy as np
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader


class RepresentationAnalyzer:
    """
    Analyzer for computing effective rank of hidden state representations.

    The effective rank measures the dimensionality being actively used by the model.
    High rank indicates full capacity usage (memorization), while low rank suggests
    the model has found compact, structured representations (generalization).
    """

    @staticmethod
    @torch.no_grad()
    def compute_effective_rank(
        hidden_states: torch.Tensor,
        threshold: float = 0.01
    ) -> Tuple[float, np.ndarray]:
        """
        Compute effective rank of hidden states using SVD.

        The effective rank is the number of singular values needed to capture
        (1 - threshold) of the total variance.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, d_model)
            threshold: Variance threshold (default: 0.01 = 99% variance explained)

        Returns:
            effective_rank: Number of dimensions capturing (1-threshold) of variance
            singular_values: Full spectrum of singular values
        """
        # Flatten batch and sequence dimensions
        B, T, D = hidden_states.shape
        X = hidden_states.view(B * T, D).cpu()  # Move to CPU for stability

        # Center the data (subtract mean)
        X = X - X.mean(dim=0, keepdim=True)

        # Compute SVD
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        except RuntimeError as e:
            print(f"Warning: SVD failed with error: {e}. Returning rank = d_model")
            return float(D), np.ones(D)

        # Compute explained variance ratio
        explained_variance = (S ** 2) / (S ** 2).sum()
        cumulative_variance = torch.cumsum(explained_variance, dim=0)

        # Effective rank: number of components to explain (1-threshold) of variance
        mask = cumulative_variance < (1 - threshold)
        effective_rank = mask.sum().item() + 1  # +1 for the threshold-crossing component

        return float(effective_rank), S.cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def compute_rank_per_layer(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 10,
        threshold: float = 0.01
    ) -> Tuple[List[float], List[np.ndarray]]:
        """
        Compute effective rank for each layer's hidden states.

        Args:
            model: The transformer model
            dataloader: DataLoader to sample batches from
            device: Device to run inference on
            num_batches: Number of batches to use for analysis
            threshold: Variance threshold for effective rank

        Returns:
            ranks: List of effective ranks per layer
            singular_values: List of singular value spectra per layer
        """
        model.eval()
        n_layers = len(model.blocks)
        all_hidden_states = [[] for _ in range(n_layers)]

        # Collect hidden states from multiple batches
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            output = model(input_ids, return_hidden_states=True)

            for layer_idx, hidden in enumerate(output['hidden_states']):
                all_hidden_states[layer_idx].append(hidden)

        # Compute rank per layer
        ranks = []
        singular_values = []

        for layer_idx in range(n_layers):
            # Concatenate all batches for this layer
            hidden_concat = torch.cat(all_hidden_states[layer_idx], dim=0)

            # Compute effective rank
            rank, sv = RepresentationAnalyzer.compute_effective_rank(
                hidden_concat,
                threshold=threshold
            )

            ranks.append(rank)
            singular_values.append(sv)

        return ranks, singular_values


class AttentionAnalyzer:
    """
    Analyzer for computing entropy of attention distributions.

    Attention entropy measures the focus vs. diffusion of attention patterns.
    High entropy indicates uniform attention (uncertain/exploratory), while
    low entropy indicates focused attention (learned structure).
    """

    @staticmethod
    @torch.no_grad()
    def compute_entropy(attn_weights: torch.Tensor) -> float:
        """
        Compute average Shannon entropy of attention distributions.

        Args:
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)

        Returns:
            Average entropy as a single scalar float, averaged over batch, heads, and query positions
        """
        # Add small epsilon for numerical stability (avoid log(0))
        eps = 1e-9
        attn_weights = attn_weights.clamp(min=eps)

        # Compute Shannon entropy: H = -sum(p * log(p))
        # Sum over last dimension (attention distribution)
        entropy = -(attn_weights * torch.log(attn_weights)).sum(dim=-1)  # (B, H, seq_len)

        # Average over all dimensions: batch, heads, query positions
        avg_entropy = entropy.mean().item()

        return avg_entropy

    @staticmethod
    @torch.no_grad()
    def compute_attention_entropy(attn_weights: torch.Tensor) -> np.ndarray:
        """
        Compute entropy of attention distributions.

        Entropy H(p) = -sum(p * log(p)) measures the concentration of the distribution.

        Args:
            attn_weights: Attention weights of shape (batch_size, n_heads, seq_len_q, seq_len_k)

        Returns:
            entropy_per_head: Average entropy per head, shape (n_heads,)
        """
        B, H, T_q, T_k = attn_weights.shape

        # Add small epsilon for numerical stability
        eps = 1e-9
        attn_weights = attn_weights + eps

        # Compute entropy: -sum(p * log(p))
        # Note: log is natural log (ln)
        entropy = -(attn_weights * torch.log(attn_weights)).sum(dim=-1)  # (B, H, T_q)

        # Average over batch and query positions
        entropy_per_head = entropy.mean(dim=(0, 2))  # (H,)

        return entropy_per_head.cpu().numpy()

    @staticmethod
    @torch.no_grad()
    def compute_entropy_per_layer(
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 10
    ) -> List[np.ndarray]:
        """
        Compute attention entropy for each layer and head.

        Args:
            model: The transformer model
            dataloader: DataLoader to sample batches from
            device: Device to run inference on
            num_batches: Number of batches to use for analysis

        Returns:
            entropies: List of entropy arrays per layer, each of shape (n_heads,)
        """
        model.eval()
        n_layers = len(model.blocks)
        all_attention_weights = [[] for _ in range(n_layers)]

        # Collect attention weights from multiple batches
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids'].to(device)
            _, attentions = model(input_ids)

            for layer_idx, attn in enumerate(attentions):
                all_attention_weights[layer_idx].append(attn)

        # Compute entropy per layer
        entropies = []

        for layer_idx in range(n_layers):
            # Concatenate all batches for this layer
            attn_concat = torch.cat(all_attention_weights[layer_idx], dim=0)

            # Compute entropy
            entropy = AttentionAnalyzer.compute_attention_entropy(attn_concat)

            entropies.append(entropy)

        return entropies


class AnalysisLogger:
    """
    Logger for storing and managing analysis metrics over training.
    """

    def __init__(self):
        self.data = {
            'step': [],
            'representation_rank': [],      # List of lists: [layer_ranks] per step
            'singular_values': [],          # List of lists: [layer_sv] per step
            'attention_entropy': []         # List of lists: [layer_entropies] per step
        }

    def log(
        self,
        step: int,
        ranks: List[float],
        singular_values: List[np.ndarray],
        entropies: List[np.ndarray]
    ):
        """
        Log analysis metrics for a training step.

        Args:
            step: Training step number
            ranks: Effective rank per layer
            singular_values: Singular values per layer
            entropies: Attention entropy per layer
        """
        self.data['step'].append(step)
        self.data['representation_rank'].append(ranks)
        self.data['singular_values'].append(singular_values)
        self.data['attention_entropy'].append(entropies)

    def get_final_layer_rank(self) -> List[float]:
        """Get rank time series for the final layer."""
        if not self.data['representation_rank']:
            return []
        return [ranks[-1] for ranks in self.data['representation_rank']]

    def get_average_entropy(self) -> List[float]:
        """Get average attention entropy across all layers and heads."""
        if not self.data['attention_entropy']:
            return []

        avg_entropies = []
        for entropies_per_layer in self.data['attention_entropy']:
            # Average across all layers and heads
            all_entropies = np.concatenate(entropies_per_layer)
            avg_entropies.append(all_entropies.mean())

        return avg_entropies

    def to_dict(self) -> Dict:
        """
        Convert to JSON-serializable dictionary.

        Note: Singular values are omitted from serialization due to size.
        """
        return {
            'step': self.data['step'],
            'representation_rank': self.data['representation_rank'],
            'attention_entropy': [
                [entropy.tolist() for entropy in entropies_per_layer]
                for entropies_per_layer in self.data['attention_entropy']
            ]
        }
