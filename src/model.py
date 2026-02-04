"""
Minimal research-grade decoder-only Transformer for modular arithmetic.

Architecture:
    - Causal (autoregressive) Transformer
    - Token + positional embeddings
    - Stacked Transformer blocks
    - Final layer normalization
    - Linear output head

Each block contains:
    - LayerNorm -> Multi-head causal self-attention -> Residual
    - LayerNorm -> Feedforward MLP (GELU) -> Residual
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with explicit lower-triangular masking.

    Returns attention weights for analysis.
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Causal mask: lower-triangular matrix
        # Register as buffer so it moves with model to device
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with causal masking.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            out: Output tensor of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, d_model)

        # Reshape for multi-head attention: (B, T, d_model) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask: set future positions to -inf
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float('-inf')
        )

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, T, T)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (B, num_heads, T, T) @ (B, num_heads, T, head_dim) -> (B, num_heads, T, head_dim)
        out = attn_weights @ v

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_proj(out)

        return out, attn_weights


class FeedForward(nn.Module):
    """
    Position-wise feedforward network with GELU activation.

    Architecture: Linear -> GELU -> Dropout -> Linear
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch_size, seq_len, d_model)

        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block.

    Architecture:
        LayerNorm -> Causal Self-Attention -> Residual
        LayerNorm -> FeedForward MLP -> Residual
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.

        Args:
            x: Input of shape (batch_size, seq_len, d_model)

        Returns:
            x: Output of shape (batch_size, seq_len, d_model)
            attn_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out

        # Feedforward with residual
        x = x + self.mlp(self.ln2(x))

        return x, attn_weights


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer for causal language modeling.

    Architecture:
        Token Embeddings + Positional Embeddings
        -> N x TransformerBlock
        -> LayerNorm
        -> Linear (output head)

    Default configuration targets ~1-2M parameters:
        - d_model: 128
        - num_layers: 4
        - num_heads: 4
        - d_ff: 512
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 32,
        dropout: float = 0.1
    ):
        """
        Initialize the Transformer model.

        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension (default: 128)
            num_layers: Number of Transformer blocks (default: 4)
            num_heads: Number of attention heads (default: 4)
            d_ff: Feedforward hidden dimension (default: 512)
            max_seq_len: Maximum sequence length (default: 32)
            dropout: Dropout probability (default: 0.1)
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_len, dropout)
            for _ in range(num_layers)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(d_model)

        # Output head (projects to vocabulary)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: tie token embedding and output head weights
        self.head.weight = self.token_embedding.weight

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor, return_hidden_states: bool = False):
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len)
            return_hidden_states: If True, return hidden states from each layer

        Returns:
            If return_hidden_states=False:
                logits: Output logits of shape (batch_size, seq_len, vocab_size)
                attentions: List of attention tensors, one per layer
                           Each has shape (batch_size, num_heads, seq_len, seq_len)
            If return_hidden_states=True:
                dict with keys 'logits', 'attentions', 'hidden_states'
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"

        # Token embeddings + positional embeddings
        token_emb = self.token_embedding(input_ids)  # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=input_ids.device))  # (T, d_model)
        x = self.dropout(token_emb + pos_emb)  # (B, T, d_model)

        # Pass through Transformer blocks and collect attention weights and hidden states
        attentions = []
        hidden_states = []
        for block in self.blocks:
            if return_hidden_states:
                hidden_states.append(x)
            x, attn_weights = block(x)
            attentions.append(attn_weights)

        # Final layer norm
        x = self.ln_f(x)  # (B, T, d_model)

        # Output projection
        logits = self.head(x)  # (B, T, vocab_size)

        if return_hidden_states:
            return {
                'logits': logits,
                'attentions': attentions,
                'hidden_states': hidden_states
            }
        else:
            return logits, attentions

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# SANITY CHECK: Minimal example demonstrating model usage
# ============================================================================

def sanity_check():
    """
    Sanity check: Verify model instantiation and forward pass.

    Checks:
        1. Model instantiation with default hyperparameters
        2. Forward pass on random token IDs
        3. Output shapes are correct
        4. Attention tensors are returned correctly
    """
    print("="*60)
    print("Decoder-Only Transformer: Sanity Check")
    print("="*60)

    # Configuration
    vocab_size = 104  # From tokenizer (0-97 integers + 6 special tokens)
    batch_size = 4
    seq_len = 16

    # 1. Model instantiation
    print("\n1. Instantiating model with default hyperparameters...")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=128,       # Default
        num_layers=4,      # Default
        num_heads=4,       # Default
        d_ff=512,          # Default
        max_seq_len=32,    # Default
        dropout=0.0        # No dropout for sanity check
    )

    num_params = model.count_parameters()
    print(f"   [OK] Model created")
    print(f"   [OK] Total parameters: {num_params:,}")
    print(f"   [OK] Target: ~1-2M parameters")

    if 0.8e6 <= num_params <= 2.2e6:
        print(f"   [OK] Parameter count is within target range!")
    else:
        print(f"   [WARNING] Parameter count outside target range")

    # 2. Forward pass on random token IDs
    print("\n2. Running forward pass on random token IDs...")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape} = (batch_size={batch_size}, seq_len={seq_len})")

    with torch.no_grad():
        logits, attentions = model(input_ids)

    print(f"   [OK] Forward pass successful")

    # 3. Verify output shapes
    print("\n3. Verifying output shapes...")

    # Check logits shape
    expected_logits_shape = (batch_size, seq_len, vocab_size)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Expected: {expected_logits_shape}")
    assert logits.shape == expected_logits_shape, f"Logits shape mismatch!"
    print(f"   [OK] Logits shape is correct")

    # Check number of attention tensors
    print(f"\n   Number of attention tensors: {len(attentions)}")
    print(f"   Expected (num_layers): {model.num_layers}")
    assert len(attentions) == model.num_layers, f"Wrong number of attention tensors!"
    print(f"   [OK] Correct number of attention tensors")

    # Check attention tensor shapes
    expected_attn_shape = (batch_size, model.num_heads, seq_len, seq_len)
    print(f"\n   Attention tensor shape (per layer): {attentions[0].shape}")
    print(f"   Expected: {expected_attn_shape}")
    for i, attn in enumerate(attentions):
        assert attn.shape == expected_attn_shape, f"Layer {i} attention shape mismatch!"
    print(f"   [OK] All attention tensor shapes are correct")

    # 4. Additional checks
    print("\n4. Additional checks...")

    # Check that attention weights sum to 1 (softmax property)
    attn_sum = attentions[0].sum(dim=-1)  # Sum over key dimension
    expected_sum = torch.ones(batch_size, model.num_heads, seq_len)
    assert torch.allclose(attn_sum, expected_sum, atol=1e-6), "Attention weights don't sum to 1!"
    print(f"   [OK] Attention weights sum to 1 (valid probability distribution)")

    # Check causal masking: attention to future positions should be zero
    # For position i, attention to positions > i should be zero
    for t in range(seq_len - 1):
        future_attn = attentions[0][:, :, t, t+1:]  # Attention from position t to future
        assert torch.all(future_attn == 0), f"Causal mask violated at position {t}!"
    print(f"   [OK] Causal masking is correct (no attention to future positions)")

    # Check that logits have reasonable range (not NaN or Inf)
    assert not torch.isnan(logits).any(), "Logits contain NaN!"
    assert not torch.isinf(logits).any(), "Logits contain Inf!"
    print(f"   [OK] Logits are finite (no NaN or Inf)")

    print("\n" + "="*60)
    print("[OK] All sanity checks passed!")
    print("="*60)

    # Print model summary
    print("\nModel Summary:")
    print(f"  Architecture: Decoder-only Transformer")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension (d_model): {model.d_model}")
    print(f"  Number of layers: {model.num_layers}")
    print(f"  Number of heads: {model.num_heads}")
    print(f"  Feedforward dimension (d_ff): 512")
    print(f"  Max sequence length: {model.max_seq_len}")
    print(f"  Total parameters: {num_params:,}")
    print()
    print("Model is ready for training!")
    print()


if __name__ == "__main__":
    # Run sanity check when module is executed directly
    sanity_check()
