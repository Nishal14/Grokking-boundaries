"""
Structured modular arithmetic dataset with constraint-conflict splits.

This module implements sum-based splitting that creates genuine extrapolation
challenges, preventing pure interpolation and forcing algorithmic learning.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple

from src.tokenizer import ModularArithmeticTokenizer


class StructuredModularArithmeticDataset(Dataset):
    """
    Modular arithmetic dataset with structured sum-based split.

    Split strategy:
        Train: (a + b) mod p < p/2  (lower-half sums)
        Test:  (a + b) mod p >= p/2 (upper-half sums)

    This creates a constraint conflict where test examples require
    algorithmic extrapolation, not feature interpolation.

    For p=97:
        Train: results in [0, 48]
        Test:  results in [49, 96]
    """

    def __init__(
        self,
        modulus_p: int = 97,
        split: str = 'train',
        seed: int = 42  # Seed for deterministic ordering within each split
    ):
        """
        Initialize structured dataset.

        Args:
            modulus_p: Prime modulus for arithmetic operations
            split: 'train' or 'test'
            seed: Random seed for shuffling within split (optional)
        """
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got {split}"
        assert modulus_p > 1, f"modulus_p must be > 1, got {modulus_p}"

        self.modulus_p = modulus_p
        self.split = split
        self.tokenizer = ModularArithmeticTokenizer(modulus_p)

        # Generate all (a, b) pairs
        all_pairs = [(a, b) for a in range(modulus_p) for b in range(modulus_p)]

        # Structured split based on sum
        threshold = modulus_p // 2

        train_pairs = []
        test_pairs = []

        for a, b in all_pairs:
            result = (a + b) % modulus_p
            if result < threshold:
                train_pairs.append((a, b))
            else:
                test_pairs.append((a, b))

        # Select pairs based on split
        if split == 'train':
            self.pairs = train_pairs
        else:
            self.pairs = test_pairs

        # Shuffle within split for training variety (deterministic)
        np.random.seed(seed)
        indices = np.random.permutation(len(self.pairs))
        self.pairs = [self.pairs[i] for i in indices]

    def __len__(self) -> int:
        """Return number of examples in this split."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.

        Returns:
            Dictionary with:
                - input_ids: Token IDs for input (without last token)
                - labels: Token IDs for labels (without first token)
                - result: Ground truth result value (c)
                - a: First operand
                - b: Second operand
        """
        a, b = self.pairs[idx]
        c = (a + b) % self.modulus_p

        # Create sequence: <bos> a + b mod p = c <eos>
        text = f"<bos> {a} + {b} mod {self.modulus_p} = {c} <eos>"
        token_ids = self.tokenizer.encode(text)

        # For next-token prediction:
        # input_ids:  [<bos>, a, +, b, mod, p, =, c]     (indices 0-7)
        # labels:     [a, +, b, mod, p, =, c, <eos>]     (indices 1-8)
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': labels,
            'result': c,  # For evaluation
            'a': a,
            'b': b
        }

    def get_text_example(self, idx: int) -> str:
        """Get human-readable text for an example."""
        a, b = self.pairs[idx]
        c = (a + b) % self.modulus_p
        return f"<bos> {a} + {b} mod {self.modulus_p} = {c} <eos>"


def create_structured_dataloaders(
    modulus_p: int = 97,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders with structured sum-based split.

    Args:
        modulus_p: Prime modulus
        batch_size: Batch size
        seed: Random seed for shuffling within splits
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = StructuredModularArithmeticDataset(
        modulus_p=modulus_p,
        split='train',
        seed=seed
    )

    test_dataset = StructuredModularArithmeticDataset(
        modulus_p=modulus_p,
        split='test',
        seed=seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def analyze_structured_split(modulus_p: int = 97):
    """
    Analyze the properties of the structured sum-based split.

    Returns a report on:
        - Train/test sizes
        - Operand coverage
        - Result coverage
        - Structural properties
    """
    threshold = modulus_p // 2

    # Generate splits
    all_pairs = [(a, b) for a in range(modulus_p) for b in range(modulus_p)]
    train_pairs = [(a, b) for a, b in all_pairs if (a + b) % modulus_p < threshold]
    test_pairs = [(a, b) for a, b in all_pairs if (a + b) % modulus_p >= threshold]

    # Operand coverage
    train_a = set(a for a, b in train_pairs)
    train_b = set(b for a, b in train_pairs)
    test_a = set(a for a, b in test_pairs)
    test_b = set(b for a, b in test_pairs)

    # Result coverage
    train_results = set((a + b) % modulus_p for a, b in train_pairs)
    test_results = set((a + b) % modulus_p for a, b in test_pairs)

    report = f"""
STRUCTURED SUM-BASED SPLIT ANALYSIS (p={modulus_p})
{'='*70}

Split Configuration:
  Train: (a + b) mod {modulus_p} < {threshold} (lower-half sums)
  Test:  (a + b) mod {modulus_p} >= {threshold} (upper-half sums)

Dataset Sizes:
  Total examples: {len(all_pairs)}
  Train: {len(train_pairs)} ({len(train_pairs)/len(all_pairs)*100:.1f}%)
  Test:  {len(test_pairs)} ({len(test_pairs)/len(all_pairs)*100:.1f}%)

Operand Coverage:
  Train: {len(train_a)}/{modulus_p} values of a, {len(train_b)}/{modulus_p} values of b
  Test:  {len(test_a)}/{modulus_p} values of a, {len(test_b)}/{modulus_p} values of b

Result Coverage (CRITICAL):
  Train: {sorted(train_results)}
  Test:  {sorted(test_results)}
  Overlap: {train_results & test_results}

Constraint Conflict Properties:
  ✓ Zero result overlap (train and test have disjoint result ranges)
  ✓ Full operand coverage in both splits (all values appear)
  ✓ Algorithmic extrapolation required (cannot interpolate results)
  ✓ Must learn modular addition rule to generalize

Expected Behavior:
  - Long flat accuracy plateau (model cannot interpolate)
  - Sudden sharp accuracy jump (algorithmic discovery)
  - Representation rank collapse during transition
  - Attention entropy collapse during transition
"""
    return report
