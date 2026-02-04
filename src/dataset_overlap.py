"""
Overlapping constraint-conflict split for v3 experiment.

Train: (a + b) mod p ∈ [0, 62]
Test:  (a + b) mod p ∈ [32, 96]

Overlap region: [32, 62] (appears in both train and test)
Train-only: [0, 31]
Test-only: [63, 96]

This creates constraint conflict while maintaining interpolation capability.
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict

from src.tokenizer import ModularArithmeticTokenizer


class OverlapModularArithmeticDataset(Dataset):
    """
    Modular arithmetic dataset with overlapping output-based split.

    Creates constraint conflict by having different output ranges for
    train vs test, with partial overlap to prevent pure extrapolation.
    """

    def __init__(self, modulus_p=97, split='train', seed=42):
        """
        Args:
            modulus_p: Modulus for arithmetic (default 97, prime)
            split: 'train' or 'test'
            seed: Random seed (for consistency)
        """
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got {split}"

        self.modulus_p = modulus_p
        self.split = split
        self.tokenizer = ModularArithmeticTokenizer(modulus_p)

        # Generate all possible pairs
        all_pairs = [(a, b) for a in range(modulus_p) for b in range(modulus_p)]

        # Define ranges
        # Train: results in [0, 62]
        # Test: results in [32, 96]
        train_range = list(range(0, 63))  # [0, 62]
        test_range = list(range(32, 97))   # [32, 96]

        # Split based on result
        train_pairs = []
        test_pairs = []

        for a, b in all_pairs:
            result = (a + b) % modulus_p
            if result in train_range:
                train_pairs.append((a, b))
            if result in test_range:
                test_pairs.append((a, b))

        # Select data based on split
        if split == 'train':
            self.pairs = train_pairs
        elif split == 'test':
            self.pairs = test_pairs
        else:
            raise ValueError(f"split must be 'train' or 'test', got {split}")

        # Shuffle within split for training variety (deterministic)
        np.random.seed(seed)
        indices = np.random.permutation(len(self.pairs))
        self.pairs = [self.pairs[i] for i in indices]

        # Store statistics
        self.train_size = len(train_pairs)
        self.test_size = len(test_pairs)
        overlap_count = len(set(train_pairs) & set(test_pairs))
        self.overlap_size = overlap_count

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
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

    def get_split_info(self):
        """Return information about the split."""
        return {
            'modulus': self.modulus_p,
            'split': self.split,
            'size': len(self.pairs),
            'train_size': self.train_size,
            'test_size': self.test_size,
            'overlap_size': self.overlap_size,
            'overlap_pct': 100 * self.overlap_size / min(self.train_size, self.test_size)
        }


def create_overlap_dataloaders(modulus_p=97, batch_size=256, seed=42, num_workers=0):
    """
    Create train and test dataloaders with overlapping split.

    Returns:
        train_loader, test_loader
    """
    train_dataset = OverlapModularArithmeticDataset(
        modulus_p=modulus_p,
        split='train',
        seed=seed
    )

    test_dataset = OverlapModularArithmeticDataset(
        modulus_p=modulus_p,
        split='test',
        seed=seed
    )

    # Print split statistics
    train_info = train_dataset.get_split_info()
    test_info = test_dataset.get_split_info()

    print(f"Overlap Split Configuration:")
    print(f"  Train range: [0, 62]  -> {train_info['size']:,} examples")
    print(f"  Test range:  [32, 96] -> {test_info['size']:,} examples")
    print(f"  Overlap:     [32, 62] -> {train_info['overlap_size']:,} examples ({train_info['overlap_pct']:.1f}%)")
    print(f"  Train-only:  [0, 31]  -> {train_info['size'] - train_info['overlap_size']:,} examples")
    print(f"  Test-only:   [63, 96] -> {test_info['size'] - test_info['overlap_size']:,} examples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def validate_split():
    """Validate the split configuration."""
    train_dataset = OverlapModularArithmeticDataset(modulus_p=97, split='train')
    test_dataset = OverlapModularArithmeticDataset(modulus_p=97, split='test')

    # Get all results from each split
    train_results = set()
    test_results = set()

    for a, b in train_dataset.pairs:
        train_results.add((a + b) % 97)

    for a, b in test_dataset.pairs:
        test_results.add((a + b) % 97)

    overlap_results = train_results & test_results
    train_only_results = train_results - test_results
    test_only_results = test_results - train_results

    print("Split Validation:")
    print(f"  Train results: {sorted(train_results)[:10]}...{sorted(train_results)[-10:]}")
    print(f"  Test results: {sorted(test_results)[:10]}...{sorted(test_results)[-10:]}")
    print(f"  Overlap: {sorted(overlap_results)}")
    print(f"  Train-only: {sorted(train_only_results)}")
    print(f"  Test-only: {sorted(test_only_results)}")

    # Verify ranges
    assert train_results == set(range(0, 63)), "Train range incorrect"
    assert test_results == set(range(32, 97)), "Test range incorrect"
    assert overlap_results == set(range(32, 63)), "Overlap range incorrect"
    assert train_only_results == set(range(0, 32)), "Train-only range incorrect"
    assert test_only_results == set(range(63, 97)), "Test-only range incorrect"

    print("\nSplit validation passed")


if __name__ == '__main__':
    import sys
    import os
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    validate_split()
