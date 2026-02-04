"""
Modular arithmetic dataset for grokking experiments.
"""
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, List

from src.tokenizer import ModularArithmeticTokenizer


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic: a + b mod p = c

    Generates all possible (a, b) pairs where a, b ∈ {0, ..., p-1}
    and splits them deterministically into train/test sets.

    Sequence format: <bos> a + b mod p = c <eos>
    Example: <bos> 5 + 3 mod 97 = 8 <eos>

    Total examples: p × p = 97 × 97 = 9,409
    Train/Test split: 80/20 → 7,527 train, 1,882 test
    """

    def __init__(
        self,
        modulus_p: int = 97,
        split: str = 'train',
        train_split: float = 0.8,
        seed: int = 42
    ):
        """
        Initialize dataset.

        Args:
            modulus_p: Prime modulus for arithmetic operations
            split: 'train' or 'test'
            train_split: Fraction of data for training
            seed: Random seed for deterministic split
        """
        assert split in ['train', 'test'], f"split must be 'train' or 'test', got {split}"
        assert 0.0 < train_split < 1.0, f"train_split must be in (0, 1), got {train_split}"

        self.modulus_p = modulus_p
        self.split = split
        self.tokenizer = ModularArithmeticTokenizer(modulus_p)

        # Generate all (a, b) pairs
        all_pairs = [(a, b) for a in range(modulus_p) for b in range(modulus_p)]

        # Deterministic structured split
        np.random.seed(seed)
        indices = np.random.permutation(len(all_pairs))
        split_idx = int(len(all_pairs) * train_split)

        if split == 'train':
            self.pairs = [all_pairs[i] for i in indices[:split_idx]]
        else:
            self.pairs = [all_pairs[i] for i in indices[split_idx:]]

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


def create_dataloaders(
    modulus_p: int = 97,
    train_split: float = 0.8,
    batch_size: int = 256,
    seed: int = 42,
    num_workers: int = 0  # Set to 0 on Windows to avoid multiprocessing issues
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        modulus_p: Prime modulus
        train_split: Fraction for training
        batch_size: Batch size
        seed: Random seed
        num_workers: Number of workers for data loading

    Returns:
        (train_loader, test_loader)
    """
    train_dataset = ModularArithmeticDataset(
        modulus_p=modulus_p,
        split='train',
        train_split=train_split,
        seed=seed
    )

    test_dataset = ModularArithmeticDataset(
        modulus_p=modulus_p,
        split='test',
        train_split=train_split,
        seed=seed
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Always True for GPU execution
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True  # Always True for GPU execution
    )

    return train_loader, test_loader
