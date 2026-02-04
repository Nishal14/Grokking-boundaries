"""
Fixed vocabulary tokenizer for modular arithmetic sequences.
"""
from typing import List, Union


class ModularArithmeticTokenizer:
    """
    Tokenizer for sequences of the form: <bos> a + b mod p = c <eos>

    Vocabulary:
        - Tokens 0-97: integers 0 to p (where p=97, includes the modulus value)
        - Token 98: '+'
        - Token 99: 'mod'
        - Token 100: '='
        - Token 101: '<bos>'
        - Token 102: '<eos>'
        - Token 103: '<pad>'

    Total vocab size: 104
    """

    def __init__(self, modulus_p: int = 97):
        """
        Initialize tokenizer with fixed vocabulary.

        Args:
            modulus_p: Prime modulus (default: 97)
        """
        self.modulus_p = modulus_p
        self.vocab = self._build_vocab()
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab_size = len(self.vocab)

        # Special token IDs
        self.bos_token_id = self.token_to_id['<bos>']
        self.eos_token_id = self.token_to_id['<eos>']
        self.pad_token_id = self.token_to_id['<pad>']

    def _build_vocab(self) -> List[str]:
        """Build vocabulary list."""
        vocab = []

        # Add integers 0 to p (inclusive, to include the modulus value in sequences)
        for i in range(self.modulus_p + 1):
            vocab.append(str(i))

        # Add operators and symbols
        vocab.extend(['+', 'mod', '='])

        # Add special tokens
        vocab.extend(['<bos>', '<eos>', '<pad>'])

        return vocab

    def encode(self, text: str) -> List[int]:
        """
        Convert text sequence to token IDs.

        Args:
            text: String of the form "<bos> a + b mod p = c <eos>"

        Returns:
            List of token IDs

        Example:
            >>> tokenizer.encode("<bos> 5 + 3 mod 97 = 8 <eos>")
            [100, 5, 97, 3, 98, 97, 99, 8, 101]
        """
        tokens = text.split()
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to text.

        Args:
            ids: List of token IDs

        Returns:
            String representation
        """
        tokens = [self.id_to_token[id] for id in ids if id != self.pad_token_id]
        return ' '.join(tokens)

    def pad_sequence(self, ids: List[int], max_len: int) -> List[int]:
        """
        Pad sequence to max_len with <pad> token.

        Args:
            ids: List of token IDs
            max_len: Target length

        Returns:
            Padded list of token IDs
        """
        if len(ids) >= max_len:
            return ids[:max_len]
        return ids + [self.pad_token_id] * (max_len - len(ids))

    def __len__(self):
        """Return vocabulary size."""
        return self.vocab_size
