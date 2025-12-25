#!/usr/bin/env python3
"""
Shared utility functions for RhoDesign MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
Original source: repo/RhoDesign/src/util.py, repo/RhoDesign/src/alphabet.py
"""

import numpy as np
import torch
import random
from typing import Sequence, List, Tuple


def set_random_seeds(seed: int = 1) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def seq_recovery_rate(seq1: str, seq2: str) -> float:
    """
    Calculate the sequence recovery rate between two sequences.

    Inlined from repo/RhoDesign/src/util.py:seq_rec_rate

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Recovery rate as fraction (0.0 to 1.0)
    """
    assert len(seq1) == len(seq2), f"Sequences must have same length: {len(seq1)} vs {len(seq2)}"

    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


class SimpleAlphabet:
    """
    Simplified RNA alphabet for MCP use.

    Extracted from repo/RhoDesign/src/alphabet.py and simplified.
    Removes unnecessary complexity for MCP tools.
    """

    def __init__(self, standard_tokens: List[str] = None):
        if standard_tokens is None:
            standard_tokens = ['A', 'G', 'C', 'U', 'X']

        self.standard_tokens = list(standard_tokens)
        self.prepend_tokens = ["<null_0>", "<pad>", "<eos>", "<unk>"]
        self.append_tokens = ["<cls>", "<mask>", "<sep>"]

        self.prepend_bos = True
        self.append_eos = False

        # Build complete token list
        self.all_tokens = list(self.prepend_tokens)
        self.all_tokens.extend(self.standard_tokens)

        # Add null tokens for alignment
        for i in range((8 - (len(self.all_tokens) % 8)) % 8):
            self.all_tokens.append(f"<null_{i + 1}>")
        self.all_tokens.extend(self.append_tokens)

        # Create token-to-index mapping
        self.token_to_idx = {token: i for i, token in enumerate(self.all_tokens)}
        self.idx_to_token = {i: token for token, i in self.token_to_idx.items()}

        # Special indices
        self.unk_idx = self.token_to_idx["<unk>"]
        self.pad_idx = self.token_to_idx["<pad>"]
        self.cls_idx = self.token_to_idx["<cls>"]
        self.mask_idx = self.token_to_idx["<mask>"]
        self.eos_idx = self.token_to_idx["<eos>"]

    def encode(self, sequence: str) -> List[int]:
        """Encode sequence to token indices."""
        tokens = []
        if self.prepend_bos:
            tokens.append(self.cls_idx)

        for char in sequence:
            if char in self.token_to_idx:
                tokens.append(self.token_to_idx[char])
            else:
                tokens.append(self.unk_idx)

        if self.append_eos:
            tokens.append(self.eos_idx)

        return tokens

    def decode(self, tokens: List[int]) -> str:
        """Decode token indices to sequence."""
        chars = []
        for token in tokens:
            if token in self.idx_to_token:
                char = self.idx_to_token[token]
                # Skip special tokens in output
                if char in self.standard_tokens:
                    chars.append(char)
        return ''.join(chars)

    def __len__(self):
        return len(self.all_tokens)


def validate_rna_sequence(sequence: str) -> bool:
    """
    Validate that sequence contains only valid RNA nucleotides.

    Args:
        sequence: RNA sequence string

    Returns:
        True if valid, False otherwise
    """
    valid_chars = set('AGCUX')
    return all(char.upper() in valid_chars for char in sequence)


def format_sequence_for_display(sequence: str, line_length: int = 80) -> str:
    """
    Format sequence for display with line breaks.

    Args:
        sequence: RNA sequence
        line_length: Maximum characters per line

    Returns:
        Formatted sequence string
    """
    lines = []
    for i in range(0, len(sequence), line_length):
        lines.append(sequence[i:i + line_length])
    return '\n'.join(lines)