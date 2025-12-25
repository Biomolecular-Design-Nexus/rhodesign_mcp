#!/usr/bin/env python3
"""
Shared I/O functions for RhoDesign MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
Original source: repo/RhoDesign/src/util.py
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Any, Tuple
import json

def load_json(file_path: Union[str, Path]) -> dict:
    """Load JSON file."""
    with open(file_path) as f:
        return json.load(f)

def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def save_fasta(sequence: str, header: str, file_path: Union[str, Path]) -> None:
    """Save sequence to FASTA file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(f'>{header}\n{sequence}\n')

def load_numpy(file_path: Union[str, Path]) -> np.ndarray:
    """Load numpy array from .npy file."""
    return np.load(file_path)

def save_numpy(data: np.ndarray, file_path: Union[str, Path]) -> None:
    """Save numpy array to .npy file."""
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(file_path, data)