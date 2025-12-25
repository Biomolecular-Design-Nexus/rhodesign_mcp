#!/usr/bin/env python3
"""
Structure loading functions for RhoDesign MCP scripts.

These maintain dependency on repo code because they involve complex
PDB parsing and coordinate extraction that would be error-prone to reimplement.

Original source: repo/RhoDesign/src/util.py (load_structure, extract_coords_from_structure)
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Tuple, Union


def get_repo_path() -> Path:
    """Get path to the RhoDesign repository source."""
    script_dir = Path(__file__).parent
    mcp_root = script_dir.parent.parent  # scripts/lib -> scripts -> root
    repo_src_path = mcp_root / "repo" / "RhoDesign" / "src"

    if not repo_src_path.exists():
        raise FileNotFoundError(
            f"RhoDesign repository not found at {repo_src_path}. "
            f"Please ensure the repo is cloned in the correct location."
        )

    return repo_src_path


def load_structure_and_coords(pdb_path: Union[str, Path]) -> Tuple[np.ndarray, str]:
    """
    Load PDB structure and extract coordinates and sequence.

    This function uses the original RhoDesign implementation for reliability.
    Lazy loading is used to minimize import overhead.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Tuple of (coords, sequence) where:
        - coords: numpy array of shape (L, 7, 3) with atom coordinates
        - sequence: RNA sequence string

    Raises:
        FileNotFoundError: If PDB file or repo not found
        ImportError: If required dependencies are missing
    """
    pdb_path = Path(pdb_path)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    # Add repo to path for imports
    repo_src_path = get_repo_path()
    if str(repo_src_path) not in sys.path:
        sys.path.insert(0, str(repo_src_path))

    try:
        # Lazy import to minimize startup overhead
        from util import load_structure, extract_coords_from_structure

        # Load structure using original implementation
        structure = load_structure(str(pdb_path))

        # Extract coordinates and sequence
        coords, sequence = extract_coords_from_structure(structure)

        return coords, sequence

    except ImportError as e:
        raise ImportError(
            f"Failed to import RhoDesign modules: {e}. "
            f"Please ensure the env_py39 environment is activated and "
            f"all dependencies are installed."
        ) from e

    finally:
        # Clean up sys.path (optional - prevents pollution)
        if str(repo_src_path) in sys.path:
            sys.path.remove(str(repo_src_path))


def load_secondary_structure(ss_path: Union[str, Path]) -> np.ndarray:
    """
    Load secondary structure contact map from numpy file.

    Args:
        ss_path: Path to .npy file containing secondary structure data

    Returns:
        Secondary structure contact map as numpy array

    Raises:
        FileNotFoundError: If file not found
        ValueError: If file format is invalid
    """
    ss_path = Path(ss_path)
    if not ss_path.exists():
        raise FileNotFoundError(f"Secondary structure file not found: {ss_path}")

    try:
        ss_data = np.load(ss_path)
        return ss_data
    except Exception as e:
        raise ValueError(f"Failed to load secondary structure from {ss_path}: {e}") from e


def validate_structure_data(coords: np.ndarray, sequence: str) -> bool:
    """
    Validate that structure data is consistent.

    Args:
        coords: Coordinate array
        sequence: RNA sequence

    Returns:
        True if valid, False otherwise
    """
    if coords is None or sequence is None:
        return False

    if len(coords.shape) != 3 or coords.shape[1] != 7 or coords.shape[2] != 3:
        return False

    if coords.shape[0] != len(sequence):
        return False

    return True