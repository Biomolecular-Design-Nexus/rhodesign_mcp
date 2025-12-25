#!/usr/bin/env python3
"""
Script: rna_structure_to_sequence_no_ss.py
Description: RNA structure-to-sequence design using only 3D structure (no secondary structure)

Original Use Case: examples/use_case_2_structure_to_sequence_no_ss.py
Dependencies Removed: Hardcoded paths, complex argument parsing
Dependencies Maintained: RhoDesign model (repo dependency), biotite (structure parsing)

Usage:
    python scripts/rna_structure_to_sequence_no_ss.py --pdb FILE --output FILE

Example:
    python scripts/rna_structure_to_sequence_no_ss.py \
        --pdb examples/data/2zh6_B.pdb \
        --output results/output.fasta
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import sys
import os
import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Dict, Any

# Local utilities
try:
    from lib.io import save_fasta, save_json
    from lib.utils import set_random_seeds, seq_recovery_rate, SimpleAlphabet
    from lib.structure import load_structure_and_coords, validate_structure_data
except ImportError:
    # Fallback for running directly
    sys.path.insert(0, str(Path(__file__).parent / "lib"))
    import lib.io as rna_io
    save_fasta, save_json = rna_io.save_fasta, rna_io.save_json
    from utils import set_random_seeds, seq_recovery_rate, SimpleAlphabet
    from structure import load_structure_and_coords, validate_structure_data

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_checkpoint": "checkpoint/no_ss_apexp_best.pth",
    "device": -1,  # -1 for CPU, 0+ for CUDA device
    "temperature": 1.0,
    "random_seed": 1,
    "max_sequence_length": 1000,
    "model_args": {
        # These will be loaded from the model or config
        "d_model": 512,
        "n_head": 8,
        "n_layers": 8
    }
}

# ==============================================================================
# Model Arguments Class (simplified from repo)
# ==============================================================================
class ModelArgs:
    """Simplified model arguments for RhoDesign without secondary structure."""
    def __init__(self):
        self.encoder_layers = 8
        self.encoder_attention_heads = 8
        self.encoder_embed_dim = 512
        self.encoder_ffn_embed_dim = 2048
        self.activation_fn = 'gelu'
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.max_positions = 1024
        self.embed_scale = None
        self.encoder_normalize_before = False
        self.no_encoder_attn = False
        self.encoder_learned_pos = False

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_structure_to_sequence_no_ss(
    pdb_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate RNA sequence from 3D structure only (no secondary structure constraints).

    Args:
        pdb_file: Path to PDB file containing 3D structure
        output_file: Path to save output FASTA file (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - predicted_sequence: Generated RNA sequence
            - original_sequence: Original sequence from PDB
            - recovery_rate: Sequence recovery rate (similarity to original)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_structure_to_sequence_no_ss(
        ...     "examples/data/2zh6_B.pdb",
        ...     "output/result.fasta"
        ... )
        >>> print(f"Recovery rate: {result['recovery_rate']:.3f}")
    """
    # Setup configuration
    pdb_file = Path(pdb_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Set random seeds for reproducibility
    set_random_seeds(config["random_seed"])

    # Validate input files
    if not pdb_file.exists():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")

    # Setup device
    device_id = config["device"]
    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    try:
        # Import RhoDesign modules (lazy loading)
        repo_path = Path(__file__).parent.parent / "repo" / "RhoDesign" / "src"
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from RhoDesign_without2d import RhoDesignModel
        from alphabet import Alphabet

        print("RhoDesign: RNA Structure-to-Sequence Design (3D Structure Only)")
        print("=" * 70)

        # Initialize model components
        model_args = ModelArgs()
        alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])

        # Load model
        print("Loading RhoDesign model (without secondary structure)...")
        model = RhoDesignModel(model_args, alphabet)
        model = model.to(device)

        # Load checkpoint
        checkpoint_path = Path(config["model_checkpoint"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}. "
                f"Please download from Google Drive: "
                f"https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz"
            )

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"Model loaded from: {checkpoint_path}")

        # Load input data
        print("Loading input structure...")
        coords, original_seq = load_structure_and_coords(pdb_file)

        # Validate data
        if not validate_structure_data(coords, original_seq):
            raise ValueError("Invalid structure data")

        print(f"Structure loaded: {len(original_seq)} residues")
        print(f"Original sequence: {original_seq}")

        # Run inference
        print("Generating sequence...")
        with torch.no_grad():
            # Convert to tensors
            coords_tensor = torch.from_numpy(coords).unsqueeze(0).float().to(device)

            # Generate sequence (no secondary structure input)
            output = model.generate(
                coords_tensor,
                temperature=config["temperature"]
            )

            # Decode sequence
            predicted_seq = alphabet.decode(output[0].cpu().numpy())

        # Calculate recovery rate
        recovery_rate = seq_recovery_rate(original_seq, predicted_seq)

        print(f"Predicted sequence: {predicted_seq}")
        print(f"Recovery rate: {recovery_rate:.3f}")

        # Prepare results
        result = {
            "predicted_sequence": predicted_seq,
            "original_sequence": original_seq,
            "recovery_rate": recovery_rate,
            "output_file": None,
            "metadata": {
                "pdb_file": str(pdb_file),
                "config": config,
                "sequence_length": len(predicted_seq),
                "device": str(device),
                "model_type": "no_secondary_structure"
            }
        }

        # Save output if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save FASTA
            pdb_name = pdb_file.stem
            header = f"RhoDesign_prediction_no_SS_{pdb_name}_recovery_{recovery_rate:.3f}"
            save_fasta(predicted_seq, header, output_path)

            # Save metadata
            metadata_path = output_path.with_suffix('.json')
            save_json(result["metadata"], metadata_path)

            result["output_file"] = str(output_path)
            print(f"Results saved to: {output_path}")

        return result

    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you're running in the env_py39 environment with RhoDesign dependencies installed.")
        raise


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--pdb', '-p', required=True,
                      help='Path to PDB file containing 3D structure')
    parser.add_argument('--output', '-o',
                      help='Path to save output FASTA file')
    parser.add_argument('--config', '-c',
                      help='Path to configuration JSON file')
    parser.add_argument('--device', type=int, default=DEFAULT_CONFIG["device"],
                      help='Device ID (-1 for CPU, 0+ for CUDA)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG["temperature"],
                      help='Sampling temperature')
    parser.add_argument('--checkpoint', default=DEFAULT_CONFIG["model_checkpoint"],
                      help='Path to model checkpoint file')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        from lib.io import load_json
        config = load_json(args.config)

    # Override with CLI arguments
    config_overrides = {
        "device": args.device,
        "temperature": args.temperature,
        "model_checkpoint": args.checkpoint
    }

    # Run
    try:
        result = run_structure_to_sequence_no_ss(
            pdb_file=args.pdb,
            output_file=args.output,
            config=config,
            **config_overrides
        )

        print("\n✅ Success!")
        print(f"   Predicted sequence length: {len(result['predicted_sequence'])}")
        print(f"   Recovery rate: {result['recovery_rate']:.3f}")
        if result['output_file']:
            print(f"   Output saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


if __name__ == '__main__':
    main()