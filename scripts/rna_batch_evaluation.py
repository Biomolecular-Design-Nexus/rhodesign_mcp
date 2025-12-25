#!/usr/bin/env python3
"""
Script: rna_batch_evaluation.py
Description: Batch evaluation and benchmarking of RNA sequence generation

Original Use Case: examples/use_case_3_batch_evaluation.py
Dependencies Removed: Hardcoded paths, complex argument parsing
Dependencies Maintained: RhoDesign model (repo dependency), biotite (structure parsing)

Usage:
    python scripts/rna_batch_evaluation.py --test_dir DIR --ss_dir DIR --output FILE

Example:
    python scripts/rna_batch_evaluation.py \
        --test_dir data/test \
        --ss_dir data/test_ss \
        --output results/batch_results.json \
        --max_files 10
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
from typing import Union, Optional, Dict, Any, List, Tuple
from tqdm import tqdm
import time

# Local utilities
try:
    from lib.io import save_fasta, save_json, load_json
    from lib.utils import set_random_seeds, seq_recovery_rate, SimpleAlphabet
    from lib.structure import load_structure_and_coords, load_secondary_structure, validate_structure_data
except ImportError:
    # Fallback for running directly
    sys.path.insert(0, str(Path(__file__).parent / "lib"))
    import lib.io as rna_io
    save_fasta, save_json, load_json = rna_io.save_fasta, rna_io.save_json, rna_io.load_json
    from utils import set_random_seeds, seq_recovery_rate, SimpleAlphabet
    from structure import load_structure_and_coords, load_secondary_structure, validate_structure_data

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "model_checkpoint": "checkpoint/ss_apexp_best.pth",
    "device": -1,  # -1 for CPU, 0+ for CUDA device
    "temperature": 1.0,
    "random_seed": 1,
    "max_files": None,  # None for all files
    "file_extensions": [".pdb"],
    "statistics": {
        "compute_recovery_stats": True,
        "compute_length_stats": True,
        "save_individual_results": True
    }
}

# ==============================================================================
# Model Arguments Class (simplified from repo)
# ==============================================================================
class ModelArgs:
    """Simplified model arguments for RhoDesign."""
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
# Utility Functions
# ==============================================================================
def find_matching_files(test_dir: Path, ss_dir: Path, max_files: Optional[int] = None) -> List[Tuple[Path, Path]]:
    """
    Find matching PDB and secondary structure files.

    Args:
        test_dir: Directory containing PDB files
        ss_dir: Directory containing secondary structure files
        max_files: Maximum number of files to process

    Returns:
        List of (pdb_path, ss_path) tuples
    """
    pdb_files = list(test_dir.glob("*.pdb"))
    matching_pairs = []

    for pdb_file in pdb_files:
        # Look for matching .npy file
        ss_file = ss_dir / f"{pdb_file.stem}.npy"
        if ss_file.exists():
            matching_pairs.append((pdb_file, ss_file))

    if max_files and len(matching_pairs) > max_files:
        matching_pairs = matching_pairs[:max_files]

    return matching_pairs

def compute_statistics(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute summary statistics from batch results.

    Args:
        results: List of individual result dictionaries

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    recovery_rates = [r["recovery_rate"] for r in results]
    sequence_lengths = [r["sequence_length"] for r in results]
    processing_times = [r.get("processing_time", 0) for r in results]

    stats = {
        "total_files": len(results),
        "successful_files": sum(1 for r in results if r["success"]),
        "failed_files": sum(1 for r in results if not r["success"]),
        "recovery_rate": {
            "mean": np.mean(recovery_rates) if recovery_rates else 0.0,
            "std": np.std(recovery_rates) if recovery_rates else 0.0,
            "min": np.min(recovery_rates) if recovery_rates else 0.0,
            "max": np.max(recovery_rates) if recovery_rates else 0.0,
            "median": np.median(recovery_rates) if recovery_rates else 0.0
        },
        "sequence_length": {
            "mean": np.mean(sequence_lengths) if sequence_lengths else 0.0,
            "std": np.std(sequence_lengths) if sequence_lengths else 0.0,
            "min": int(np.min(sequence_lengths)) if sequence_lengths else 0,
            "max": int(np.max(sequence_lengths)) if sequence_lengths else 0,
            "median": int(np.median(sequence_lengths)) if sequence_lengths else 0
        },
        "processing_time": {
            "total": sum(processing_times),
            "mean": np.mean(processing_times) if processing_times else 0.0,
            "per_file": np.mean(processing_times) if processing_times else 0.0
        }
    }

    return stats

# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_evaluation(
    test_dir: Union[str, Path],
    ss_dir: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform batch evaluation of RNA sequence generation.

    Args:
        test_dir: Directory containing PDB test files
        ss_dir: Directory containing secondary structure files (.npy)
        output_file: Path to save batch results JSON (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - results: List of individual file results
            - statistics: Summary statistics
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_evaluation(
        ...     "data/test",
        ...     "data/test_ss",
        ...     "results/batch_results.json",
        ...     max_files=5
        ... )
        >>> print(f"Mean recovery: {result['statistics']['recovery_rate']['mean']:.3f}")
    """
    # Setup configuration
    test_dir = Path(test_dir)
    ss_dir = Path(ss_dir)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Set random seeds for reproducibility
    set_random_seeds(config["random_seed"])

    # Validate input directories
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not ss_dir.exists():
        raise FileNotFoundError(f"Secondary structure directory not found: {ss_dir}")

    # Setup device
    device_id = config["device"]
    if device_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        print(f"Using GPU: {device}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # Find matching files
    print("Finding matching PDB and secondary structure files...")
    file_pairs = find_matching_files(test_dir, ss_dir, config.get("max_files"))

    if not file_pairs:
        raise ValueError(f"No matching files found in {test_dir} and {ss_dir}")

    print(f"Found {len(file_pairs)} matching file pairs")

    try:
        # Import RhoDesign modules (lazy loading)
        repo_path = Path(__file__).parent.parent / "repo" / "RhoDesign" / "src"
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        from RhoDesign import RhoDesignModel
        from alphabet import Alphabet

        print("\nRhoDesign: Batch Evaluation and Benchmarking")
        print("=" * 50)

        # Initialize model components
        model_args = ModelArgs()
        alphabet = Alphabet(['A', 'G', 'C', 'U', 'X'])

        # Load model
        print("Loading RhoDesign model...")
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

        # Process files
        print(f"\nProcessing {len(file_pairs)} files...")
        results = []
        start_time = time.time()

        for pdb_path, ss_path in tqdm(file_pairs, desc="Processing files"):
            file_start_time = time.time()
            file_result = {
                "pdb_file": str(pdb_path),
                "ss_file": str(ss_path),
                "success": False,
                "error": None,
                "predicted_sequence": None,
                "original_sequence": None,
                "recovery_rate": 0.0,
                "sequence_length": 0,
                "processing_time": 0.0
            }

            try:
                # Load structure and secondary structure
                coords, original_seq = load_structure_and_coords(pdb_path)
                ss_contact_map = load_secondary_structure(ss_path)

                # Validate data
                if not validate_structure_data(coords, original_seq):
                    raise ValueError("Invalid structure data")

                # Run inference
                with torch.no_grad():
                    coords_tensor = torch.from_numpy(coords).unsqueeze(0).float().to(device)
                    ss_tensor = torch.from_numpy(ss_contact_map).unsqueeze(0).float().to(device)

                    # Generate sequence
                    output = model.generate(
                        coords_tensor,
                        ss_tensor,
                        temperature=config["temperature"]
                    )

                    # Decode sequence
                    predicted_seq = alphabet.decode(output[0].cpu().numpy())

                # Calculate metrics
                recovery_rate = seq_recovery_rate(original_seq, predicted_seq)

                # Update result
                file_result.update({
                    "success": True,
                    "predicted_sequence": predicted_seq,
                    "original_sequence": original_seq,
                    "recovery_rate": recovery_rate,
                    "sequence_length": len(predicted_seq),
                    "processing_time": time.time() - file_start_time
                })

            except Exception as e:
                file_result["error"] = str(e)
                file_result["processing_time"] = time.time() - file_start_time

            results.append(file_result)

        total_time = time.time() - start_time

        # Compute statistics
        print("\nComputing statistics...")
        statistics = compute_statistics(results)
        statistics["total_processing_time"] = total_time

        # Print summary
        successful = statistics["successful_files"]
        total = statistics["total_files"]
        mean_recovery = statistics["recovery_rate"]["mean"]

        print(f"\nBatch Evaluation Results:")
        print(f"  Files processed: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {statistics['failed_files']}")
        print(f"  Mean recovery rate: {mean_recovery:.3f}")
        print(f"  Total processing time: {total_time:.2f}s")

        # Prepare final result
        final_result = {
            "results": results,
            "statistics": statistics,
            "output_file": None,
            "metadata": {
                "test_dir": str(test_dir),
                "ss_dir": str(ss_dir),
                "config": config,
                "total_files": len(file_pairs),
                "device": str(device),
                "total_processing_time": total_time
            }
        }

        # Save results if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            save_json(final_result, output_path)
            final_result["output_file"] = str(output_path)
            print(f"Results saved to: {output_path}")

        return final_result

    except Exception as e:
        print(f"Error during batch evaluation: {e}")
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
    parser.add_argument('--test_dir', '-t', required=True,
                      help='Directory containing PDB test files')
    parser.add_argument('--ss_dir', '-s', required=True,
                      help='Directory containing secondary structure files (.npy)')
    parser.add_argument('--output', '-o',
                      help='Path to save batch results JSON file')
    parser.add_argument('--config', '-c',
                      help='Path to configuration JSON file')
    parser.add_argument('--device', type=int, default=DEFAULT_CONFIG["device"],
                      help='Device ID (-1 for CPU, 0+ for CUDA)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG["temperature"],
                      help='Sampling temperature')
    parser.add_argument('--checkpoint', default=DEFAULT_CONFIG["model_checkpoint"],
                      help='Path to model checkpoint file')
    parser.add_argument('--max_files', type=int,
                      help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        config = load_json(args.config)

    # Override with CLI arguments
    config_overrides = {
        "device": args.device,
        "temperature": args.temperature,
        "model_checkpoint": args.checkpoint
    }
    if args.max_files:
        config_overrides["max_files"] = args.max_files

    # Run
    try:
        result = run_batch_evaluation(
            test_dir=args.test_dir,
            ss_dir=args.ss_dir,
            output_file=args.output,
            config=config,
            **config_overrides
        )

        print("\n✅ Batch evaluation completed!")
        stats = result["statistics"]
        print(f"   Files processed: {stats['total_files']}")
        print(f"   Success rate: {stats['successful_files']}/{stats['total_files']}")
        print(f"   Mean recovery rate: {stats['recovery_rate']['mean']:.3f} ± {stats['recovery_rate']['std']:.3f}")
        print(f"   Processing time: {stats['total_processing_time']:.2f}s")

        if result['output_file']:
            print(f"   Results saved to: {result['output_file']}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


if __name__ == '__main__':
    main()