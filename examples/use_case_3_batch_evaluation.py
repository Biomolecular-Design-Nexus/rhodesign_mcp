#!/usr/bin/env python3
"""
RhoDesign Use Case 3: Batch Evaluation and Benchmarking

This script performs batch evaluation of the RhoDesign model on multiple PDB structures.
It processes a directory of PDB files and their corresponding secondary structure files,
generating sequences and computing recovery rates for benchmarking purposes.

Usage:
    python examples/use_case_3_batch_evaluation.py --test_dir data/test/ --ss_dir data/test_ss/ --output results/

Requirements:
    - Needs to run in the env_py39 environment
    - Requires model checkpoint: checkpoint/ss_apexp_best.pth
    - GPU recommended (CUDA)
    - Test data directory with PDB files
    - Secondary structure directory with corresponding .npy files

Features:
    - Batch processing of multiple structures
    - Comprehensive benchmarking and recovery rate statistics
    - Progress tracking with tqdm
    - Results summary and analysis
"""

import sys
import os
import argparse
import numpy as np
import torch
import random
from pathlib import Path
from tqdm import tqdm

# Add the repo source directory to the path
repo_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'repo', 'RhoDesign', 'src'))
sys.path.insert(0, repo_src_path)

# Set random seeds for reproducibility
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def main():
    parser = argparse.ArgumentParser(
        description='RhoDesign: Batch evaluation and benchmarking',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--test_dir', type=str, default='data/test/',
                      help='Directory containing test PDB files')
    parser.add_argument('--ss_dir', type=str, default='data/test_ss/',
                      help='Directory containing secondary structure .npy files')
    parser.add_argument('-o', '--output', type=str, default='results/',
                      help='Output directory for results')
    parser.add_argument('-t', '--temperature', type=float, default=1e-5,
                      help='Temperature for sampling (lower = more conservative)')
    parser.add_argument('-d', '--device', type=int, default=0,
                      help='GPU device ID (use -1 for CPU)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/ss_apexp_best.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--max_files', type=int, default=None,
                      help='Maximum number of files to process (for testing)')

    args = parser.parse_args()

    # Check inputs
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        print("You may need to download the test data from Google Drive")
        return 1

    if not os.path.exists(args.ss_dir):
        print(f"Error: Secondary structure directory not found: {args.ss_dir}")
        print("You may need to download the test data from Google Drive")
        return 1

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model checkpoint not found: {args.checkpoint}")
        print("You may need to download the model checkpoint from Google Drive:")
        print("https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz?usp=sharing")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    try:
        # Import required modules from RhoDesign
        from RhoDesign import RhoDesignModel
        from alphabet import Alphabet
        from util import load_structure, extract_coords_from_structure, seq_rec_rate

        print("RhoDesign: Batch Evaluation and Benchmarking")
        print("=" * 50)
        print(f"Test directory: {args.test_dir}")
        print(f"Secondary structure directory: {args.ss_dir}")
        print(f"Output directory: {args.output}")
        print(f"Temperature: {args.temperature}")
        print(f"Device: {'GPU ' + str(args.device) if args.device >= 0 else 'CPU'}")
        print()

        # Get list of PDB files
        pdb_files = [f for f in os.listdir(args.test_dir) if f.endswith('.pdb')]
        if args.max_files:
            pdb_files = pdb_files[:args.max_files]

        if not pdb_files:
            print("No PDB files found in test directory")
            return 1

        print(f"Found {len(pdb_files)} PDB files to process")
        print()

        # Model configuration (same as original)
        class ModelArgs:
            def __init__(self):
                self.encoder_embed_dim = 512
                self.decoder_embed_dim = 512
                self.dropout = 0.1
                self.local_rank = int(os.getenv("LOCAL_RANK", -1))
                self.device_id = [0, 1, 2, 3, 4, 5, 6, 7]
                self.epochs = 100
                self.lr = 1e-5
                self.batch_size = 1
                self.gvp_top_k_neighbors = 15
                self.gvp_node_hidden_dim_vector = 256
                self.gvp_node_hidden_dim_scalar = 512
                self.gvp_edge_hidden_dim_scalar = 32
                self.gvp_edge_hidden_dim_vector = 1
                self.gvp_num_encoder_layers = 3
                self.gvp_dropout = 0.1
                self.encoder_layers = 3
                self.encoder_attention_heads = 4
                self.attention_dropout = 0.1
                self.encoder_ffn_embed_dim = 512
                self.decoder_layers = 3
                self.decoder_attention_heads = 4
                self.decoder_ffn_embed_dim = 512

        # Initialize model
        model_args = ModelArgs()
        alphabet = Alphabet(['A','G','C','U','X'])

        # Setup device
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device(f'cuda:{args.device}')
            print(f"Using GPU: {device}")
        else:
            device = torch.device('cpu')
            print("Using CPU")

        # Load model
        print("Loading RhoDesign model...")
        model = RhoDesignModel(model_args, alphabet)
        model = model.to(device)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
        print()

        # Process files
        recovery_rates = []
        results = []

        print("Processing structures...")
        for pdb_file in tqdm(pdb_files, desc="Evaluating structures"):
            try:
                # File paths
                pdb_path = os.path.join(args.test_dir, pdb_file)
                pdb_name = os.path.splitext(pdb_file)[0]
                ss_file = pdb_name + '.npy'
                ss_path = os.path.join(args.ss_dir, ss_file)

                # Check if secondary structure file exists
                if not os.path.exists(ss_path):
                    print(f"Warning: Secondary structure file not found for {pdb_file}, skipping...")
                    continue

                # Load structure and secondary structure
                pdb_structure = load_structure(pdb_path)
                coords, original_seq = extract_coords_from_structure(pdb_structure)
                ss_contact_map = np.load(ss_path)

                # Generate sequence
                with torch.no_grad():
                    predicted_seq = model.sample(coords, ss_contact_map, device, temperature=args.temperature)

                # Calculate recovery rate
                recovery_rate = seq_rec_rate(original_seq, predicted_seq)
                recovery_rates.append(recovery_rate)

                # Store results
                result = {
                    'pdb_file': pdb_file,
                    'length': len(original_seq),
                    'original_seq': original_seq,
                    'predicted_seq': predicted_seq,
                    'recovery_rate': recovery_rate
                }
                results.append(result)

                # Save individual result
                output_file = os.path.join(args.output, f'{pdb_name}_predicted.fasta')
                with open(output_file, 'w') as f:
                    f.write(f'>{pdb_name}_RhoDesign_prediction\n')
                    f.write(predicted_seq + '\n')

            except Exception as e:
                print(f"Error processing {pdb_file}: {e}")
                continue

        # Compute and save statistics
        if recovery_rates:
            mean_recovery = np.mean(recovery_rates)
            std_recovery = np.std(recovery_rates)
            min_recovery = np.min(recovery_rates)
            max_recovery = np.max(recovery_rates)

            print()
            print("Evaluation Results:")
            print("=" * 30)
            print(f"Structures processed: {len(results)}")
            print(f"Mean recovery rate: {mean_recovery:.4f} ± {std_recovery:.4f}")
            print(f"Min recovery rate:  {min_recovery:.4f}")
            print(f"Max recovery rate:  {max_recovery:.4f}")

            # Save detailed results
            results_file = os.path.join(args.output, 'batch_evaluation_results.txt')
            with open(results_file, 'w') as f:
                f.write("RhoDesign Batch Evaluation Results\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Test directory: {args.test_dir}\n")
                f.write(f"Model checkpoint: {args.checkpoint}\n")
                f.write(f"Temperature: {args.temperature}\n")
                f.write(f"Device: {device}\n\n")

                f.write("Summary Statistics:\n")
                f.write(f"Structures processed: {len(results)}\n")
                f.write(f"Mean recovery rate: {mean_recovery:.4f} ± {std_recovery:.4f}\n")
                f.write(f"Min recovery rate:  {min_recovery:.4f}\n")
                f.write(f"Max recovery rate:  {max_recovery:.4f}\n\n")

                f.write("Individual Results:\n")
                f.write("PDB_File\tLength\tRecovery_Rate\tOriginal_Sequence\tPredicted_Sequence\n")
                for result in results:
                    f.write(f"{result['pdb_file']}\t{result['length']}\t{result['recovery_rate']:.4f}\t{result['original_seq']}\t{result['predicted_seq']}\n")

            print(f"Detailed results saved to: {results_file}")

        else:
            print("No structures were successfully processed")

        return 0

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this in the env_py39 environment with RhoDesign dependencies installed.")
        return 1
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1

if __name__ == '__main__':
    exit(main())