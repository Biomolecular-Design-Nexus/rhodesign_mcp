#!/usr/bin/env python3
"""
RhoDesign Use Case 2: RNA Structure-to-Sequence Design without Secondary Structure

This script performs RNA inverse folding using RhoDesign model with only 3D structure
constraints (no secondary structure). It takes a PDB file and generates RNA sequences
based purely on the 3D structural information.

Usage:
    python examples/use_case_2_structure_to_sequence_no_ss.py --pdb examples/data/2zh6_B.pdb --output output/

Requirements:
    - Needs to run in the env_py39 environment
    - Requires model checkpoint: checkpoint/no_ss_apexp_best.pth
    - GPU recommended (CUDA)

Features:
    - Uses only 3D structure constraints (no secondary structure input required)
    - Temperature-controlled sampling for diversity vs accuracy trade-off
    - Calculates sequence recovery rate
    - Saves output in FASTA format
"""

import sys
import os
import argparse
import numpy as np
import torch
import random
from pathlib import Path

# Add the repo source directory to the path
repo_src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'repo', 'RhoDesign', 'src'))
sys.path.insert(0, repo_src_path)

# Set random seeds for reproducibility
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

def main():
    parser = argparse.ArgumentParser(
        description='RhoDesign: RNA structure-to-sequence design without secondary structure constraints',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-p', '--pdb', type=str, default='examples/data/2zh6_B.pdb',
                      help='Path to input PDB file')
    parser.add_argument('-o', '--output', type=str, default='output/',
                      help='Output directory for results')
    parser.add_argument('-t', '--temperature', type=float, default=1.0,
                      help='Temperature for sampling (lower = more conservative)')
    parser.add_argument('-d', '--device', type=int, default=0,
                      help='GPU device ID (use -1 for CPU)')
    parser.add_argument('--checkpoint', type=str, default='checkpoint/no_ss_apexp_best.pth',
                      help='Path to model checkpoint')

    args = parser.parse_args()

    # Check inputs
    if not os.path.exists(args.pdb):
        print(f"Error: PDB file not found: {args.pdb}")
        print("You may need to download the model checkpoint and data from Google Drive")
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
        from RhoDesign_without2d import RhoDesignModel
        from alphabet import Alphabet
        from util import load_structure, extract_coords_from_structure, seq_rec_rate

        print("RhoDesign: RNA Structure-to-Sequence Design (3D Structure Only)")
        print("=" * 70)
        print(f"Input PDB: {args.pdb}")
        print(f"Output directory: {args.output}")
        print(f"Temperature: {args.temperature}")
        print(f"Device: {'GPU ' + str(args.device) if args.device >= 0 else 'CPU'}")
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
        print("Loading RhoDesign model (without secondary structure)...")
        model = RhoDesignModel(model_args, alphabet)
        model = model.to(device)

        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        print("Model loaded successfully!")
        print()

        # Load input data
        print("Loading input structure...")
        pdb_structure = load_structure(args.pdb)
        coords, original_seq = extract_coords_from_structure(pdb_structure)

        print(f"Structure loaded: {len(original_seq)} residues")
        print(f"Original sequence: {original_seq}")
        print()

        # Generate sequence
        print("Generating RNA sequence (3D structure only)...")
        with torch.no_grad():
            predicted_seq = model.sample(coords, device, temperature=args.temperature)

        # Calculate recovery rate
        recovery_rate = seq_rec_rate(original_seq, predicted_seq)

        # Save results
        pdb_name = os.path.splitext(os.path.basename(args.pdb))[0]
        output_file = os.path.join(args.output, f'{pdb_name}_without2d.fasta')
        with open(output_file, 'w') as f:
            f.write(f'>{pdb_name}_without2d\n')
            f.write(predicted_seq + '\n')

        # Print results
        print("Results:")
        print(f"Original sequence:  {original_seq}")
        print(f"Predicted sequence: {predicted_seq}")
        print(f"Recovery rate: {recovery_rate:.4f}")
        print(f"Output saved to: {output_file}")

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