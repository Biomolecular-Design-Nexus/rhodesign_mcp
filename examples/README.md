# RhoDesign Use Cases and Examples

This directory contains standalone Python scripts demonstrating the main use cases of RhoDesign for RNA inverse folding and structure-to-sequence design.

## Overview

RhoDesign is a structure-to-sequence model for RNA design that leverages Geometric Vector Perceptrons (GVP) encoding and a Transformer encoder-decoder to capture structural details and generate meaningful RNA sequences.

## Use Cases

### UC-001: RNA Structure-to-Sequence Design with Secondary Structure
**Script**: `use_case_1_structure_to_sequence_with_ss.py`

Performs RNA inverse folding using both 3D structure and secondary structure constraints.

**Features:**
- Uses both 3D structure (PDB) and secondary structure (contact map) as constraints
- Temperature-controlled sampling for diversity vs accuracy trade-off
- Calculates sequence recovery rate
- Saves output in FASTA format

**Usage:**
```bash
# Activate the legacy Python 3.9 environment
mamba activate ./env_py39

# Basic usage with default example data
python examples/use_case_1_structure_to_sequence_with_ss.py

# Custom usage
python examples/use_case_1_structure_to_sequence_with_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy \
    --output output/ \
    --temperature 1e-5
```

**Inputs:**
- PDB file: 3D structure coordinates
- NPY file: Secondary structure contact map
- Temperature: Sampling temperature (1e-5 for high accuracy, 1.0 for diversity)

**Outputs:**
- FASTA file with predicted RNA sequence
- Recovery rate compared to original sequence

---

### UC-002: RNA Structure-to-Sequence Design without Secondary Structure
**Script**: `use_case_2_structure_to_sequence_no_ss.py`

Performs RNA inverse folding using only 3D structure constraints (no secondary structure required).

**Features:**
- Uses only 3D structure constraints
- No secondary structure input required
- Temperature-controlled sampling
- Calculates sequence recovery rate

**Usage:**
```bash
# Activate the legacy Python 3.9 environment
mamba activate ./env_py39

# Basic usage with default example data
python examples/use_case_2_structure_to_sequence_no_ss.py

# Custom usage
python examples/use_case_2_structure_to_sequence_no_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --output output/ \
    --temperature 1.0
```

**Inputs:**
- PDB file: 3D structure coordinates
- Temperature: Sampling temperature

**Outputs:**
- FASTA file with predicted RNA sequence
- Recovery rate compared to original sequence

---

### UC-003: Batch Evaluation and Benchmarking
**Script**: `use_case_3_batch_evaluation.py`

Performs batch evaluation of the RhoDesign model on multiple PDB structures for benchmarking purposes.

**Features:**
- Batch processing of multiple structures
- Comprehensive benchmarking and recovery rate statistics
- Progress tracking with tqdm
- Results summary and analysis

**Usage:**
```bash
# Activate the legacy Python 3.9 environment
mamba activate ./env_py39

# Batch evaluation (requires test dataset)
python examples/use_case_3_batch_evaluation.py \
    --test_dir data/test/ \
    --ss_dir data/test_ss/ \
    --output results/ \
    --max_files 10
```

**Inputs:**
- Test directory: Directory containing PDB files
- SS directory: Directory containing corresponding .npy secondary structure files
- Max files: Limit number of files for testing

**Outputs:**
- Individual FASTA files for each prediction
- Comprehensive results summary with statistics
- Recovery rate analysis

---

## Demo Data

The `examples/data/` directory contains example data:

- `2zh6_B.pdb`: Example RNA 3D structure (PDB format)
- `2zh6_B.npy`: Corresponding secondary structure contact map
- `pred_seq.fasta`: Example prediction output

## Requirements

**Environment**: All scripts must be run in the `./env_py39` environment (Python 3.9) with PyTorch and RhoDesign dependencies.

**Model Checkpoints**: Download from Google Drive:
- [RhoDesign Model Checkpoints](https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz?usp=sharing)

Required checkpoints:
- `checkpoint/ss_apexp_best.pth` - Model with secondary structure constraints
- `checkpoint/no_ss_apexp_best.pth` - Model without secondary structure constraints

**Hardware**: GPU recommended for faster inference (CUDA support required)

## Getting Started

1. **Activate environment:**
   ```bash
   mamba activate ./env_py39
   ```

2. **Download model checkpoints** from Google Drive and place in `checkpoint/` directory

3. **Run a simple example:**
   ```bash
   python examples/use_case_1_structure_to_sequence_with_ss.py
   ```

## Notes

- Temperature parameter controls the trade-off between accuracy and diversity:
  - `1e-5`: Very conservative, high recovery rate
  - `1.0`: More diverse sequences, lower recovery rate
- GPU is recommended but CPU fallback is available
- All scripts include comprehensive error handling and help documentation
- Use `--help` flag with any script to see all available options

## Troubleshooting

- **Import errors**: Ensure you're using the `./env_py39` environment
- **Missing checkpoints**: Download from Google Drive link above
- **CUDA errors**: Use `--device -1` to force CPU usage
- **File not found**: Check paths to input files and model checkpoints