# RhoDesign MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Utility functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `rna_structure_to_sequence_with_ss.py` | Generate RNA sequence from 3D structure + secondary structure | Yes (model) | `configs/structure_to_sequence_with_ss_config.json` |
| `rna_structure_to_sequence_no_ss.py` | Generate RNA sequence from 3D structure only | Yes (model) | `configs/structure_to_sequence_no_ss_config.json` |
| `rna_batch_evaluation.py` | Batch processing and benchmarking | Yes (model) | `configs/batch_evaluation_config.json` |

## Dependencies

### Essential Dependencies (All Scripts)
- `numpy` - Numerical computing
- `torch` - PyTorch framework
- `pathlib` - Path handling

### Repo Dependencies (Cannot be eliminated)
- **RhoDesign models**: Complex ML models requiring repo code
- **biotite**: PDB structure parsing (complex format)
- **Model checkpoints**: Must be downloaded separately

### Inlined Dependencies
- Sequence recovery rate calculation (from `util.py`)
- RNA alphabet handling (simplified from `alphabet.py`)
- Basic I/O functions (FASTA, JSON, numpy)

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env_py39  # or: conda activate ./env_py39

# Single structure with secondary structure
python scripts/rna_structure_to_sequence_with_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy \
    --output results/output_with_ss.fasta

# Single structure without secondary structure
python scripts/rna_structure_to_sequence_no_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --output results/output_no_ss.fasta

# Batch evaluation (requires test datasets)
python scripts/rna_batch_evaluation.py \
    --test_dir data/test \
    --ss_dir data/test_ss \
    --output results/batch_results.json \
    --max_files 5

# With custom config
python scripts/rna_structure_to_sequence_with_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy \
    --config configs/custom_config.json
```

## Shared Library

Common functions are in `scripts/lib/`:
- `io.py`: File loading/saving (FASTA, JSON, numpy)
- `utils.py`: RNA utilities (alphabet, validation, recovery rate)
- `structure.py`: Structure loading (maintains repo dependency)

## Model Dependencies

### Required Checkpoints
Scripts require model checkpoints that must be downloaded manually:

```bash
# Create checkpoint directory
mkdir -p checkpoint/

# Download from Google Drive (manual step):
# https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz
# - ss_apexp_best.pth (~50MB)
# - no_ss_apexp_best.pth (~50MB)
```

### Environment Requirements
- **Python**: 3.9 (env_py39)
- **PyTorch**: 1.12.1+cu113
- **CUDA**: 11.3 or CPU mode
- **Memory**: ~2GB for model loading

## Configuration

Each script supports configuration files in `configs/`:

```json
{
  "model": {
    "checkpoint": "checkpoint/ss_apexp_best.pth",
    "type": "with_secondary_structure"
  },
  "inference": {
    "device": -1,
    "temperature": 1.0,
    "random_seed": 1
  },
  "output": {
    "format": "fasta",
    "include_metadata": true
  }
}
```

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from scripts.rna_structure_to_sequence_with_ss import run_structure_to_sequence_with_ss

# In MCP tool:
@mcp.tool()
def predict_rna_with_ss(pdb_file: str, ss_file: str, output_file: str = None):
    return run_structure_to_sequence_with_ss(pdb_file, ss_file, output_file)
```

## Performance Notes

- **CPU Mode**: Use `--device -1` for CPU-only execution
- **Memory**: ~2GB required for model loading
- **Speed**: ~10-30 seconds per sequence on CPU
- **Batch**: Progress bars with tqdm for long operations

## Error Handling

Scripts provide clear error messages for common issues:
- Missing model checkpoints → Download instructions
- Wrong environment → Environment activation reminder
- Invalid files → File format validation
- CUDA issues → CPU fallback suggestion

## Testing

Test with provided example data:
```bash
# Quick test (should work if models are downloaded)
python scripts/rna_structure_to_sequence_with_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy \
    --output test_output.fasta \
    --device -1
```