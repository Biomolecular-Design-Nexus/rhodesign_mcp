# RhoDesign MCP Configuration Files

Configuration files for RhoDesign MCP scripts, extracted from use case parameters.

## Configuration Files

| File | Purpose | Script |
|------|---------|--------|
| `default_config.json` | Global defaults for all scripts | All |
| `structure_to_sequence_with_ss_config.json` | With secondary structure constraints | `rna_structure_to_sequence_with_ss.py` |
| `structure_to_sequence_no_ss_config.json` | 3D structure only | `rna_structure_to_sequence_no_ss.py` |
| `batch_evaluation_config.json` | Batch processing settings | `rna_batch_evaluation.py` |

## Configuration Structure

### Model Configuration
```json
{
  "model": {
    "checkpoint": "checkpoint/ss_apexp_best.pth",
    "type": "with_secondary_structure",
    "args": {
      "encoder_layers": 8,
      "encoder_attention_heads": 8,
      "encoder_embed_dim": 512,
      "activation_fn": "gelu",
      "dropout": 0.1
    }
  }
}
```

### Inference Configuration
```json
{
  "inference": {
    "device": -1,           // -1 for CPU, 0+ for CUDA device
    "temperature": 1.0,     // Sampling temperature
    "random_seed": 1,       // For reproducibility
    "max_sequence_length": 1000
  }
}
```

### Input/Output Configuration
```json
{
  "input": {
    "pdb_file": "examples/data/2zh6_B.pdb",
    "ss_file": "examples/data/2zh6_B.npy",
    "required_formats": ["pdb", "npy"]
  },
  "output": {
    "format": "fasta",
    "include_metadata": true,
    "save_recovery_rate": true
  }
}
```

## Usage

### Loading Configuration
```python
from scripts.lib.io import load_json

config = load_json("configs/structure_to_sequence_with_ss_config.json")
result = run_structure_to_sequence_with_ss(
    pdb_file="input.pdb",
    ss_file="input.npy",
    config=config
)
```

### CLI Usage
```bash
python scripts/rna_structure_to_sequence_with_ss.py \
    --config configs/structure_to_sequence_with_ss_config.json \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy
```

### Runtime Overrides
```python
# Config file provides defaults, but can be overridden
result = run_structure_to_sequence_with_ss(
    pdb_file="input.pdb",
    ss_file="input.npy",
    config=config,
    device=0,              # Override: use GPU instead of CPU
    temperature=0.8        # Override: lower temperature
)
```

## Configuration Hierarchy

1. **Default values** in script `DEFAULT_CONFIG`
2. **Config file** values (override defaults)
3. **CLI arguments** (override config file)
4. **Function kwargs** (override everything)

## Model Checkpoints

The configuration files reference model checkpoints that must be downloaded:

```json
{
  "model_checkpoints": {
    "with_ss": {
      "path": "checkpoint/ss_apexp_best.pth",
      "url": "https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz"
    },
    "no_ss": {
      "path": "checkpoint/no_ss_apexp_best.pth",
      "url": "https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz"
    }
  }
}
```

## Environment Settings

Configurations include environment requirements:

```json
{
  "environment": {
    "python_version": "3.9",
    "conda_env": "./env_py39",
    "required_packages": [
      "torch==1.12.1",
      "numpy>=1.20.0",
      "biotite>=0.37.0",
      "tqdm>=4.60.0",
      "torch-geometric>=2.0.0"
    ]
  }
}
```

## Creating Custom Configurations

1. **Copy a base configuration**:
   ```bash
   cp configs/structure_to_sequence_with_ss_config.json configs/my_config.json
   ```

2. **Modify parameters**:
   ```json
   {
     "inference": {
       "device": 0,        // Use GPU
       "temperature": 0.5  // Lower temperature for more conservative predictions
     }
   }
   ```

3. **Use with script**:
   ```bash
   python scripts/rna_structure_to_sequence_with_ss.py \
       --config configs/my_config.json \
       --pdb my_structure.pdb \
       --ss my_structure.npy
   ```