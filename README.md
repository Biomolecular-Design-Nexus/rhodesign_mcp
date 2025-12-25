# RhoDesign MCP

> AI-powered RNA sequence design from 3D structure using deep learning inverse folding

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

RhoDesign MCP provides AI-powered RNA sequence design from 3D structures using transformer-based inverse folding models. Given a PDB structure file (with optional secondary structure constraints), it generates RNA sequences that are likely to fold into the given 3D structure.

### Features
- **Dual Design Modes**: With or without secondary structure constraints
- **High Accuracy**: 60-90% sequence recovery rates on benchmark datasets
- **Batch Processing**: Handle multiple structures efficiently
- **GPU/CPU Support**: Automatic device detection with CPU fallback
- **Job Management**: Background processing for large datasets
- **Multiple Interfaces**: Direct scripts, MCP server, or Claude Code integration

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment (Python 3.10 for MCP)
├── env_py39/              # Legacy environment (Python 3.9 for RhoDesign)
├── src/
│   └── server.py           # MCP server with 14 tools
├── scripts/
│   ├── rna_structure_to_sequence_with_ss.py    # RNA design with secondary structure
│   ├── rna_structure_to_sequence_no_ss.py      # RNA design (3D structure only)
│   ├── rna_batch_evaluation.py                 # Batch processing and benchmarking
│   └── lib/                                     # Shared utilities
├── examples/
│   └── data/               # Demo data (2zh6_B.pdb, 2zh6_B.npy, pred_seq.fasta)
├── configs/                # Configuration files
├── checkpoint/             # Model files (download required)
└── repo/                   # Original RhoDesign repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.9+ and 3.10+
- CUDA 11.3+ (optional, for GPU acceleration)
- 8GB+ RAM
- ~5GB storage for environments

### Create Environment
Please strictly follow the information in `reports/step3_environment.md` for the complete setup procedure. The workflow below shows the essential steps:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhodesign_mcp

# Create main MCP environment (Python 3.10 for MCP server)
mamba create -p ./env python=3.10 pip -y
# or: conda create -p ./env python=3.10 pip -y

# Create legacy environment (Python 3.9 for RhoDesign models)
mamba env create -f repo/RhoDesign/environment.yml -p ./env_py39 -y
# or: conda env create -f repo/RhoDesign/environment.yml -p ./env_py39 -y

# Install MCP dependencies
mamba activate ./env
pip install fastmcp loguru --force-reinstall --no-cache-dir

# Verify installations
mamba activate ./env_py39
python -c "import torch; print('PyTorch version:', torch.__version__)"
```

### Download Model Checkpoints (Required)
```bash
# Download from Google Drive (manual step required)
# URL: https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz
# Place files in checkpoint/ directory:
# - ss_apexp_best.pth (~50MB) - for secondary structure mode
# - no_ss_apexp_best.pth (~50MB) - for 3D-only mode
```

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `rna_structure_to_sequence_with_ss.py` | RNA sequence design with secondary structure constraints | See below |
| `rna_structure_to_sequence_no_ss.py` | RNA sequence design from 3D structure only | See below |
| `rna_batch_evaluation.py` | Batch processing and benchmarking | See below |

### Script Examples

#### RNA Design with Secondary Structure

```bash
# Activate environment
mamba activate ./env_py39

# Run script
python scripts/rna_structure_to_sequence_with_ss.py \
  --pdb examples/data/2zh6_B.pdb \
  --ss examples/data/2zh6_B.npy \
  --output results/with_ss.fasta \
  --device -1
```

**Parameters:**
- `--pdb, -p`: Input PDB file with 3D RNA structure (required)
- `--ss, -s`: Secondary structure contact map (.npy file) (required)
- `--output, -o`: Output FASTA file path (default: auto-generated)
- `--device, -d`: GPU device ID (-1 for CPU, 0+ for GPU) (default: -1)
- `--temperature, -t`: Sampling temperature (default: 1.0)
- `--config, -c`: Configuration file (optional)

#### RNA Design (3D Structure Only)

```bash
python scripts/rna_structure_to_sequence_no_ss.py \
  --pdb examples/data/2zh6_B.pdb \
  --output results/no_ss.fasta \
  --device -1
```

**Parameters:**
- `--pdb, -p`: Input PDB file with 3D RNA structure (required)
- `--output, -o`: Output FASTA file path (default: auto-generated)
- `--device, -d`: GPU device ID (-1 for CPU, 0+ for GPU) (default: -1)
- `--temperature, -t`: Sampling temperature (default: 1.0)
- `--config, -c`: Configuration file (optional)

#### Batch Evaluation

```bash
python scripts/rna_batch_evaluation.py \
  --test_dir data/test \
  --ss_dir data/test_ss \
  --output results/batch.json \
  --max_files 10
```

**Parameters:**
- `--test_dir`: Directory containing test PDB files (required)
- `--ss_dir`: Directory containing secondary structure files (required)
- `--output, -o`: Output JSON file path (default: auto-generated)
- `--max_files`: Maximum number of files to process (optional)
- `--device, -d`: GPU device ID (default: -1)

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Install MCP server for Claude Code
mamba activate ./env
fastmcp install src/server.py --name RhoDesign
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add RhoDesign -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "RhoDesign": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhodesign_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhodesign_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from RhoDesign?
```

#### Basic Usage
```
Use predict_rna_sequence_with_ss with pdb_file @examples/data/2zh6_B.pdb and ss_file @examples/data/2zh6_B.npy
```

#### Design without Secondary Structure
```
Run predict_rna_sequence_no_ss on @examples/data/2zh6_B.pdb and save to results/output.fasta
```

#### Long-Running Tasks (Submit API)
```
Submit RNA sequence prediction with secondary structure for @examples/data/2zh6_B.pdb
Then check the job status
```

#### Batch Processing
```
Process these files in batch:
- @examples/data/2zh6_B.pdb
- @data/structure2.pdb
- @data/structure3.pdb
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/2zh6_B.pdb` | Reference the demo PDB file |
| `@examples/data/2zh6_B.npy` | Reference the demo secondary structure |
| `@configs/default_config.json` | Reference a config file |
| `@results/` | Reference output directory |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "RhoDesign": {
      "command": "/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhodesign_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/NucleicMCP/NucleicMCP/tool-mcps/rhodesign_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use predict_rna_sequence_with_ss with file examples/data/2zh6_B.pdb
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict_rna_sequence_with_ss` | Generate RNA sequence from 3D structure with secondary structure | `pdb_file`, `ss_file`, `output_file`, `device`, `temperature` |
| `predict_rna_sequence_no_ss` | Generate RNA sequence from 3D structure only | `pdb_file`, `output_file`, `device`, `temperature` |
| `validate_rna_structure` | Validate RNA structure file format and content | `pdb_file` |
| `get_example_data` | Get information about available example datasets | (none) |
| `get_available_configs` | Get information about configuration files | (none) |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_rna_sequence_prediction_with_ss` | Submit RNA sequence generation with secondary structure | `pdb_file`, `ss_file`, `output_dir`, `device`, `temperature`, `job_name` |
| `submit_rna_sequence_prediction_no_ss` | Submit RNA sequence generation without secondary structure | `pdb_file`, `output_dir`, `device`, `temperature`, `job_name` |
| `submit_rna_batch_evaluation` | Submit batch processing and benchmarking | `test_dir`, `ss_dir`, `output_dir`, `max_files`, `device`, `job_name` |
| `submit_batch_rna_prediction` | Submit batch processing for multiple structures | `input_files`, `prediction_type`, `ss_files`, `output_dir`, `device`, `temperature`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs |

---

## Examples

### Example 1: Single RNA Structure Design

**Goal:** Design an RNA sequence from a 3D structure with secondary structure constraints

**Using Script:**
```bash
mamba activate ./env_py39
python scripts/rna_structure_to_sequence_with_ss.py \
  --pdb examples/data/2zh6_B.pdb \
  --ss examples/data/2zh6_B.npy \
  --output results/example1.fasta
```

**Using MCP (in Claude Code):**
```
Use predict_rna_sequence_with_ss to process @examples/data/2zh6_B.pdb with secondary structure @examples/data/2zh6_B.npy and save results to results/example1.fasta
```

**Expected Output:**
- FASTA file with predicted RNA sequence
- Recovery rate metric (similarity to original sequence)
- Processing metadata

### Example 2: Structure-Only Design

**Goal:** Design RNA sequence from 3D structure only (no secondary structure)

**Using Script:**
```bash
mamba activate ./env_py39
python scripts/rna_structure_to_sequence_no_ss.py \
  --pdb examples/data/2zh6_B.pdb \
  --output results/example2.fasta
```

**Using MCP (in Claude Code):**
```
Run predict_rna_sequence_no_ss on @examples/data/2zh6_B.pdb and save to results/example2.fasta
```

**Expected Output:**
- FASTA file with predicted sequence
- Recovery rate and confidence metrics

### Example 3: Batch Processing

**Goal:** Process multiple structures at once

**Using Script:**
```bash
for f in data/*.pdb; do
  python scripts/rna_structure_to_sequence_no_ss.py --pdb "$f" --output results/batch/
done
```

**Using MCP (in Claude Code):**
```
Submit batch RNA prediction for all PDB files in @data/ directory with prediction_type "no_ss"
```

**Expected Output:**
- Batch job ID for tracking
- Individual FASTA files for each structure
- Summary statistics

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With |
|------|-------------|----------|
| `2zh6_B.pdb` | Sample RNA 3D structure (58KB) | Both prediction tools |
| `2zh6_B.npy` | Secondary structure contact map (1.3KB) | With-SS prediction tool |
| `pred_seq.fasta` | Example prediction output (59B) | Reference output |

---

## Configuration Files

The `configs/` directory contains configuration templates:

| Config | Description | Parameters |
|--------|-------------|------------|
| `structure_to_sequence_with_ss_config.json` | For RNA design with secondary structure | model checkpoint, inference settings |
| `structure_to_sequence_no_ss_config.json` | For RNA design without secondary structure | model checkpoint, inference settings |
| `batch_evaluation_config.json` | For batch processing | evaluation settings, file limits |
| `default_config.json` | Default settings for all scripts | paths, model info, validation rules |

### Config Example

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

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environments
mamba create -p ./env python=3.10 -y
mamba env create -f repo/RhoDesign/environment.yml -p ./env_py39 -y
mamba activate ./env
pip install fastmcp loguru
```

**Problem:** Import errors
```bash
# Verify installation
mamba activate ./env_py39
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import biotite; print('Biotite available')"
```

**Problem:** CUDA issues
```bash
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
# If False, models will use CPU (slower but functional)
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove RhoDesign
claude mcp add RhoDesign -- $(pwd)/env/bin/python $(pwd)/src/server.py
```

**Problem:** Tools not working
```bash
# Test server directly
mamba activate ./env
python src/server.py --help
```

### Model Issues

**Problem:** Model checkpoints missing
```bash
# Download from Google Drive manually:
# https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz
# Place in checkpoint/ directory:
ls checkpoint/
# Should show: ss_apexp_best.pth, no_ss_apexp_best.pth
```

**Problem:** Out of memory errors
```bash
# Use CPU instead of GPU
python scripts/rna_structure_to_sequence_with_ss.py --device -1 --pdb examples/data/2zh6_B.pdb --ss examples/data/2zh6_B.npy
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la jobs/

# View job log
cat jobs/<job_id>/job.log
```

**Problem:** Job failed
```
Use get_job_log with job_id "<job_id>" and tail 100 to see error details
```

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env_py39

# Test scripts directly
python scripts/rna_structure_to_sequence_with_ss.py --help
python scripts/rna_structure_to_sequence_no_ss.py --help
python scripts/rna_batch_evaluation.py --help
```

### Starting Dev Server

```bash
# Run MCP server in dev mode
mamba activate ./env
fastmcp dev src/server.py
```

---

## Performance Notes

- **Single Structure**: ~30 seconds - 2 minutes per structure
- **GPU Acceleration**: 5-10x faster than CPU
- **Memory Requirements**: ~2GB per job
- **Batch Processing**: Efficient parallel processing supported
- **Sequence Recovery**: Typically 60-90% on benchmark datasets

---

## License

MIT License - Based on the original RhoDesign repository

## Credits

Based on [RhoDesign](https://github.com/rindkind/rhoDesign): Deep Learning for RNA Structure-to-Sequence Design

**Citation:**
```
@article{rhodesign2024,
  title={RhoDesign: Deep Learning for RNA Structure-to-Sequence Design},
  authors={[Original Authors]},
  year={2024}
}
```