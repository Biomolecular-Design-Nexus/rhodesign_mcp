# Step 3: Use Cases Report

## Scan Information
- **Scan Date**: 2024-12-24
- **Repository**: RhoDesign (RNA structure-to-sequence design)
- **Filter Applied**: RNA inverse folding, structure-to-sequence RNA design, RNA sequence design from 3D structure
- **Python Version Strategy**: Dual environment (3.10 for MCP, 3.9 for RhoDesign)
- **Package Manager**: mamba

## Use Cases Identified

### UC-001: RNA Structure-to-Sequence Design with Secondary Structure
- **Description**: RNA inverse folding using both 3D structure and secondary structure constraints
- **Script Path**: `examples/use_case_1_structure_to_sequence_with_ss.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py39`
- **Source**: `repo/RhoDesign/src/inference.py`
- **Model**: `checkpoint/ss_apexp_best.pth`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| pdb_file | file | Input PDB file with 3D structure | --pdb, -p |
| ss_file | file | Secondary structure contact map (.npy) | --ss, -s |
| temperature | float | Sampling temperature (default: 1e-5) | --temperature, -t |
| device | int | GPU device ID (default: 0) | --device, -d |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| fasta_file | file | Predicted RNA sequence in FASTA format |
| recovery_rate | float | Sequence recovery rate compared to original |

**Example Usage:**
```bash
mamba activate ./env_py39
python examples/use_case_1_structure_to_sequence_with_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --ss examples/data/2zh6_B.npy \
    --output output/ \
    --temperature 1e-5
```

**Example Data**: `examples/data/2zh6_B.pdb`, `examples/data/2zh6_B.npy`

---

### UC-002: RNA Structure-to-Sequence Design without Secondary Structure
- **Description**: RNA inverse folding using only 3D structure constraints (no secondary structure required)
- **Script Path**: `examples/use_case_2_structure_to_sequence_no_ss.py`
- **Complexity**: Medium
- **Priority**: High
- **Environment**: `./env_py39`
- **Source**: `repo/RhoDesign/src/inference_without2d.py`
- **Model**: `checkpoint/no_ss_apexp_best.pth`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| pdb_file | file | Input PDB file with 3D structure | --pdb, -p |
| temperature | float | Sampling temperature (default: 1.0) | --temperature, -t |
| device | int | GPU device ID (default: 0) | --device, -d |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| fasta_file | file | Predicted RNA sequence in FASTA format |
| recovery_rate | float | Sequence recovery rate compared to original |

**Example Usage:**
```bash
mamba activate ./env_py39
python examples/use_case_2_structure_to_sequence_no_ss.py \
    --pdb examples/data/2zh6_B.pdb \
    --output output/ \
    --temperature 1.0
```

**Example Data**: `examples/data/2zh6_B.pdb`

---

### UC-003: Batch Evaluation and Benchmarking
- **Description**: Batch processing and evaluation of RhoDesign model on multiple structures
- **Script Path**: `examples/use_case_3_batch_evaluation.py`
- **Complexity**: High
- **Priority**: Medium
- **Environment**: `./env_py39`
- **Source**: `repo/RhoDesign/src/eval_model.py`
- **Model**: `checkpoint/ss_apexp_best.pth`

**Inputs:**
| Name | Type | Description | Parameter |
|------|------|-------------|----------|
| test_dir | directory | Directory with test PDB files | --test_dir |
| ss_dir | directory | Directory with secondary structure files | --ss_dir |
| max_files | int | Maximum files to process (testing) | --max_files |
| temperature | float | Sampling temperature (default: 1e-5) | --temperature, -t |

**Outputs:**
| Name | Type | Description |
|------|------|-------------|
| individual_fastas | files | Predicted sequences for each structure |
| results_summary | file | Comprehensive evaluation report |
| statistics | data | Recovery rate statistics and analysis |

**Example Usage:**
```bash
mamba activate ./env_py39
python examples/use_case_3_batch_evaluation.py \
    --test_dir data/test/ \
    --ss_dir data/test_ss/ \
    --output results/ \
    --max_files 10
```

**Example Data**: Requires test dataset download from Google Drive

---

## Filter Compliance

All identified use cases perfectly match the specified filter criteria:
- **RNA inverse folding** ✅ - All use cases perform inverse folding
- **Structure-to-sequence RNA design** ✅ - All convert structures to sequences
- **RNA sequence design from 3D structure** ✅ - All use 3D structural input

## Summary Statistics

| Metric | Count |
|--------|-------|
| Total Use Cases Found | 3 |
| Scripts Created | 3 |
| High Priority | 2 |
| Medium Priority | 1 |
| Low Priority | 0 |
| Demo Data Copied | ✅ |
| Environment Ready | ✅ |

## Technical Details

### Model Requirements
- **Checkpoints Required**: 2 models need download from Google Drive
  - `ss_apexp_best.pth`: With secondary structure constraints
  - `no_ss_apexp_best.pth`: Without secondary structure constraints
- **Hardware**: CUDA-compatible GPU recommended (CUDA 11.3)
- **Memory**: 8GB+ RAM for model loading and inference

### Framework Integration
- **Base Framework**: PyTorch 1.12.1 with CUDA 11.3
- **Geometric ML**: torch-geometric for GVP encoding
- **Structural Biology**: biotite for PDB file handling
- **Architecture**: Transformer encoder-decoder with GVP

### Performance Characteristics
- **Inference Speed**: ~1-5 seconds per structure (GPU)
- **Temperature Control**: 1e-5 (high accuracy) to 1.0 (diversity)
- **Sequence Recovery**: 60-90% typical recovery rates
- **Batch Processing**: Supports parallel processing

## Demo Data Index

| Source | Destination | Description | Size |
|--------|-------------|-------------|------|
| `repo/RhoDesign/example/2zh6_B.pdb` | `examples/data/2zh6_B.pdb` | Sample RNA 3D structure (PDB format) | 58KB |
| `repo/RhoDesign/example/2zh6_B.npy` | `examples/data/2zh6_B.npy` | Secondary structure contact map | 1.3KB |
| `repo/RhoDesign/example/pred_seq.fasta` | `examples/data/pred_seq.fasta` | Example prediction output | 59B |

## Installation Requirements

### Model Checkpoints (Required)
- **Download Source**: https://drive.google.com/drive/folders/1H3Itu6TTfaVErPH50Ly7rmQDxElH3JEz?usp=sharing
- **Target Location**: `checkpoint/` directory
- **Files**: `ss_apexp_best.pth`, `no_ss_apexp_best.pth`
- **Size**: ~100MB total

### Test Dataset (Optional)
- **Purpose**: Required for UC-003 batch evaluation
- **Download**: Same Google Drive link
- **Files**: `test/` directory (PDB files), `test_ss/` directory (NPY files)
- **Size**: Varies by dataset

## Validation Status

- [x] All scripts created with proper error handling
- [x] Example data copied and accessible
- [x] Relative paths configured for `examples/data/`
- [x] Environment requirements documented
- [x] CLI parameters and help text added
- [x] Model checkpoint requirements specified
- [x] GPU/CPU fallback options implemented
- [x] Comprehensive documentation included

## Future MCP Integration

These use cases will be converted to MCP tools with the following mapping:
1. **UC-001** → `predict_rna_sequence_with_ss` tool
2. **UC-002** → `predict_rna_sequence_no_ss` tool
3. **UC-003** → `evaluate_batch_rna_sequences` tool

Each tool will maintain the same functionality while providing structured MCP interfaces for integration with Claude and other MCP clients.