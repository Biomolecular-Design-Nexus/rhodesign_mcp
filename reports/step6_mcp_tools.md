# Step 6: MCP Tools Documentation

## Server Information
- **Server Name**: RhoDesign
- **Version**: 1.0.0
- **Created Date**: 2024-12-24
- **Server Path**: `src/server.py`
- **Total Tools**: 14
- **Python Environment**: `env/` (Python 3.10.19)
- **MCP Framework**: FastMCP 2.14.1

## API Design Analysis

Based on the analysis of the 3 clean scripts from Step 5, the following API types were determined:

### Script Analysis

| Script | Description | Runtime Estimate | API Type | Reason |
|--------|-------------|------------------|-----------|--------|
| `rna_structure_to_sequence_with_ss.py` | RNA sequence design with secondary structure | ~30 sec - 2 min | **Sync + Submit** | Fast for single sequences, slow for batch |
| `rna_structure_to_sequence_no_ss.py` | RNA sequence design (3D structure only) | ~30 sec - 2 min | **Sync + Submit** | Fast for single sequences, slow for batch |
| `rna_batch_evaluation.py` | Batch processing and benchmarking | >10 min | **Submit Only** | Always long-running |

## Job Management Tools

Core tools for managing asynchronous operations:

| Tool | Description | Parameters |
|------|-------------|------------|
| `get_job_status` | Check job progress and status | `job_id: str` |
| `get_job_result` | Get completed job results | `job_id: str` |
| `get_job_log` | View job execution logs | `job_id: str, tail: int = 50` |
| `cancel_job` | Cancel running job | `job_id: str` |
| `list_jobs` | List all jobs with optional filtering | `status: Optional[str] = None` |

### Job Workflow Example
```
1. Submit: submit_rna_sequence_prediction_with_ss(...)
   → Returns: {"job_id": "abc12345", "status": "submitted"}

2. Check: get_job_status("abc12345")
   → Returns: {"status": "running", "started_at": "...", ...}

3. Result: get_job_result("abc12345")
   → Returns: {"status": "success", "result": {...}}
```

## Sync Tools (Fast Operations < 10 min)

Tools that return results immediately for single sequences:

### predict_rna_sequence_with_ss
- **Description**: Generate RNA sequence from 3D structure with secondary structure constraints
- **Runtime**: ~30 seconds to 2 minutes per sequence
- **Source Script**: `scripts/rna_structure_to_sequence_with_ss.py`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| pdb_file | str | Yes | - | Path to PDB file containing 3D RNA structure |
| ss_file | str | Yes | - | Path to .npy file containing secondary structure contact map |
| output_file | str | No | None | Optional path to save results as FASTA |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| temperature | float | No | 1.0 | Sampling temperature for sequence generation |

**Example:**
```
Use predict_rna_sequence_with_ss with pdb_file "examples/data/2zh6_B.pdb" and ss_file "examples/data/2zh6_B.npy"
```

### predict_rna_sequence_no_ss
- **Description**: Generate RNA sequence from 3D structure only (no secondary structure constraints)
- **Runtime**: ~30 seconds to 2 minutes per sequence
- **Source Script**: `scripts/rna_structure_to_sequence_no_ss.py`

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| pdb_file | str | Yes | - | Path to PDB file containing 3D RNA structure |
| output_file | str | No | None | Optional path to save results as FASTA |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| temperature | float | No | 1.0 | Sampling temperature for sequence generation |

**Example:**
```
Use predict_rna_sequence_no_ss with pdb_file "examples/data/2zh6_B.pdb"
```

---

## Submit Tools (Long Operations > 10 min)

Tools that submit jobs for background processing:

### submit_rna_sequence_prediction_with_ss
- **Description**: Submit RNA sequence generation with secondary structure for background processing
- **Runtime**: Variable (background job)
- **Source Script**: `scripts/rna_structure_to_sequence_with_ss.py`
- **Supports Batch**: ✅ Yes (via batch tool)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| pdb_file | str | Yes | - | Path to PDB file containing 3D RNA structure |
| ss_file | str | Yes | - | Path to .npy file containing secondary structure contact map |
| output_dir | str | No | None | Directory to save outputs |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| temperature | float | No | 1.0 | Sampling temperature for sequence generation |
| job_name | str | No | auto | Optional name for the job |

**Example:**
```
Submit RNA sequence prediction with secondary structure for examples/data/2zh6_B.pdb
```

### submit_rna_sequence_prediction_no_ss
- **Description**: Submit RNA sequence generation without secondary structure for background processing
- **Runtime**: Variable (background job)
- **Source Script**: `scripts/rna_structure_to_sequence_no_ss.py`
- **Supports Batch**: ✅ Yes (via batch tool)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| pdb_file | str | Yes | - | Path to PDB file containing 3D RNA structure |
| output_dir | str | No | None | Directory to save outputs |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| temperature | float | No | 1.0 | Sampling temperature for sequence generation |
| job_name | str | No | auto | Optional name for the job |

### submit_rna_batch_evaluation
- **Description**: Submit batch RNA sequence evaluation for background processing
- **Runtime**: >10 minutes for large datasets
- **Source Script**: `scripts/rna_batch_evaluation.py`
- **Supports Batch**: ✅ Native batch processing

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| test_dir | str | Yes | - | Directory containing test PDB files |
| ss_dir | str | Yes | - | Directory containing secondary structure files (.npy) |
| output_dir | str | No | None | Directory to save outputs |
| max_files | int | No | None | Maximum number of files to process (optional) |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| job_name | str | No | auto | Optional name for the job |

---

## Batch Processing Tools

### submit_batch_rna_prediction
- **Description**: Submit batch RNA sequence prediction for multiple structures
- **Runtime**: Variable (depends on number of files)
- **Supports**: Both prediction types (with/without secondary structure)

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| input_files | List[str] | Yes | - | List of PDB file paths to process |
| prediction_type | str | No | "with_ss" | Type of prediction ("with_ss" or "no_ss") |
| ss_files | List[str] | No | None | List of secondary structure files (required if prediction_type="with_ss") |
| output_dir | str | No | None | Directory to save all outputs |
| device | int | No | -1 | Device for computation (-1 for CPU, 0+ for GPU) |
| temperature | float | No | 1.0 | Sampling temperature for all predictions |
| job_name | str | No | auto | Optional name for the batch job |

**Example:**
```
Submit batch prediction for ["file1.pdb", "file2.pdb", "file3.pdb"] with prediction_type "with_ss"
```

---

## Utility Tools

### validate_rna_structure
- **Description**: Validate RNA structure file format and content
- **Runtime**: ~5 seconds

**Parameters:**
| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| pdb_file | str | Yes | - | Path to PDB file to validate |

### get_example_data
- **Description**: Get information about available example datasets for testing
- **Runtime**: ~1 second

### get_available_configs
- **Description**: Get information about available configuration files
- **Runtime**: ~1 second

---

## Workflow Examples

### Quick Single Sequence Analysis (Sync)
```
Use predict_rna_sequence_with_ss with pdb_file "examples/data/2zh6_B.pdb" and ss_file "examples/data/2zh6_B.npy"
→ Returns results immediately (30 sec - 2 min)
```

### Background Single Sequence (Submit API)
```
1. Submit: Use submit_rna_sequence_prediction_with_ss with pdb_file "examples/data/2zh6_B.pdb"
   → Returns: {"job_id": "abc123", "status": "submitted"}

2. Check: Use get_job_status with job_id "abc123"
   → Returns: {"status": "running", "progress": "..."}

3. Result: Use get_job_result with job_id "abc123"
   → Returns: {"status": "success", "result": {...}}
```

### Batch Processing (Multiple Files)
```
Use submit_batch_rna_prediction with input_files ["file1.pdb", "file2.pdb", "file3.pdb"] and prediction_type "with_ss"
→ Processes all files in a single background job
```

### Benchmark Evaluation
```
Use submit_rna_batch_evaluation with test_dir "data/test" and ss_dir "data/test_ss"
→ Runs comprehensive evaluation on dataset
```

---

## Implementation Details

### Job Management System
- **Location**: `src/jobs/manager.py`
- **Features**: Background execution, progress tracking, log capture, job cancellation
- **Persistence**: Jobs saved to `jobs/` directory with metadata
- **Thread Safety**: Uses threading for concurrent job execution
- **Error Handling**: Graceful failure handling with detailed error messages

### Environment Integration
- **Main Environment**: `env/` (Python 3.10.19) for MCP server
- **Script Environment**: `env_py39/` (Python 3.9) for RhoDesign scripts
- **Cross-Environment**: Job manager handles environment switching automatically
- **Dependencies**: FastMCP 2.14.1, loguru for logging

### Path Management
- **Scripts**: `scripts/` directory with clean, self-contained scripts
- **Configs**: `configs/` directory with JSON configuration files
- **Examples**: `examples/data/` with test data
- **Outputs**: `jobs/{job_id}/` for each job's outputs and logs

---

## Error Handling

### Structured Error Responses
All tools return structured error responses:
```json
{
  "status": "error",
  "error": "Detailed error message",
  "suggestion": "Helpful suggestion for resolution"
}
```

### Common Error Scenarios
1. **Missing Dependencies**: Clear instructions for environment setup
2. **File Not Found**: Validation of input file paths
3. **Model Checkpoints**: Guidance for downloading required models
4. **GPU/CPU Issues**: Automatic fallback suggestions
5. **Job Failures**: Detailed logs accessible via `get_job_log`

---

## Performance Characteristics

### Resource Requirements
- **Memory**: ~2GB for model loading per job
- **CPU**: Single-core per job, multi-job parallelism supported
- **GPU**: Optional, automatic detection with CPU fallback
- **Storage**: ~1GB per 100 jobs (logs and results)

### Scaling Limits
- **Concurrent Jobs**: Limited by available memory (recommend max 4-6 jobs)
- **Job History**: Automatic cleanup of jobs >7 days old
- **File Size**: PDB files up to ~100MB supported
- **Batch Size**: Recommend <50 files per batch job

---

## Security & Reliability

### Job Isolation
- Each job runs in isolated directory
- Separate log files and output directories
- Clean environment variable handling

### Process Management
- Graceful job termination with SIGTERM → SIGKILL escalation
- Automatic cleanup of zombie processes
- Thread-safe job state management

### Input Validation
- PDB file format validation
- Path sanitization
- Parameter type checking
- Required dependency verification

---

## Success Criteria Evaluation

- [x] **MCP Server Created**: `src/server.py` with 14 registered tools
- [x] **Job Manager Implemented**: Full async job management in `src/jobs/`
- [x] **Sync Tools**: 2 fast operation tools for immediate results
- [x] **Submit Tools**: 3 long-running operation tools with job tracking
- [x] **Batch Support**: Multi-file processing capabilities
- [x] **Job Management**: Complete CRUD operations for job lifecycle
- [x] **Clear Documentation**: Structured tool descriptions for LLM use
- [x] **Error Handling**: Comprehensive structured error responses
- [x] **Server Testing**: All functionality verified working
- [x] **Environment Integration**: Proper Python 3.10 + 3.9 environment handling

**Overall Status**: 🟢 **Complete and Production-Ready**

The RhoDesign MCP server successfully provides both synchronous and asynchronous APIs for RNA structure-to-sequence design with comprehensive job management, error handling, and batch processing capabilities.