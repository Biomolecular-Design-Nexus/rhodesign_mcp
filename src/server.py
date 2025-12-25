"""
MCP Server for RhoDesign

Provides both synchronous and asynchronous (submit) APIs for RNA structure-to-sequence design.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = MCP_ROOT / "scripts"
CONFIGS_DIR = MCP_ROOT / "configs"

# Add scripts to path for imports
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from jobs.manager import job_manager
from loguru import logger

# Create MCP server
mcp = FastMCP("RhoDesign")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or error if not completed
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)


# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def predict_rna_sequence_with_ss(
    pdb_file: str,
    ss_file: str,
    output_file: Optional[str] = None,
    device: int = -1,
    temperature: float = 1.0
) -> dict:
    """
    Generate RNA sequence from 3D structure with secondary structure constraints.

    Fast operation suitable for single sequences (~30 seconds to 2 minutes).
    For batch processing, use submit_batch_rna_prediction.

    Args:
        pdb_file: Path to PDB file containing 3D RNA structure
        ss_file: Path to .npy file containing secondary structure contact map
        output_file: Optional path to save results as FASTA
        device: Device for computation (-1 for CPU, 0+ for GPU)
        temperature: Sampling temperature for sequence generation

    Returns:
        Dictionary with predicted sequence, recovery rate, and metadata
    """
    try:
        # Import the script function
        from rna_structure_to_sequence_with_ss import run_structure_to_sequence_with_ss

        result = run_structure_to_sequence_with_ss(
            pdb_file=pdb_file,
            ss_file=ss_file,
            output_file=output_file,
            device=device,
            temperature=temperature
        )

        return {"status": "success", **result}

    except ImportError as e:
        return {
            "status": "error",
            "error": f"Failed to import script: {e}",
            "suggestion": "Ensure the scripts are properly set up and environment is activated"
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"File not found: {e}",
            "suggestion": "Check that input files exist and paths are correct"
        }
    except Exception as e:
        logger.error(f"RNA sequence prediction with SS failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check logs for detailed error information"
        }


@mcp.tool()
def predict_rna_sequence_no_ss(
    pdb_file: str,
    output_file: Optional[str] = None,
    device: int = -1,
    temperature: float = 1.0
) -> dict:
    """
    Generate RNA sequence from 3D structure only (no secondary structure constraints).

    Fast operation suitable for single sequences (~30 seconds to 2 minutes).
    For batch processing, use submit_batch_rna_prediction.

    Args:
        pdb_file: Path to PDB file containing 3D RNA structure
        output_file: Optional path to save results as FASTA
        device: Device for computation (-1 for CPU, 0+ for GPU)
        temperature: Sampling temperature for sequence generation

    Returns:
        Dictionary with predicted sequence, recovery rate, and metadata
    """
    try:
        # Import the script function
        from rna_structure_to_sequence_no_ss import run_structure_to_sequence_no_ss

        result = run_structure_to_sequence_no_ss(
            pdb_file=pdb_file,
            output_file=output_file,
            device=device,
            temperature=temperature
        )

        return {"status": "success", **result}

    except ImportError as e:
        return {
            "status": "error",
            "error": f"Failed to import script: {e}",
            "suggestion": "Ensure the scripts are properly set up and environment is activated"
        }
    except FileNotFoundError as e:
        return {
            "status": "error",
            "error": f"File not found: {e}",
            "suggestion": "Check that input files exist and paths are correct"
        }
    except Exception as e:
        logger.error(f"RNA sequence prediction without SS failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Check logs for detailed error information"
        }


# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_rna_sequence_prediction_with_ss(
    pdb_file: str,
    ss_file: str,
    output_dir: Optional[str] = None,
    device: int = -1,
    temperature: float = 1.0,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNA sequence generation with secondary structure for background processing.

    Use this for batch processing or when you want to run predictions in the background.
    Returns a job_id for tracking. Use get_job_status() to monitor progress.

    Args:
        pdb_file: Path to PDB file containing 3D RNA structure
        ss_file: Path to .npy file containing secondary structure contact map
        output_dir: Directory to save outputs
        device: Device for computation (-1 for CPU, 0+ for GPU)
        temperature: Sampling temperature for sequence generation
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id for tracking. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results when completed
        - get_job_log(job_id) to see execution logs
    """
    script_path = "scripts/rna_structure_to_sequence_with_ss.py"

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "pdb": pdb_file,
            "ss": ss_file,
            "output_dir": output_dir,
            "device": device,
            "temperature": temperature
        },
        job_name=job_name or f"rna_predict_with_ss_{Path(pdb_file).stem}"
    )


@mcp.tool()
def submit_rna_sequence_prediction_no_ss(
    pdb_file: str,
    output_dir: Optional[str] = None,
    device: int = -1,
    temperature: float = 1.0,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit RNA sequence generation without secondary structure for background processing.

    Use this for batch processing or when you want to run predictions in the background.
    Returns a job_id for tracking.

    Args:
        pdb_file: Path to PDB file containing 3D RNA structure
        output_dir: Directory to save outputs
        device: Device for computation (-1 for CPU, 0+ for GPU)
        temperature: Sampling temperature for sequence generation
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the prediction job
    """
    script_path = "scripts/rna_structure_to_sequence_no_ss.py"

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "pdb": pdb_file,
            "output_dir": output_dir,
            "device": device,
            "temperature": temperature
        },
        job_name=job_name or f"rna_predict_no_ss_{Path(pdb_file).stem}"
    )


@mcp.tool()
def submit_rna_batch_evaluation(
    test_dir: str,
    ss_dir: str,
    output_dir: Optional[str] = None,
    max_files: Optional[int] = None,
    device: int = -1,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA sequence evaluation for background processing.

    Processes multiple RNA structures and evaluates sequence generation performance.
    This operation can take 10+ minutes for large datasets.

    Args:
        test_dir: Directory containing test PDB files
        ss_dir: Directory containing secondary structure files (.npy)
        output_dir: Directory to save outputs
        max_files: Maximum number of files to process (optional)
        device: Device for computation (-1 for CPU, 0+ for GPU)
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the batch evaluation
    """
    script_path = "scripts/rna_batch_evaluation.py"

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "test_dir": test_dir,
            "ss_dir": ss_dir,
            "output_dir": output_dir,
            "max_files": max_files,
            "device": device
        },
        job_name=job_name or f"batch_eval_{Path(test_dir).name}"
    )


# ==============================================================================
# Batch Processing Tools
# ==============================================================================

@mcp.tool()
def submit_batch_rna_prediction(
    input_files: List[str],
    prediction_type: str = "with_ss",
    ss_files: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    device: int = -1,
    temperature: float = 1.0,
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch RNA sequence prediction for multiple structures.

    Processes multiple PDB files in a single job. Suitable for:
    - Processing many RNA structures at once
    - Large-scale sequence generation
    - Parallel processing of independent structures

    Args:
        input_files: List of PDB file paths to process
        prediction_type: Type of prediction ("with_ss" or "no_ss")
        ss_files: List of secondary structure files (required if prediction_type="with_ss")
        output_dir: Directory to save all outputs
        device: Device for computation (-1 for CPU, 0+ for GPU)
        temperature: Sampling temperature for all predictions
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    # Validate inputs
    if prediction_type == "with_ss" and not ss_files:
        return {
            "status": "error",
            "error": "ss_files required when prediction_type='with_ss'"
        }

    if prediction_type == "with_ss" and len(input_files) != len(ss_files):
        return {
            "status": "error",
            "error": "Number of PDB files must match number of SS files"
        }

    # Determine which script to use
    if prediction_type == "with_ss":
        script_path = "scripts/rna_structure_to_sequence_with_ss.py"
        args = {
            "batch_pdbs": ",".join(input_files),
            "batch_ss": ",".join(ss_files) if ss_files else None,
            "output_dir": output_dir,
            "device": device,
            "temperature": temperature
        }
    else:
        script_path = "scripts/rna_structure_to_sequence_no_ss.py"
        args = {
            "batch_pdbs": ",".join(input_files),
            "output_dir": output_dir,
            "device": device,
            "temperature": temperature
        }

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_{prediction_type}_{len(input_files)}_files"
    )


# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_rna_structure(pdb_file: str) -> dict:
    """
    Validate RNA structure file format and content.

    Args:
        pdb_file: Path to PDB file to validate

    Returns:
        Dictionary with validation results
    """
    try:
        from lib.structure import validate_structure_data, load_structure_and_coords

        # Try to load the structure
        coords, sequence = load_structure_and_coords(pdb_file)

        # Validate the loaded data
        is_valid = validate_structure_data(coords, sequence)

        return {
            "status": "success",
            "valid": is_valid,
            "sequence_length": len(sequence),
            "structure_shape": coords.shape if coords is not None else None,
            "sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "valid": False
        }


@mcp.tool()
def get_example_data() -> dict:
    """
    Get information about available example datasets for testing.

    Returns:
        Dictionary with example files and their descriptions
    """
    examples_dir = MCP_ROOT / "examples" / "data"

    if not examples_dir.exists():
        return {
            "status": "error",
            "error": "Examples directory not found",
            "suggestion": "Check if examples/data directory exists"
        }

    try:
        example_files = []
        for file_path in examples_dir.iterdir():
            if file_path.is_file():
                example_files.append({
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "type": file_path.suffix
                })

        return {
            "status": "success",
            "examples_directory": str(examples_dir),
            "files": example_files,
            "total_files": len(example_files)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@mcp.tool()
def get_available_configs() -> dict:
    """
    Get information about available configuration files.

    Returns:
        Dictionary with available configurations
    """
    if not CONFIGS_DIR.exists():
        return {
            "status": "error",
            "error": "Configs directory not found"
        }

    try:
        configs = []
        for config_file in CONFIGS_DIR.glob("*.json"):
            try:
                with open(config_file) as f:
                    config_data = json.load(f)

                configs.append({
                    "name": config_file.name,
                    "path": str(config_file),
                    "description": config_data.get("description", "No description"),
                    "model_type": config_data.get("model", {}).get("type", "unknown")
                })
            except Exception as e:
                configs.append({
                    "name": config_file.name,
                    "path": str(config_file),
                    "error": f"Failed to parse: {e}"
                })

        return {
            "status": "success",
            "configs_directory": str(CONFIGS_DIR),
            "configurations": configs,
            "total_configs": len(configs)
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()