"""
Pipeline runner for CAUTI data processing pipeline.

This script allows running:
1. Full pipeline (raw_to_bronze -> bronze_to_silver -> model)
2. Only raw_to_bronze (notebooks 1-57)
3. Only bronze_to_silver (notebooks 1-6)
4. Build model only (ANN_LOSO.ipynb)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# Get the project root directory (parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent

# Paths
RAW_TO_BRONZE_DIR = SCRIPTS_DIR / "raw_to_bronze"
BRONZE_TO_SILVER_DIR = SCRIPTS_DIR / "bronze_to_silver"
MODELS_DIR = SCRIPTS_DIR / "models"
TESTING_CSVS_DIR = PROJECT_ROOT / "testing_csvs"

# Data directories (relative to project root)
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

# Required input files for model creation
REQUIRED_MODEL_INPUTS = [
    SILVER_DIR / "silver_dataset.csv"
]

# Output files that will be created by model (checked for existence)
MODEL_OUTPUT_FILES = [
    MODELS_DIR / "bronze_scaler.pkl",
    MODELS_DIR / "cauti_ann_loso_model.h5",
    MODELS_DIR / "cauti_ann_loso_model.keras"
]


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_step(step_num: int, total: int, description: str):
    """Print a step indicator."""
    print(f"[{step_num}/{total}] {description}")


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists()


def check_testing_csvs() -> bool:
    """Check if testing_csvs directory has files."""
    if not TESTING_CSVS_DIR.exists():
        return False
    csv_files = list(TESTING_CSVS_DIR.glob("*.csv"))
    return len(csv_files) > 0


def verify_model_prerequisites() -> Tuple[bool, List[str], List[str]]:
    """
    Verify that all required input files exist before model creation.
    Also check if output files already exist.
    Returns (all_inputs_exist, missing_inputs, existing_outputs)
    """
    missing = []
    existing_outputs = []
    
    # Check required input files
    for filepath in REQUIRED_MODEL_INPUTS:
        if not check_file_exists(filepath):
            missing.append(str(filepath))
    
    # Check if output files already exist (for warning)
    for filepath in MODEL_OUTPUT_FILES:
        if check_file_exists(filepath):
            existing_outputs.append(str(filepath))
    
    # Check testing_csvs (should exist, but not critical)
    if not check_testing_csvs():
        # This is a warning, not a blocker
        pass
    
    return len(missing) == 0, missing, existing_outputs


def get_notebook_files(directory: Path, pattern: str) -> List[Path]:
    """
    Get notebook files matching a pattern, sorted by number.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern like "1_*.ipynb" or "*.ipynb"
    
    Returns:
        Sorted list of notebook paths
    """
    notebooks = list(directory.glob(pattern))
    
    # Sort by the number prefix
    def get_number(path: Path) -> int:
        name = path.stem
        # Extract number from start of filename
        parts = name.split('_')
        try:
            return int(parts[0])
        except ValueError:
            return 9999  # Put non-numbered files at the end
    
    notebooks.sort(key=get_number)
    return notebooks


def check_jupyter_available() -> bool:
    """Check if jupyter/nbconvert is available."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


def execute_notebook(notebook_path: Path) -> bool:
    """
    Execute a Jupyter notebook using nbconvert.
    
    Args:
        notebook_path: Path to the notebook file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Executing: {notebook_path.name}")
        
        # Use nbconvert to execute the notebook
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                str(notebook_path)
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout per notebook
        )
        
        if result.returncode == 0:
            print(f"  ✓ Success: {notebook_path.name}")
            return True
        else:
            print(f"  ✗ Error in {notebook_path.name}:")
            if result.stderr:
                # Print last few lines of stderr for context
                stderr_lines = result.stderr.strip().split('\n')
                print(f"    {stderr_lines[-5:] if len(stderr_lines) > 5 else stderr_lines}")
            if result.stdout:
                # Check for error messages in stdout too
                stdout_lines = result.stdout.strip().split('\n')
                error_lines = [line for line in stdout_lines if 'error' in line.lower() or 'exception' in line.lower()]
                if error_lines:
                    print(f"    {error_lines[-3:]}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  ✗ Timeout: {notebook_path.name} (exceeded 1 hour)")
        return False
    except FileNotFoundError:
        print(f"  ✗ Jupyter not found. Please install jupyter: pip install jupyter")
        return False
    except Exception as e:
        print(f"  ✗ Exception in {notebook_path.name}: {e}")
        return False


def run_raw_to_bronze() -> bool:
    """Run raw_to_bronze pipeline (notebooks 1-57)."""
    print_header("RAW TO BRONZE PIPELINE")
    
    # Get notebooks 1-57 (numbered notebooks)
    notebooks = []
    for i in range(0, 58):
        pattern = f"{i}_*.ipynb"
        matching = get_notebook_files(RAW_TO_BRONZE_DIR, pattern)
        notebooks.extend(matching)
    
    # Also include 0_build_base_dataset if it exists
    base_notebook = RAW_TO_BRONZE_DIR / "0_build_base_dataset.ipynb"
    if base_notebook.exists():
        notebooks.insert(0, base_notebook)
    
    if not notebooks:
        print("  ✗ No notebooks found in raw_to_bronze directory")
        return False
    
    print(f"  Found {len(notebooks)} notebooks to execute\n")
    
    failed = []
    for idx, notebook in enumerate(notebooks, 1):
        print_step(idx, len(notebooks), f"Processing {notebook.name}")
        if not execute_notebook(notebook):
            failed.append(notebook.name)
    
    if failed:
        print(f"\n  ✗ Failed notebooks: {', '.join(failed)}")
        return False
    
    print("\n  ✓ Raw to Bronze pipeline completed successfully!")
    return True


def run_bronze_to_silver() -> bool:
    """Run bronze_to_silver pipeline (notebooks 1-6)."""
    print_header("BRONZE TO SILVER PIPELINE")
    
    # Get notebooks 1-6 (numbered notebooks)
    notebooks = []
    for i in range(1, 7):
        pattern = f"{i}_*.ipynb"
        matching = get_notebook_files(BRONZE_TO_SILVER_DIR, pattern)
        notebooks.extend(matching)
    
    # Also include 0_Bronze_Data_Analysis if it exists
    base_notebook = BRONZE_TO_SILVER_DIR / "0_Bronze_Data_Analysis.ipynb"
    if base_notebook.exists():
        notebooks.insert(0, base_notebook)
    
    if not notebooks:
        print("  ✗ No notebooks found in bronze_to_silver directory")
        return False
    
    print(f"  Found {len(notebooks)} notebooks to execute\n")
    
    failed = []
    for idx, notebook in enumerate(notebooks, 1):
        print_step(idx, len(notebooks), f"Processing {notebook.name}")
        if not execute_notebook(notebook):
            failed.append(notebook.name)
    
    if failed:
        print(f"\n  ✗ Failed notebooks: {', '.join(failed)}")
        return False
    
    print("\n  ✓ Bronze to Silver pipeline completed successfully!")
    return True


def run_model_build() -> bool:
    """Run model building (ANN_LOSO.ipynb)."""
    print_header("BUILDING MODEL")
    
    # Verify prerequisites
    print("  Checking prerequisites...")
    all_exist, missing, existing_outputs = verify_model_prerequisites()
    
    if not all_exist:
        print("  ✗ Missing required input files:")
        for file in missing:
            print(f"    - {file}")
        print("\n  Please ensure all required files exist before building the model.")
        return False
    
    print("  ✓ All required input files found")
    
    # Warn if output files already exist
    if existing_outputs:
        print("\n  ⚠ Warning: The following model files already exist and will be overwritten:")
        for file in existing_outputs:
            print(f"    - {file}")
        response = input("\n  Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("  Model building cancelled.")
            return False
    
    # Check testing_csvs (informational)
    if check_testing_csvs():
        csv_count = len(list(TESTING_CSVS_DIR.glob("*.csv")))
        print(f"  ℹ Found {csv_count} CSV file(s) in testing_csvs directory")
    else:
        print("  ℹ testing_csvs directory is empty (will be populated by model notebook)")
    
    # Execute ANN_LOSO.ipynb
    model_notebook = MODELS_DIR / "ANN_LOSO.ipynb"
    if not model_notebook.exists():
        print(f"  ✗ Model notebook not found: {model_notebook}")
        return False
    
    print(f"\n  Executing model notebook: {model_notebook.name}")
    if execute_notebook(model_notebook):
        print("\n  ✓ Model building completed successfully!")
        
        # Verify outputs were created
        print("\n  Verifying output files...")
        all_created = True
        for filepath in MODEL_OUTPUT_FILES:
            if check_file_exists(filepath):
                print(f"    ✓ {filepath.name}")
            else:
                print(f"    ✗ {filepath.name} (not found)")
                all_created = False
        
        if not all_created:
            print("\n  ⚠ Warning: Some expected output files were not created.")
        
        return True
    else:
        print("\n  ✗ Model building failed!")
        return False


def run_full_pipeline() -> bool:
    """Run the full pipeline (raw_to_bronze -> bronze_to_silver -> model)."""
    print_header("FULL PIPELINE")
    
    # Step 1: Raw to Bronze
    if not run_raw_to_bronze():
        print("\n  ✗ Full pipeline failed at: Raw to Bronze")
        return False
    
    # Step 2: Bronze to Silver
    if not run_bronze_to_silver():
        print("\n  ✗ Full pipeline failed at: Bronze to Silver")
        return False
    
    # Step 3: Build Model
    if not run_model_build():
        print("\n  ✗ Full pipeline failed at: Model Building")
        return False
    
    print_header("FULL PIPELINE COMPLETED SUCCESSFULLY!")
    return True


def display_menu() -> str:
    """Display menu and get user choice."""
    print("\n" + "=" * 70)
    print("  CAUTI DATA PROCESSING PIPELINE")
    print("=" * 70)
    print("\n  Select an option:")
    print("    1. Run Full Pipeline (raw_to_bronze -> bronze_to_silver -> model)")
    print("    2. Run Only Raw to Bronze (notebooks 1-57)")
    print("    3. Run Only Bronze to Silver (notebooks 1-6)")
    print("    4. Build Model Only (ANN_LOSO.ipynb)")
    print("    5. Exit")
    print("\n" + "-" * 70)
    
    while True:
        choice = input("\n  Enter your choice (1-5): ").strip()
        if choice in ['1', '2', '3', '4', '5']:
            return choice
        print("  Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def main():
    """Main function."""
    # Change to project root directory
    os.chdir(PROJECT_ROOT)
    
    # Check if jupyter is available
    if not check_jupyter_available():
        print("\n" + "=" * 70)
        print("  ERROR: Jupyter/nbconvert is not available!")
        print("=" * 70)
        print("\n  Please install jupyter:")
        print("    pip install jupyter")
        print("\n  Or if using conda:")
        print("    conda install jupyter")
        print("=" * 70 + "\n")
        sys.exit(1)
    
    # Display menu and get choice
    choice = display_menu()
    
    if choice == '5':
        print("\n  Exiting...")
        return
    
    # Execute selected option
    success = False
    if choice == '1':
        success = run_full_pipeline()
    elif choice == '2':
        success = run_raw_to_bronze()
    elif choice == '3':
        success = run_bronze_to_silver()
    elif choice == '4':
        success = run_model_build()
    
    # Final status
    print("\n" + "=" * 70)
    if success:
        print("  ✓ Pipeline execution completed successfully!")
    else:
        print("  ✗ Pipeline execution failed!")
    print("=" * 70 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

