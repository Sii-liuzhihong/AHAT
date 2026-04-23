"""
AHAT Command Line Interface
================================

This module provides the main entry point for the AHAT system.
It utilizes `typer` to expose the pipeline as accessible command-line utilities.

Usage:
    python cli.py [COMMAND] [ARGS]...
    python cli.py --help
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

# Initialize the main application
app = typer.Typer(
    help="AHAT CLI: AHAT Household Task Planner.",
    no_args_is_help=True,
    rich_markup_mode="rich"
)

# Create a subcommand group for download operations
download_app = typer.Typer(help="Download models and datasets")
app.add_typer(download_app, name="download")

# Create a subcommand group for pipeline operations
pipeline_app = typer.Typer(help="Run the planning pipeline")
app.add_typer(pipeline_app, name="pipeline")

# ==========================================
# Module: Download Model
# ==========================================
@download_app.command("model")
def download_model_entry(
    repo_id: Optional[str] = typer.Option(None, help="Hugging Face repository ID for the model"),
    output_dir: Optional[str] = typer.Option(None, help="Target directory for the downloaded model"),
):
    """
    [bold green]Download[/bold green] the AHAT-TGPO model from Hugging Face.
    (This calls scripts/download_model.py)
    """

    # Locate download script
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    script_path = project_root / "scripts" / "download_model.py"

    if not script_path.exists():
        print(f"[Error] Could not find download script at: {script_path}")
        print("Are you running this from the source repository?")
        raise typer.Exit(code=1)

    # Construct command
    cmd = [sys.executable, str(script_path)]
    
    if repo_id:
        cmd.extend(["--repo-id", repo_id])
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    print(f"Executing: {' '.join(cmd)}")
    
    # Call download script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[Error] Model download failed.")
        raise typer.Exit(code=1)

# ==========================================
# Module: Download Eval Set
# ==========================================
@download_app.command("eval_set")
def download_eval_set_entry(
    repo_id: Optional[str] = typer.Option(None, help="Hugging Face dataset repository ID"),
    output_dir: Optional[str] = typer.Option(None, help="Local directory to save dataset"),
):
    """
    [bold green]Download[/bold green] the AHAT evaluation set from Hugging Face.
    (This calls scripts/download_data.py with dataset parameters)
    """

    # Locate download script
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    script_path = project_root / "scripts" / "download_data.py"

    if not script_path.exists():
        print(f"[Error] Could not find download script at: {script_path}")
        print("Are you running this from the source repository?")
        raise typer.Exit(code=1)

    # Construct command
    cmd = [sys.executable, str(script_path)]
    
    if repo_id:
        cmd.extend(["--repo-id", repo_id])
    if output_dir:
        cmd.extend(["--output-dir", output_dir])

    print(f"Executing: {' '.join(cmd)}")
    
    # Call download script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[Error] Evaluation set download failed.")
        raise typer.Exit(code=1)


def _run_pipeline(mode: str) -> None:
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "src" / "ahat" / "planning" / "pipeline.py"

    if not script_path.exists():
        print(f"[Error] Could not find pipeline script at: {script_path}")
        print("Are you running this from the source repository?")
        raise typer.Exit(code=1)

    cmd = [sys.executable, str(script_path), "--mode", mode]

    print(f"Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("[Error] Pipeline execution failed.")
        raise typer.Exit(code=1)


@pipeline_app.command("local")
def pipeline_local() -> None:
    """
    Run the pipeline using a local model for task decomposition.
    """
    _run_pipeline(mode="local")


@pipeline_app.command("api")
def pipeline_api() -> None:
    """
    Run the pipeline using an API model for task decomposition.
    """
    _run_pipeline(mode="api")

if __name__ == "__main__":
    app()
