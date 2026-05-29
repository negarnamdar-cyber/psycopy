#!/usr/bin/env python3
"""Entry point for the Speech Gating Experiment with Medoc thermal stimulation."""

import shutil
import sys
from pathlib import Path

# Nuke stale Python cache before any imports so __pycache__ doesn't shadow edits.
_project_root = Path(__file__).parent.resolve()
for pycache in _project_root.rglob("__pycache__"):
    if pycache.is_dir():
        shutil.rmtree(pycache, ignore_errors=True)
for pyc in _project_root.rglob("*.pyc"):
    if pyc.is_file():
        pyc.unlink(missing_ok=True)

if __name__ == "__main__":
    package_dir = Path(__file__).parent / "psycopy"
    if package_dir.exists():
        sys.path.insert(0, str(package_dir.parent))

try:
    from psycopy.medoc_experiment import MedocExperiment
    from psycopy.medoc import MedocConnectionError
    from psycopy.config import ExperimentConfig, MedocConfig, show_startup_dialog

    def main():
        config = show_startup_dialog()
        try:
            experiment = MedocExperiment(config)
            experiment.run()
        except MedocConnectionError as exc:
            from psychopy import gui

            gui.msgBox(
                title="Medoc Connection Error",
                msg=f"Failed to connect to Medoc device at {exc.ip}:{exc.port}.\n\nError: {exc}",
                warn=False,
            )

    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nMake sure you have installed the required packages:")
    print("  pip install -e .")
    print("  pip install -r requirements.txt")
    print("\nOr activate the virtual environment:")
    print("  source venv/bin/activate  (Linux/Mac)")
    print("  venv\\Scripts\\activate     (Windows)")
    sys.exit(1)
except Exception as e:
    print(f"Experiment error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
