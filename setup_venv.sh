#!/usr/bin/env bash
#
# Setup script (venv + pip only)
#

set -euo pipefail

echo "=========================================="
echo "Speech Gating Experiment - Setup"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

PYTHON_CMD=""
for py in python3.10 python3.11 python3; do
    if command_exists "$py"; then
        PYTHON_CMD="$py"
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python 3.10+ not found."
    exit 1
fi

echo "Using Python: $("$PYTHON_CMD" --version)"

if [ -d "venv" ]; then
    read -r -p "Existing venv found. Remove and recreate it? [Y/n]: " reply
    case "${reply:-Y}" in
        [Nn]|[Nn][Oo]) ;;
        *)
            rm -rf venv
            ;;
    esac
fi

if [ ! -d "venv" ]; then
    "$PYTHON_CMD" -m venv venv
fi

# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Run:"
echo "  bash run_experiment.sh"

