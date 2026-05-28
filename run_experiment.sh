#!/usr/bin/env bash
#
# Run the Speech Gating Experiment (venv only)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    echo "Using venv environment"
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "Starting Speech Gating Experiment..."
    python main.py
    exit $?
fi

echo "Error: No Python environment found"
echo ""
echo "Please run setup first:"
echo "  bash setup_venv.sh"
echo ""
echo "Or manually create an environment:"
echo "  python3.10 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python main.py"
exit 1

