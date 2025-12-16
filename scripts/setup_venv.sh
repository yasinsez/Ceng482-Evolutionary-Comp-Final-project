#!/usr/bin/env bash
set -euo pipefail

# Create a local venv in .venv/ and install requirements.
# Works on macOS/Linux with python3 available.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: '$PYTHON_BIN' not found. Install Python 3 and retry." >&2
  exit 1
fi

if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
  echo "Created virtual environment at .venv/"
else
  echo "Virtual environment already exists at .venv/ (reusing)"
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo
echo "Done."
echo "Activate with: source .venv/bin/activate"


