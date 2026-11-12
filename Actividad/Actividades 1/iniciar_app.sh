#!/usr/bin/env bash
# Starts the local MNIST app at http://127.0.0.1:8501 and opens the browser.
set -euo pipefail
cd "$(dirname "$0")"
VENV="$HOME/.venvs/mnist-a1"
PYTHON="$(command -v python3.12 || command -v python3)"

if [ ! -x "$VENV/bin/python" ]; then
  echo "[1/2] Creating virtual environment in $VENV ..."
  "$PYTHON" -m venv "$VENV"
fi
if ! "$VENV/bin/python" -c "import tensorflow, streamlit, streamlit_drawable_canvas" 2>/dev/null; then
  echo "[2/2] Installing dependencies, this takes a few minutes the first time ..."
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r requirements-app.txt
fi

echo "Opening http://127.0.0.1:8501  (press Ctrl+C to stop)"
exec "$VENV/bin/python" -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
