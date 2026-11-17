#!/usr/bin/env bash
# One-time setup (macOS / Linux). Usage: ./scripts/setup.sh [--ml]
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt -r frontend/requirements.txt
if [ "$1" = "--ml" ]; then pip install -r backend/requirements-ml.txt; fi
[ -f .env ] || cp .env.example .env
echo "Done. Activate with: source .venv/bin/activate  then run: python run.py"
