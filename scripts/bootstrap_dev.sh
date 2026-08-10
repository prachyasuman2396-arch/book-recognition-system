#!/usr/bin/env bash
# Bootstrap a local development environment.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Creating virtual environment"
python3 -m venv .venv
source .venv/bin/activate

echo "==> Installing dependencies"
pip install --upgrade pip
pip install -r requirements-dev.txt

echo "==> Installing pre-commit hooks"
pre-commit install

if [ ! -f .env ]; then
  echo "==> Creating .env from .env.example"
  cp .env.example .env
  echo "    Edit .env and set GEMINI_API_KEY / GOOGLE_BOOKS_API_KEY before running."
fi

echo "==> Creating data directories"
mkdir -p data/uploads data/crops data/enhanced data/checkpoints models weights

echo "==> Done. Activate with: source .venv/bin/activate"
echo "==> Run the API with: uvicorn app.main:app --reload"
