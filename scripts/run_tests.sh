#!/usr/bin/env bash
# Run the full test suite with coverage.
set -euo pipefail

cd "$(dirname "$0")/.."

export GEMINI_API_KEY="${GEMINI_API_KEY:-test-key}"
export GOOGLE_BOOKS_API_KEY="${GOOGLE_BOOKS_API_KEY:-test-key}"
export API_KEY_ENABLED="${API_KEY_ENABLED:-false}"

pytest "$@"
