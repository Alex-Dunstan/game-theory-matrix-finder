#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLE_DIR="$ROOT_DIR/.build/hf_space"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_BIN="${NODE_BIN:-node}"

if ! command -v "$NODE_BIN" >/dev/null 2>&1; then
  echo "error: node not found for static classifier validation" >&2
  exit 1
fi

"$ROOT_DIR/scripts/export_hf_space.sh" "$BUNDLE_DIR" >/dev/null

if ! "$PYTHON_BIN" -c 'import numpy, pytest' >/dev/null 2>&1; then
  echo "error: validation requires numpy and pytest; install engine/requirements.txt" >&2
  exit 1
fi

PYTHONPATH="$ROOT_DIR/spaces" "$PYTHON_BIN" -m pytest -q "$ROOT_DIR/spaces/tests/test_analysis.py"
"$NODE_BIN" "$ROOT_DIR/spaces/static/test_classifier.mjs"

grep -q '^sdk: static$' "$BUNDLE_DIR/README.md"
grep -q '^app_file: index.html$' "$BUNDLE_DIR/README.md"
grep -q '^emoji: 🎲$' "$BUNDLE_DIR/README.md"

EXPECTED_FILES="$(
  cat <<EOF
$BUNDLE_DIR/.gitignore
$BUNDLE_DIR/README.md
$BUNDLE_DIR/app.mjs
$BUNDLE_DIR/classifier.mjs
$BUNDLE_DIR/index.html
$BUNDLE_DIR/styles.css
EOF
)"

ACTUAL_FILES="$(find "$BUNDLE_DIR" -maxdepth 1 -type f | sort)"

if [[ "$ACTUAL_FILES" != "$EXPECTED_FILES" ]]; then
  printf 'Unexpected export bundle contents\nExpected:\n%s\nActual:\n%s\n' "$EXPECTED_FILES" "$ACTUAL_FILES" >&2
  exit 1
fi

grep -q 'href="./styles.css"' "$BUNDLE_DIR/index.html"
grep -q 'src="./app.mjs"' "$BUNDLE_DIR/index.html"
grep -q 'Game Theory Matrix Classifier' "$BUNDLE_DIR/index.html"
grep -q 'classifyFull' "$BUNDLE_DIR/classifier.mjs"
if grep -q 'gradio-lite' "$BUNDLE_DIR/index.html"; then
  echo "error: exported static bundle still references gradio-lite" >&2
  exit 1
fi

printf '%s\n' "$ACTUAL_FILES"
