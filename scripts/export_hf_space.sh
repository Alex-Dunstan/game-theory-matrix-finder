#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SRC_DIR="$ROOT_DIR/spaces"

if (( $# > 0 )) && [[ -z "$1" ]]; then
    echo "error: export destination cannot be empty" >&2
    exit 2
fi

DEST_ARG="${1:-$ROOT_DIR/.build/hf_space}"

ROOT_DIR="$ROOT_DIR" SRC_DIR="$SRC_DIR" DEST_ARG="$DEST_ARG" python3 - <<'PY'
from pathlib import Path
import os
import shutil
import sys

root = Path(os.environ["ROOT_DIR"]).resolve()
src = Path(os.environ["SRC_DIR"]).resolve()
dest = Path(os.environ["DEST_ARG"]).expanduser()
if not dest.is_absolute():
    dest = (Path.cwd() / dest).resolve()
else:
    dest = dest.resolve()

build_dir = (root / ".build").resolve()
if build_dir not in dest.parents and dest != build_dir:
    print(f"error: refusing unsafe export destination: {dest}", file=sys.stderr)
    sys.exit(2)

if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)

files_to_copy = [
    "README.md",
    ".gitignore",
    "static/index.html",
    "static/app.mjs",
    "static/classifier.mjs",
    "static/styles.css",
]

for rel in files_to_copy:
    source_path = src / rel
    dest_name = Path(rel).name if rel.startswith("static/") else rel
    shutil.copy2(source_path, dest / dest_name)

print(dest)
PY
