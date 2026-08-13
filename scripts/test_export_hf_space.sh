#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
EXPORT_SCRIPT="$ROOT_DIR/scripts/export_hf_space.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf -- "$TMP_DIR"' EXIT

mkdir -p "$TMP_DIR/bin"
for command in rm mkdir cp; do
  cat >"$TMP_DIR/bin/$command" <<'SHIM'
#!/usr/bin/env bash
printf '%s\n' "$(basename "$0") $*" >>"$EXPORT_TEST_COMMAND_LOG"
exit 0
SHIM
  chmod +x "$TMP_DIR/bin/$command"
done

assert_rejected_without_file_operations() {
  local label="$1"
  local destination="$2"
  local sentinel="$3"
  local log="$TMP_DIR/${label//[^A-Za-z0-9]/_}.log"
  : >"$log"

  if PATH="$TMP_DIR/bin:$PATH" EXPORT_TEST_COMMAND_LOG="$log" \
      bash "$EXPORT_SCRIPT" "$destination" >"$TMP_DIR/stdout" 2>"$TMP_DIR/stderr"; then
    printf 'FAIL: unsafe destination was accepted: %s (%q)\n' "$label" "$destination" >&2
    return 1
  fi

  if [[ -s "$log" ]]; then
    printf 'FAIL: file operation attempted for %s (%q):\n' "$label" "$destination" >&2
    cat "$log" >&2
    return 1
  fi

  if [[ ! -e "$sentinel" ]]; then
    printf 'FAIL: sentinel was deleted for %s: %s\n' "$label" "$sentinel" >&2
    return 1
  fi
}

assert_rejected_without_file_operations empty '' "$EXPORT_SCRIPT"
assert_rejected_without_file_operations root '/' /bin/sh
assert_rejected_without_file_operations repository-root "$ROOT_DIR" "$ROOT_DIR/README.md"
assert_rejected_without_file_operations source-tree "$ROOT_DIR/spaces" "$ROOT_DIR/spaces/README.md"
assert_rejected_without_file_operations repository-scripts "$ROOT_DIR/scripts" "$EXPORT_SCRIPT"
assert_rejected_without_file_operations repository-parent "$(dirname "$ROOT_DIR")" "$ROOT_DIR"
assert_rejected_without_file_operations external-arbitrary '/tmp/evil-export-dest' "$EXPORT_SCRIPT"

printf 'unsafe export destination checks passed\n'
