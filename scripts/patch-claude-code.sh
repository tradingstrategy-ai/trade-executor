#!/usr/bin/env bash
# ================================================
# Claude Code User-Agent Patcher
# Patches "axios/1.8.4" → real browser UA in CLI + VS Code extension
# Supports: macOS & Linux only
# Usage:
#   ./patch-claude-ua.sh --dry-run     # Preview only
#   ./patch-claude-ua.sh               # Apply patches (creates backups)
# ================================================

set -euo pipefail

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "=== DRY-RUN MODE (no files will be changed) ==="
fi

OLD_UA="axios/1.8.4"
NEW_UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

echo "Claude Code UA Patcher"
echo "Old UA : $OLD_UA"
echo "New UA : $NEW_UA"
echo "Mode   : ${DRY_RUN:+DRY-RUN}${DRY_RUN:-PATCH}"
echo "==================================================="

backup_file() {
  local f="$1"
  if [[ $DRY_RUN == true ]]; then
    echo "[DRY] Would backup: $f"
    return
  fi
  local bak="${f}.bak.$(date +%Y%m%d-%H%M%S)"
  cp "$f" "$bak"
  echo "✅ Backed up → $bak"
}

patch_file() {
  local f="$1"
  [[ -f "$f" ]] || return 1

  # Only patch files that actually contain the old UA
  if ! grep -qF "$OLD_UA" "$f" 2>/dev/null; then
    return 1
  fi

  echo "Found match in: $f"

  if [[ $DRY_RUN == true ]]; then
    echo "  [DRY] Would replace '$OLD_UA' with '$NEW_UA'"
    grep -oE 'axios/[0-9a-zA-Z.-]+' "$f" 2>/dev/null | head -3 || true
    return 0
  fi

  backup_file "$f"

  # Safe string replacement (works on both JS bundles and native binaries)
  perl -pi -e 's|\Q'"$OLD_UA"'\E|'"$NEW_UA"'|g' "$f"

  echo "✅ Patched: $f"

  # Re-sign macOS binaries (required after editing)
  if [[ "$(uname)" == "Darwin" ]] && [[ -x "$f" ]]; then
    codesign -f -s - "$f" 2>/dev/null || true
    echo "   Re-signed for macOS"
  fi
}

# ====================== CLI (native binary) ======================
echo "Scanning CLI binary..."
CLI_PATHS=(
  "$(command -v claude 2>/dev/null || true)"
  "$HOME/.local/bin/claude"
  "/usr/local/bin/claude"
  "/opt/homebrew/bin/claude"
  "$HOME/bin/claude"
)

for p in "${CLI_PATHS[@]}"; do
  [[ -x "$p" ]] && patch_file "$p"
done

# ====================== VS Code Extension ======================
echo "Scanning VS Code extension..."
EXT_BASES=(
  "$HOME/.vscode/extensions"
  "$HOME/Library/Application Support/Code/extensions"   # macOS fallback
  "$HOME/.vscode-oss/extensions"
)

for base in "${EXT_BASES[@]}"; do
  [[ -d "$base" ]] || continue
  find "$base" -maxdepth 3 -name "*anthropic.claude-code-*" -type d 2>/dev/null | while read -r ext_dir; do
    echo "  Found extension: $ext_dir"
    # Patch all JS files (bundled code) that contain the UA
    find "$ext_dir" -type f \( -name "*.js" -o -name "*.bundle.js" -o -name "index.js" \) 2>/dev/null | while read -r js; do
      patch_file "$js"
    done
  done
done

echo "==================================================="
if [[ $DRY_RUN == true ]]; then
  echo "Dry-run complete. Review the output above."
  echo "To apply: run the script WITHOUT --dry-run"
else
  echo "Patching finished!"
  echo "• Restart your terminal and/or VS Code"
  echo "• Test WebFetch on Gitbook (ask Claude to read a page)"
  echo "• After any Claude Code update you must re-run the script"
  echo "• All originals are backed up with .bak. timestamps"
fi
echo "Use at your own risk — this is an unofficial patch."