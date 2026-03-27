#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_REPO_ROOT="${PARENT_REPO_ROOT:-/Users/moo/code/trade-executor}"
PYTHON_BIN="${PYTHON_BIN:-$PARENT_REPO_ROOT/.venv/bin/python}"
PYTEST_BIN="${PYTEST_BIN:-$PARENT_REPO_ROOT/.venv/bin/pytest}"
SAMPLE_STATE_FILE="${SAMPLE_STATE_FILE:-$HOME/Downloads/hyper-ai-3.json}"
STRATEGY_FILE="$REPO_ROOT/strategies/test_only/hyper-ai-test.py"
TEST_PRIVATE_KEY="0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83"

if [[ ! -f "$REPO_ROOT/.local-test.env" ]]; then
  echo "Missing $REPO_ROOT/.local-test.env"
  exit 1
fi

if [[ ! -f "$SAMPLE_STATE_FILE" ]]; then
  echo "Missing sample state file: $SAMPLE_STATE_FILE"
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN"
  exit 1
fi

if [[ ! -x "$PYTEST_BIN" ]]; then
  echo "Missing pytest wrapper: $PYTEST_BIN"
  exit 1
fi

cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "$REPO_ROOT/.local-test.env"

: "${JSON_RPC_HYPERLIQUID:?Missing JSON_RPC_HYPERLIQUID after sourcing .local-test.env}"
: "${TRADING_STRATEGY_API_KEY:?Missing TRADING_STRATEGY_API_KEY after sourcing .local-test.env}"

TMPDIR_RUN="$(mktemp -d "${TMPDIR:-/tmp}/hypercore-account-checks.XXXXXX")"
STATE_FILE="$TMPDIR_RUN/hyper-ai-3-copy.json"
CACHE_PATH="$TMPDIR_RUN/cache"
mkdir -p "$CACHE_PATH"
cp "$SAMPLE_STATE_FILE" "$STATE_FILE"

STATE_DETAILS="$(
  STATE_PATH="$STATE_FILE" "$PYTHON_BIN" - <<'PY'
import os
from tradeexecutor.state.state import State

state = State.read_json_file(os.environ["STATE_PATH"])
vault_address = state.sync.deployment.address
assert vault_address, "State file did not contain a Lagoon vault address"

module_addresses = {
    tx.contract_address
    for position in state.portfolio.open_positions.values()
    for trade in position.trades.values()
    for tx in trade.blockchain_transactions
    if tx.contract_address
}
assert len(module_addresses) == 1, module_addresses

hypercore_positions = [
    position
    for position in state.portfolio.open_positions.values()
    if position.pair.is_hyperliquid_vault()
]

print(vault_address)
print(next(iter(module_addresses)))
print(len(hypercore_positions))
PY
)"
VAULT_ADDRESS="$(printf '%s\n' "$STATE_DETAILS" | sed -n '1p')"
VAULT_ADAPTER_ADDRESS="$(printf '%s\n' "$STATE_DETAILS" | sed -n '2p')"
HYPERCORE_POSITION_COUNT="$(printf '%s\n' "$STATE_DETAILS" | sed -n '3p')"

COMMON_ENV=(
  "HOME=$HOME"
  "PATH=$PATH"
  "PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  "EXECUTOR_ID=manual_hypercore_account_checks"
  "STRATEGY_FILE=$STRATEGY_FILE"
  "STATE_FILE=$STATE_FILE"
  "CACHE_PATH=$CACHE_PATH"
  "ASSET_MANAGEMENT_MODE=lagoon"
  "VAULT_ADDRESS=$VAULT_ADDRESS"
  "VAULT_ADAPTER_ADDRESS=$VAULT_ADAPTER_ADDRESS"
  "PRIVATE_KEY=$TEST_PRIVATE_KEY"
  "JSON_RPC_HYPERLIQUID=$JSON_RPC_HYPERLIQUID"
  "TRADING_STRATEGY_API_KEY=$TRADING_STRATEGY_API_KEY"
  "UNIT_TESTING=true"
  "LOG_LEVEL=info"
)

run_cli() {
  local command="$1"
  local logfile="$2"
  local expected_codes="${3:-0}"
  local status

  set +e
  env -i "${COMMON_ENV[@]}" \
    "$PYTHON_BIN" -c "from tradeexecutor.cli.main import app; app([\"$command\"], standalone_mode=False)" \
    2>&1 | tee "$logfile"
  status=${PIPESTATUS[0]}
  set -e

  case " $expected_codes " in
    *" $status "*) ;;
    *)
      echo
      echo "Command '$command' failed with exit code $status"
      echo "See log: $logfile"
      exit "$status"
      ;;
  esac
}

run_pytest() {
  local test_name="$1"
  local logfile="$2"

  "$PYTEST_BIN" -s "$test_name" 2>&1 | tee "$logfile"
}

CHECK_BEFORE_LOG="$TMPDIR_RUN/check-accounts-before.log"
CORRECT_LOG="$TMPDIR_RUN/correct-accounts.log"
CHECK_AFTER_LOG="$TMPDIR_RUN/check-accounts-after.log"
PYTEST_CHECK_LOG="$TMPDIR_RUN/pytest-check-accounts.log"
PYTEST_CORRECT_LOG="$TMPDIR_RUN/pytest-correct-accounts.log"

echo "Worktree root: $REPO_ROOT"
echo "Temporary run directory: $TMPDIR_RUN"
echo "Copied state file: $STATE_FILE"
echo "Detected Hypercore vault positions in state: $HYPERCORE_POSITION_COUNT"
echo
echo "Running manual check-accounts before correction"
run_cli "check-accounts" "$CHECK_BEFORE_LOG" "0 1"
echo
echo "Running manual correct-accounts"
run_cli "correct-accounts" "$CORRECT_LOG" "0"
echo
echo "Running manual check-accounts after correction"
run_cli "check-accounts" "$CHECK_AFTER_LOG" "0"
echo
echo "Running pytest regression: check-accounts"
run_pytest \
  "tests/hyperliquid/test_hypercore_account_checks_sample_state.py::test_check_accounts_lists_hypercore_vault_positions" \
  "$PYTEST_CHECK_LOG"
echo
echo "Running pytest regression: correct-accounts"
run_pytest \
  "tests/hyperliquid/test_hypercore_account_checks_sample_state.py::test_correct_accounts_syncs_hypercore_vault_positions" \
  "$PYTEST_CORRECT_LOG"
echo
echo "Finished."
echo "Logs:"
echo "  $CHECK_BEFORE_LOG"
echo "  $CORRECT_LOG"
echo "  $CHECK_AFTER_LOG"
echo "  $PYTEST_CHECK_LOG"
echo "  $PYTEST_CORRECT_LOG"
echo
echo "Quick things to inspect:"
echo "  - '$CHECK_BEFORE_LOG' should contain 'pmalt' and '[ Systemic Strategies ] L/S Grids'"
echo "  - '$CORRECT_LOG' should contain 'Vault equity sync:'"
echo "  - '$CHECK_AFTER_LOG' should end with a clean account table and 'All accounts match'"
