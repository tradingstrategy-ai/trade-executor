#!/bin/bash
# Manual tutorial: an Arbitrum Lagoon vault trading synchronous ERC-4626 vaults.
#
# Read top to bottom before running. These commands spend real USDC and gas when
# JSON_RPC_ARBITRUM points to mainnet. Run a dedicated test vault with no daemon
# or other state writer. This follows manual-lighter-subaccounts-cli.sh's format.
#
# 0. Check funds and save public resume configuration.
# 1. Deploy Lagoon, its Safe and Guard with the strategy's ERC-4626 permissions.
# 2. Initialise the state and subscribe the investor's capital.
# 3. Open and close each declared implementation with perform-test-trade.
# 4. Reconcile books and NAV, redeem the investor and verify no shares remain.
#
# Prerequisites: Poetry environment, jq, a funded ERC4626_TEST_PRIVATE_KEY,
# JSON_RPC_ARBITRUM, TRADING_STRATEGY_API_KEY (check-wallet constructs a client)
# and ETHERSCAN_API_KEY for verified mainnet deployment.
# The static strategy declares generic, Morpho V1/V2, Yearn V3 and Compounder
# representatives. They are test candidates, not allocation recommendations.
# Closed/full/illiquid vaults stop the run; do not silently omit their rows.
# Optional inputs: ERC4626_RUN_DIR for an existing run, SUBSCRIPTION_USDC
# (default 10), and the existing executor AMOUNT (default 1 USDC per test).
#
# Start:
#   source .local-test.env
#   export ERC4626_TEST_PRIVATE_KEY=...  # keep in your shell; never pass as an argument
#   bash scripts/lagoon/manual-erc4626-cli.sh --deploy --subscribe
# Continue trading after deployment/subscription:
#   source <run-directory>/env.sh
#   bash scripts/lagoon/manual-erc4626-cli.sh
# If HyperSync is behind the chain tip, unset HYPERSYNC_API_KEY before resuming
# to use the existing RPC settlement log reader. Keep the original state.
#
# Fork rehearsal: tests/lagoon/test_erc4626_manual_cli_e2e.py exercises the public
# commands and balance helpers on a pinned fork; it does not execute this shell
# wrapper. Only investor funding is injected, never protocol liquidity. The test
# additionally syncs settlement before snapshots for nested simulation forks.
# Its results do not establish current mainnet availability.

set -Eeuo pipefail
DEPLOY=false
SUBSCRIBE=false
for argument in "$@"; do
    case "$argument" in
        --deploy) DEPLOY=true ;;
        --subscribe) SUBSCRIBE=true ;;
        *) echo "Usage: $0 [--deploy] [--subscribe]" >&2; exit 2 ;;
    esac
done
umask 077
cd "$(dirname "$0")/../.."
: "${JSON_RPC_ARBITRUM:?Set JSON_RPC_ARBITRUM}"
: "${ERC4626_TEST_PRIVATE_KEY:?Set ERC4626_TEST_PRIVATE_KEY in your shell}"
: "${TRADING_STRATEGY_API_KEY:?Set TRADING_STRATEGY_API_KEY for check-wallet}"
command -v poetry >/dev/null
command -v jq >/dev/null

export ERC4626_RUN_DIR="${ERC4626_RUN_DIR:-$HOME/.tradingstrategy/examples/erc4626-cli-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$ERC4626_RUN_DIR/state"
chmod 700 "$ERC4626_RUN_DIR"
export SUBSCRIPTION_USDC="${SUBSCRIPTION_USDC:-10}"
export AMOUNT="${AMOUNT:-1}"
# Match the existing Lagoon funding/redemption commands' minimum gas balance.
export MIN_GAS_BALANCE="${MIN_GAS_BALANCE:-0.001}"
export EXECUTOR_ID=erc4626-cli
export NAME="ERC-4626 CLI tutorial"
export PRIVATE_KEY="$ERC4626_TEST_PRIVATE_KEY"
export STATE_FILE="$ERC4626_RUN_DIR/state/$EXECUTOR_ID.json"
export CACHE_PATH="$ERC4626_RUN_DIR/cache"
export VAULT_RECORD_FILE="$ERC4626_RUN_DIR/vault-record.txt"
export STRATEGY_FILE="$ERC4626_RUN_DIR/strategy.py"
export LOG_LEVEL=info
export SIMULATE=false
export ASSET_MANAGEMENT_MODE=lagoon
# Only Arbitrum belongs to this universe. Passing unrelated RPC chains would
# change deployment configuration; keep those credentials out of env.sh/logs.
for variable in ${!JSON_RPC_@}; do
    if [ "$variable" != JSON_RPC_ARBITRUM ]; then unset "$variable"; fi
done

SOURCE_STRATEGY=strategies/test_only/erc4626_manual_strategy.py
if [ -s "$STRATEGY_FILE" ]; then
    cmp -s "$SOURCE_STRATEGY" "$STRATEGY_FILE" || { echo "Strategy changed: inspect the saved run before reusing its permissions" >&2; exit 1; }
else
    cp "$SOURCE_STRATEGY" "$STRATEGY_FILE"
fi
# Public settings only. Required credentials remain references to the shell.
{
    printf 'export ERC4626_RUN_DIR=%q\n' "$ERC4626_RUN_DIR"
    printf 'export SUBSCRIPTION_USDC=%q AMOUNT=%q\n' "$SUBSCRIPTION_USDC" "$AMOUNT"
    printf ': "${ERC4626_TEST_PRIVATE_KEY:?Set the original investor key}"\n'
    printf ': "${JSON_RPC_ARBITRUM:?Set the original Arbitrum RPC}"\n'
    printf 'export PRIVATE_KEY="$ERC4626_TEST_PRIVATE_KEY"\n'
    for setting in EXECUTOR_ID NAME STRATEGY_FILE STATE_FILE CACHE_PATH VAULT_RECORD_FILE ASSET_MANAGEMENT_MODE LOG_LEVEL SIMULATE MIN_GAS_BALANCE; do
        printf 'export %s=%q\n' "$setting" "${!setting}"
    done
} > "$ERC4626_RUN_DIR/env.sh"
exec > >(tee -a "$ERC4626_RUN_DIR/run.log") 2>&1

CURRENT_STEP=0
LAST_COMMAND="preflight"
ACTIVE_PID=""
HEARTBEAT_PID=""
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
step() { CURRENT_STEP="$1"; log "Step $1: $2"; }
cleanup() {
    if [ -n "$HEARTBEAT_PID" ]; then kill "$HEARTBEAT_PID" 2>/dev/null || true; fi
}
failed() {
    cleanup
    log "Stopped at step $CURRENT_STEP: $LAST_COMMAND"
    log "State and transaction identities remain in $ERC4626_RUN_DIR. Inspect the failure before retrying."
    log "Do not discard state, correct discrepancies away or add capital automatically."
    log "Resume: source $ERC4626_RUN_DIR/env.sh && bash scripts/lagoon/manual-erc4626-cli.sh"
}
trap failed ERR
trap 'cleanup; if [ -n "$ACTIVE_PID" ]; then kill "$ACTIVE_PID" 2>/dev/null || true; fi; failed; exit 130' INT TERM
run() {
    local started=$SECONDS status=0 reply=""
    if [ "${1:-}" = --stdin ]; then reply="$2"; shift 2; fi
    LAST_COMMAND="$(printf '%q ' "$@")"
    log "$LAST_COMMAND"
    if [ -n "$reply" ]; then
        printf '%b' "$reply" | "$@" &
    else
        "$@" < /dev/null &
    fi
    ACTIVE_PID=$!
    (while sleep 60; do log "Still running step $CURRENT_STEP"; done) &
    HEARTBEAT_PID=$!
    wait "$ACTIVE_PID" || status=$?
    cleanup
    ACTIVE_PID="" HEARTBEAT_PID=""
    log "Finished after $((SECONDS - started)) seconds (exit $status)"
    return "$status"
}
MANUAL=(poetry run python -m tradeexecutor.testing.erc4626_manual --strategy-file "$STRATEGY_FILE")

# Step 0: deployment and subscription are explicit operator actions. A rerun
# without flags only reuses existing deployment/state and continues trading.
step 0 "Check tools and signer"
DEPLOYER_ADDRESS="$("${MANUAL[@]}" identity)"
DEPLOYMENT_FILE="$ERC4626_RUN_DIR/state/$EXECUTOR_ID.deployment.json"
if [ "$DEPLOY" = true ] || [ "$SUBSCRIBE" = true ]; then
    run "${MANUAL[@]}" preflight --amount "$SUBSCRIPTION_USDC"
fi

# Step 1: the deployment command's artefact is authoritative. After an
# interrupted deployment without this artefact, inspect existing transactions
# and recover it before continuing. Do not repeat --deploy blindly.
step 1 "Deploy or load the Arbitrum Lagoon vault, Safe and Guard"
if [ "$DEPLOY" = true ]; then
    [ ! -e "$DEPLOYMENT_FILE" ] || { log "Deployment already exists; omit --deploy"; exit 1; }
    [ ! -e "$STATE_FILE" ] || { log "Executor state already exists; use a fresh run directory for a new deployment."; exit 1; }
    if [ "${UNIT_TESTING:-false}" != true ]; then
        : "${ETHERSCAN_API_KEY:?Set ETHERSCAN_API_KEY before verified mainnet deployment}"
    fi
    run --stdin 'y\n' poetry run trade-executor lagoon-deploy-vault \
        --strategy-file "$STRATEGY_FILE" --vault-record-file "$VAULT_RECORD_FILE" \
        --fund-name "ERC4626 CLI test" --fund-symbol ECLI \
        --multisig-owners "$DEPLOYER_ADDRESS" --performance-fee 0 --management-fee 0
fi
[ -s "$DEPLOYMENT_FILE" ] || { log "No deployment artefact. For a new vault use --deploy; after an interruption recover the existing deployment first."; exit 1; }
export VAULT_ADDRESS="$(jq -er '.deployments.arbitrum.vault_address' "$DEPLOYMENT_FILE")"
export VAULT_ADAPTER_ADDRESS="$(jq -er '.deployments.arbitrum.module_address' "$DEPLOYMENT_FILE")"
printf 'export VAULT_ADDRESS=%q VAULT_ADAPTER_ADDRESS=%q\n' "$VAULT_ADDRESS" "$VAULT_ADAPTER_ADDRESS" >> "$ERC4626_RUN_DIR/env.sh"
log "Lagoon vault: https://arbiscan.io/address/$VAULT_ADDRESS"
log "Guard/module: https://arbiscan.io/address/$VAULT_ADAPTER_ADDRESS"

# Step 2: the existing command requests, settles and claims the first deposit.
# --subscribe is for a fresh vault. After an interruption, reconcile through
# existing Lagoon commands, then rerun without --subscribe. No marker files.
step 2 "Initialise state and optionally subscribe $SUBSCRIPTION_USDC USDC"
if [ ! -s "$STATE_FILE" ]; then
    [ "$DEPLOY" = true ] || { log "Executor state is missing. Recover it before continuing. If this is a verified fresh deployment that never had a ledger, run trade-executor init manually."; exit 1; }
    run poetry run trade-executor init
fi
run poetry run trade-executor check-wallet
if [ "$SUBSCRIBE" = true ]; then
    run poetry run trade-executor lagoon-first-deposit --deposit-amount "$SUBSCRIPTION_USDC"
fi

# Step 3: one position at a time. The ledger, not a .done marker, decides which
# leg to run. A completed buy is never repeated after interruption. The close
# uses its actual test-owned shares and receipt proceeds, not a 1:1 USD guess.
step 3 "Open and close each declared protocol using perform-test-trade"
"${MANUAL[@]}" pairs > "$ERC4626_RUN_DIR/pairs.tsv"
while IFS=$'\t' read -r label target_address; do
    log "Testing $label at $target_address"
    phase="$("${MANUAL[@]}" phase --address "$target_address" < /dev/null)"
    if [ "$phase" = open ]; then
        run poetry run trade-executor perform-test-trade \
            --pair "(arbitrum, $target_address)" --amount "$AMOUNT" --buy-only --no-test-short
        phase="$("${MANUAL[@]}" phase --address "$target_address" < /dev/null)"
    fi
    if [ "$phase" = close ]; then
        run poetry run trade-executor perform-test-trade \
            --pair "(arbitrum, $target_address)" --close-only --no-test-short
    fi
    [ "$("${MANUAL[@]}" phase --address "$target_address" < /dev/null)" = done ]
    run poetry run trade-executor check-accounts
    run poetry run trade-executor show-valuation
done < "$ERC4626_RUN_DIR/pairs.tsv"

# Step 4: all protocol shares must be redeemed before the investor exits.
# Honour Lagoon Guard settlement limits and cooldowns: wait/resume on a refusal,
# never bypass them. Keep investor shares and protocol holdings intact when an
# exit is unavailable. lagoon-redeem has request/settle/claim recovery built in.
step 4 "Reconcile, redeem the investor and check final balances"
run "${MANUAL[@]}" audit
run poetry run trade-executor check-accounts
run poetry run trade-executor check-wallet
run poetry run trade-executor lagoon-settle --force-valuation
run poetry run trade-executor show-valuation
run poetry run trade-executor lagoon-redeem
run "${MANUAL[@]}" finished
log "All declared protocols opened and closed. Investor redeemed. Evidence: $ERC4626_RUN_DIR"
