#!/bin/bash
# Manual test: an Arbitrum Lagoon vault trading a Lighter pool, Atoma and Upshift.
#
# Read top to bottom before running. In live mode these commands spend real
# USDC and gas on Arbitrum and Ethereum. Run a dedicated test vault with no
# daemon or other state writer. This follows manual-erc4626-cli.sh's format.
#
# The vault's hub is Arbitrum. Atoma Index is traded on the hub; Upshift Core
# USDC and the pmalt Lighter public pool are reached through the Ethereum
# satellite Safe, funded over CCTP. Deployment activates the satellite Safe's
# Lighter account and registers its API key.
#
#  0. Check tools, signer, funds and that the Lighter pool accepts deposits.
#  1. Deploy the hub vault, Ethereum satellite and Lighter account.
#  2. Initialise the state and subscribe the investor's capital.
#  3. Lighter pool: bridge to Ethereum, deposit, mint; burn, withdraw, claim, bridge back.
#  4. Atoma Index: deposit; after the share lock, request, settle and claim.
#  5. Upshift Core USDC: bridge, deposit; redeem, bridge back.
#  6. Reconcile books and NAV (state audit and check-wallet: check-accounts
#     supports one chain only).
#  7. Settle NAV, redeem the investor and check that no shares remain.
#
# Live funding, the minimum for the default amounts (see
# docs/plans/arbitrum-multi-vault-live-plan.md, "Minimum funding"):
#
#   131 native Arbitrum USDC  20 bootstrap (consumed by deployment: 1 USDC
#                             activates Lighter, the rest stays in the vault)
#                             + 111 subscription (100 Atoma, 5 Lighter,
#                             5 Upshift, 1 CCTP fees and headroom)
#   0.02 ETH on Ethereum      satellite deployment, Lighter activation,
#                             deposits, claims and CCTP receives
#   0.005 ETH on Arbitrum     hub deployment, Atoma and CCTP burns
#
# Everything returns to the deployer apart from fees, operator fees and
# sub-cent dust. The activation USDC is minted into the pool on the first
# Lighter buy and withdrawn with it.
#
# Simulated mode (SIMULATE=true) runs the same commands against two Anvil
# forks: Arbitrum at Atoma's reviewed block and Ethereum at the shared
# midnight block. Lighter's sequencer is the fake of
# tradeexecutor/testing/lighter_core.py, Circle's attestations are forged for
# real burns, and only Atoma's share lock is advanced on the clock. All
# Safe, Guard, Lagoon, Atoma, Upshift, CCTP and ZkLighter transactions are
# real fork transactions. Simulation files go to /tmp/multi-vault-sim-<run>/,
# never into the run directory.
#
# Inputs:
#   JSON_RPC_ARBITRUM, JSON_RPC_ETHEREUM  Required. Forked in simulated mode.
#   TRADING_STRATEGY_API_KEY              Required: commands create a client.
#   MULTI_VAULT_TEST_PRIVATE_KEY          Live only: funded deployer, which
#                                         becomes the Safe owner and asset
#                                         manager. Never printed.
#   ETHERSCAN_API_KEY                     Live only: contract verification.
#   SIMULATE                              Optional, default false.
#   MULTI_VAULT_RUN_DIR                   Optional: resume an earlier run.
#   SUBSCRIPTION_USDC, AMOUNT_LIGHTER, AMOUNT_ATOMA, AMOUNT_UPSHIFT
#                                         Optional amounts, defaults above.
#
# Targets: --targets lighter,atoma,upshift (the default) selects which
# positions steps 3-5 open and close in this invocation. Steps 6-7 run only
# once every target is done, so the ones with no lock can finish first:
# Lighter (5-minute mint lockup, ~25-minute withdrawal) and Upshift (no lock)
# in one sitting, Atoma (shares lock until the end of its 3-day epoch, exit
# settled at the end of the next) days later with --targets atoma.
#
# Start:
#   source .local-test.env
#   SIMULATE=true bash scripts/lagoon/manual-multi-vault-cli.sh --deploy --subscribe
# Live, in two sittings:
#   export MULTI_VAULT_TEST_PRIVATE_KEY=...  # keep in your shell; never pass as an argument
#   bash scripts/lagoon/manual-multi-vault-cli.sh --deploy --subscribe --targets lighter,upshift
#   source <run directory>/env.sh && bash scripts/lagoon/manual-multi-vault-cli.sh --targets atoma
# Continue after an interruption or a wait:
#   source <run directory>/env.sh
#   bash scripts/lagoon/manual-multi-vault-cli.sh [--targets ...]
#
# Resuming: every step reads the state file. A completed buy is never
# repeated; a pending exit is resumed, never requested again. Atoma's live
# exit waits for its share lock and for the operator's epoch settlement:
# the script stops and says when to rerun.

set -Eeuo pipefail
DEPLOY=false
SUBSCRIBE=false
TARGETS=lighter,atoma,upshift
while [ $# -gt 0 ]; do
    case "$1" in
        --deploy) DEPLOY=true ;;
        --subscribe) SUBSCRIBE=true ;;
        --targets) TARGETS="$2"; shift ;;
        --targets=*) TARGETS="${1#--targets=}" ;;
        *) echo "Usage: $0 [--deploy] [--subscribe] [--targets lighter,atoma,upshift]" >&2; exit 2 ;;
    esac
    shift
done
for target in ${TARGETS//,/ }; do
    case "$target" in lighter|atoma|upshift) ;; *) echo "Unknown target '$target'; use lighter, atoma or upshift" >&2; exit 2 ;; esac
done
is_selected() { case ",$TARGETS," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
umask 077
cd "$(dirname "$0")/../.."
command -v jq >/dev/null
: "${JSON_RPC_ARBITRUM:?Set JSON_RPC_ARBITRUM}"
: "${JSON_RPC_ETHEREUM:?Set JSON_RPC_ETHEREUM}"
: "${TRADING_STRATEGY_API_KEY:?Set TRADING_STRATEGY_API_KEY}"

export MULTI_VAULT_SIMULATE="${MULTI_VAULT_SIMULATE:-${SIMULATE:-false}}"
# A Python command prefix; override with the interpreter of an existing environment.
read -r -a PYTHON_CMD <<< "${MULTI_VAULT_PYTHON:-poetry run python}"
HELPER=("${PYTHON_CMD[@]}" -m tradeexecutor.testing.multi_vault_manual)

export MULTI_VAULT_RUN_DIR="${MULTI_VAULT_RUN_DIR:-$HOME/.tradingstrategy/examples/multi-vault-cli-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$MULTI_VAULT_RUN_DIR/state"
chmod 700 "$MULTI_VAULT_RUN_DIR"
export SUBSCRIPTION_USDC="${SUBSCRIPTION_USDC:-111}"
export AMOUNT_LIGHTER="${AMOUNT_LIGHTER:-5}" AMOUNT_ATOMA="${AMOUNT_ATOMA:-100}" AMOUNT_UPSHIFT="${AMOUNT_UPSHIFT:-5}"
export EXECUTOR_ID=multi-vault-cli
export NAME="Multi-vault CLI tutorial"
export STATE_FILE="$MULTI_VAULT_RUN_DIR/state/$EXECUTOR_ID.json"
export CACHE_PATH="$MULTI_VAULT_RUN_DIR/cache"
export VAULT_RECORD_FILE="$MULTI_VAULT_RUN_DIR/vault-record.txt"
# The owner-only operator record with the Lighter API key, written at deployment.
export LIGHTER_OPERATOR_RECORD_FILE="$MULTI_VAULT_RUN_DIR/vault-record.json"
export STRATEGY_FILE="$MULTI_VAULT_RUN_DIR/strategy.py"
export LOG_LEVEL="${LOG_LEVEL:-info}"
export ASSET_MANAGEMENT_MODE=lagoon
export MIN_GAS_BALANCE="${MIN_GAS_BALANCE:-0.001}"
# The executor's own --simulate mode writes no state; this tutorial always uses a real state file.
export SIMULATE=false
# Only the hub and the Ethereum satellite belong to this universe. Other RPCs
# would add deployment chains; keep their credentials out of env.sh and logs.
for variable in ${!JSON_RPC_@}; do
    case "$variable" in JSON_RPC_ARBITRUM|JSON_RPC_ETHEREUM) ;; *) unset "$variable" ;; esac
done

if [ "$MULTI_VAULT_SIMULATE" = true ]; then
    # Public Anvil account zero, funded on the forks: never a production key.
    export PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
    export MULTI_VAULT_SIM_DIR="${MULTI_VAULT_SIM_DIR:-/tmp/multi-vault-sim-$(basename "$MULTI_VAULT_RUN_DIR")}"
    # Fork-only settings: no contract verification, one confirmation block.
    export UNIT_TESTING=true CONFIRMATION_BLOCK_COUNT=1
else
    : "${MULTI_VAULT_TEST_PRIVATE_KEY:?Set MULTI_VAULT_TEST_PRIVATE_KEY in your shell}"
    export PRIVATE_KEY="$MULTI_VAULT_TEST_PRIVATE_KEY"
fi

SOURCE_STRATEGY=strategies/test_only/multi_vault_arbitrum.py
if [ -s "$STRATEGY_FILE" ]; then
    cmp -s "$SOURCE_STRATEGY" "$STRATEGY_FILE" || { echo "Strategy changed: inspect the saved run before reusing its permissions" >&2; exit 1; }
else
    cp "$SOURCE_STRATEGY" "$STRATEGY_FILE"
fi
exec > >(tee -a "$MULTI_VAULT_RUN_DIR/run.log") 2>&1

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
    log "State and transaction identities remain in $MULTI_VAULT_RUN_DIR. Inspect the failure before retrying."
    log "Do not discard state, correct discrepancies away or add capital automatically."
    log "Resume: source $MULTI_VAULT_RUN_DIR/env.sh && bash scripts/lagoon/manual-multi-vault-cli.sh"
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

# Step 0: in simulated mode, start the forks once and point the RPCs at them.
step 0 "Check tools, signer, funds and the Lighter pool"
if [ "$MULTI_VAULT_SIMULATE" = true ]; then
    if [ ! -s "$MULTI_VAULT_SIM_DIR/simulation.json" ]; then
        [ "$DEPLOY" = true ] || { log "No running simulation in $MULTI_VAULT_SIM_DIR; a simulated run cannot resume after its forks stop. Start a fresh run with --deploy."; exit 1; }
        FORK_EXPORTS="$("${HELPER[@]}" simulate-start --directory "$MULTI_VAULT_SIM_DIR" --deployer-usdc "$((20 + SUBSCRIPTION_USDC))")"
        eval "$FORK_EXPORTS"
    else
        eval "$(jq -r '"export JSON_RPC_ARBITRUM=\(.arbitrum_rpc) JSON_RPC_ETHEREUM=\(.ethereum_rpc)"' "$MULTI_VAULT_SIM_DIR/simulation.json")"
    fi
    log "Simulated forks: Arbitrum $JSON_RPC_ARBITRUM, Ethereum $JSON_RPC_ETHEREUM; simulation files in $MULTI_VAULT_SIM_DIR"
    EXECUTOR=("${HELPER[@]}" cli --)
else
    read -r -a EXECUTOR <<< "${MULTI_VAULT_EXECUTOR:-poetry run trade-executor}"
fi
# Public settings only. Credentials remain references to the shell.
{
    printf 'export MULTI_VAULT_RUN_DIR=%q MULTI_VAULT_SIMULATE=%q\n' "$MULTI_VAULT_RUN_DIR" "$MULTI_VAULT_SIMULATE"
    printf 'export SUBSCRIPTION_USDC=%q AMOUNT_LIGHTER=%q AMOUNT_ATOMA=%q AMOUNT_UPSHIFT=%q\n' "$SUBSCRIPTION_USDC" "$AMOUNT_LIGHTER" "$AMOUNT_ATOMA" "$AMOUNT_UPSHIFT"
    if [ "$MULTI_VAULT_SIMULATE" = true ]; then
        printf 'export MULTI_VAULT_SIM_DIR=%q\n' "$MULTI_VAULT_SIM_DIR"
    else
        printf ': "${MULTI_VAULT_TEST_PRIVATE_KEY:?Set the original deployer key}"\n'
    fi
    printf ': "${JSON_RPC_ARBITRUM:?Set the original Arbitrum RPC}"\n'
    printf ': "${JSON_RPC_ETHEREUM:?Set the original Ethereum RPC}"\n'
} > "$MULTI_VAULT_RUN_DIR/env.sh"

DEPLOYER_ADDRESS="$("${HELPER[@]}" identity)"
log "Deployer and investor: $DEPLOYER_ADDRESS"
if [ "$DEPLOY" = true ] || [ "$SUBSCRIBE" = true ]; then
    PREFLIGHT=(preflight --subscription-usdc "$([ "$SUBSCRIBE" = true ] && echo "$SUBSCRIPTION_USDC" || echo 0)")
    [ "$DEPLOY" = true ] && PREFLIGHT+=(--bootstrap)
    run "${HELPER[@]}" "${PREFLIGHT[@]}"
fi

# Step 1: the deployment artefact is authoritative. After an interrupted
# deployment without it, recover from the transactions and the private
# activation report beside the vault record. Do not repeat --deploy blindly.
step 1 "Deploy or load the Arbitrum vault, Ethereum satellite and Lighter account"
DEPLOYMENT_FILE="$MULTI_VAULT_RUN_DIR/state/$EXECUTOR_ID.deployment.json"
if [ "$DEPLOY" = true ]; then
    [ ! -e "$DEPLOYMENT_FILE" ] || { log "Deployment already exists; omit --deploy"; exit 1; }
    [ ! -e "$STATE_FILE" ] || { log "Executor state already exists; use a fresh run directory for a new deployment."; exit 1; }
    if [ "$MULTI_VAULT_SIMULATE" != true ]; then
        : "${ETHERSCAN_API_KEY:?Set ETHERSCAN_API_KEY before verified mainnet deployment}"
    fi
    # The command asks once to confirm a deployment that activates Lighter.
    run --stdin 'y\n' "${EXECUTOR[@]}" lagoon-deploy-vault \
        --strategy-file "$STRATEGY_FILE" --vault-record-file "$VAULT_RECORD_FILE" \
        --fund-name "Multi-vault CLI test" --fund-symbol MVCLI \
        --multisig-owners "$DEPLOYER_ADDRESS" --performance-fee 0 --management-fee 0 \
        --generate-lighter-api-key
fi
[ -s "$DEPLOYMENT_FILE" ] || { log "No deployment artefact. For a new vault use --deploy; after an interruption recover the existing deployment first."; exit 1; }
[ -s "$LIGHTER_OPERATOR_RECORD_FILE" ] || { log "No Lighter operator record at $LIGHTER_OPERATOR_RECORD_FILE; recover it from the activation report before trading."; exit 1; }
export VAULT_ADDRESS="$(jq -er '.deployments.arbitrum.vault_address' "$DEPLOYMENT_FILE")"
export VAULT_ADAPTER_ADDRESS="$(jq -er '.deployments.arbitrum.module_address' "$DEPLOYMENT_FILE")"
printf 'export VAULT_ADDRESS=%q VAULT_ADAPTER_ADDRESS=%q\n' "$VAULT_ADDRESS" "$VAULT_ADAPTER_ADDRESS" >> "$MULTI_VAULT_RUN_DIR/env.sh"
log "Lagoon vault: https://arbiscan.io/address/$VAULT_ADDRESS"
log "Ethereum satellite module: $(jq -r '.deployments.ethereum.module_address' "$DEPLOYMENT_FILE")"

# Step 2: init imports the 20 USDC bootstrap subscription from the deployment
# block. The vault is no longer fresh, so the trading capital is an ordinary
# subscription: the investor requests a deposit, the asset manager settles it
# with lagoon-settle and the investor claims the shares. The request reads the
# vault: a rerun never requests twice while one is outstanding. Pass
# --subscribe once per run.
step 2 "Initialise state and optionally subscribe $SUBSCRIPTION_USDC USDC"
if [ ! -s "$STATE_FILE" ]; then
    [ "$DEPLOY" = true ] || { log "Executor state is missing. Recover it before continuing."; exit 1; }
    run "${EXECUTOR[@]}" init
fi
run "${EXECUTOR[@]}" check-wallet
if [ "$SUBSCRIBE" = true ]; then
    LAST_COMMAND="multi_vault_manual request-deposit"
    deposit_status="$("${HELPER[@]}" request-deposit --amount "$SUBSCRIPTION_USDC" < /dev/null | tail -n 1)"
    log "Subscription status: $deposit_status"
    if [ "$deposit_status" = requested ] || [ "$deposit_status" = pending ]; then
        run "${EXECUTOR[@]}" lagoon-settle
    fi
    run "${HELPER[@]}" claim-shares
fi

# Steps 3-5: one target at a time; the ledger, not a marker, decides the leg.
test_target() {
    local target="$1" pair="$2" amount="$3" phase
    phase="$("${HELPER[@]}" phase "$target")"
    log "$target is at phase: $phase"
    if [ "$phase" = open ]; then
        run "${EXECUTOR[@]}" perform-test-trade --pair "$pair" --amount "$amount" --buy-only --no-test-short
        phase="$("${HELPER[@]}" phase "$target")"
    fi
    if [ "$phase" = pending ]; then
        # Resumes the saved operation or ticket; never creates another request.
        run "${EXECUTOR[@]}" perform-test-trade --pair "$pair" --close-only --no-test-short
        phase="$("${HELPER[@]}" phase "$target")"
    fi
    if [ "$phase" = close ] && [ "$target" = atoma ]; then
        if [ "$MULTI_VAULT_SIMULATE" = true ]; then
            run "${HELPER[@]}" atoma-unlock
        elif ! "${HELPER[@]}" atoma-unlocked; then
            log "atoma is waiting for its epoch lock. Keep this state and rerun --targets atoma after the time above."
            return 0
        fi
    fi
    if [ "$phase" = close ]; then
        run "${EXECUTOR[@]}" perform-test-trade --pair "$pair" --close-only --no-test-short
        phase="$("${HELPER[@]}" phase "$target")"
    fi
    if [ "$phase" != done ]; then
        log "$target is waiting ($phase). Keep this state and rerun after the protocol settles."
        return 0
    fi
    # check-accounts is single-chain; check-wallet reads both chains' Safes.
    run "${EXECUTOR[@]}" check-wallet
}

step 3 "Lighter pool: deposit, mint, burn, withdraw and claim"
if is_selected lighter; then test_target lighter "(lighter, lighter-pool-281474976552918)" "$AMOUNT_LIGHTER"; else log "lighter not selected"; fi
step 4 "Atoma Index: deposit, request withdrawal and claim"
if is_selected atoma; then test_target atoma "(arbitrum, 0xcc56410e1a136af0eceb7241c6ae394f4d8b581c)" "$AMOUNT_ATOMA"; else log "atoma not selected"; fi
step 5 "Upshift Core USDC: deposit and redeem through the satellite"
if is_selected upshift; then test_target upshift "(ethereum, 0xE9B725010A9E419412ed67d0fA5f3A5f40159D32)" "$AMOUNT_UPSHIFT"; else log "upshift not selected"; fi

# The investor is redeemed only once every target, selected or not, is done.
REMAINING=""
for target in lighter atoma upshift; do
    phase="$("${HELPER[@]}" phase "$target")"
    [ "$phase" = done ] || REMAINING="$REMAINING $target($phase)"
done
if [ -n "$REMAINING" ]; then
    log "Targets not yet done:$REMAINING. Rerun with --targets <name> when they can proceed; the investor is redeemed once all are done."
    [ "$MULTI_VAULT_SIMULATE" != true ] || log "Simulation forks keep running in $MULTI_VAULT_SIM_DIR"
    exit 0
fi

# Step 6: no automatic corrections; a discrepancy stops the run.
# check-accounts and correct-accounts support one chain only, so this
# multichain vault is reconciled by the state audit, check-wallet's balances
# on both chains and the NAV.
step 6 "Reconcile books and NAV"
run "${HELPER[@]}" audit
run "${EXECUTOR[@]}" check-wallet
run "${EXECUTOR[@]}" show-valuation

# Step 7: honour Guard settlement limits and cooldowns; wait and resume on a
# refusal, never bypass them. lagoon-redeem recovers its own request/claim.
step 7 "Settle NAV, redeem the investor and check final balances"
run "${EXECUTOR[@]}" lagoon-settle --force-valuation
run "${EXECUTOR[@]}" show-valuation
run "${EXECUTOR[@]}" lagoon-redeem
run "${HELPER[@]}" finished
log "Lighter pool, Atoma and Upshift opened and closed. Investor redeemed. Evidence: $MULTI_VAULT_RUN_DIR"
if [ "$MULTI_VAULT_SIMULATE" = true ]; then
    run "${HELPER[@]}" simulate-stop --directory "$MULTI_VAULT_SIM_DIR"
fi
