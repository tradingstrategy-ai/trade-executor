#!/bin/bash
#
# Manual test: a Lagoon vault with a Lighter master account and two
# sub-accounts, driven only by trade-executor command-line commands.
#
# This script is also the runbook of what such a deployment needs. Read it
# top to bottom: every step says what it does to the vault and why, and
# every action is a plain command you could type yourself.
#
# Steps, with their durations in the live run of 2026-10-28 (1h 55m in
# total). Steps 8 to 10 and the check-wallet runs were added after that run
# and have no live duration yet; each takes a few minutes.
#
#  0. Check the deployer can pay for the run                            6s
#  1. Deploy the vault: Safe, Lagoon vault, guard, Lighter master
#     account and two sub-accounts, each with its own API key      14m 44s
#  2. Read the deployment's public addresses and Lighter indices        0s
#  3. Initialise executor state, book the 1 USDC activation and
#     check the wallet                                                40s
#  4. Subscribe test capital to the vault and claim the shares      4m 49s
#  5. perform-test-trade: fund sub-account "trend" from the Safe    3m 11s
#  6. perform-test-trade: fund sub-account "arb" from the Safe      3m 13s
#  7. perform-test-trade: move USDC from "trend" to "arb"           1m 02s
#  8. Audit clean books: check-wallet, correct-accounts books
#     nothing, lagoon-settle posts the NAV the balances show
#  9. arb's trader moves USDC to "trend" outside the executor:
#     check-accounts reports it, correct-accounts books it
# 10. USDC reaches the Safe outside the vault: check-accounts
#     reports it, correct-accounts books it, the NAV rises
# 11. Withdraw "arb" to the Safe (Lighter L1 withdrawal)           29m 31s
# 12. Withdraw "trend" to the Safe (Lighter L1 withdrawal)         28m 45s
# 13. Withdraw the master to the Safe (secure withdrawal)          24m 14s
# 14. Redeem every share back to the deployer, check the wallet     4m 14s
# 15. Sweep the Safe and reconcile the deployer's balances             13s
#
# The live transaction links below use the step numbers of that run, in
# which the withdrawals were steps 8 to 10, the redemption step 11 and the
# sweep step 12.
#
# The three withdrawals are mostly Lighter's withdrawal delay; the rest is
# Ethereum confirmations. Gas prices were about 0.1 gwei.
#
# Example run, 2026-10-28: every USDC came back to the deployer, Lighter
# charged no transfer fee and the run spent 0.0074 ETH of gas.
#
#   Vault   https://etherscan.io/address/0x208353A55b5eDC3F07563e46fA910a305A5BFA55
#   Safe    https://etherscan.io/address/0x47D479f32D88f0e0898d85EF6FC98ef2b0A62380
#   Guard   https://etherscan.io/address/0xE31A546f24BC38D8B17bA9e8505Ae76b0DB48fC5
#   Lighter master 764640, sub-accounts trend 281474976447040 and arb 281474976447039
#
#   Step 1, 1 USDC activation deposit to Lighter
#     https://etherscan.io/tx/0x8f5691bf4575031c830915c9361b673430c5d6db4e8602810d5ec7a9ec3713b1
#   Step 1, API key registrations (Safe changePubKey) for master, trend and arb
#     https://etherscan.io/tx/0x86d2f68b0fe3c37e53a7e2f26b4c17a56a9454d65541641453991f63c44d521b
#     https://etherscan.io/tx/0x1ecc90d000852cd73097bffc120290505229d81a34c8da616f0b2bc25c714aba
#     https://etherscan.io/tx/0x43c94be6b96a7cf95ab61876be7b40469e7b3cc7188e115bfabc05afb17ac7c0
#   Step 4, deposit request, NAV posting, settlement and share claim
#     https://etherscan.io/tx/0xd6c1e6d968e38fa9f391f023902332666b033346d5c9650b39d7ec730dde6a2f
#     https://etherscan.io/tx/0x9ed6013f57179f66c1e9aca4fbbd4a4047b01800a8a28426ea3f9b42ef827e5c
#     https://etherscan.io/tx/0x0fbd10d3075d312b8ff0a0a1a6c3149eda7d2fed2669b2b0eca6068243cbe1cc
#     https://etherscan.io/tx/0x26ee3e338a800271f4a799f4f9607eb1373273acc09a7d99ee51fca63ffa7eb3
#   Steps 5 and 6, Safe deposits to the Lighter master (approve, deposit)
#     https://etherscan.io/tx/0x2b4f567a4c305130db083ecd4ae511f4cfc963d32d5058581010542cf12d3109
#     https://etherscan.io/tx/0x9609bc9108ed325be90bb8ffbeadb76c70012146636e2175030f16e122111961
#     https://etherscan.io/tx/0xb94c3d0477afdd08e63e22d746fdb2372122c09b80662c9d76a093be1c2e3d47
#     https://etherscan.io/tx/0xd2b7fa4453b211198cc35da91b0480ce7ce1c354d4a3305210c9f74e8cf083f1
#     The transfers between Lighter accounts in steps 5 to 7 happen on
#     Lighter, not Ethereum, so they have no Etherscan link.
#   Step 8, arb L1 withdrawal request and claim into the Safe
#     https://etherscan.io/tx/0x2e05ad2a75d529125bee9a48f435d4e584eabfc067f5a9ea173f8d859e2717a0
#     https://etherscan.io/tx/0x15fe80fb7afc6ead5c5d81c4b1cdef650956fccfdefbb7ddfe644adfaaa82f34
#   Step 9, trend L1 withdrawal request and claim into the Safe
#     https://etherscan.io/tx/0x69c745744bda1b90d40bbaeea44ad25ac46b22c2e33bde974efea64d08769cb1
#     https://etherscan.io/tx/0x044ed4f0f51693bcc7c134bffdc6c0797efd45839b66c286a74da53250d757c5
#   Step 10, claim of the master's secure withdrawal into the Safe; the
#   request itself is signed with the master's key on Lighter
#     https://etherscan.io/tx/0x54f491e8038fe2a8552daeaebcd5f445c3ba3ee6974e90955a4ae2c26bf6a483
#   Step 11, NAV posting, redemption settlement and claim
#     https://etherscan.io/tx/0x4f80f3f3f6eca9bb676a709e00ddec5ffbe4255db1447c31d7d45e2eeb034612
#     https://etherscan.io/tx/0x68a6739bb0d028e94b81c0afcf2cd1eae5d9ef6fe9cce4f33b8de0b4c39ebc6a
#     https://etherscan.io/tx/0xa038d97cbdbf30c14744b052e34b570ae63574e25c60b9f282bc3c0212cb205c
#
# Real money, Ethereum mainnet only: Lighter's sequencer cannot run on a
# fork, so there is no simulation mode. The deployer needs ETH for a full
# Lagoon vault deployment with contract verification plus about 30 more
# transactions (the run of 2026-10-28 spent 0.0074 ETH of gas before steps
# 8 to 10 added three NAV settlements and a transfer), and
# 1 + SUBSCRIPTION_USDC + 1 native USDC (12 by default). It gets the USDC
# back apart from Lighter transfer fees, if Lighter charges any, and dust
# below Lighter's 1 USDC withdrawal minimum.
#
# Inputs (export before running):
#
#   JSON_RPC_ETHEREUM         Ethereum mainnet archive RPC
#   LIGHTER_TEST_PRIVATE_KEY  Funded deployer; becomes the sole Safe owner
#                             and the asset manager. Never printed.
#   ETHERSCAN_API_KEY         Contract verification at deployment
#   TRADING_STRATEGY_API_KEY  Trading Strategy client every command creates
#   SUBSCRIPTION_USDC         Optional test capital, whole USDC, default 10
#   LIGHTER_RUN_DIR           Optional; set to an earlier run's directory to
#                             resume it
#   HEARTBEAT_SECONDS         Optional progress interval, default 60
#
# Prompts: lagoon-deploy-vault asks once to confirm the deployment,
# correct-accounts asks before applying corrections and lighter-move-funds
# asks for the direction, the amount and a confirmation. The script answers
# them and says so before each command. Every other command gets no input,
# so an unexpected prompt fails instead of hanging.
#
# Resuming: everything a run writes lives in its run directory. Rerun with
# LIGHTER_RUN_DIR set to that directory; completed steps are skipped, and a
# money-moving step first checks the books and the balances, so money is
# never moved twice. Where an interruption left a state the script cannot
# resolve by itself, such as a transfer that moved but was not saved or a
# withdrawal whose request was not recorded, it stops and prints what to do;
# commands to run by hand start with "source <run directory>/env.sh", which
# loads this run's configuration. If the script stops, its last lines say
# what failed and how to continue.
#
# Run from anywhere:
#
#   export JSON_RPC_ETHEREUM=... LIGHTER_TEST_PRIVATE_KEY=... ETHERSCAN_API_KEY=...
#   scripts/lagoon/manual-lighter-subaccounts-cli.sh
#
# See docs/plans/lagoon-lighter-cli-manual-test.md for the design.
#

set -Eeuo pipefail

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

: "${JSON_RPC_ETHEREUM:?Set JSON_RPC_ETHEREUM to an Ethereum mainnet archive RPC}"
: "${LIGHTER_TEST_PRIVATE_KEY:?Set LIGHTER_TEST_PRIVATE_KEY to the funded deployer key}"
: "${ETHERSCAN_API_KEY:?Set ETHERSCAN_API_KEY for contract verification}"
# The strategy builds its universe in code, but the commands are only tested with a Trading Strategy client.
: "${TRADING_STRATEGY_API_KEY:?Set TRADING_STRATEGY_API_KEY}"
export TRADING_STRATEGY_API_KEY

if [ "${SIMULATE:-false}" != "false" ]; then
    echo "Lighter activation needs live Ethereum and Lighter state; this script runs on mainnet only. Unset SIMULATE."
    exit 1
fi

command -v jq >/dev/null || { echo "jq is required to read the deployment artefact"; exit 1; }
command -v poetry >/dev/null || { echo "poetry is required"; exit 1; }

SUBSCRIPTION_USDC="${SUBSCRIPTION_USDC:-10}"
if ! [[ "$SUBSCRIPTION_USDC" =~ ^[0-9]+$ ]] || [ "$SUBSCRIPTION_USDC" -lt 4 ]; then
    echo "SUBSCRIPTION_USDC must be a whole number of USDC, at least 4: the run funds two sub-accounts with 2 USDC each"
    exit 1
fi

# Run from the repository root, so strategy paths and poetry resolve.
cd "$(dirname "$0")/../.."

# ---------------------------------------------------------------------------
# Configuration: the environment every trade-executor command reads
# ---------------------------------------------------------------------------

# All artefacts, records, state and logs of this run. Mode 0700: it holds
# the mode-0600 files with the generated Lighter API keys.
RUN_DIR="${LIGHTER_RUN_DIR:-$HOME/.tradingstrategy/examples/lighter-cli-$(date -u +%Y%m%d-%H%M%S)}"
mkdir -p "$RUN_DIR/state"
chmod 700 "$RUN_DIR"

# Executor id: names the state and the deployment artefact.
export EXECUTOR_ID="lighter-cli"

# Strategy module: declares the sub-accounts "trend" and "arb". Deployment
# creates one Lighter sub-account per declaration and every command builds
# one trading pair per account from it. This script never runs `start`, so
# the module's own cash policy never acts.
export STRATEGY_FILE="strategies/test_only/lighter_multi_subaccount_strategy.py"

# Executor state. The deployment artefact is written beside it as
# lighter-cli.deployment.json, where init and every later command look.
export STATE_FILE="$RUN_DIR/state/lighter-cli.json"

# The deployer signs everything: it deploys the vault, is the Safe's sole
# owner and the asset manager, and is the test investor.
export PRIVATE_KEY="$LIGHTER_TEST_PRIVATE_KEY"

# Contract verification at deployment, read from the environment so it
# never appears on an echoed command line.
export ETHERSCAN_API_KEY

# Live run, never a fork.
export SIMULATE=false

# Executor logging: info shows each transaction, transfer and withdrawal wait as it happens.
export LOG_LEVEL=info

# Token and market metadata cache of this run.
export CACHE_PATH="$RUN_DIR/cache"

# The deployer keeps only enough ETH for this run; skip the generic
# minimum-gas check that live executors use.
export MIN_GAS_BALANCE=0

# Public deployment report; the private records are written beside it:
#   vault-record.json                Master API key, mode 0600
#   vault-record.nautilus.<id>.json  One sub-account key each, mode 0600
export VAULT_RECORD_FILE="$RUN_DIR/vault-record.txt"

# Each generated API key is saved here before it is registered on Lighter,
# and deleted once every record is written. Kept if the deployment fails.
export LIGHTER_KEY_BACKUP_DIR="$RUN_DIR/key-backup"

# The master's API key: perform-test-trade and lighter-move-funds sign with
# it, and check-accounts and correct-accounts read its transfer history.
export LIGHTER_OPERATOR_RECORD_FILE="$RUN_DIR/vault-record.json"

# The sub-accounts' keys: a transfer sent from a sub-account (step 7) is
# signed with that sub-account's own key, and the accounting commands read
# the sub-accounts' transfer history with them. The files exist from step 1.
export LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES="$RUN_DIR/vault-record.nautilus.trend.json,$RUN_DIR/vault-record.nautilus.arb.json"

# Longest wait for a Lighter withdrawal to become claimable: one hour.
# perform-test-trade reads the first name, lighter-move-funds the second.
export LIGHTER_WITHDRAWAL_TIMEOUT=3600
export LIGHTER_WITHDRAW_TIMEOUT=3600

# Longest wait for a transfer between Lighter accounts to appear in history.
export LIGHTER_TRANSFER_TIMEOUT=600

# lagoon-deploy-vault with a strategy file connects to every chain that has a
# JSON_RPC_* variable and refuses when the strategy does not use them all. A
# shell set up for tests or other vaults exports many; keep only Ethereum.
for name in $(compgen -e | grep '^JSON_RPC_' | grep -vx 'JSON_RPC_ETHEREUM'); do
    unset "$name"
done

# lighter-move-funds acts on the account this selects; the withdrawal steps
# pass --lighter-account-index explicitly, so a value inherited from the
# shell must not redirect the master's withdrawal.
unset LIGHTER_ACCOUNT_INDEX

# Progress interval while a command runs.
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-60}"

# The testing module's command, used by the helpers below; every step
# spells its commands out in full.
LIGHTER_MANUAL=(poetry run python -m tradeexecutor.testing.lighter_manual)

# Amounts of the custody steps, in whole USDC.
FUNDING_USDC=2
TRANSFER_USDC=1
# Moved back from arb to trend by arb's trader in step 9, outside the executor.
TRADER_TRANSFER_USDC=1
# Sent by the deployer straight to the Safe in step 10, outside the vault.
DONATION_USDC=1

# Expected balances after each step, as safe, master, trend and arb, and
# the vault's NAV. The 1 USDC activation deposit stays on the master until
# step 13. They assume no fee for a transfer between accounts of one owner,
# as Lighter charged none on 2026-10-26 and 2026-10-28; with a fee, the
# balance checks stop the run after the transfer and show the difference.
SAFE_AFTER_SUBSCRIPTION=$SUBSCRIPTION_USDC
SAFE_AFTER_TREND=$((SUBSCRIPTION_USDC - FUNDING_USDC))
SAFE_AFTER_ARB=$((SUBSCRIPTION_USDC - 2 * FUNDING_USDC))
SAFE_AFTER_DONATION=$((SAFE_AFTER_ARB + DONATION_USDC))
TREND_AFTER_TRANSFER=$((FUNDING_USDC - TRANSFER_USDC))
ARB_AFTER_TRANSFER=$((FUNDING_USDC + TRANSFER_USDC))
TREND_AFTER_TRADER=$((TREND_AFTER_TRANSFER + TRADER_TRANSFER_USDC))
ARB_AFTER_TRADER=$((ARB_AFTER_TRANSFER - TRADER_TRANSFER_USDC))
NAV_INVESTED=$((1 + SUBSCRIPTION_USDC))
NAV_AFTER_DONATION=$((NAV_INVESTED + DONATION_USDC))

# ---------------------------------------------------------------------------
# Observability: one log, timestamps, step banners, echoed and timed
# commands, a heartbeat while they run, and a failure report
# ---------------------------------------------------------------------------

RUN_LOG="$RUN_DIR/run.log"

# Everything printed from here on goes to the terminal and to the run log.
exec > >(tee -a "$RUN_LOG") 2>&1

CURRENT_STEP="setup"
STEP_STARTED=$SECONDS
LAST_COMMAND=""
FAILURE_REPORTED=0

# Print a timestamped line. Script messages go to standard error, so a
# command's captured result on standard output stays clean.
log() {
    echo "[$(date -u '+%H:%M:%S') UTC] $*" >&2
}

# Format seconds as "Xm YYs".
format_duration() {
    printf '%dm %02ds' $(($1 / 60)) $(($1 % 60))
}

# Short name of a command for the heartbeat: "trade-executor lagoon-settle".
describe() {
    local args=("$@")
    if [ "${args[0]}" = "poetry" ]; then args=("${args[@]:2}"); fi
    if [ "${args[0]}" = "python" ]; then args=("${args[@]:2}"); fi
    echo "${args[0]##*.} ${args[1]:-}"
}

# Print "still running" every HEARTBEAT_SECONDS until killed.
heartbeat() {
    local what="$1" started=$SECONDS
    while sleep "$HEARTBEAT_SECONDS"; do
        log "… still running $what ($(format_duration $((SECONDS - started))) elapsed)"
    done
}

# Run one command: echo it, show a heartbeat while it runs, report how it
# ended and how long it took. "--stdin TEXT" answers the command's prompts;
# without it the command gets no input, so a prompt fails instead of hanging.
# "--probe" marks a balance check whose mismatch is an expected answer, not
# an error, so it is reported as "no match" rather than as a failure.
run() {
    local input="" probe=0
    if [ "$1" = "--probe" ]; then
        probe=1
        shift
    fi
    if [ "$1" = "--stdin" ]; then
        input="$2"
        shift 2
    fi
    LAST_COMMAND="$(printf '%q ' "$@")"
    if [ -n "$input" ]; then
        log "→ $LAST_COMMAND  (answering its prompts with: $(printf '%b' "$input" | paste -sd ',' -))"
    else
        log "→ $LAST_COMMAND"
    fi
    local started=$SECONDS status=0
    # Standard output to /dev/null: inside plan="$(run ...)" the heartbeat's
    # sleep would otherwise hold the capture open for up to a heartbeat.
    heartbeat "$(describe "$@")" </dev/null >/dev/null &
    local heartbeat_pid=$!
    if [ -n "$input" ]; then
        printf '%b' "$input" | "$@" || status=$?
    else
        "$@" </dev/null || status=$?
    fi
    kill "$heartbeat_pid" 2>/dev/null || true
    wait "$heartbeat_pid" 2>/dev/null || true
    if [ "$status" -eq 0 ]; then
        log "✓ finished in $(format_duration $((SECONDS - started)))"
    elif [ "$probe" -eq 1 ]; then
        log "· no match"
    else
        log "✗ failed with exit code $status after $(format_duration $((SECONDS - started)))"
    fi
    return "$status"
}

# Start a step: print its banner, or skip it when an earlier run completed it.
step() {
    local number="$1" title="$2" duration="${3:-}"
    CURRENT_STEP="$number"
    STEP_STARTED=$SECONDS
    if [ -e "$RUN_DIR/step-$number.done" ]; then
        log "Step $number/12 already done in an earlier run, skipping: $title"
        return 1
    fi
    echo >&2
    log "════ Step $number/12: $title"
    if [ -n "$duration" ]; then
        log "     Typical duration: $duration"
    fi
    return 0
}

# Mark a step complete.
done_step() {
    touch "$RUN_DIR/step-$1.done"
    log "════ Step $1/12 done in $(format_duration $((SECONDS - STEP_STARTED)))"
}

# Write env.sh: this run's public configuration, so an operator can run its
# commands by hand. Secrets stay in the operator's shell: the file refers to
# JSON_RPC_ETHEREUM, LIGHTER_TEST_PRIVATE_KEY and TRADING_STRATEGY_API_KEY
# without storing them.
ENV_FILE="$RUN_DIR/env.sh"
write_env_file() {
    {
        echo "# Configuration of the run in $RUN_DIR. From the repository root: source $ENV_FILE"
        echo "# Needs JSON_RPC_ETHEREUM, LIGHTER_TEST_PRIVATE_KEY and TRADING_STRATEGY_API_KEY exported in your shell."
        echo 'export PRIVATE_KEY="$LIGHTER_TEST_PRIVATE_KEY"'
        echo '# As the script does: only the Ethereum RPC, and the master account from the records.'
        echo 'unset $(env | sed -n "s/^\(JSON_RPC_[A-Za-z0-9_]*\)=.*/\1/p" | grep -vx JSON_RPC_ETHEREUM) 2>/dev/null'
        echo 'unset LIGHTER_ACCOUNT_INDEX'
        local name
        for name in EXECUTOR_ID STRATEGY_FILE STATE_FILE SIMULATE LOG_LEVEL CACHE_PATH MIN_GAS_BALANCE VAULT_RECORD_FILE \
            LIGHTER_KEY_BACKUP_DIR LIGHTER_OPERATOR_RECORD_FILE LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES \
            LIGHTER_WITHDRAWAL_TIMEOUT LIGHTER_WITHDRAW_TIMEOUT LIGHTER_TRANSFER_TIMEOUT \
            ASSET_MANAGEMENT_MODE VAULT_ADDRESS VAULT_ADAPTER_ADDRESS; do
            if [ -n "${!name:-}" ]; then printf 'export %s=%q\n' "$name" "${!name}"; fi
        done
    } > "$ENV_FILE"
}

# What to do after a failure in each step.
recovery_hint() {
    case "$CURRENT_STEP" in
        1) echo "A rerun does not deploy again while $RUN_DIR/deploy-started exists without a deployment artefact. Generated keys, if any, are in $LIGHTER_KEY_BACKUP_DIR. Inspect $RUN_LOG and the Safe first." ;;
        5|6|7) echo "If check-accounts reports an unbooked deposit or transfer: source $ENV_FILE && printf 'y\\n' | poetry run trade-executor correct-accounts, then rerun." ;;
        8) echo "Nothing moved in this step; rerun. A failed NAV check means lagoon-settle posted a NAV that differs from the balances or the books: inspect with trade-executor show-valuation and check-accounts." ;;
        9) echo "If $RUN_DIR/trader-transfer records only 'requested', the transfer may or may not have been sent: check arb's transfer history on Lighter, then delete the marker only if nothing was sent. Otherwise rerun; it does not send again." ;;
        10) echo "If the donation transaction failed, delete $RUN_DIR/donation and rerun; if it succeeded, rerun and it does not send again." ;;
        11|12|13) echo "A withdrawal with a recorded request is resumed by rerunning. One without is reported as unconfirmed; see README-Lighter.md, Recovery." ;;
        14) echo "lagoon-redeem resumes a pending or claimable redemption when rerun." ;;
        *) echo "Fix the cause shown above, then rerun." ;;
    esac
}

# Print the failure report once.
report_failure() {
    if [ "$FAILURE_REPORTED" -eq 1 ]; then return; fi
    FAILURE_REPORTED=1
    echo >&2
    log "✗✗ Run stopped in step $CURRENT_STEP after $(format_duration $((SECONDS - STEP_STARTED))) in that step"
    log "   Last command: ${LAST_COMMAND:-none}"
    log "   What to do: $(recovery_hint)"
    log "   Full log: $RUN_LOG"
    log "   Resume with: LIGHTER_RUN_DIR=$RUN_DIR $0"
}

# Stop with a message and the failure report.
fail() {
    log "✗ $*"
    report_failure
    exit 1
}

# A failure inside a captured command, such as plan="$(run ...)", also
# fails the assignment in the main shell; report once, from the main shell.
trap 'status=$?; if [ "$BASH_SUBSHELL" -eq 0 ]; then report_failure; fi; exit $status' ERR
trap 'log "Interrupted"; report_failure; exit 130' INT TERM

# Show the deployer's, the Safe's and every Lighter account's balances.
show_balances() {
    run "${LIGHTER_MANUAL[@]}" balances
}

# Resume guard of a money-moving step (steps 5 to 7). Sets GUARD_RESULT to
# "run" when the step should move its money, or "done" when an earlier run
# already moved it. Stops the script in any other state.
#
#   guard_money_step N "<start balances>" "<target balances>" ["<halfway balances>" "<completion options>"]
guard_money_step() {
    local number="$1" start="$2" target="$3" halfway="${4:-}" completion="${5:-}"
    log "Resume guard for step $number: the books must match the chain, then the balances must be the step's start or its target"
    if ! run poetry run trade-executor check-accounts; then
        # perform-test-trade saves the state only when it finishes, so an
        # interrupted step leaves moved money unbooked. Say which state it is in.
        # shellcheck disable=SC2086
        if run --probe "${LIGHTER_MANUAL[@]}" expect-balances --quiet $target >/dev/null; then
            fail "Step $number moved its money in an earlier run but did not save it. Book it: source $ENV_FILE && printf 'y\\n' | poetry run trade-executor correct-accounts, then rerun this script."
        fi
        # shellcheck disable=SC2086
        if [ -n "$halfway" ] && run --probe "${LIGHTER_MANUAL[@]}" expect-balances --quiet $halfway >/dev/null; then
            fail "Step $number stopped halfway in an earlier run: the deposit reached the master but the transfer to the sub-account did not happen. Book the deposit: source $ENV_FILE && printf 'y\\n' | poetry run trade-executor correct-accounts, then rerun this script; it then gives the command that completes the transfer."
        fi
        fail "The books do not match the chain before step $number; see the check-accounts output above."
    fi
    # shellcheck disable=SC2086  # The expectations are separate words on purpose.
    if run --probe "${LIGHTER_MANUAL[@]}" expect-balances --quiet $target >/dev/null; then
        log "The target balances already hold: an earlier run moved this step's money. Nothing is moved again."
        done_step "$number"
        GUARD_RESULT="done"
        return
    fi
    # shellcheck disable=SC2086
    if run --probe "${LIGHTER_MANUAL[@]}" expect-balances --quiet $start >/dev/null; then
        log "The start balances hold; running step $number."
        GUARD_RESULT="run"
        return
    fi
    # shellcheck disable=SC2086
    if [ -n "$halfway" ] && run --probe "${LIGHTER_MANUAL[@]}" expect-balances --quiet $halfway >/dev/null; then
        fail "Step $number stopped halfway in an earlier run: the deposit reached the master and is booked, but the transfer to the sub-account did not happen. Complete it: source $ENV_FILE && poetry run trade-executor perform-test-trade $completion, then rerun this script."
    fi
    # shellcheck disable=SC2086
    run --probe "${LIGHTER_MANUAL[@]}" expect-balances $start >/dev/null || true
    fail "The balances match neither the start nor the target of step $number; see the table above and inspect with trade-executor check-accounts."
}

# Checks after a money-moving step: the books match the chain, the
# balances are the step's target, and the step is marked done.
finish_money_step() {
    local number="$1" target="$2"
    run poetry run trade-executor check-accounts
    # shellcheck disable=SC2086
    run "${LIGHTER_MANUAL[@]}" expect-balances $target >/dev/null
    show_balances
    done_step "$number"
}

# Withdraw one Lighter account to the Safe (steps 11 to 13). The decision
# comes from withdrawal-plan: resume a withdrawal an earlier run submitted,
# withdraw the account's free USDC, or skip an account below Lighter's
# 1 USDC withdrawal minimum. A withdrawal an earlier run started without
# recording its request stops the script: only Lighter's or Ethereum's
# history can tell whether it was sent, and sending another could withdraw twice.
#
#   withdraw_account N NAME INDEX [sub-account]
withdraw_account() {
    local number="$1" name="$2" index="$3" kind="${4:-master}" plan amount
    local account=()
    if [ "$kind" = "sub-account" ]; then
        # Without the option lighter-move-funds acts on the master.
        account=(--lighter-account-index "$index")
    fi
    LAST_COMMAND="lighter_manual withdrawal-plan $index"
    plan="$(run "${LIGHTER_MANUAL[@]}" withdrawal-plan "$index")"
    log "Withdrawal plan for $name (account $index): $plan"
    case "$plan" in
        resume)
            log "Resuming the withdrawal an earlier run submitted; lighter-move-funds waits until it is claimable and claims it into the Safe."
            run poetry run trade-executor lighter-move-funds "${account[@]}"
            ;;
        withdraw\ *)
            amount="${plan#withdraw }"
            log "Withdrawing $amount USDC from $name to the Safe at $(date -u '+%H:%M UTC'); Lighter makes it claimable after its withdrawal delay."
            # lighter-move-funds asks for the direction (w), the amount and a confirmation (y).
            run --stdin "w\n$amount\ny\n" poetry run trade-executor lighter-move-funds "${account[@]}"
            ;;
        skip)
            log "$name holds less than Lighter's 1 USDC withdrawal minimum; nothing to withdraw. lagoon-redeem leaves it out of the final NAV."
            ;;
        unconfirmed\ *)
            local history="the master's withdrawal history on Lighter"
            if [ "$kind" = "sub-account" ]; then
                history="the Safe's transactions on Etherscan for an L1 withdraw from account $index"
            fi
            fail "lighter-move-funds started withdrawal trade ${plan#unconfirmed } for $name in an earlier run but did not record its request. Check $history, then follow README-Lighter.md, Recovery, before rerunning. Do not withdraw again until you know whether it was sent."
            ;;
        *)
            fail "Unexpected withdrawal plan for $name: $plan"
            ;;
    esac
    run poetry run trade-executor check-accounts
    run "${LIGHTER_MANUAL[@]}" expect-balances "$name<=0.99" >/dev/null
    show_balances
    done_step "$number"
}

# check-accounts must fail now: the chain changed outside the executor
# (steps 9 and 10). Passing would mean it cannot see such a change.
expect_unclean_books() {
    local what="$1"
    log "check-accounts must now report $what; a failure below is expected"
    if run --probe poetry run trade-executor check-accounts; then
        fail "check-accounts did not report $what: its books match a chain that changed outside the executor."
    fi
    log "check-accounts reported $what, as it should"
}

# correct-accounts in its two modes. --dry-run must book nothing; the real
# run books whatever differs, answering y to its confirmation, and leaves
# books that check-accounts accepts. With "clean", the books matched
# before, so the real run must book nothing too.
#
#   check_correct_accounts [clean]
check_correct_accounts() {
    local expectation="${1:-}" before after
    LAST_COMMAND="lighter_manual books"
    before="$(run "${LIGHTER_MANUAL[@]}" books)"
    run poetry run trade-executor correct-accounts --dry-run
    LAST_COMMAND="lighter_manual books"
    after="$(run "${LIGHTER_MANUAL[@]}" books)"
    [ "$before" = "$after" ] || fail "correct-accounts --dry-run changed the books: $before before, $after after."
    run --stdin "y\n" poetry run trade-executor correct-accounts
    LAST_COMMAND="lighter_manual books"
    after="$(run "${LIGHTER_MANUAL[@]}" books)"
    if [ "$expectation" = "clean" ] && [ "$before" != "$after" ]; then
        fail "correct-accounts booked something although the books matched: $before before, $after after. Inspect the new balance updates with trade-executor show-positions."
    fi
    log "Books before correct-accounts: $before, after: $after"
    run poetry run trade-executor check-accounts
}

# Post the NAV with lagoon-settle while no deposit or redemption waits, then
# check it equals the expected amount, the live balances and the books.
#
#   settle_and_check_nav NAV
settle_and_check_nav() {
    local nav="$1"
    run poetry run trade-executor lagoon-settle
    run "${LIGHTER_MANUAL[@]}" expect-nav --nav "$nav" >/dev/null
    run poetry run trade-executor show-valuation
}

write_env_file
log "Lighter sub-account manual test, run directory $RUN_DIR"
log "Log of this run: $RUN_LOG"
log "Repository commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), subscription $SUBSCRIPTION_USDC USDC"

# ---------------------------------------------------------------------------
# Step 0: check the deployer can pay for the run
# ---------------------------------------------------------------------------
#
# The deployer pays 1 USDC for Lighter's activation deposit, subscribes
# SUBSCRIPTION_USDC as test capital and sends DONATION_USDC straight to the
# Safe in step 10, plus gas. check-funds also records the starting
# balances in start.env; step 15 reconciles against them.

if step 0 "Check the deployer's ETH and USDC"; then
    run poetry run python -m tradeexecutor.testing.lighter_manual check-funds --usdc "$((1 + SUBSCRIPTION_USDC + DONATION_USDC))" --start-file "$RUN_DIR/start.env"
    done_step 0
fi

# start.env holds START_USDC, START_ETH_WEI and DEPLOYER_ADDRESS.
# shellcheck source=/dev/null
source "$RUN_DIR/start.env"
log "Deployer: https://etherscan.io/address/$DEPLOYER_ADDRESS"

# ---------------------------------------------------------------------------
# Step 1: deploy the vault with a Lighter master and two keyed sub-accounts
# ---------------------------------------------------------------------------
#
# lagoon-deploy-vault deploys a Safe, a Lagoon vault and its guard module,
# then activates a Lighter account owned by the Safe: it subscribes 1 USDC
# to the vault, deposits it from the Safe to Lighter (the minimum deposit,
# which creates the master account) and registers an API key on the master
# with a Safe changePubKey transaction. It then creates one sub-account per
# SubAccount the strategy declares, trend and arb, and registers a key on
# each the same way. The deployer is the Safe's sole owner, so it signs the
# Safe transactions directly.
#
# Records written: the public report vault-record.txt, the master key in
# vault-record.json, each sub-account key in vault-record.nautilus.<id>.json,
# and the public artefact state/lighter-cli.deployment.json.
#
# The command asks once to confirm the deployment; the script answers y.

# Written last by a successful deployment, so its presence means step 1 ran.
DEPLOYMENT_FILE="$RUN_DIR/state/$EXECUTOR_ID.deployment.json"

if step 1 "Deploy the Lagoon vault, the Lighter master account and the sub-accounts trend and arb" "10 to 20 minutes: contract deployment and verification, then Lighter activation"; then
    if [ -s "$DEPLOYMENT_FILE" ]; then
        log "$DEPLOYMENT_FILE exists: an earlier run deployed the vault before it was interrupted. Not deploying again."
    elif [ -e "$RUN_DIR/deploy-started" ]; then
        # The deployment spends gas and the 1 USDC activation deposit before it
        # generates any key, so only this marker shows an attempt was made.
        fail "A deployment started at $(cat "$RUN_DIR/deploy-started") did not complete. Find its Safe and vault in $RUN_LOG; generated keys, if any, are in $LIGHTER_KEY_BACKUP_DIR. Recover or abandon that deployment, then start a new run in a new directory."
    else
        date -u '+%Y-%m-%d %H:%M:%S UTC' > "$RUN_DIR/deploy-started"
        run --stdin "y\n" poetry run trade-executor lagoon-deploy-vault \
            --strategy-file "$STRATEGY_FILE" \
            --generate-lighter-api-key \
            --lighter-api-key-index 4 \
            --lighter-sub-account-api-key-index 4 \
            --lighter-key-backup-dir "$LIGHTER_KEY_BACKUP_DIR" \
            --vault-record-file "$VAULT_RECORD_FILE" \
            --fund-name "Lighter CLI test" \
            --fund-symbol "LCLI" \
            --multisig-owners "$DEPLOYER_ADDRESS" \
            --any-asset \
            --performance-fee 0 \
            --management-fee 0
    fi
    done_step 1
fi

# ---------------------------------------------------------------------------
# Step 2: read the deployment
# ---------------------------------------------------------------------------
#
# The public artefact holds the vault, Safe and guard module addresses and
# each Lighter account's index. Every later command needs the vault and the
# module; the unwind needs the account indices. Saved to deployment.env.

if step 2 "Read the deployed addresses and Lighter account indices"; then
    jq -r '.deployments.ethereum as $d | [
        "VAULT_ADDRESS=\($d.vault_address)",
        "VAULT_ADAPTER_ADDRESS=\($d.module_address)",
        "SAFE_ADDRESS=\($d.safe_address)",
        "LIGHTER_MASTER_INDEX=\($d.lighter_account_setup.account_index)",
        ($d.lighter_account_setup.sub_accounts[] | "LIGHTER_\(.label | ascii_upcase)_INDEX=\(.account_index)")
    ] | .[]' "$DEPLOYMENT_FILE" > "$RUN_DIR/deployment.env"
    done_step 2
fi

# shellcheck source=/dev/null
source "$RUN_DIR/deployment.env"
export VAULT_ADDRESS VAULT_ADAPTER_ADDRESS
# From here on every command acts on the deployed Lagoon vault.
export ASSET_MANAGEMENT_MODE=lagoon
write_env_file
log "Commands of this run can be run by hand after: source $ENV_FILE"

log "Vault:  https://etherscan.io/address/$VAULT_ADDRESS"
log "Safe:   https://etherscan.io/address/$SAFE_ADDRESS"
log "Guard:  https://etherscan.io/address/$VAULT_ADAPTER_ADDRESS"
log "Lighter master $LIGHTER_MASTER_INDEX: https://mainnet.zklighter.elliot.ai/api/v1/account?by=index&value=$LIGHTER_MASTER_INDEX"
log "Lighter trend  $LIGHTER_TREND_INDEX: https://mainnet.zklighter.elliot.ai/api/v1/account?by=index&value=$LIGHTER_TREND_INDEX"
log "Lighter arb    $LIGHTER_ARB_INDEX: https://mainnet.zklighter.elliot.ai/api/v1/account?by=index&value=$LIGHTER_ARB_INDEX"

# ---------------------------------------------------------------------------
# Step 3: initialise executor state and book the activation
# ---------------------------------------------------------------------------
#
# init creates the state from the deployment artefact, recording which
# Lighter account index each id (master, trend, arb) maps to.
# correct-accounts --process-redemption books what deployment already did:
# the 1 USDC investor subscription and the 1 USDC now on the master. It asks
# before applying the corrections; the script answers y. check-accounts then
# confirms the books match the chain, and check-wallet that the asset
# manager has gas and that no deposit or redemption is left unclaimed.

if step 3 "Initialise executor state and book the Lighter activation deposit"; then
    if [ -s "$STATE_FILE" ]; then
        log "The state file exists from an earlier run; not initialising it again"
    else
        run poetry run trade-executor init
    fi
    run --stdin "y\n" poetry run trade-executor correct-accounts --process-redemption
    run poetry run trade-executor check-accounts
    run poetry run trade-executor check-wallet
    run poetry run python -m tradeexecutor.testing.lighter_manual expect-balances safe=0 master=1 trend=0 arb=0 >/dev/null
    show_balances
    done_step 3
fi

# ---------------------------------------------------------------------------
# Step 4: subscribe test capital and claim the shares
# ---------------------------------------------------------------------------
#
# A Lagoon subscription has three parts. The investor requests a deposit,
# which waits in the vault's silo; the asset manager settles the queue,
# posting NAV and moving the USDC into the Safe; the investor claims the
# shares. lagoon-settle is the asset manager's part. request-deposit and
# claim-shares are the investor's, outside the executor, so they are done
# by the testing module. All three are safe to rerun.

if step 4 "Subscribe $SUBSCRIPTION_USDC USDC to the vault and claim the shares" "about 5 minutes: five Ethereum transactions"; then
    LAST_COMMAND="lighter_manual request-deposit"
    deposit_status="$(run poetry run python -m tradeexecutor.testing.lighter_manual request-deposit "$SUBSCRIPTION_USDC" --marker "$RUN_DIR/deposit-requested")"
    log "Subscription status: $deposit_status"
    if [ "$deposit_status" = "requested" ] || [ "$deposit_status" = "pending" ]; then
        run poetry run trade-executor lagoon-settle
    fi
    run poetry run python -m tradeexecutor.testing.lighter_manual claim-shares >/dev/null
    run poetry run trade-executor check-accounts
    run poetry run python -m tradeexecutor.testing.lighter_manual expect-balances "safe=$SAFE_AFTER_SUBSCRIPTION" master=1 trend=0 arb=0 >/dev/null
    show_balances
    done_step 4
fi

# ---------------------------------------------------------------------------
# Step 5: fund sub-account trend with perform-test-trade
# ---------------------------------------------------------------------------
#
# --pair selects the trend sub-account's trading pair. --buy-only funds it
# and leaves the money there: a deposit from the Safe to the master, then a
# transfer from the master to trend, signed with the master's key. Each
# movement is the custody trade `start` would make, booked in the state.

if step 5 "Fund sub-account trend with $FUNDING_USDC USDC" "a few minutes: one Lighter deposit and one transfer"; then
    guard_money_step 5 \
        "safe=$SAFE_AFTER_SUBSCRIPTION master=1 trend=0 arb=0" \
        "safe=$SAFE_AFTER_TREND master=1 trend=$FUNDING_USDC arb=0" \
        "safe=$SAFE_AFTER_TREND master=$((1 + FUNDING_USDC)) trend=0 arb=0" \
        "--lighter-transfer master:trend --buy-only --amount $FUNDING_USDC"
    if [ "$GUARD_RESULT" = "run" ]; then
        run poetry run trade-executor perform-test-trade \
            --pair "(ethereum, lighter, LIGHTER-trend, USDC)" \
            --buy-only \
            --amount "$FUNDING_USDC"
        finish_money_step 5 "safe=$SAFE_AFTER_TREND master=1 trend=$FUNDING_USDC arb=0"
    fi
fi

# ---------------------------------------------------------------------------
# Step 6: fund sub-account arb with perform-test-trade
# ---------------------------------------------------------------------------

if step 6 "Fund sub-account arb with $FUNDING_USDC USDC" "a few minutes: one Lighter deposit and one transfer"; then
    guard_money_step 6 \
        "safe=$SAFE_AFTER_TREND master=1 trend=$FUNDING_USDC arb=0" \
        "safe=$SAFE_AFTER_ARB master=1 trend=$FUNDING_USDC arb=$FUNDING_USDC" \
        "safe=$SAFE_AFTER_ARB master=$((1 + FUNDING_USDC)) trend=$FUNDING_USDC arb=0" \
        "--lighter-transfer master:arb --buy-only --amount $FUNDING_USDC"
    if [ "$GUARD_RESULT" = "run" ]; then
        run poetry run trade-executor perform-test-trade \
            --pair "(ethereum, lighter, LIGHTER-arb, USDC)" \
            --buy-only \
            --amount "$FUNDING_USDC"
        finish_money_step 6 "safe=$SAFE_AFTER_ARB master=1 trend=$FUNDING_USDC arb=$FUNDING_USDC"
    fi
fi

# ---------------------------------------------------------------------------
# Step 7: move USDC from trend to arb with perform-test-trade
# ---------------------------------------------------------------------------
#
# --lighter-transfer trend:arb sends one transfer between the sub-accounts,
# signed with trend's own key from LIGHTER_SUB_ACCOUNT_OPERATOR_RECORD_FILES.
# It is booked as a trade on arb's position, with the amount (and any
# Lighter fee) taken from trend's position as balance updates.

if step 7 "Move $TRANSFER_USDC USDC from sub-account trend to arb" "a minute: one Lighter transfer"; then
    guard_money_step 7 \
        "safe=$SAFE_AFTER_ARB master=1 trend=$FUNDING_USDC arb=$FUNDING_USDC" \
        "safe=$SAFE_AFTER_ARB master=1 trend=$((FUNDING_USDC - TRANSFER_USDC)) arb=$((FUNDING_USDC + TRANSFER_USDC))"
    if [ "$GUARD_RESULT" = "run" ]; then
        run poetry run trade-executor perform-test-trade \
            --lighter-transfer trend:arb \
            --buy-only \
            --amount "$TRANSFER_USDC"
        finish_money_step 7 "safe=$SAFE_AFTER_ARB master=1 trend=$((FUNDING_USDC - TRANSFER_USDC)) arb=$((FUNDING_USDC + TRANSFER_USDC))"
    fi
fi

# ---------------------------------------------------------------------------
# Step 8: audit clean books with capital on every account
# ---------------------------------------------------------------------------
#
# With USDC on the Safe, the master and both sub-accounts, the books must
# need no correction: correct-accounts, in dry-run mode and for real, books
# nothing. lagoon-settle then posts the NAV with no deposit or redemption
# waiting, and expect-nav checks the vault's NAV equals the Safe's USDC plus
# every Lighter account's equity and the state's total equity. check-wallet
# checks the asset manager's gas and the vault's queues. Only gas is spent.

if step 8 "Audit the books, the NAV and the wallet with capital on every account" "a few minutes: one NAV settlement"; then
    run poetry run trade-executor check-accounts
    run poetry run trade-executor check-wallet
    check_correct_accounts clean
    settle_and_check_nav "$NAV_INVESTED"
    run poetry run trade-executor show-positions
    show_balances
    done_step 8
fi

# ---------------------------------------------------------------------------
# Step 9: a trader moves USDC between sub-accounts outside the executor
# ---------------------------------------------------------------------------
#
# A sub-account's trader holds its API key and may move USDC to another
# account of the Safe at any time; the executor books nothing. trader-transfer
# signs such a transfer with arb's own key, from its hand-over record.
# check-accounts must then report both sub-accounts differing from the books,
# and correct-accounts must accept Lighter's equity as balance updates. The
# NAV does not change: the USDC only moved between the vault's accounts.
# A marker file stops a rerun from sending the transfer again.

if step 9 "Move $TRADER_TRANSFER_USDC USDC from arb to trend with arb's own key, outside the executor" "a minute: one Lighter transfer and corrections"; then
    if [ ! -e "$RUN_DIR/trader-transfer" ]; then
        run poetry run trade-executor check-accounts
        run "${LIGHTER_MANUAL[@]}" expect-balances "safe=$SAFE_AFTER_ARB" master=1 "trend=$TREND_AFTER_TRANSFER" "arb=$ARB_AFTER_TRANSFER" >/dev/null
    fi
    LAST_COMMAND="lighter_manual trader-transfer"
    sent="$(run poetry run python -m tradeexecutor.testing.lighter_manual trader-transfer arb trend "$TRADER_TRANSFER_USDC" --marker "$RUN_DIR/trader-transfer" --timeout "$LIGHTER_TRANSFER_TIMEOUT")"
    run poetry run python -m tradeexecutor.testing.lighter_manual expect-balances "safe=$SAFE_AFTER_ARB" master=1 "trend=$TREND_AFTER_TRADER" "arb=$ARB_AFTER_TRADER" >/dev/null
    if [ "$sent" = "sent" ]; then
        expect_unclean_books "the transfer arb's trader made"
    else
        log "An earlier run sent the transfer; its corrections may already be booked"
    fi
    check_correct_accounts
    settle_and_check_nav "$NAV_INVESTED"
    show_balances
    done_step 9
fi

# ---------------------------------------------------------------------------
# Step 10: USDC reaches the Safe outside the vault
# ---------------------------------------------------------------------------
#
# The deployer sends USDC straight to the Safe, as an airdrop or a mistaken
# transfer would, without a deposit request. No investor subscribed it, so
# it is income the executor has not booked: check-accounts must report the
# reserve differing from the books, correct-accounts books it, and the next
# NAV settlement raises the share price. The deployer gets it back with its
# shares in step 14. A marker file stops a rerun from sending again.

if step 10 "Send $DONATION_USDC USDC straight to the Safe, outside the vault" "a few minutes: one Ethereum transfer, corrections and a NAV settlement"; then
    if [ ! -e "$RUN_DIR/donation" ]; then
        run poetry run trade-executor check-accounts
        run "${LIGHTER_MANUAL[@]}" expect-balances "safe=$SAFE_AFTER_ARB" >/dev/null
    fi
    LAST_COMMAND="lighter_manual donate"
    sent="$(run poetry run python -m tradeexecutor.testing.lighter_manual donate "$DONATION_USDC" --marker "$RUN_DIR/donation")"
    run poetry run python -m tradeexecutor.testing.lighter_manual expect-balances "safe=$SAFE_AFTER_DONATION" master=1 "trend=$TREND_AFTER_TRADER" "arb=$ARB_AFTER_TRADER" >/dev/null
    if [ "$sent" = "sent" ]; then
        expect_unclean_books "USDC the vault did not book"
    else
        log "An earlier run sent the USDC; its correction may already be booked"
    fi
    check_correct_accounts
    settle_and_check_nav "$NAV_AFTER_DONATION"
    show_balances
    done_step 10
fi

# ---------------------------------------------------------------------------
# Steps 11 to 13: withdraw every Lighter account to the Safe
# ---------------------------------------------------------------------------
#
# lighter-move-funds withdraws one account's free USDC to the Safe and
# books it. A sub-account (LIGHTER_ACCOUNT_INDEX set) uses Lighter's
# key-free L1 withdraw(subIndex) through the Safe; the master uses the
# secure withdrawal signed with its key. Both wait until Lighter makes the
# USDC claimable, then claim it into the Safe. The command asks for the
# direction, the amount and a confirmation; the script answers with w, the
# account's free USDC and y.

if step 11 "Withdraw sub-account arb to the Safe" "25 to 35 minutes: Lighter L1 withdrawal delay"; then
    withdraw_account 11 arb "$LIGHTER_ARB_INDEX" sub-account
fi

if step 12 "Withdraw sub-account trend to the Safe" "25 to 35 minutes: Lighter L1 withdrawal delay"; then
    withdraw_account 12 trend "$LIGHTER_TREND_INDEX" sub-account
fi

if step 13 "Withdraw the master to the Safe" "15 to 25 minutes: Lighter secure withdrawal delay"; then
    withdraw_account 13 master "$LIGHTER_MASTER_INDEX"
fi

# ---------------------------------------------------------------------------
# Step 14: redeem every share back to the deployer
# ---------------------------------------------------------------------------
#
# lagoon-redeem requests redemption of all the deployer's shares, posts NAV,
# settles the redemption queue and claims the USDC. It finds every Lighter
# account from the state. Lagoon pays a redemption only from the Safe, so
# lagoon-redeem leaves Lighter balances that total less than 5 USDC, such as
# dust below the withdrawal minimum, out of the NAV it posts. check-wallet
# then confirms no deposit or redemption is left unclaimed.

if step 14 "Redeem all shares back to the deployer" "a few minutes"; then
    LAST_COMMAND="lighter_manual redeem-status"
    redeem_status="$(run poetry run python -m tradeexecutor.testing.lighter_manual redeem-status)"
    if [ "$redeem_status" = "redeem" ]; then
        run poetry run trade-executor lagoon-redeem
    else
        log "The deployer has no shares or redemption left; nothing to redeem"
    fi
    run poetry run trade-executor check-accounts
    run poetry run trade-executor check-wallet
    show_balances
    done_step 14
fi

# ---------------------------------------------------------------------------
# Step 15: sweep the Safe and reconcile
# ---------------------------------------------------------------------------
#
# Once no shares are outstanding, whatever the Safe still holds belongs to
# nobody but its owner: sweep-safe moves it to the deployer. reconcile then
# explains the deployer's USDC change since step 0 by Lighter fees, including
# the fee of the trader's transfer in step 9, and dust, and fails on
# anything else.

if step 15 "Sweep the Safe and reconcile the deployer's balances"; then
    run poetry run python -m tradeexecutor.testing.lighter_manual sweep-safe >/dev/null
    run poetry run python -m tradeexecutor.testing.lighter_manual reconcile --start-file "$RUN_DIR/start.env" --trader-transfer-marker "$RUN_DIR/trader-transfer" >/dev/null
    show_balances
    done_step 15
fi

log "Manual test complete. Artefacts, records and the log are in $RUN_DIR"
