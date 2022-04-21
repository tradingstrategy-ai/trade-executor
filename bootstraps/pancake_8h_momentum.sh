#!/bin/bash
#
# Pancake 8h momentum trading launch script.
#
# This script combines a strategy specific trade executor environt variables
# with secret environment variables before running the trade-executor command line entry point.
#
# Check that `~/secrets.env` and `~/$EXECUTOR_ID.secrets.env` are set up.
#
# Then run the checks:
#
#     bootstraps/pancake_8h_momentum.sh check-universe --max-data-delay-minutes=1440
#
# Then start:
#
#     bootstraps/pancake_8h_momentum.sh start
#
# The strategy is mapped to a webhook in its secrets file:
#
# - https://t-100.tradingstrategy.ai/state
#

set -e

# https://stackoverflow.com/a/65396324/315168
check_secret_envs()
{
    var_names=("$@")
    for var_name in "${var_names[@]}"; do
        [ -z "${!var_name}" ] && echo "$var_name is unset." && var_unset=true
    done
    [ -n "$var_unset" ] && exit 1
    return 0
}

# This id is used in various paths and such
export EXECUTOR_ID=pancake_8h_momentum

# Strategy specific secrets
export STRATEGY_SECRETS_FILE=~/$EXECUTOR_ID.secrets.env
if [ ! -f "$STRATEGY_SECRETS_FILE" ] ; then
  echo "Strategy secrets missing: $STRATEGY_SECRETS_FILE"
  exit 1
fi


# Read generic secrets, then strategy specific secrets
source ~/secrets.env
source $STRATEGY_SECRETS_FILE

# These variables must come from the secrets file
check_secret_envs PRIVATE_KEY JSON_RPC_BINANCE TRADING_STRATEGY_API_KEY DISCORD_WEBHOOK_URL HTTP_ENABLED

# Metadata
export NAME="Pancake 8h momentum tick"
export SHORT_DESCRIPTION="A data collection test strategy that trades on 8h ticks on PancakeSwap"
export LONG_DESCRIPTION="For each two 4h candles, the strategy calculates the momentum and picks highest gainers. The portfolio holds 4 tokens at once. Only BUSD quoted pairs are considered."
export ICON_URL="https://upload.wikimedia.org/wikipedia/commons/4/43/Blueberry_pancakes_%283%29.jpg"

export STRATEGY_FILE="${PWD}/strategies/${EXECUTOR_ID}.py"

# As exported from scripts/show-pancake-info.py
export UNISWAP_V2_FACTORY_ADDRESS=0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73
export UNISWAP_V2_ROUTER_ADDRESS=0x10ED43C718714eb63d5aA57B78B54704E256024E
export UNISWAP_V2_INIT_CODE_HASH=0x00fb7f630766e6a796048ea87d01acd3068e8ff67d078148a3fa3f4a84f69bd5

export JSON_RPC=$JSON_RPC_BINANCE
export GAS_PRICE_METHOD="legacy"
export STATE_FILE="$EXECUTOR_ID.json"
export EXECUTION_TYPE="uniswap_v2_hot_wallet"
export APPROVAL_TYPE="unchecked"
export CACHE_PATH="${PWD}/.cache/${EXECUTOR_ID}"
export TICK_OFFSET_MINUTES="10"
export TICK_SIZE="8h"
# 12 hours
export MAX_DATA_DELAY_MINUTES=720

echo "HTTP enabled is $HTTP_ENABLED"

# https://stackoverflow.com/a/1537695/315168
poetry run trade-executor "$@"
