#!/bin/bash
#
# Pancake 8h momentum trading launch script.
#
# This script combines a strategy specific trade executor environt variables
# with secret environment variables before running the trade-executor command line entry point.
#
# First source `~/secrets.env` and `~/$EXECUTOR_ID.secrets.env` files to set up the enviroment.
#
# source ~/$EXECUTOR_ID.secrets.env
# source ~/secrets.env
#
# Then run the checks:
#
# bootstraps/pancake_8h_momentum.sh check-universe --max-data-delay-minutes=1440
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


# These variables must come from the secrets file
check_secret_envs PRIVATE_KEY JSON_RPC_BINANCE TRADING_STRATEGY_API_KEY DISCORD_WEBHOOK_URL

# This id is used in various paths and such
EXECUTOR_ID=pancake_8h_momentum

export NAME="Pancake 8h momentum tick"
export STRATEGY_FILE="${PWD}/strategies/${EXECUTOR_ID}.py"
export HTTP_ENABLED=false

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

# https://stackoverflow.com/a/1537695/315168
poetry run trade-executor "$@"
