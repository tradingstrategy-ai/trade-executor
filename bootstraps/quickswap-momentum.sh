#!/bin/bash
#
# QuickSwap momentum trading launch script.
#
# This script combines a strategy specific trade executor environt variables
# with secret environment variables before running the trade-executor command line entry point.
#
# Check that `~/secrets.env` and `~/$EXECUTOR_ID.secrets.env` are set up.
#
# Then run the checks:
#
#     bootstraps/polygon_momentum.sh check-universe --max-data-delay-minutes=1440
#
# Then start:
#
#     bootstraps/polygon_momentum.sh start
#
# The strategy is mapped to a webhook in its secrets file:
#
# - https://robocop.tradingstrategy.ai/state
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

# This id is used in various paths and such.
# It is taken from the shell script name.
export EXECUTOR_ID=`basename "$0" .sh`
echo "Starting trade executor $EXECUTOR_ID"

# Strategy specific secrets.
# This file will give us private key needed to access the hot wallet
# plus Discord webhook keys.
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
export NAME="Polygon momentum"
export SHORT_DESCRIPTION="A data collection test that trades on Polygon MATIC/USDC pairs."
export LONG_DESCRIPTION="This strategy is a test strategy that does not aim to generated profit. The strategy trades on MATIC/USDC pairs on QuickSwap. The strategy rebalances every 16 hours to generate maximum amount of data. Based on 4h candles, the strategy calculates the momentum and picks highest gainers. The portfolio can hold 6 positions at once."
export ICON_URL="src="https://i0.wp.com/bloody-disgusting.com/wp-content/uploads/2022/01/watch-robocop-the-series.png?w=1270"

export STRATEGY_FILE="${PWD}/strategies/${EXECUTOR_ID}.py"

export JSON_RPC=$JSON_RPC_BINANCE
export GAS_PRICE_METHOD="legacy"
export STATE_FILE="$EXECUTOR_ID.json"
export EXECUTION_TYPE="uniswap_v2_hot_wallet"
export APPROVAL_TYPE="unchecked"
export CACHE_PATH="${PWD}/.cache/${EXECUTOR_ID}"
export TICK_OFFSET_MINUTES="10"
export TICK_SIZE="16h"
# 12 hours
export MAX_DATA_DELAY_MINUTES=720

echo "HTTP enabled is $HTTP_ENABLED"

# https://stackoverflow.com/a/1537695/315168
poetry run trade-executor "$@"
