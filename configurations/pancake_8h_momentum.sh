#!/bin/bash
#
# Pancake 8h momentum strategy configuration script.
#
# Splices the .env file for the trade executor from several configs.
# See docker.md for more details.
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
export NAME="Pancake momentum alpha v0"
export SHORT_DESCRIPTION="A data collection test that trades on PancakeSwap BUSD pairs."
export LONG_DESCRIPTION="This strategy is a test strategy that does not aim to generated profit. The strategy trades on BUSD pairs on PancakeSwap. The strategy rebalances every 8 hours to generate maximum amount of data. For each two 4h candles, the strategy calculates the momentum and picks highest gainers. The portfolio can hold 4 positions at once."
export ICON_URL="https://upload.wikimedia.org/wikipedia/commons/4/43/Blueberry_pancakes_%283%29.jpg"
export STRATEGY_FILE="strategies/${EXECUTOR_ID}.py"
export CACHE_PATH=cache/${EXECUTOR_ID}
export JSON_RPC=$JSON_RPC_BINANCE
export GAS_PRICE_METHOD="legacy"
export STATE_FILE="$EXECUTOR_ID.json"
export EXECUTION_TYPE="uniswap_v2_hot_wallet"
export APPROVAL_TYPE="unchecked"
export TICK_OFFSET_MINUTES="10"
export TICK_SIZE="8h"
# 12 hours
export MAX_DATA_DELAY_MINUTES=720

# Webhook server configuration for Docker
export HTTP_PORT=3456
export HTTP_HOST=0.0.0.0


OUTPUT=~/$EXECUTOR_ID.env
poetry run python scripts/prepare-env.py > $OUTPUT
echo "Created $OUTPUT"