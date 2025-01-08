#!/bin/bash
#
# Deploy Lagoon vault for a strategy defined in docker-compose.yml
#
# Set up
# - Gnosis Safe
# - Vault smart contract
# - TradingStrategyModuleV0 guard with allowed assets
# - trade executor hot wallet as the asset manager role
#
# To run:
#
#   SIMULATE=true deploy/deploy-base-memex.sh
#

set -e
set -x
# set -u

unset JSON_RPC_ETHEREUM
unset JSON_RPC_BINANCE
unset JSON_RPC_POLYGON
unset JSON_RPC_ARBITRUM

if [ "$SIMULATE" = "" ]; then
    echo "Set SIMULATE=true or SIMULATE=false"
    exit 1
fi

# docker composer entry name
ID="base-memex"

# ERC-20 share token symbol
export FUND_SYMBOL="MEMEX"

# ERC-20 share toke  name
export FUND_NAME="Memex memecoin index strategy (Base)"

# The vault is nominated in USDC on Base
export DENOMINATION_ASSET="0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# Set as the initial owners or deployed Safe + deployer will be threre
# Safe signing threshold is number of cosigners minus one.
export MULTISIG_OWNERS="0xa7208b5c92d4862b3f11c0047b57a00Dc304c0f8, 0xbD35322AA7c7842bfE36a8CF49d0F063bf83a100, 0x05835597cAf9e04331dfe1f62C2Ec0C2aDc0d4a2, 0x5C46ab9e42824c51b55DcD3Cf5876f1132F9FbA9"

# Terms of service manager smart contract address.
# This one is deployed on Polygon.
# export TERMS_OF_SERVICE_ADDRESS="0xDCD7C644a6AA72eb2f86781175b18ADc30Aa4f4d"

# Run the command
# - Pass private key and JSON-RPC node from environment variables
# - Set vault-info.json to be written to a local file system

trade-executor \
    lagoon-deploy-vault \
    --vault-record-file="deploy/$ID-vault-info.json" \
    --fund-name="$FUND_NAME" \
    --fund-symbol="$FUND_SYMBOL" \
    --denomination-asset="$DENOMINATION_ASSET" \
    --any-asset \
    --uniswap-v2 \
    --uniswap-v3 \
    --multisig-owners="$MULTISIG_OWNERS"

