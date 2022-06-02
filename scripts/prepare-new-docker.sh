#!/bin/bash
#
# Prepare new docker images with master source code
# and latest configs.
#

set -e

# Update all source
git pull origin master
(cd deps/web3-ethereum-defi && git pull origin master)
(cd deps/trading-strategy && git pull origin master)

# Build new docker image
docker build -t trading-strategy/trade-executor:latest .

# Recreate .env files for docker-compose
configurations/pancake_8h_momentum.sh
configurations/quickswap-momentum.sh

echo "All done"

