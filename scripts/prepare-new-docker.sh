#!/bin/bash
#
# Prepare a new docker image with master source code
#

set -e
git pull
(cd deps/web3-ethereum-defi && git pull)
(cd deps/trading-strategy && git pull)
docker build -t trading-strategy/trade-executor:latest .