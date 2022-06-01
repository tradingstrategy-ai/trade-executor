#!/bin/bash
#
# Prepare a new docker image with master source code
#

set -e
git pull origin master
(cd deps/web3-ethereum-defi && git pull origin master)
(cd deps/trading-strategy && git pull origin master)
docker build -t trading-strategy/trade-executor:latest .