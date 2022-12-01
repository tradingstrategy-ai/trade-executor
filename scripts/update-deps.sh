#!/bin/bash
#
# Update all git submodule dependenecies
#

set -e

(cd spec && git checkout main && git pull origin main)
(cd deps/trading-strategy && git pull origin master)
(cd deps/web3-ethereum-defi && git pull origin master)