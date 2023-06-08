#!/bin/bash
#
# Start trade-executor under docker compose
#

# /root/.local/bin/poetry --directory=/usr/src/trade-executor --quiet run trade-executor "$@"
/root/.local/bin/poetry --directory /usr/src/trade-executor run trade-executor "$@"
