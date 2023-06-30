#!/bin/bash
#
# Start trade-executor under docker compose
#

/root/.local/bin/poetry run  --quiet --directory /usr/src/trade-executor trade-executor "$@"
