#!/bin/bash
#
# Start trade-executor under docker compose
#

poetry --quiet run trade-executor "$@"
