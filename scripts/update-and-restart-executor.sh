#!/bin/bash
#
# Update and restart particular trade executor
#
# Usage:
#
#   scripts/generic/update-and-restart-executor.sh quickswap-momentum
#
# - Find the latest release tag from ghcr.io registry
# - Pull the latest released image
# - Restart the named trade executor container using the new release tag
#
set -e

# https://stackoverflow.com/a/59921864/315168
: ${1?' You forgot to supply a trade executor Docker container name. Use docker-compose ps to see available.'}

EXECUTOR_NAME=$1

echo "Updating and restarting $EXECUTOR_NAME"

if [ -z "$GITHUB_TOKEN"] ; then
    echo "GITHUB_TOKEN env variable must be available to use ghcr.io"
    exit 1
fi

# Get the latest release
docker login ghcr.io -u miohtama -p $GITHUB_TOKEN
export TRADE_EXECUTOR_VERSION=`poetry run get-latest-release`

if [[ "${TRADE_EXECUTOR_VERSION:0:1}" == "v" ]]; then
  echo "Trade executor version is now $TRADE_EXECUTOR_VERSION"
else
  echo "Version corrupted: $TRADE_EXECUTOR_VERSION"
  echo "Make sure you run from poetry shell. Type 'poetry shell' to active, then 'source ~/secrets.env'"
  exit 1
fi

if command docker-compose /dev/null ; then
  docker-compose stop $EXECUTOR_NAME
  docker-compose up -d --force-recreate $EXECUTOR_NAME
else
  # New Docker versions have command "docker compose"
  echo "Stopping and restarting $EXECUTOR_NAME"
  # Stop sends graceful shutdown?
  docker compose stop $EXECUTOR_NAME
  # Restart with new version
  docker compose up -d --force-recreate $EXECUTOR_NAME
fi

sleep 5

echo "All ok - check that uptime is zero and the container is running"
docker ps|grep $EXECUTOR_NAME
