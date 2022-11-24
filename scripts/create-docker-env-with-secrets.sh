#
# Creates .env file for a trade-executor Docker container
#
# - Reads public .env configuration variables
# - Reads secret environment variables
# - Generates some file locations based on the strategy id
# - Validates environment variables
# - Writes the output to a single file
#
# Usage:
#
#   # Load your non-committed secrets from the server root
#   source ~/my-secrets.env
#
#   # Combine the env configuration file with the secrets
#   scripts/create-docker-env-with-secrets.sh env/pancake-eth-usd-sma.env > ~/pancake-eth-usd-sma-final.env
#



#
# Generated environment variables.
# These maps to the physical paths on the server,
# also mapped in docker-compose.yml
#
export CACHE_PATH="cache/${EXECUTOR_ID}"
export STRATEGY_FILE="strategies/${EXECUTOR_ID}.py"
