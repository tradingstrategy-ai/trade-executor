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
# Helper function to check that required secret environment variables are set
#
# https://stackoverflow.com/a/65396324/315168
check_secret_envs()
{
    var_names=("$@")
    for var_name in "${var_names[@]}"; do
        [ -z "${!var_name}" ] && echo "$var_name is unset." && var_unset=true
    done
    [ -n "$var_unset" ] && exit 1
    return 0
}

# These variables must come from the secrets
check_secret_envs PRIVATE_KEY TRADING_STRATEGY_API_KEY DISCORD_WEBHOOK_URL HTTP_ENABLED

#
# Generated environment variables.
# These maps to the physical paths on the server,
# also mapped in docker-compose.yml
#
export CACHE_PATH="cache/${EXECUTOR_ID}"
export STRATEGY_FILE="strategies/${EXECUTOR_ID}.py"
