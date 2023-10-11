# Export the latest trade-executor release tag
#
# This will set TRADE_EXECUTOR_VERSION environment variable to the latest release.
#
# Usage:
#
#    source scripts/set-latest-tag.sh
#

tag=`curl -s "https://api.github.com/repos/tradingstrategy-ai/trade-executor/tags" | jq -r '.[0].name'`
export TRADE_EXECUTOR_VERSION=$tag
echo "TRADE_EXECUTOR_VERSION environemnt variable is now set to the latest version"
echo "export TRADE_EXECUTOR_VERSION=${TRADE_EXECUTOR_VERSION}"