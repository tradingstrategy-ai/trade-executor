#!/bin/bash
#
# Check the trade-executor status through the webhook
#
# - Return 1 if the trade-executor main loop has crashed
#
# - Echo the crash reason
#
# - Read the webhook URL from the command line argumetn
#
# See https://tradingstrategy.ai/docs/deployment/troubleshooting.html for more information
#

set -e

if [ -z "$1" ]; then
    echo "Error: Give the webhook URL as the first argument"
    exit 1
fi

set -u

webhook_url=$1

# /status gives 200 in the case the trade-executor has crashed
# and you need to check for the exception record in the status output.
#
# Have some built-in curl retry logic to reading the webhook,
# instead of relying on Docker healthcheck retries.
# https://stackoverflow.com/a/64041910/315168
#
failure_reason=$(curl --silent --fail --retry 6 --max-time 5 --retry-delay 10 --retry-max-time 60 "$webhook_url/status" | jq ".exception")

if [ "$failure_reason" != "null" ] ; then
    echo "trade-executor has crashed: $failure_reason"
    exit 1
fi

echo "Ok"
exit 0
