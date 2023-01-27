#!/bin/bash
#
# This script will set up an examples folder for notebooks.
#
# It combines notebooks from tradd-executor and docs repos
#

set -e

find ./notebooks -iname "*.ipynb" -exec cp {} examples \;
git clone https://github.com/tradingstrategy-ai/docs.git /tmp/docs
find /tmp/docs -iname "*.ipynb" -exec cp {} examples \;