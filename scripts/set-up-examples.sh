#!/bin/bash
#
# This script will set up an examples folder for notebooks.
#
# It combines notebooks from tradd-executor and docs repos
#

set -e

RUN find ./notebooks -iname "*.ipynb" -exec cp {} examples \;
git clone https://github.com/tradingstrategy-ai/docs.git /tmp/docs
RUN find /tmp/docs -iname "*.ipynb" -exec cp {} examples \;