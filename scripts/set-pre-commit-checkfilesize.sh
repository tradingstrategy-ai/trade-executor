#!/bin/bash
set -e

pre_commit_path=".git/hooks/pre-commit"
pre_commit_checkfilesize_sample_path="scripts/pre-commit-sample/pre-commit-checkfilesize.sample"
pre_commit_config_path="scripts/pre-commit-sample/pre-commit-config.yaml"

# Install pre-commit 
pip install pre-commit

# Setup .pre-commit-config.yaml file
cp -r "$pre_commit_config_path" ./.pre-commit-config.yaml

cp -r "$pre_commit_checkfilesize_sample_path" "$pre_commit_path"
chmod +x "$pre_commit_path"   # Set executable permission
echo "Copied pre-commit-checkfilesize from pre-commit-checkfilesize.sample."
        
# Add pre-commit-checkfilesize to the staged area
git add "$pre_commit_path" 
        
# Commit pre-commit-checkfilesize
git commit -m "Add pre-commit-checkfilesize hook"

# Run pre-commit on all files
pre-commit run --all-files