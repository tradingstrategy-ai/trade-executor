"""Helper script for the production server releases.

See docker.md.
"""
import base64
import os
import sys

import requests


def main():

    org = "tradingstrategy-ai"

    repo = "trade-executor"

    github_token = os.environ.get("GITHUB_TOKEN")

    if not github_token:
        sys.exit("GITHUB_TOKEN missing. Please include ~/secrets.env")

    ghcr_token = base64.b64encode(github_token.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Bearer {ghcr_token}"
    }

    req = requests.get(f"https://ghcr.io/v2/{org}/{repo}/tags/list", headers=headers)

    # {'name': 'tradingstrategy-ai/oracle', 'tags': ['pr-85', 'v2', 'latest', 'v3', 'v4']}
    tag_data = req.json()

    # Get version tags. These are identified by b prefix
    version_tags = [tag for tag in tag_data["tags"] if tag.startswith("v")]

    # Assume running counter
    version_tags = sorted(version_tags, reverse=True, key=lambda tag: int(tag[1:]))

    latest_version = version_tags[0]

    print(latest_version)

