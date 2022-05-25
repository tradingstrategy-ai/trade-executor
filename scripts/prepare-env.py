"""Prepare environment variable list.

Because the configuration options list is so long, it is hard to manage by hand.

- Reads the current environment variable list
- Writes out cleaned up .env file to stdout

Very useful with Docker.
"""

import os

from tradeexecutor.cli.env import get_available_env_vars

vars = get_available_env_vars()

# print("Strategy execution settings are:", ", ".join(vars))

for desc in vars:
    value = os.environ.get(desc.name)
    print(f"# {desc.help}")
    print(f"# Type: {desc.type}")
    if value is not None:
        print(f"{desc.name}={value}")
    else:
        print(f"{desc.name}=")
    print()







