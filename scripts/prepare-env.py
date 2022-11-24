"""Prepare environment variable list.

Because the configuration options list is so long, it is hard to manage by hand.

- Reads the current environment variable list
- Writes out cleaned up .env file to stdout

Very useful with Docker.
"""

import os
import sys

from tradeexecutor.cli.env import get_available_env_vars

out = open(sys.argv[1], "wt")

assert sys.version_info >= (3, 9), f"Watch out for old system Python, got version {sys.version_info}"

#: Read environment variables we use directly from Typer
vars = get_available_env_vars()

# print("Strategy execution settings are:", ", ".join(vars))

for desc in vars:
    value = os.environ.get(desc.name)
    print(f"# {desc.help}", file=out)
    print(f"# Type: {desc.type}", file=out)
    if value is not None:
        print(f"{desc.name}={value}", file=out)
    else:
        print(f"{desc.name}=", file=out)
    print(file=out)







