
"""version CLi command."""

from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.version_info import VersionInfo


@app.command()
def version():
    """Print out the version information."""
    version_info = VersionInfo.read_docker_version()
    print(f"Version: {version_info.tag}")
    print(f"Commit hash: {version_info.commit_hash}")
    print(f"Commit message: {version_info.commit_message}")
    print("")
    print("Version information is only available within Docker image.")
