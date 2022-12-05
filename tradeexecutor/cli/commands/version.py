from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.version_info import VersionInfo


@app.command()
def version():
    """Check that the application loads without doing anything."""
    version_info = VersionInfo.read_docker_version()
    print(f"Version: {version_info.tag}")
    print(f"Commit hash: {version_info.commit_hash}")
    print(f"Commit message: {version_info.commit_message}")
    print("")
    print("Version information is only available within Docker image.")
