"""Read Docker version info.

Based on this idea: https://stackoverflow.com/a/74694676/315168
"""
import os
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class VersionInfo:
    """Reflect the version information embedded in the Docker image during CI build.

    See `Dockerimage` and `.githyb/workflows/tests.yml` for details.


    """

    #: v50
    tag: Optional[str] = None

    #: The latest commit message before release.sh was run
    commit_message: Optional[str] = None

    #: Git commit SHA hash
    commit_hash: Optional[str] = None

    def __repr__(self):
        return f"Trade-executor Docker version: {self.tag}\nCommit hash: {self.commit_hash}\nCommit message: {self.commit_message}"

    @staticmethod
    def read_version_file(name: str) -> Optional[str]:
        """See Dockerfile"""
        if os.path.exists(name):
             with open(name, "rt") as inp:
                return inp.read().strip()
        return None

    @staticmethod
    def read_docker_version() -> "VersionInfo":
        """Read version information burnt within Docker file-system during image building.

        :return:
            Populated version info or `None` for every field
            if it does not look like we are inside a Docker container.
        """
        return VersionInfo(
            tag=VersionInfo.read_version_file("GIT_VERSION_TAG.txt"),
            commit_message=VersionInfo.read_version_file("GIT_COMMIT_MESSAGE.txt"),
            commit_hash=VersionInfo.read_version_file("GIT_VERSION_HASH.txt"),
        )

