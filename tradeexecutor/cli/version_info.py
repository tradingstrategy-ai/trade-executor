"""Read Docker version info."""
import os
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class VersionInfo:
    #: v50
    tag: Optional[str] = None

    #: The latest commit message before release.sh was run
    commit_message: Optional[str] = None

    #: Git commit SHA hash
    commit_hash: Optional[str] = None

    @staticmethod
    def read_version_file(name: str) -> Optional[str]:
        """See Dockerfile"""
        if os.path.exists(name):
             with open(name, "rt") as inp:
                return inp.read(name)
        return None

    @staticmethod
    def read_docker_version() -> "VersionInfo":
        return VersionInfo(
            tag=VersionInfo.read_version_file("GIT_VERSION_TAG.txt"),
            commit_message=VersionInfo.read_version_file("GIT_COMMIT_MESSAGE.txt"),
            commit_hash=VersionInfo.read_version_file("$GIT_VERSION_HASH.txt"),
        )
