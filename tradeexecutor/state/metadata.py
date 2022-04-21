"""Metadata describes strategy for website rendering.

Metadata is not stored as the part of the state, but configured
on the executor start up.
"""
import datetime
from dataclasses import dataclass
from typing import Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Metadata:
    """Strategy metadata."""

    name: str

    #: 1 sentence
    short_description: Optional[str]

    #: Multiple paragraphs.
    long_description: Optional[str]

    #: For <img src>
    icon_url: Optional[str]

    #: When the instance was started last time, UTC
    started_at: datetime.datetime
