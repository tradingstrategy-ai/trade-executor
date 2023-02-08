"""Uptime statistics.

Record uptime and completion statistics as the part of the state.
"""

import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Uptime:
    """Update statistics of past trade execution."""

    # Contains list of timestamps when uptime check was recorded on the state
    uptime_checks: List[datetime.datetime] = field(default_factory=list)

    #: When a strategy cycle was complete
    #:
    #: Contains strategy cycle number -> UTC wall clock time when a cycle was completed
    cycles_completed_at: Dict[int, datetime.datetime] = field(default_factory=dict)

    def record_cycle_complete(self, cycle_number: int, now_: Optional[datetime.datetime] = None):
        """Mark the execution cycle successfully completed"""
        assert isinstance(cycle_number, int)
        if now_ is None:
            now_ = datetime.datetime.utcnow()
        assert isinstance(now_, datetime.datetime)
        assert cycle_number not in self.cycles_completed_at, f"Cycle completion for #{cycle_number} already recorded"
        self.cycles_completed_at[cycle_number] = now_
