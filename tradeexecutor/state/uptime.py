"""Uptime statistics.

Record uptime and completion statistics as the part of the state.
"""

import logging
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataclasses_json import dataclass_json


logger = logging.getLogger(__name__)

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

        # Should not ever happen, but may happen if state crashes at the right moment
        # Do warning only, as uptime recording is not critical for correct function of the strategy
        if cycle_number in self.cycles_completed_at:
            logger.warning(f"Cycle completion for #%d already recorded", cycle_number)

        self.cycles_completed_at[cycle_number] = now_
