"""Storing of custom variables in the backtesting state."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from dataclasses_json import dataclass_json

JsonSerialisableObject: TypeAlias = Any


def _dict_dict():
    return defaultdict(dict)


@dataclass_json
@dataclass(slots=True)
class OtherData:
    """Store custom variables in the backtesting state.

    - For each cycle, you can record custom variables here

    - All historical cycle values are stored

    - All values must be JSON serialisable.

    - You can then read the variables back

    - This can be used in live trade execution as well **with care**.
      Because of the underlying infrastructure may crash (blockchain halt, server crash)
      cycles might be skipped.
    """

    #: Cycle number -> dict mapping
    data: dict[int, JsonSerialisableObject] = field(default_factory=_dict_dict)

    def get_latest_stored_cycle(self) -> int:
        """Get the cycle for which we have recorded any data.

        :return:
            0 if no data
        """
        if len(self.data) == 0:
            return 0
        return max(self.data.keys())

    def save(self, cycle: int, name: str, value: JsonSerialisableObject):
        """Save the value on this cycle."""
        assert type(cycle) == int, f"Got {cycle}"
        assert type(name) == str, f"Got {name}"
        self.data[cycle][name] = value

    def load_latest(self, name: str) -> JsonSerialisableObject | None:
        """Load the latest named value from the store.

        - Take the value whatever is the last cycle

        :return:
            If the last cycle did not store this var, then return `None`.
        """
        latest_cycle = self.get_latest_stored_cycle()
        return self.data.get(latest_cycle).get(name, None)