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

    Example of storing and loading custom variables:

    .. code-block:: python

        def decide_trades(input: StrategyInput) -> list[TradeExecution]:
            cycle = input.cycle
            state = input.state

            # Saving values by cycle
            state.other_data.save(cycle, "my_value", 1)
            state.other_data.save(cycle, "my_value_2", [1, 2])
            state.other_data.save(cycle, "my_value_3", {1: 2})

            if cycle >= 2:
                # Loading latest values
                assert state.other_data.load_latest("my_value") == 1
                assert state.other_data.load_latest("my_value_2") == [1, 2]
                assert state.other_data.load_latest("my_value_3") == {1: 2}

            return []

    You can also read these variables after the backtest is complete:

    .. code-block::

        result = run_backtest_inline(
            client=None,
            decide_trades=decide_trades,
            create_indicators=create_indicators,
            universe=strategy_universe,
            reserve_currency=ReserveCurrency.usdc,
            engine_version="0.5",
            parameters=StrategyParameters.from_class(Parameters),
            mode=ExecutionMode.unit_testing,
        )

        # Variables are readable after the backtest
        state = result.state
        assert len(state.other_data.data.keys()) == 29  # We stored data for 29 decide_trades cycles
        assert state.other_data.data[1]["my_value"] == 1      # We can read historic values
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

    def load_latest(self, name: str, default=None) -> JsonSerialisableObject | None:
        """Load the latest named value from the store.

        - Take the value whatever is the last cycle

        :return:
            If the last cycle did not store this var, then return `None`.
        """
        latest_cycle = self.get_latest_stored_cycle()
        return self.data.get(latest_cycle, {}).get(name, default)