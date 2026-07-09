"""Unit tests for position valuation update statistics recording."""

import datetime

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.routing import RoutingState
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.strategy.valuation_update import update_position_valuations


class NoOpRoutingState(RoutingState):
    """Routing state stand-in — the portfolio has no open positions, so routing is never used."""

    def __init__(self, universe: StrategyExecutionUniverse):
        super().__init__(universe)


class NoOpValuationModel(ValuationModel):
    """Valuation model stand-in — never called because the portfolio has no open positions."""

    def __call__(self, ts, position):
        raise NotImplementedError("No open positions to value in this test")


@pytest.fixture()
def universe() -> StrategyExecutionUniverse:
    """Minimal strategy universe with no reserve assets."""
    return StrategyExecutionUniverse(reserve_assets=[])


@pytest.fixture()
def state() -> State:
    """Empty portfolio state with no recorded statistics."""
    return State()


def test_update_position_valuations_skip_statistics(
    state: State,
    universe: StrategyExecutionUniverse,
):
    """skip_statistics controls whether a portfolio statistics entry is recorded.

    The live stats refresh holds back the statistics write until treasury settlement
    has reconciled reserve cash to on-chain, so a stale reserve balance is never
    combined with a fresh position valuation in a recorded statistics point.

    1. Call update_position_valuations() with skip_statistics=True on an empty portfolio.
    2. Verify no portfolio statistics entry was recorded.
    3. Call update_position_valuations() again with the default skip_statistics=False.
    4. Verify exactly one portfolio statistics entry was recorded.
    """
    ts = datetime.datetime(2026, 7, 9, 6, 0)

    # 1. Call update_position_valuations() with skip_statistics=True on an empty portfolio.
    update_position_valuations(
        timestamp=ts,
        state=state,
        universe=universe,
        execution_context=unit_test_execution_context,
        valuation_model=NoOpValuationModel(),
        routing_state=NoOpRoutingState(universe),
        skip_statistics=True,
    )

    # 2. Verify no portfolio statistics entry was recorded.
    assert len(state.stats.portfolio) == 0

    # 3. Call update_position_valuations() again with the default skip_statistics=False.
    update_position_valuations(
        timestamp=ts,
        state=state,
        universe=universe,
        execution_context=unit_test_execution_context,
        valuation_model=NoOpValuationModel(),
        routing_state=NoOpRoutingState(universe),
    )

    # 4. Verify exactly one portfolio statistics entry was recorded.
    assert len(state.stats.portfolio) == 1
