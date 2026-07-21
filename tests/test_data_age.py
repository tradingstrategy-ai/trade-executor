"""Check we correctly handle fresh and expired data."""
import os
import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tradeexecutor.strategy.execution_context import ExecutionMode, unit_test_execution_context, unit_test_trading_execution_context
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse, DataTooOld, UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel, load_partial_data, TradingStrategyUniverse
from tradeexecutor.utils.timer import timed_task
from tradingstrategy.timebucket import TimeBucket
from eth_defi.compat import native_datetime_utc_now
from tradingstrategy.universe import Universe

from tradeexecutor.analysis.vault import display_vaults

# https://docs.pytest.org/en/latest/how-to/skipping.html#skip-all-test-functions-of-a-class-or-module
pytestmark = pytest.mark.skipif(os.environ.get("TRADING_STRATEGY_API_KEY") is None, reason="Set TRADING_STRATEGY_API_KEY environment variable to run this test")


class DataAgeTestUniverseModel(TradingStrategyUniverseModel):
    """Load 6 months data."""

    def construct_universe(self, ts: datetime.datetime, mode: ExecutionMode) -> StrategyExecutionUniverse:
        assert isinstance(mode, ExecutionMode)

        client = self.client

        dataset = load_partial_data(
            client,
            unit_test_trading_execution_context,
            TimeBucket.d30,
            pairs=((ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),),
            universe_options=UniverseOptions(history_period=datetime.timedelta(days=6*30)),
        )

        # Pair index takes long time to construct and is not needed for the test
        universe = TradingStrategyUniverse.create_from_dataset(dataset)
        return universe


# @pytest.mark.slow_test_group
@pytest.mark.skip(reason="Live integration - need to rewrite this test")
def test_data_fresh(persistent_test_client):
    """Fresh data passes our data check."""
    # d1 data is used by other tests and cached
    universe_model = DataAgeTestUniverseModel(persistent_test_client, timed_task)
    best_before_duration = datetime.timedelta(weeks=1000)  # Our unit test is good for next 1000 years
    ts = native_datetime_utc_now()
    universe = universe_model.construct_universe(ts, ExecutionMode.unit_testing_trading)
    universe_model.check_data_age(ts, universe, best_before_duration)


@pytest.mark.skip(reason="Live integration - need to rewrite this test")
def test_data_aged(persistent_test_client):
    """Aged data raises an exception."""
    universe_model = DataAgeTestUniverseModel(persistent_test_client, timed_task)
    ts = native_datetime_utc_now()
    best_before_duration = datetime.timedelta(seconds=1)  # We can never have one second old data
    universe = universe_model.construct_universe(ts, ExecutionMode.backtesting)
    with pytest.raises(DataTooOld):
        universe_model.check_data_age(ts, universe, best_before_duration)


def test_data_aged_liquidity_reports_stale_pair() -> None:
    """Report the stale liquidity pair so a live-trading failure is actionable.

    1. Create a universe with a named vault pair and stale liquidity samples.
    2. Run the freshness check with a one-day liquidity tolerance.
    3. Verify the raised exception identifies both the pair ID and name.
    """
    # 1. Create a universe with a named vault pair and stale liquidity samples.
    pair = MagicMock()
    pair.pair_id = 123
    pair.other_data = {"vault_name": "Example vault"}
    pair.base_token_symbol = "vUSDC"
    pair.quote_token_symbol = "USDC"

    liquidity = MagicMock()
    liquidity.df = pd.DataFrame({
        "pair_id": [123],
        "timestamp": [pd.Timestamp("2026-07-18")],
    })
    liquidity.get_timestamp_range.return_value = (
        pd.Timestamp("2023-08-28"),
        pd.Timestamp("2026-07-18"),
    )

    data_universe = MagicMock(spec=Universe)
    data_universe.candles = None
    data_universe.liquidity = liquidity
    data_universe.pairs.get_pair_by_id.return_value = pair
    strategy_universe = TradingStrategyUniverse(data_universe=data_universe, reserve_assets=[])

    # 2. Run the freshness check with a one-day liquidity tolerance.
    with pytest.raises(DataTooOld) as exception_info:
        TradingStrategyUniverseModel.check_data_age(
            ts=datetime.datetime(2026, 7, 20),
            strategy_universe=strategy_universe,
            best_before_duration=datetime.timedelta(days=2),
            best_before_duration_liquidity=datetime.timedelta(days=1),
        )

    # 3. Verify the raised exception identifies both the pair ID and name.
    exception_message = str(exception_info.value)
    assert "Pair #123" in exception_message
    assert "Example vault" in exception_message


def test_display_vaults_logs_complete_live_checklist_row() -> None:
    """Render the full live vault checklist row without pandas abbreviation.

    1. Create a vault diagnostic row with deliberately long pair details.
    2. Render the checklist in a live-trading execution mode.
    3. Verify the log output contains the full pair ID and status text.
    """
    # 1. Create a vault diagnostic row with deliberately long pair details.
    pair = MagicMock()
    pair.internal_id = 123
    pair.get_vault_name.return_value = "Example vault"
    pair.get_vault_protocol.return_value = "example_protocol"
    pair.quote.token_symbol = "USDC"

    status = "No price data found for vault: <Pair #123: Example vault with complete details>"
    strategy_universe = MagicMock()
    strategy_universe.get_vault_error.return_value = status
    strategy_universe.get_pair_by_smart_contract.return_value = pair
    output = []

    # 2. Render the checklist in a live-trading execution mode.
    display_vaults(
        vaults=[(ChainId.ethereum, "0x1234567890123456789012345678901234567890")],
        strategy_universe=strategy_universe,
        execution_mode=ExecutionMode.real_trading,
        printer=output.append,
    )

    # 3. Verify the log output contains the full pair ID and status text.
    checklist_output = output[1]
    assert "123" in checklist_output
    assert status in checklist_output





