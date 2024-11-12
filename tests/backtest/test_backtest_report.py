"""Generate Python notebook report from the backtest results.
"""
import datetime
import logging
import os
import time
import webbrowser
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Tuple, Any

import pytest

import pandas as pd
from nbformat import NotebookNode

from tradeexecutor.backtest.tearsheet import export_backtest_report
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.backtest.backtest_runner import run_backtest, setup_backtest_for_universe
from tradeexecutor.cli.log import setup_pytest_logging
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles

@pytest.fixture(scope="module")
def logger(request):
    """Setup test logger."""
    return setup_pytest_logging(request, mute_requests=False)


@pytest.fixture(scope="module")
def mock_chain_id() -> ChainId:
    """Mock a chai id."""
    return ChainId.ethereum


@pytest.fixture(scope="module")
def mock_exchange(mock_chain_id) -> Exchange:
    """Mock an exchange."""
    return generate_exchange(exchange_id=1, chain_id=mock_chain_id, address=generate_random_ethereum_address())


@pytest.fixture(scope="module")
def usdc() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "USDC", 6, 1)


@pytest.fixture(scope="module")
def weth() -> AssetIdentifier:
    """Mock some assets"""
    return AssetIdentifier(ChainId.ethereum.value, generate_random_ethereum_address(), "WETH", 18, 2)


@pytest.fixture(scope="module")
def weth_usdc(mock_exchange, usdc, weth) -> TradingPairIdentifier:
    """Mock some assets"""
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=555,
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030
    )


@pytest.fixture(scope="module")
def strategy_path() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.join(os.path.dirname(__file__), "..", "..", "strategies", "test_only", "pandas_buy_every_second_day.py"))


@pytest.fixture(scope="module")
def synthetic_universe(mock_chain_id, mock_exchange, weth_usdc) -> TradingStrategyUniverse:
    """Generate synthetic trading data universe for a single trading pair.

    - Single mock exchange

    - Single mock trading pair

    - Random candles

    - No liquidity data available
    """

    start_date = datetime.datetime(2021, 6, 1)
    end_date = datetime.datetime(2022, 1, 1)

    time_bucket = TimeBucket.d1

    pair_universe = create_pair_universe_from_code(mock_chain_id, [weth_usdc])

    # Generate candles for pair_id = 1
    candles = generate_ohlcv_candles(time_bucket, start_date, end_date, pair_id=weth_usdc.internal_id)
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={mock_chain_id},
        exchanges={mock_exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[weth_usdc.quote])


@pytest.fixture(scope="module")
def routing_model(synthetic_universe) -> BacktestRoutingModel:
    return generate_simple_routing_model(synthetic_universe)


@pytest.fixture(scope="module")
def state(
        logger: logging.Logger,
        strategy_path: Path,
        synthetic_universe: TradingStrategyUniverse,
        routing_model: BacktestRoutingModel,
    ):
    """Run a simple strategy backtest.

    Calculate some statistics based on it.
    """

    # Run the test
    setup = setup_backtest_for_universe(
        strategy_path,
        start_at=datetime.datetime(2021, 6, 1),
        end_at=datetime.datetime(2022, 1, 1),
        cycle_duration=CycleDuration.cycle_1d,  # Override to use 24h cycles despite what strategy file says
        candle_time_frame=TimeBucket.d1,  # Override to use 24h cycles despite what strategy file says
        initial_deposit=10_000,
        universe=synthetic_universe,
        routing_model=routing_model,
    )

    state, universe, debug_dump = run_backtest(setup)
    state.name = "Backtest example"
    return state

@pytest.fixture(scope="module")
def notebook(state, synthetic_universe) -> NotebookNode:
    with NamedTemporaryFile(suffix='.ipynb', prefix=os.path.basename(__file__)) as temp:
        nb = export_backtest_report(
            state,
            synthetic_universe,
            output_notebook=Path(temp.name),
        )
        return nb


@pytest.fixture(scope="module")
def html_report(state, synthetic_universe) -> Path:
    with NamedTemporaryFile(suffix='.ipynb', prefix=os.path.basename(__file__)) as temp_notebook, \
        NamedTemporaryFile(suffix='.html', prefix=os.path.basename(__file__)) as temp_html:
        html_report = Path(temp_html.name)
        nb = export_backtest_report(
            state,
            synthetic_universe,
            output_notebook=Path(temp_notebook.name),
            output_html=html_report
        )
        yield html_report


@pytest.mark.slow_test_group
def test_generate_backtest_pass_state(notebook: NotebookNode):
    """We can convert any of our statistics to dataframes"""
    # We filled in parameter cell correctly
    parameter_cell = notebook.cells[0]
    assert ".json" in parameter_cell.source  # Check for generated .json file path


@pytest.mark.slow_test_group
def test_generate_html_report(html_report: Path):
    """We can generate a HTML report"""
    assert os.path.exists(html_report), f"Did not create: {html_report}"

    # Check we injected CSS correctly
    html_content = html_report.open("rt").read()
    assert "/* trade-executor backtest report generator custom CSS */" in html_content


@pytest.mark.slow_test_group
@pytest.mark.skipif(os.environ.get("SHOW_REPORT_IN_BROWSER") is None, reason="Manual web browser based test")
def test_show_backtest_report(html_report: Path):
    """Manual test to view the result.

    To run

        SHOW_REPORT_IN_BROWSER=true pytest -k test_show_backtest_report

    """
    webbrowser.open(f"file://{html_report}")
    time.sleep(5)  # Give it time to open it before /tmp autodelete triggers in


