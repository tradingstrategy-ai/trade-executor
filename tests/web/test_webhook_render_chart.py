"""Rendering server-side charts."""

import datetime
import random
import tempfile
from pathlib import Path
from queue import Queue

import numpy as np
import pandas as pd
import pytest
import requests
from eth_defi.utils import find_free_port
import plotly.graph_objects as go

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.state import State
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.strategy.chart.definition import ChartRegistry, ChartKind, ChartInput
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_lending_data import generate_lending_reserve, generate_lending_universe
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.webhook.server import create_webhook_server
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture()
def store() -> JSONFileStore:
    """Dummy state and store for the tests."""
    portfolio = Portfolio()
    state = State(portfolio=portfolio)
    store = JSONFileStore("/tmp/webhook-test.json")
    store.sync(state)
    return store


@pytest.fixture()
def strategy_universe() -> TradingStrategyUniverse:
    """Set up a mock universe."""

    time_bucket = TimeBucket.d1

    start_at = datetime.datetime(2023, 1, 1)
    end_at = datetime.datetime(2023, 6, 1)

    # Set up fake assets
    chain_id = ChainId.ethereum
    mock_exchange = generate_exchange(
        exchange_id=random.randint(1, 1000),
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="uniswap-v2"
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
    weth_usdc = TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        mock_exchange.address,
        internal_id=random.randint(1, 1000),
        internal_exchange_id=mock_exchange.exchange_id,
        fee=0.0030,
    )

    usdc_reserve = generate_lending_reserve(usdc, chain_id, 1)
    weth_reserve = generate_lending_reserve(weth, chain_id, 2)

    _, lending_candle_universe = generate_lending_universe(
        time_bucket,
        start_at,
        end_at,
        reserves=[usdc_reserve, weth_reserve],
        aprs={
            "supply": 2,
            "variable": 5,
        }
    )

    candles = generate_ohlcv_candles(
        time_bucket,
        start_at,
        end_at,
        start_price=1800,
        pair_id=weth_usdc.internal_id,
        exchange_id=mock_exchange.exchange_id,
    )

    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=mock_exchange,
        time_bucket=time_bucket,
        pair=weth_usdc,
        lending_candles=lending_candle_universe,
    )


@pytest.fixture()
def strategy_input_indicators(strategy_universe: TradingStrategyUniverse) -> StrategyInputIndicators:
    """Create a mock StrategyInputIndicators."""
    return StrategyInputIndicators(
        strategy_universe=strategy_universe,
        available_indicators=IndicatorSet(),
        indicator_results={},
    )



@pytest.fixture()
def chart_registry() -> ChartRegistry:
    """Test functiosn exposed for rendering."""
    def chart_func(chart_input: ChartInput):
        """Render sine wave as a Plotly figure."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines', name='Sine Wave'))
        fig.update_layout(title='Sine Wave', xaxis_title='x', yaxis_title='sin(x)')
        return fig

    def table_func(chart_input: ChartInput):
        """Render sine wave as DataFrame table."""
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        return pd.DataFrame({
            "x": x,
            "y": y,
        })

    chart_registry = ChartRegistry()
    chart_registry.register(chart_func, ChartKind.indicator_all_pairs)
    chart_registry.register(table_func, ChartKind.indicator_all_pairs)
    return chart_registry


@pytest.fixture()
def server_url(store, chart_registry, strategy_input_indicators):
    execution_state = RunState()
    execution_state.source_code = "Foobar"
    execution_state.visualisation.small_image = b"1"
    execution_state.visualisation.large_image = b"2"
    execution_state.visualisation.small_image_dark = b"3"
    execution_state.visualisation.large_image_dark = b"4"
    execution_state.version.tag = "v1"
    execution_state.version.commit_message = "Foobar"
    execution_state.read_only_state_copy = store.load()
    execution_state.chart_registry = chart_registry
    execution_state.latest_indicators = strategy_input_indicators

    queue = Queue()

    metadata = Metadata(
        "Foobar",
        "Short desc",
        "Long desc",
        None,
        datetime.datetime.utcnow(),
        True,
        badges=Metadata.parse_badges_configuration("polygon, metamask, eth, usdc"),
        tags={StrategyTag.beta},
        fees=dict(
            management_fee=0.00,
            trading_strategy_protocol_fee=0.02,
            strategy_developer_fee=0.05,
        )
    )

    # Inject some fake files for the backtest content
    notebook_result = Path(tempfile.mkdtemp()) / 'test_cli_backtest.ipynb'
    notebook_result.open("wt").write("Foo")
    html_result = Path(tempfile.mkdtemp()) / 'test_cli_backtest.html'
    html_result.open("wt").write("Bar")
    metadata.backtest_notebook = notebook_result
    metadata.backtest_html = html_result

    port = find_free_port(20_000, 40_000, 20)
    server = create_webhook_server("127.0.0.1", port, "test", "test", queue, store, metadata, execution_state)
    server_url = f"http://test:test@127.0.0.1:{port}"
    yield server_url
    server.shutdown()


def test_web_render_figure(logger, server_url):
    """Render PNG and HTML output on the server-side for strategy charts"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry/render",
        params={"chart_id": "chart_func"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "image/png", f"Got: {resp.text}"
    assert int(resp.headers["content-length"]) > 100


def test_web_render_table(logger, server_url):
    """Render PNG and HTML output on the server-side for strategy charts"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry/render",
        params={"chart_id": "table_func"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "text/html; charset=UTF-8", f"Got: {resp.text}"
    assert int(resp.headers["content-length"]) > 100


def test_web_chart_registry(logger, server_url):
    """Render PNG and HTML output on the server-side for strategy charts"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry",
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "application/json", f"Got: {resp.text}"


def test_web_chart_pairs(logger, server_url):
    """"Get pair list"""

    # Check image output
    resp = requests.get(
        f"{server_url}/chart-registry/pairs",
    )
    assert resp.status_code == 200
    assert resp.headers.get("content-type") == "application/json", f"Got: {resp.text}"
