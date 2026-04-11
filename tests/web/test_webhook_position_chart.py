"""Position chart webhook endpoint tests."""

import copy
import datetime
import tempfile
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from queue import Queue

import pytest
import requests
from eth_defi.compat import native_datetime_utc_now

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PositionStatistics
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInputIndicators
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.testing.webhook import create_webhook_server_with_retries, get_webhook_test_url
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange
from tradeexecutor.testing.synthetic_price_data import generate_ohlcv_candles
from tradeexecutor.testing.synthetic_universe_data import create_synthetic_single_pair_universe
from tradeexecutor.webhook.server import WebhookServer
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


@pytest.fixture()
def chart_pair() -> TradingPairIdentifier:
    """Create a single synthetic trading pair for position chart tests."""
    chain_id = ChainId.ethereum
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
        exchange_slug="uniswap-v2",
    )
    usdc = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "USDC", 6, 1)
    weth = AssetIdentifier(chain_id.value, generate_random_ethereum_address(), "WETH", 18, 2)
    return TradingPairIdentifier(
        weth,
        usdc,
        generate_random_ethereum_address(),
        exchange.address,
        internal_id=1,
        internal_exchange_id=exchange.exchange_id,
        fee=0.0030,
    )


@pytest.fixture()
def strategy_universe(chart_pair: TradingPairIdentifier) -> TradingStrategyUniverse:
    """Create a candle-backed universe for the endpoint tests."""
    chain_id = ChainId.ethereum
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=chain_id,
        address=chart_pair.exchange_address,
        exchange_slug="uniswap-v2",
    )
    candles = generate_ohlcv_candles(
        TimeBucket.d1,
        datetime.datetime(2024, 1, 1),
        datetime.datetime(2024, 1, 10),
        start_price=100,
        pair_id=chart_pair.internal_id,
        exchange_id=exchange.exchange_id,
    )
    return create_synthetic_single_pair_universe(
        candles=candles,
        chain_id=chain_id,
        exchange=exchange,
        time_bucket=TimeBucket.d1,
        pair=chart_pair,
    )


def _create_state_for_position_chart(
    chart_pair: TradingPairIdentifier,
    *,
    use_pair_internal_id: bool,
) -> tuple[State, int]:
    """Create a state with one open position, two trades, and statistics."""
    state = State(portfolio=Portfolio())
    trade_pair = copy.deepcopy(chart_pair)

    if not use_pair_internal_id:
        trade_pair.internal_id = None

    opened_at = datetime.datetime(2024, 1, 2)
    reduced_at = datetime.datetime(2024, 1, 5)

    position, buy_trade, _ = state.create_trade(
        strategy_cycle_at=opened_at,
        pair=trade_pair,
        quantity=None,
        reserve=Decimal("1000"),
        assumed_price=100.0,
        trade_type=TradeType.rebalance,
        reserve_currency=trade_pair.quote,
        reserve_currency_price=1.0,
    )
    buy_trade.mark_success(
        executed_at=opened_at,
        executed_price=100.0,
        executed_quantity=Decimal("10"),
        executed_reserve=Decimal("1000"),
        lp_fees=0.0,
        native_token_price=0.0,
        force=True,
    )

    _, sell_trade, _ = state.create_trade(
        strategy_cycle_at=reduced_at,
        pair=trade_pair,
        quantity=Decimal("-4"),
        reserve=None,
        assumed_price=105.0,
        trade_type=TradeType.rebalance,
        reserve_currency=trade_pair.quote,
        reserve_currency_price=1.0,
        position=position,
    )
    sell_trade.mark_success(
        executed_at=reduced_at,
        executed_price=105.0,
        executed_quantity=Decimal("-4"),
        executed_reserve=Decimal("420"),
        lp_fees=0.0,
        native_token_price=0.0,
        force=True,
    )

    position.last_token_price = 106.0
    position.last_pricing_at = reduced_at

    state.stats.add_positions_stats(
        position.position_id,
        PositionStatistics(
            calculated_at=opened_at,
            last_valuation_at=opened_at,
            profitability=0.0,
            profit_usd=0.0,
            quantity=10.0,
            value=1000.0,
            underlying_price=100.0,
        ),
    )
    state.stats.add_positions_stats(
        position.position_id,
        PositionStatistics(
            calculated_at=reduced_at,
            last_valuation_at=reduced_at,
            profitability=0.06,
            profit_usd=60.0,
            quantity=6.0,
            value=636.0,
            underlying_price=106.0,
        ),
    )

    return state, position.position_id


@pytest.fixture()
def create_position_chart_server() -> Callable[[State, TradingStrategyUniverse | None], str]:
    """Create webhook servers for position chart tests and clean them up afterwards."""
    servers: list[WebhookServer] = []

    def _create_server(state: State, universe: TradingStrategyUniverse | None) -> str:
        store = JSONFileStore(str(Path(tempfile.mkdtemp()) / "webhook-position-chart.json"))
        store.sync(state)

        run_state = RunState()
        run_state.read_only_state_copy = store.load()
        run_state.version.tag = "v1"
        run_state.version.commit_message = "Position chart test"

        if universe is not None:
            run_state.latest_indicators = StrategyInputIndicators(
                strategy_universe=universe,
                available_indicators=IndicatorSet(),
                indicator_results={},
            )

        metadata = Metadata(
            "Foobar",
            "Short desc",
            "Long desc",
            None,
            native_datetime_utc_now(),
            True,
            badges=Metadata.parse_badges_configuration("polygon, metamask, eth, usdc"),
            tags={StrategyTag.beta},
            fees=dict(
                management_fee=0.00,
                trading_strategy_protocol_fee=0.02,
                strategy_developer_fee=0.05,
            ),
        )

        notebook_result = Path(tempfile.mkdtemp()) / "test_cli_backtest.ipynb"
        notebook_result.open("wt").write("Foo")
        html_result = Path(tempfile.mkdtemp()) / "test_cli_backtest.html"
        html_result.open("wt").write("Bar")
        metadata.backtest_notebook = notebook_result
        metadata.backtest_html = html_result

        server = create_webhook_server_with_retries("127.0.0.1", "test", "test", Queue(), store, metadata, run_state)
        servers.append(server)
        return get_webhook_test_url(server, "test", "test")

    yield _create_server

    for server in servers:
        server.shutdown()


def test_position_chart_happy_path(
    chart_pair: TradingPairIdentifier,
    strategy_universe: TradingStrategyUniverse,
    create_position_chart_server: Callable[[State, TradingStrategyUniverse | None], str],
) -> None:
    """Position chart returns candles, trades, and statistics for a valid position.

    1. Create a state with one position, executed trades, and stored statistics.
    2. Query the position chart endpoint with candle-backed universe data.
    3. Check the response returns price history, trades, statistics, and no warnings.
    """
    state, position_id = _create_state_for_position_chart(chart_pair, use_pair_internal_id=True)
    server_url = create_position_chart_server(state, strategy_universe)

    # 1. Create a state with one position, executed trades, and stored statistics.

    # 2. Query the position chart endpoint with candle-backed universe data.
    response = requests.get(f"{server_url}/position-chart/{position_id}")

    # 3. Check the response returns price history, trades, statistics, and no warnings.
    assert response.status_code == 200
    payload = response.json()
    assert payload["position_number"] == position_id
    assert len(payload["price_history"]) > 0
    for point in payload["price_history"]:
        assert len(point) == 2
        assert isinstance(point[0], float)
        assert point[0] > 1_700_000_000
        assert isinstance(point[1], float)
        assert point[1] > 0
    assert [trade["trade_id"] for trade in payload["trades"]] == [1, 2]
    assert len(payload["position_statistics"]) == 2
    assert payload["price_history_status_message"] == "Historical price data is available."
    assert payload["warnings"] == []


def test_position_chart_without_candles_returns_partial_data(
    chart_pair: TradingPairIdentifier,
    create_position_chart_server: Callable[[State, TradingStrategyUniverse | None], str],
) -> None:
    """Position chart returns partial data when candle history is unavailable.

    1. Create a state with a valid position, trades, and statistics.
    2. Query the position chart endpoint without loading universe candles.
    3. Check the endpoint still returns position data with empty price history and warnings.
    """
    state, position_id = _create_state_for_position_chart(chart_pair, use_pair_internal_id=True)
    server_url = create_position_chart_server(state, None)

    # 1. Create a state with a valid position, trades, and statistics.

    # 2. Query the position chart endpoint without loading universe candles.
    response = requests.get(f"{server_url}/position-chart/{position_id}")

    # 3. Check the endpoint still returns position data with empty price history and warnings.
    assert response.status_code == 200
    payload = response.json()
    assert payload["price_history"] == []
    assert len(payload["trades"]) == 2
    assert len(payload["position_statistics"]) == 2
    assert any("strategy universe" in warning.lower() for warning in payload["warnings"])


def test_position_chart_without_pair_internal_id_returns_partial_data(
    chart_pair: TradingPairIdentifier,
    strategy_universe: TradingStrategyUniverse,
    create_position_chart_server: Callable[[State, TradingStrategyUniverse | None], str],
) -> None:
    """Position chart returns partial data when the position pair lacks an internal id.

    1. Create a state whose position pair does not have an internal id.
    2. Query the position chart endpoint with universe data present.
    3. Check price history is empty and the warning explains the missing pair id.
    """
    state, position_id = _create_state_for_position_chart(chart_pair, use_pair_internal_id=False)
    server_url = create_position_chart_server(state, strategy_universe)

    # 1. Create a state whose position pair does not have an internal id.

    # 2. Query the position chart endpoint with universe data present.
    response = requests.get(f"{server_url}/position-chart/{position_id}")

    # 3. Check price history is empty and the warning explains the missing pair id.
    assert response.status_code == 200
    payload = response.json()
    assert payload["price_history"] == []
    assert any("internal id" in warning.lower() for warning in payload["warnings"])


def test_position_chart_unknown_position_returns_404(
    chart_pair: TradingPairIdentifier,
    strategy_universe: TradingStrategyUniverse,
    create_position_chart_server: Callable[[State, TradingStrategyUniverse | None], str],
) -> None:
    """Position chart returns a friendly 404 error for an unknown position id.

    1. Create a state with one valid position.
    2. Query the position chart endpoint for a missing position id.
    3. Check the endpoint returns a JSON 404 response with a friendly detail message.
    """
    state, _ = _create_state_for_position_chart(chart_pair, use_pair_internal_id=True)
    server_url = create_position_chart_server(state, strategy_universe)

    # 1. Create a state with one valid position.

    # 2. Query the position chart endpoint for a missing position id.
    response = requests.get(f"{server_url}/position-chart/9999")

    # 3. Check the endpoint returns a JSON 404 response with a friendly detail message.
    assert response.status_code == 404
    payload = response.json()
    assert "not found" in payload["detail"].lower()


def test_position_chart_invalid_position_number_returns_400(
    chart_pair: TradingPairIdentifier,
    strategy_universe: TradingStrategyUniverse,
    create_position_chart_server: Callable[[State, TradingStrategyUniverse | None], str],
) -> None:
    """Position chart returns a friendly 400 error for a bad position number.

    1. Create a state with one valid position.
    2. Query the position chart endpoint with a non-integer path value.
    3. Check the endpoint returns a JSON 400 response with a friendly detail message.
    """
    state, _ = _create_state_for_position_chart(chart_pair, use_pair_internal_id=True)
    server_url = create_position_chart_server(state, strategy_universe)

    # 1. Create a state with one valid position.

    # 2. Query the position chart endpoint with a non-integer path value.
    response = requests.get(f"{server_url}/position-chart/not-a-number")

    # 3. Check the endpoint returns a JSON 400 response with a friendly detail message.
    assert response.status_code == 400
    payload = response.json()
    assert "integer" in payload["detail"].lower()
