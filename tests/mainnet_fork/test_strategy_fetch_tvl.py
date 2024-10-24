"""Real-time TVL fetching.

- Run decide_trades() with some pairs from Ethereum mainnet
"""

import os

import pandas as pd
import pytest

from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.ethereum.tvl import fetch_quote_token_tvls
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import unit_test_trading_execution_context
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_ETHEREUM") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture
def web3():
    return create_multi_provider_web3(os.environ["JSON_RPC_ETHEREUM"])


@pytest.fixture()
def strategy_universe(
    chain_id,
    persistent_test_client
) -> TradingStrategyUniverse:
    """Load volatile tokens + exchange rate pair."""

    # Load a mixed bag of Uni v2 and v3 pairs
    pairs = [
        (ChainId.ethereum, "uniswap-v2", "WBTC", "USDC"),
        (ChainId.ethereum, "uniswap-v2", "PEPE", "WETH"),
        (ChainId.ethereum, "uniswap-v3", "SPH", "USDT"),
        (ChainId.ethereum, "uniswap-v3", "ezETH", "WETH"),
        (ChainId.ethereum, "uniswap-v3", "USDT", "USDC"),
        (ChainId.ethereum, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_trading_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=UniverseOptions(history_period=pd.Timedelta("14d")),
    )

    usdc = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
    return TradingStrategyUniverse.create_from_dataset(dataset, usdc)


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """Read pair TVLs.

    - Doesn't actually trade any
    """

    all_pairs = list(input.strategy_universe.iterate_pairs())
    pair_tvls = fetch_quote_token_tvls(
        input.web3,
        input.strategy_universe,
        all_pairs,
    )
    for pair in all_pairs:
        tvl = pair_tvls[pair]
        # print(f"Pair {pair.get_ticker()}, quote token TVL {tvl}")
        assert pair_tvls[pair] > 0

    return []



def test_fetch_tvl_in_decide_trades(
    web3,
    strategy_universe,
):
    """Check that we can directly get real-time TVL in the decide_trades().

    - Run a single cycle of decide_trades
    """

    strategy_input = StrategyInput(
        cycle=1,
        strategy_universe=strategy_universe,
        timestamp=pd.Timestamp.utcnow(),
        parameters=StrategyParameters({}),
        state=None,
        indicators=None,
        pricing_model=None,
        execution_context=unit_test_trading_execution_context,
        other_data={},
        web3=web3,
    )

    decide_trades(strategy_input)


