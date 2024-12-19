"""Real-time TVL fetching on Base.

- Run decide_trades() with some pairs from Ethereum mainnet
"""

import os

import pandas as pd
import pytest

from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.tvl import fetch_quote_token_tvls, CurrencyConversionRateMissing
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import unit_test_trading_execution_context
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket


pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_BASE") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_BASE and TRADING_STRATEGY_API_KEY environment variables to run this test")


@pytest.fixture
def web3():
    return create_multi_provider_web3(os.environ["JSON_RPC_BASE"])


@pytest.fixture()
def strategy_universe(
    chain_id,
    persistent_test_client
) -> TradingStrategyUniverse:
    """Load volatile tokens + exchange rate pair."""

    # Load a mixed bag of Uni v2 and v3 pairs
    pairs = [
        (ChainId.base, "uniswap-v3", "WETH", "USDC", 0.0005),
        (ChainId.base, "uniswap-v3", "FAI", "WETH"),
        (ChainId.base, "uniswap-v2", "WETH", "USDC", 0.0030),
        (ChainId.base, "uniswap-v2", "KEYCAT", "WETH"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_trading_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=UniverseOptions(history_period=pd.Timedelta("14d")),
    )

    usdc = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
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
        assert pair_tvls[pair] > 0

    for pair in all_pairs:
        tvl = input.pricing_model.get_usd_tvl(input.timestamp, pair)
        assert tvl > 0

    return []


@pytest.fixture()
def pricing_model(web3, strategy_universe):
    pair_configurator = EthereumPairConfigurator(
        web3,
        strategy_universe,
    )
    return GenericPricing(
        pair_configurator,
    )


def test_fetch_tvl_base(
    web3,
    strategy_universe,
    pricing_model,
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
        pricing_model=pricing_model,
        execution_context=unit_test_trading_execution_context,
        other_data={},
        web3=web3,
    )

    decide_trades(strategy_input)

