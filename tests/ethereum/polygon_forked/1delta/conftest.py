
import pytest
import pandas as pd

from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.lending import LendingProtocolType

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, load_partial_data
from tradeexecutor.strategy.execution_context import python_script_execution_context, unit_test_execution_context
from tradeexecutor.strategy.universe_model import default_universe_options
from tradeexecutor.ethereum.universe import create_exchange_universe, create_pair_universe


@pytest.fixture
def exchange_universe(web3, uniswap_v3_deployment: UniswapV3Deployment) -> ExchangeUniverse:
    """We trade on one uniswap v3 deployment on tester."""
    return create_exchange_universe(web3, [uniswap_v3_deployment])


@pytest.fixture
def pair_universe(web3, exchange_universe: ExchangeUniverse, weth_usdc_spot_pair) -> PandasPairUniverse:
    exchange = next(iter(exchange_universe.exchanges.values()))
    return create_pair_universe(web3, exchange, [weth_usdc_spot_pair])


@pytest.fixture
def trading_strategy_universe(chain_id, exchange_universe, pair_universe, asset_usdc, persistent_test_client) -> TradingStrategyUniverse:

    pairs = [
        (ChainId.polygon, "uniswap-v3", "WETH", "USDC", 0.0005),
    ]

    reverses = [
        (ChainId.polygon, LendingProtocolType.aave_v3, "WETH"),
        (ChainId.polygon, LendingProtocolType.aave_v3, "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"),
    ]

    dataset = load_partial_data(
        persistent_test_client,
        execution_context=unit_test_execution_context,
        time_bucket=TimeBucket.d1,
        pairs=pairs,
        universe_options=default_universe_options,
        start_at=pd.Timestamp("2023-12-01"),
        end_at=pd.Timestamp("2023-12-30"),
        lending_reserves=reverses,
    )

    # Convert loaded data to a trading pair universe
    return TradingStrategyUniverse.create_single_pair_universe(dataset)