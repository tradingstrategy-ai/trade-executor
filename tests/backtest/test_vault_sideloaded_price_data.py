"""Create a trading universe and a simple vault rebalance backtest.
"""
import datetime


from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.execution_context import unit_test_execution_context, ExecutionContext
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.alternative_data.vault import load_multiple_vaults, load_vault_price_data, convert_vault_prices_to_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.liquidity import GroupedLiquidityUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


class Parameters:
    id = "vault-optimiser"
    candle_time_bucket = TimeBucket.d1
    cycle_duration = CycleDuration.cycle_1d
    chain_id = ChainId.base
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 1, 10)


#
VAULTS = [
    (ChainId.base, "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"),  # Ipor Base
    (ChainId.base, "0xad20523a7dc37babc1cc74897e4977232b3d02e5"),  # Gains Network
    (ChainId.base, "0xcddcdd18a16ed441f6cb10c3909e5e7ec2b9e8f3"),  # Apostro Resolv USDC
    (ChainId.base, "0xc0c5689e6f4d256e861f65465b691aeecc0deb12"),  # Gauntled USDC core
    (ChainId.base, "0xb99b6df96d4d5448cc0a5b3e0ef7896df9507cf5"),  # 40 acres
    # https://summer.fi/earn/base/position/0x98c49e13bf99d7cad8069faa2a370933ec9ecf17
    (ChainId.base, "0x98c49e13bf99d7cad8069faa2a370933ec9ecf17"),  # Summer.fi lazy vault
    # https://app.morpho.org/base/vault/0x50b5b81Fc8B1f1873Ec7F31B0E98186ba008814D/indefi-usdc
    (ChainId.base, "0x50b5b81fc8b1f1873ec7f31b0e98186ba008814d"),  # InDefi USDc on Morpho
]


def create_trading_universe(
    timestamp: datetime.datetime,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Create a trading universe with named vaults on Base."""
    chain_id = Parameters.chain_id
    time_bucket = Parameters.candle_time_bucket

    exchanges, pairs_df = load_multiple_vaults(VAULTS)
    vault_prices_df = load_vault_price_data(pairs_df)

    # Create pair universe based on the vault data
    exchange_universe = ExchangeUniverse({e.exchange_id: e for e in exchanges})
    pair_universe = PandasPairUniverse(pairs_df, exchange_universe=exchange_universe)

    # Create price candles from vault share price scrape
    candle_df, liquidity_df = convert_vault_prices_to_candles(vault_prices_df, "1h")
    candle_universe = GroupedCandleUniverse(candle_df, time_bucket=TimeBucket.h1)
    liquidity_universe = GroupedLiquidityUniverse(liquidity_df, time_bucket=TimeBucket.h1)

    data_universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchange_universe=exchange_universe,
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=liquidity_universe,
    )

    for pair in pair_universe.iterate_pairs():
        print(pair, pair.quote_token_address)

    usdc_token = pair_universe.get_token("0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913".lower(), chain_id)
    assert usdc_token is not None
    usdc = translate_token(usdc_token)

    strategy_universe = TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets=[usdc],
    )

    return strategy_universe


def test_create_vault_universe(
    persistent_test_client: Client,
):
    """Create trading universe using fetch_tvl(min_tvl=...) filter."""
    client = persistent_test_client

    universe = create_trading_universe(
        None,
        client=client,
        execution_context=unit_test_execution_context,
        universe_options=UniverseOptions.from_strategy_parameters_class(Parameters, unit_test_execution_context)
    )

    # We have liquidity data correctly loaded
    pair = universe.get_pair_by_address("0x50b5b81fc8b1f1873ec7f31b0e98186ba008814d")
    assert pair.base.token_symbol == "indeUSDC"

