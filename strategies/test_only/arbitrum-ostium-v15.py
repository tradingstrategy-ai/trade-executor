"""Test strategy for Ostium V1.5 async vault deposit/redeem.

- Opens a position on cycle 1 (deposit 50 USDC into Ostium)
- Closes the position on cycle 2 (redeem all OLP shares)

Used by integration tests that drive the executor tick-by-tick
with forced Ostium settlement between cycles.

This strategy exercises the async vault flow:
- open_spot() → requestDeposit() → vault_settlement_pending
- close_all() → requestWithdraw() → vault_settlement_pending
- Settlement retry at next tick resolves both
"""
import datetime

from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradingstrategy.alternative_data.vault import load_single_vault
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.pair import PandasPairUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


trading_strategy_engine_version = "0.5"


# Ostium OLP vault on Arbitrum (V1.5 with async deposit/redeem)
OSTIUM_VAULT_ADDRESS = "0x20d419a8e12c45f88fda7c5760bb6923cee27f98"


class Parameters:
    id = "arbitrum-ostium-v15-test"
    cycle_duration = CycleDuration.cycle_1s
    candle_time_bucket = TimeBucket.h1
    chain_id = ChainId.arbitrum
    routing = TradeRouting.default
    slippage_tolerance = 0.05
    backtest_start = datetime.datetime(2025, 1, 1)
    backtest_end = datetime.datetime(2025, 6, 1)
    initial_cash = 500
    required_history_period = datetime.timedelta(days=1)
    deposit_value = 50.0


def create_trading_universe(
    timestamp,
    client: Client,
    execution_context: ExecutionContext,
    universe_options: UniverseOptions,
) -> TradingStrategyUniverse:
    """Load the Ostium vault as a single-pair universe.

    The vault pair data is loaded from the bundled vault metadata database.
    No OHLCV candle data is needed — vault strategies use share price only.
    """
    # Load Ostium vault metadata from bundled data (no API call needed)
    vault_exchanges, vault_pairs_df = load_single_vault(
        ChainId.arbitrum,
        OSTIUM_VAULT_ADDRESS,
    )

    exchange_universe = ExchangeUniverse.from_collection(vault_exchanges)
    pair_universe = PandasPairUniverse(vault_pairs_df, exchange_universe=exchange_universe)

    # Universe.exchanges (Set[Exchange]) must be set explicitly —
    # without it, pretick_check() fails with "len(None)" because it checks
    # data_universe.exchanges which defaults to None if not provided.
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.arbitrum},
        pairs=pair_universe,
        exchange_universe=exchange_universe,
        exchanges=set(vault_exchanges),
    )

    # USDC on Arbitrum — the vault's denomination/reserve token.
    # Hardcoded because load_single_vault() doesn't provide a reserve asset directly.
    reserve_asset = AssetIdentifier(
        chain_id=42161,
        address="0xaf88d065e77c8cC2239327C5EDb3A432268e5831",
        decimals=6,
        token_symbol="USDC",
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[reserve_asset],
    )


def create_indicators(timestamp, parameters, strategy_universe, execution_context):
    """No indicators needed — vault strategies don't use technical indicators."""
    return IndicatorSet()


def decide_trades(input: StrategyInput) -> list:
    """Simple two-cycle strategy: deposit on cycle 1, redeem on cycle 2.

    On Anvil, each trade will enter vault_settlement_pending state.
    The settlement retry module resolves them at the start of the next tick
    (after force_ostium_v15_settlement() is called externally by the test).
    """
    position_manager = input.get_position_manager()
    parameters = input.parameters
    pair = input.get_default_pair()

    # Cycle 1: no open positions → deposit into vault
    if not position_manager.is_any_open():
        trades = position_manager.open_spot(
            pair,
            value=parameters.deposit_value,
        )
        return trades

    # Cycle 2+: position exists → redeem all shares from vault
    trades = position_manager.close_all()
    return trades
