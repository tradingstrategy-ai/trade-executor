"""Minimal test strategy for Derive CLI integration.

A simple strategy that does nothing but runs through init/start CLI commands.
Uses Anvil chain and deployed USDC for testing.

The Derive subaccount ID is discovered automatically from the Derive API
at universe creation time. When running inside a Lagoon vault deployment,
the Safe address is used as the Derive wallet address.

Environment variables:
- TEST_USDC_ADDRESS: Address of the USDC token contract deployed on Anvil
"""

import datetime
import logging
import os

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.derive import (
    create_derive_exchange_account_pair,
    discover_derive_subaccount_id,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.ignore
reserve_currency = ReserveCurrency.usdc

# Chain ID for local Anvil testing
CHAIN_ID = ChainId.anvil


class Parameters:
    """Strategy parameters."""
    chain_id = ChainId.anvil
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.ignore
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create a minimal trading universe.

    Has USDC as reserve and a Derive exchange account pair.
    Discovers the real subaccount ID from the Derive API.
    """
    logger = logging.getLogger(__name__)

    usdc_address = os.environ.get("TEST_USDC_ADDRESS", "0x0000000000000000000000000000000000000001")
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=usdc_address,
        token_symbol="USDC",
        decimals=6,
    )

    # Get wallet address from Lagoon vault's Safe multisig when available
    execution_model = input.execution_model
    wallet_address = None
    if execution_model and hasattr(execution_model, "vault"):
        wallet_address = execution_model.vault.safe_address
        logger.info("Using Safe address as Derive wallet: %s", wallet_address)

    is_testnet = os.environ.get("DERIVE_NETWORK", "mainnet") == "testnet"
    subaccount_id = discover_derive_subaccount_id(wallet_address=wallet_address)

    derive_account_pair = create_derive_exchange_account_pair(
        quote=usdc,
        subaccount_id=subaccount_id,
        is_testnet=is_testnet,
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [derive_account_pair])

    derive_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="anvil",
        exchange_id=1,
        exchange_slug="derive",
        address=derive_account_pair.exchange_address,
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={derive_exchange},
        pairs=pair_universe,
        candles=None,
        liquidity=None,
    )

    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[usdc],
    )


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
):
    """No indicators needed."""
    pass


def decide_trades(
    input: StrategyInput,
) -> list[TradeExecution]:
    """No trades - just a passive strategy."""
    return []
