"""Minimal test strategy for GMX Lagoon vault integration.

A simple strategy that does nothing but runs through init/start CLI commands.
Creates a GMX exchange account position that tracks the NAV of the vault's
GMX perpetuals positions.

The Safe address is resolved at runtime from the execution model's
transaction builder, not stored in the pair or environment.

Example: deploy a Lagoon vault for this strategy on a simulated Anvil fork
--------------------------------------------------------------------------

Uses the strategy-universe-based deployment path which is required for GMX
whitelisting. The key is to provide ``--strategy-file`` without
``--denomination-asset`` — this triggers ``_deploy_multichain()`` which reads
the trading universe from the strategy, detects GMX pairs, and whitelists
GMX router addresses and markets in the guard contract.

.. code-block:: shell

    STRATEGY_FILE=strategies/test_only/minimal_gmx_strategy.py \
    SIMULATE=true \
    PRIVATE_KEY=$PRIVATE_KEY \
    VAULT_RECORD_FILE=/tmp/vault-record.json \
    FUND_NAME="GMX Test Vault" \
    FUND_SYMBOL="GMX-TV" \
    ANY_ASSET=true \
    LOG_LEVEL=info \
        trade-executor lagoon-deploy-vault


"""

import datetime
import os

from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.exchange_account.gmx import create_gmx_exchange_account_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import \
    CreateTradingUniverseInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse, create_pair_universe_from_code)

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

# Detect testnet from environment
_GMX_NETWORK = os.environ.get("GMX_NETWORK", "mainnet")
_IS_TESTNET = _GMX_NETWORK == "testnet"

if _IS_TESTNET:
    # Arbitrum Sepolia testnet — use native testnet USDC as vault denomination
    # (GMX uses USDC.SG for collateral, but the exchange account value func
    # returns USD values regardless of the vault's reserve token)
    CHAIN_ID = ChainId.arbitrum_sepolia
else:
    # Arbitrum mainnet
    CHAIN_ID = ChainId.arbitrum

USDC_ADDRESS = USDC_NATIVE_TOKEN[CHAIN_ID.value]


class Parameters:
    """Strategy parameters."""
    chain_id = CHAIN_ID
    initial_cash = 100_000
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    backtest_start = None
    backtest_end = None


def create_trading_universe(
    input: CreateTradingUniverseInput,
) -> TradingStrategyUniverse:
    """Create a minimal trading universe with a GMX exchange account pair.

    Has USDC as reserve and a GMX exchange account pair.
    """
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )

    is_testnet = os.environ.get("GMX_NETWORK", "mainnet") == "testnet"

    gmx_account_pair = create_gmx_exchange_account_pair(
        quote=usdc,
        is_testnet=is_testnet,
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [gmx_account_pair])

    # Use ExchangeType.derive as placeholder — routing identifies exchange
    # accounts via other_data["exchange_protocol"], not exchange type
    chain_slug = "arbitrum_sepolia" if _IS_TESTNET else "arbitrum"
    gmx_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug=chain_slug,
        exchange_id=1,
        exchange_slug="gmx",
        address=gmx_account_pair.exchange_address,
        exchange_type=ExchangeType.derive,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={gmx_exchange},
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
    """Create the exchange account position on the first cycle, then do nothing.

    The position is created by calling ``open_exchange_account_position()``
    which creates a spoofed trade that never goes through routing/execution.
    """
    from decimal import Decimal

    from tradeexecutor.exchange_account.state import \
        open_exchange_account_position

    state = input.state
    timestamp = input.timestamp

    # Check if position already exists
    for pos in state.portfolio.open_positions.values():
        if pos.is_exchange_account():
            return []

    # First cycle: create exchange account position
    pair = next(input.strategy_universe.iterate_pairs())
    reserve = input.strategy_universe.reserve_assets[0]

    open_exchange_account_position(
        state=state,
        strategy_cycle_at=timestamp.to_pydatetime(),
        pair=pair,
        reserve_currency=reserve,
        reserve_amount=Decimal(0),
        notes="GMX exchange account position",
    )

    return []
