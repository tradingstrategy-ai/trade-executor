"""Minimal test strategy for Hypercore vault Lagoon integration.

A simple strategy that creates a Hypercore vault position on the first
cycle and does nothing afterwards. The position tracks the NAV of the
vault's equity via the Hyperliquid info API.

Deposits and withdrawals happen externally via CoreWriter actions — this
strategy only creates the tracking position and lets the valuation model
keep its price in sync.

Example: deploy a Lagoon vault for this strategy
-------------------------------------------------

.. code-block:: shell

    CHAIN_NAME=hyperliquid \\
    STRATEGY_FILE=strategies/test_only/minimal_hyperliquid_strategy.py \\
    SIMULATE=true \\
    PRIVATE_KEY=$PRIVATE_KEY \\
    VAULT_RECORD_FILE=/tmp/vault-record.json \\
    FUND_NAME="Hypercore HLP Vault" \\
    FUND_SYMBOL="HLP-V" \\
    ANY_ASSET=true \\
    LOG_LEVEL=info \\
        trade-executor lagoon-deploy-vault
"""

import datetime
import os

from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import (
    CreateTradingUniverseInput,
)
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.strategy_type import StrategyType
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
)

trading_strategy_engine_version = "0.5"
trading_strategy_type = StrategyType.managed_positions
trading_strategy_cycle = CycleDuration.cycle_1d
trade_routing = TradeRouting.default
reserve_currency = ReserveCurrency.usdc

# Detect testnet from environment
_NETWORK = os.environ.get("HYPERLIQUID_NETWORK", "mainnet")
_IS_TESTNET = _NETWORK == "testnet"

# HyperEVM mainnet = 999, testnet = 998
if _IS_TESTNET:
    CHAIN_ID = ChainId(998)
else:
    CHAIN_ID = ChainId.hyperliquid

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
    """Create a minimal trading universe with a Hypercore vault pair.

    Has USDC as reserve and a Hypercore vault pair (HLP by default).
    """
    usdc = AssetIdentifier(
        chain_id=CHAIN_ID.value,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )

    is_testnet = os.environ.get("HYPERLIQUID_NETWORK", "mainnet") == "testnet"
    vault_address = HLP_VAULT_ADDRESS["testnet" if is_testnet else "mainnet"]

    hypercore_vault_pair = create_hypercore_vault_pair(
        quote=usdc,
        vault_address=vault_address,
        is_testnet=is_testnet,
    )

    pair_universe = create_pair_universe_from_code(CHAIN_ID, [hypercore_vault_pair])

    # Use ExchangeType.erc_4626_vault since this IS a vault position
    hypercore_exchange = Exchange(
        chain_id=CHAIN_ID,
        chain_slug="hyperliquid_testnet" if is_testnet else "hyperliquid",
        exchange_id=1,
        exchange_slug="hypercore",
        address=hypercore_vault_pair.exchange_address,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=1,
    )

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={CHAIN_ID},
        exchanges={hypercore_exchange},
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
    """Create the Hypercore vault position on the first cycle, then do nothing.

    The position is created by calling ``open_hypercore_vault_position()``
    which creates a spoofed trade that never goes through routing/execution.
    """
    from tradeexecutor.ethereum.vault.hypercore_vault import (
        open_hypercore_vault_position,
    )

    state = input.state
    timestamp = input.timestamp

    # Check if vault position already exists
    for pos in state.portfolio.open_positions.values():
        if pos.is_vault() and pos.pair.other_data.get("vault_protocol") == "hypercore":
            return []

    # First cycle: create Hypercore vault position
    pair = next(input.strategy_universe.iterate_pairs())
    reserve = input.strategy_universe.reserve_assets[0]

    open_hypercore_vault_position(
        state=state,
        strategy_cycle_at=timestamp.to_pydatetime(),
        pair=pair,
        reserve_currency=reserve,
        notes="Hypercore HLP vault position",
    )

    return []
