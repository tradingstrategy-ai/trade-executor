"""Dummy Atoma strategy for the real Typer start/Anvil integration test.

Invest half the initial cash equally in Index and RWA, then redeem each once
the shared pricing admission allows it. Pending tickets are left to the normal
executor resolver; closed positions are never reopened. The harness alone
controls time and the operator. See .claude/docs/vault-deposit-redeem.md and
tests/erc_4626/test_vault_async_lagoon_erc_7540.py for the shared mechanism.
"""

import datetime

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault_protocol.atoma.vault import ATOMA_VAULT_ADDRESSES
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code


trading_strategy_engine_version = "0.5"


class Parameters:
    """Normal strategy allocation and engine settings, without harness controls."""

    id = "atoma-async"
    chain_id = ChainId.arbitrum
    cycle_duration = CycleDuration.cycle_1s
    routing = TradeRouting.default
    initial_cash = 400
    backtest_start = None
    backtest_end = None
    required_history_period = datetime.timedelta(0)
    allocation = 0.5
    slippage_tolerance = 0.01


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Bind the two reviewed deployments without a market-data download."""
    usdc = AssetIdentifier(42161, USDC_NATIVE_TOKEN[42161], "USDC", 6)
    pairs = [
        TradingPairIdentifier(
            base=AssetIdentifier(42161, address, f"Atoma-{index}", 12),
            quote=usdc,
            pool_address=address,
            exchange_address=ZERO_ADDRESS_STR,
            internal_id=index,
            internal_exchange_id=1,
            fee=0,
            kind=TradingPairKind.vault,
            other_data={"token_metadata": VaultMetadata(
                vault_name=f"Atoma-{index}", protocol_name="Atoma", protocol_slug="atoma",
                features=[ERC4626Feature.atoma_like],
            )},
        )
        for index, address in enumerate(sorted(ATOMA_VAULT_ADDRESSES), 1)
    ]
    exchange = Exchange(
        chain_id=ChainId.arbitrum,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="atoma",
        address=ZERO_ADDRESS_STR,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=len(pairs),
    )
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.arbitrum},
        exchanges={exchange},
        pairs=create_pair_universe_from_code(ChainId.arbitrum, pairs),
    )
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[usdc])


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """Vault execution needs no candle indicators."""


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Create one deposit and redemption per vault using persisted portfolio state."""
    pm = input.get_position_manager()
    pairs = list(input.strategy_universe.iterate_pairs())
    if not list(input.state.portfolio.get_all_trades()):
        amount = input.state.portfolio.get_cash() * input.parameters.allocation / len(pairs)
        return [trade for pair in pairs for trade in pm.open_spot(pair, value=amount)]
    trades = []
    for pair in pairs:
        position = pm.get_current_position_for_pair(pair)
        if position is None or position.has_pending_vault_settlement():
            continue
        if input.pricing_model.check_redemption(input.timestamp, pair, position=position).can_redeem:
            trades.extend(pm.close_position(position))
    return trades
