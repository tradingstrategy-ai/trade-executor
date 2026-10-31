"""Fixed Upshift universe for guarded public-command integration tests.

Sentora has a separate LP token and a queued withdrawal; Core USDC uses the
legacy zero-lag request path. Normal executor managers and persisted state own
all execution and settlement. No external market-data bundle is needed.
"""

import datetime

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.core import ERC4626Feature
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

    id = "upshift"
    chain_id = ChainId.ethereum
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
    usdc = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    pairs = []
    for index, (address, share, name, lag) in enumerate((
        ("0x74aD2F789Ed583DBd141bbdafC673fE1F033718b", "0xe8aA1A9eC6b9BC455D8f33E4BDC685dedFf82407", "Sentora USD", 3),
        ("0xE9B725010A9E419412ed67d0fA5f3A5f40159D32", "0xE9B725010A9E419412ed67d0fA5f3A5f40159D32", "Core USDC", 0),
    ), 1):
        features = [ERC4626Feature.upshift_like]
        if lag:
            features.append(ERC4626Feature.upshift_multi_asset_like)
        pairs.append(TradingPairIdentifier(
            base=AssetIdentifier(1, share, name, 6), quote=usdc,
            pool_address=address, exchange_address=ZERO_ADDRESS_STR,
            internal_id=index, internal_exchange_id=1, fee=0,
            kind=TradingPairKind.vault,
            other_data={"token_metadata": VaultMetadata(
                vault_name=name, protocol_name="Upshift", protocol_slug="upshift",
                features=features, lockup_days=lag, performance_fee=0, management_fee=0,
            )},
        ))
    exchange = Exchange(
        chain_id=ChainId.ethereum,
        chain_slug="ethereum",
        exchange_id=1,
        exchange_slug="upshift",
        address=ZERO_ADDRESS_STR,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=len(pairs),
    )
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={ChainId.ethereum},
        exchanges={exchange},
        pairs=create_pair_universe_from_code(ChainId.ethereum, pairs),
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
