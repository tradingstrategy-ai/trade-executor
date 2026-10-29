"""Static Arbitrum universe for the manual Lagoon ERC-4626 CLI tutorial.

These are representatives, not a recommended allocation. The generic case uses
Euler's standard ERC-4626 entry points; other cases retain their selected reader.
Admission and liquidity are checked on chain when each test is run. No daemon is
needed: the tutorial drives the public commands directly.
"""

import datetime

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.erc_4626.core import ERC4626Feature, GENERIC_ERC4626_PROTOCOL_SLUG
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
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
from tradeexecutor.strategy.reverse_universe import create_universe_from_trading_pair_identifiers
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse

trading_strategy_engine_version = "0.5"

# label, address, share symbol, share decimals, selected reader features
SOURCE_VAULTS = [
    ("generic", "0x05d28a86e057364f6ad1a88944297e58fc6160b3", "eUSDC-5", 6, []),
    ("morpho_v1", "0x4b6f1c9e5d470b97181786b26da0d0945a7cf027", "hyperUSDCa", 18, [ERC4626Feature.morpho_like]),
    ("morpho_v2", "0x55a2b207b0074e13adcb858950a81b7a04775e0f", "gtusdcb", 18, [ERC4626Feature.morpho_v2_like]),
    ("yearn_v3", "0x2b0b6376083c6e1f376c7439f328436a673f333c", "yPT-aUSDC", 6, [ERC4626Feature.yearn_v3_like]),
    ("compounder", "0xcacc53bacce744ac7b5c1ec7eb7e3ab01330733b", "ysUSDC", 6, [ERC4626Feature.yearn_compounder_like]),
]


class Parameters:
    """One small position at a time, using Arbitrum native USDC."""

    chain_id = ChainId.arbitrum
    cycle_duration = CycleDuration.cycle_1d
    routing = TradeRouting.default
    required_history_period = datetime.timedelta(days=1)
    slippage_tolerance = 0.005
    initial_cash = 100
    backtest_start = None
    backtest_end = None


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Declare exact targets without downloading a market dataset or reading RPC settings."""
    chain = input.parameters.chain_id
    usdc = AssetIdentifier(chain.value, USDC_NATIVE_TOKEN[chain.value], "USDC", 6)
    pairs = []
    for index, (label, address, symbol, decimals, features) in enumerate(SOURCE_VAULTS, 1):
        protocol = GENERIC_ERC4626_PROTOCOL_SLUG if not features else label
        metadata = VaultMetadata(vault_name=label, protocol_name=protocol, protocol_slug=protocol, features=list(features))
        pairs.append(TradingPairIdentifier(
            base=AssetIdentifier(chain.value, address, symbol, decimals),
            quote=usdc,
            pool_address=address,
            exchange_address=ZERO_ADDRESS_STR,
            internal_id=index,
            internal_exchange_id=1,
            kind=TradingPairKind.vault,
            fee=0,
            reverse_token_order=False,
            exchange_name=label,
            other_data={"vault_features": list(features), "vault_protocol": protocol, "token_metadata": metadata},
        ))
    exchange = Exchange(chain_id=chain, chain_slug=chain.get_slug(), exchange_id=1, exchange_slug="vault", address=ZERO_ADDRESS_STR, exchange_type=ExchangeType.erc_4626_vault, pair_count=len(pairs))
    exchanges = ExchangeUniverse.from_collection([exchange])
    pair_universe = create_universe_from_trading_pair_identifiers(pairs, exchanges)
    return TradingStrategyUniverse(data_universe=Universe(time_bucket=TimeBucket.not_applicable, chains={chain}, exchange_universe=exchanges, pairs=pair_universe), reserve_assets=[usdc])


def create_indicators(parameters: StrategyParameters, indicators: IndicatorSet, strategy_universe: TradingStrategyUniverse, execution_context: ExecutionContext) -> None:
    """No market indicators are needed for manual custody tests."""


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """The tutorial issues the trades; starting a daemon must not allocate capital."""
    return []
