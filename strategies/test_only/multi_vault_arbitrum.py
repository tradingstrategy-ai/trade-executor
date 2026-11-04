"""Arbitrum hub vault trading a Lighter pool, Atoma and Upshift.

A fixed, code-built universe for the multi-vault manual tutorial
``scripts/lagoon/manual-multi-vault-cli.sh`` and its tests. The Lagoon vault
lives on Arbitrum; Atoma Index is on the hub, while Upshift Core USDC and the
pmalt Lighter public pool are reached through the Ethereum satellite Safe,
funded by the CCTP bridge pair. No market-data download is needed.

The pool is pmalt, ``lighter-pool-281474976552918``. Steady Wealth Builder,
``lighter-pool-281474976552443``, is the documented fallback; replace the
pool index here deliberately, never automatically. See
``docs/plans/arbitrum-multi-vault-live-plan.md``.

``decide_trades()`` opens each target once and closes it once its existing
redemption check allows, leaving pending settlements and pool phases to the
normal resolver. Positions opened by ``perform-test-trade`` are left to that
command.
"""

import datetime

from eth_defi.abi import ZERO_ADDRESS_STR
from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.token import USDC_NATIVE_TOKEN
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradingstrategy.vault import VaultMetadata

from tradeexecutor.ethereum.lighter.lighter_pool import create_lighter_pool_pair
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
    """Normal allocation and engine settings, without deployment or harness values."""

    id = "multi-vault-arbitrum"
    chain_id = ChainId.arbitrum
    cycle_duration = CycleDuration.cycle_1s
    routing = TradeRouting.default
    initial_cash = 131
    backtest_start = None
    backtest_end = None
    required_history_period = datetime.timedelta(0)
    #: USDC per target opened by decide_trades(); Atoma's deposit minimum is 100 USDC.
    allocations = {
        "atoma": 100,
        "upshift": 5,
        "lighter": 5,
    }
    slippage_tolerance = 0.01


def create_trading_universe(input: CreateTradingUniverseInput) -> TradingStrategyUniverse:
    """Bind the reviewed deployments and the Arbitrum→Ethereum bridge."""
    arbitrum, ethereum = ChainId.arbitrum, ChainId.ethereum
    arb_usdc = AssetIdentifier(arbitrum.value, USDC_NATIVE_TOKEN[arbitrum.value], "USDC", 6)
    eth_usdc = AssetIdentifier(ethereum.value, USDC_NATIVE_TOKEN[ethereum.value], "USDC", 6)

    bridge = TradingPairIdentifier(
        base=eth_usdc,
        quote=arb_usdc,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge",
        other_data={"bridge_protocol": "cctp", "destination_chain_id": ethereum.value},
    )
    atoma_address = "0xcc56410e1a136af0eceb7241c6ae394f4d8b581c"
    atoma = TradingPairIdentifier(
        base=AssetIdentifier(arbitrum.value, atoma_address, "Atoma-Index", 12),
        quote=arb_usdc,
        pool_address=atoma_address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=2,
        internal_exchange_id=2,
        fee=0,
        kind=TradingPairKind.vault,
        other_data={"token_metadata": VaultMetadata(
            vault_name="Atoma Index", protocol_name="Atoma", protocol_slug="atoma",
            features=[ERC4626Feature.atoma_like],
        )},
    )
    upshift_address = "0xE9B725010A9E419412ed67d0fA5f3A5f40159D32"
    upshift = TradingPairIdentifier(
        base=AssetIdentifier(ethereum.value, upshift_address, "Core USDC", 6),
        quote=eth_usdc,
        pool_address=upshift_address,
        exchange_address=ZERO_ADDRESS_STR,
        internal_id=3,
        internal_exchange_id=3,
        fee=0,
        kind=TradingPairKind.vault,
        other_data={"token_metadata": VaultMetadata(
            vault_name="Core USDC", protocol_name="Upshift", protocol_slug="upshift",
            features=[ERC4626Feature.upshift_like], lockup_days=0, performance_fee=0, management_fee=0,
        )},
    )
    pool = create_lighter_pool_pair(
        281474976552918,
        internal_id=4,
        internal_exchange_id=4,
        name="pmalt",
        operator_fee=0.20,
    )

    def exchange(exchange_id: int, chain: ChainId, slug: str, address: str) -> Exchange:
        return Exchange(
            chain_id=chain,
            chain_slug=chain.get_slug(),
            exchange_id=exchange_id,
            exchange_slug=slug,
            address=address,
            exchange_type=ExchangeType.erc_4626_vault if slug != "cctp-bridge" else ExchangeType.uniswap_v2,
            pair_count=1,
        )

    exchanges = {
        exchange(1, arbitrum, "cctp-bridge", TOKEN_MESSENGER_V2),
        exchange(2, arbitrum, "atoma", ZERO_ADDRESS_STR),
        exchange(3, ethereum, "upshift", ZERO_ADDRESS_STR),
        exchange(4, ChainId.lighter, "lighter", ZERO_ADDRESS_STR),
    }
    universe = Universe(
        time_bucket=TimeBucket.not_applicable,
        chains={arbitrum, ethereum, ChainId.lighter},
        exchanges=exchanges,
        pairs=create_pair_universe_from_code(arbitrum, [bridge, atoma, upshift, pool]),
    )
    return TradingStrategyUniverse(data_universe=universe, reserve_assets=[arb_usdc], primary_chain=arbitrum)


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """Vault execution needs no candle indicators."""


def _get_target(pair: TradingPairIdentifier) -> str | None:
    """Map a universe pair to its allocation key, ignoring the bridge."""
    if pair.is_cctp_bridge():
        return None
    if pair.is_lighter_pool():
        return "lighter"
    return pair.get_vault_protocol()


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Open each target once, then close it once redeemable.

    Bridge funding and bridge-back are planned by the normal CCTP machinery.
    A target with any earlier position, including a test position, is never
    reopened. Pending settlements are left to the resolver.
    """
    pm = input.get_position_manager()
    portfolio = input.state.portfolio
    trades = []
    for pair in input.strategy_universe.iterate_pairs():
        target = _get_target(pair)
        if target is None:
            continue
        position = pm.get_current_position_for_pair(pair)
        previous = [p for p in portfolio.get_all_positions() if p.pair == pair]
        if position is None:
            if not previous:
                trades.extend(pm.open_spot(pair, value=input.parameters.allocations[target]))
            continue
        if position.is_test() or position.has_pending_vault_settlement() or position.has_unexecuted_trades():
            continue
        if input.pricing_model.check_redemption(input.timestamp, pair, position=position).can_redeem:
            trades.extend(pm.close_position(position))
    return trades
