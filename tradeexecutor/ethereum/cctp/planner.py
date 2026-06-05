"""Cross-chain CCTP bridge trade injection planner.

Analyses alpha model rebalance trades and injects the necessary
CCTP bridge transfers to move capital between chains via the
primary chain hub.

Hub-and-spoke topology: all capital routes through the primary chain.

- Satellite sells produce excess capital that must be bridged back
  to the primary chain.
- Satellite buys require capital to be bridged out from the primary
  chain.

The planner computes *net* flows per satellite chain to avoid
unnecessary round-trips when sells and buys partially cancel out.
"""

import datetime
import logging
from collections import defaultdict
from decimal import Decimal

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType

logger = logging.getLogger(__name__)


def _find_bridge_pair(
    pairs: list[TradingPairIdentifier],
    destination_chain_id: int,
) -> TradingPairIdentifier | None:
    """Find the CCTP bridge pair targeting a specific destination chain.

    :param pairs:
        All trading pairs available in the strategy universe.

    :param destination_chain_id:
        The chain ID of the destination (satellite) chain.

    :return:
        The bridge pair, or ``None`` if no bridge pair exists for that chain.
    """
    for pair in pairs:
        if pair.kind == TradingPairKind.cctp_bridge and pair.get_destination_chain_id() == destination_chain_id:
            return pair
    return None


def inject_cctp_bridge_trades(
    state: State,
    trades: list[TradeExecution],
    strategy_universe,
    primary_chain_id: int,
    ts: datetime.datetime,
    reserve_asset: AssetIdentifier,
) -> list[TradeExecution]:
    """Inject CCTP bridge trades for cross-chain rebalancing.

    Analyses the trade list from the alpha model and injects bridge
    trades to move capital between satellite chains and the primary
    chain. Uses hub-and-spoke topology: all capital routes through
    the primary chain.

    For each satellite chain:

    - Net sell (more sells than buys): inject a bridge-back
      (satellite -> primary) for the excess capital.

    - Net buy (more buys than sells): inject a bridge-out
      (primary -> satellite) for the deficit.

    The injected trades have correct sort positions (via
    ``get_execution_sort_position()``) so they execute in the right
    order: vault redeems -> bridge-backs -> bridge-outs -> vault
    deposits.

    :param state:
        Current portfolio state.

    :param trades:
        Trade list from alpha model (vault sells + buys only).

    :param strategy_universe:
        Trading universe with CCTP bridge pairs available.
        Must support ``iterate_pairs()`` returning
        :py:class:`TradingPairIdentifier` instances.

    :param primary_chain_id:
        Chain ID of the primary chain (hub for all bridges).

    :param ts:
        Strategy cycle timestamp for the injected trades.

    :param reserve_asset:
        The portfolio reserve currency asset (e.g. USDC on primary chain).

    :return:
        Augmented trade list with bridge trades injected.
        Original trades are not modified.
    """

    # Pre-fetch all pairs from the universe so we only iterate once
    all_pairs = list(strategy_universe.iterate_pairs())

    # 1. Group trades by chain and compute net capital flow per satellite chain
    chain_flows: dict[int, Decimal] = defaultdict(lambda: Decimal(0))

    for trade in trades:
        trade_chain_id = trade.pair.chain_id
        if trade_chain_id == primary_chain_id:
            continue

        # HyperCore vault trades (chain_id 9999) have their own multi-phase
        # settlement mechanism and do not need CCTP bridging.
        if trade.pair.is_hyperliquid_vault():
            continue

        if trade.is_sell():
            # Sell frees up capital — positive flow means excess to bridge back
            sell_value = trade.planned_reserve if trade.planned_reserve else Decimal(str(trade.get_planned_value()))
            chain_flows[trade_chain_id] += abs(sell_value)
        else:
            # Buy consumes capital — negative flow means deficit to bridge out
            buy_value = trade.planned_reserve if trade.planned_reserve else Decimal(str(trade.get_planned_value()))
            chain_flows[trade_chain_id] -= abs(buy_value)

    # 2. Inject bridge trades for each satellite chain with a non-zero net flow
    bridge_trades: list[TradeExecution] = []

    for chain_id, net_flow in chain_flows.items():
        if net_flow == 0:
            continue

        bridge_pair = _find_bridge_pair(all_pairs, chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue

        if net_flow > 0:
            # Net sell: bridge-back (satellite -> primary)
            amount = net_flow
            bridge_position = state.portfolio.get_bridge_position_for_chain(chain_id)

            if bridge_position is not None:
                available = bridge_position.get_available_bridge_capital()
                closing = amount >= available
            else:
                # No existing bridge position — this is unusual for a bridge-back,
                # but we still create the trade and let execution handle it
                closing = False
                bridge_position = None

            _, trade, _ = state.create_trade(
                strategy_cycle_at=ts,
                pair=bridge_pair,
                quantity=Decimal(str(-amount)),
                reserve=None,
                assumed_price=1.0,
                trade_type=TradeType.rebalance,
                reserve_currency=reserve_asset,
                reserve_currency_price=1.0,
                position=bridge_position,
                closing=closing,
            )

            logger.info(
                "Injected bridge-back sell of %s for chain %d (closing=%s)",
                amount,
                chain_id,
                closing,
            )
            bridge_trades.append(trade)

        else:
            # Net buy: bridge-out (primary -> satellite)
            amount = abs(net_flow)

            _, trade, _ = state.create_trade(
                strategy_cycle_at=ts,
                pair=bridge_pair,
                quantity=None,
                reserve=Decimal(str(amount)),
                assumed_price=1.0,
                trade_type=TradeType.rebalance,
                reserve_currency=reserve_asset,
                reserve_currency_price=1.0,
            )

            logger.info(
                "Injected bridge-out buy of %s for chain %d",
                amount,
                chain_id,
            )
            bridge_trades.append(trade)

    return trades + bridge_trades
