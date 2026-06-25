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
from tradeexecutor.state.portfolio import NotEnoughMoney
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


def _available_bridge_capital(state: State, chain_id: int) -> Decimal:
    """Per-chain available bridge capital, clamped to be non-negative.

    This is the per-chain liquidity ledger the planner sizes both bridge
    directions against: the physical satellite-chain USDC minus capital already
    committed to unsettled (async) satellite deposits.

    ``bridge_capital_allocated`` can go negative by design — a profitable
    satellite round-trip returns more than was allocated — so the raw available
    figure can be negative; never treat that as spare cash.

    :return:
        ``max(available_bridge_capital, 0)`` for the chain's open bridge
        position, or zero when no bridge position is open.
    """
    bridge_position = state.portfolio.get_bridge_position_for_chain(chain_id)
    if bridge_position is None:
        return Decimal(0)
    return max(bridge_position.get_available_bridge_capital(), Decimal(0))


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
      (primary -> satellite) for the deficit, but only for the part not
      already covered by capital idle on the satellite chain (available
      bridge capital), and never more than the primary chain can fund.
      Raises :py:class:`tradeexecutor.state.portfolio.NotEnoughMoney` if a
      genuinely needed bridge-out cannot be funded from the primary reserve.

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

    # 1. Group trades by chain and compute net capital flow per satellite chain.
    #    Also track primary-chain reserve in/outflows so bridge-outs can be
    #    bounded by what the primary chain can actually fund.
    chain_flows: dict[int, Decimal] = defaultdict(lambda: Decimal(0))
    primary_sells = Decimal(0)
    primary_buys = Decimal(0)

    for trade in trades:
        trade_chain_id = trade.pair.chain_id
        value = trade.planned_reserve if trade.planned_reserve else Decimal(str(trade.get_planned_value()))
        value = abs(value)

        if trade_chain_id == primary_chain_id:
            # Primary-chain sells free reserve (they execute before bridge-outs);
            # primary-chain buys consume it (they execute after bridge-outs).
            if trade.is_sell():
                primary_sells += value
            else:
                primary_buys += value
            continue

        # HyperCore vault trades (chain_id 9999) have their own multi-phase
        # settlement mechanism and do not need CCTP bridging.
        if trade.pair.is_hyperliquid_vault():
            continue

        if trade.is_sell():
            # Sell frees up capital — positive flow means excess to bridge back
            chain_flows[trade_chain_id] += value
        else:
            # Buy consumes capital — negative flow means deficit to bridge out
            chain_flows[trade_chain_id] -= value

    bridge_trades: list[TradeExecution] = []

    # 2. Bridge-backs first (net-sell satellites). These free capital back onto
    #    the primary chain and execute before bridge-outs, so their proceeds are
    #    available to fund same-cycle bridge-outs.
    total_bridge_back = Decimal(0)
    for chain_id in sorted(chain_flows):
        net_flow = chain_flows[chain_id]
        if net_flow <= 0:
            continue

        bridge_pair = _find_bridge_pair(all_pairs, chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue

        # Never bridge back more than the per-chain available bridge capital:
        # bridging back the full net sell would move satellite USDC that an
        # in-flight deposit on the same chain still needs to settle, starving it
        # (OutOfSimulatedBalance at deposit settlement). Any excess net sell
        # stays on the satellite and bridges back once the deposits have settled.
        #
        # Known limitation (conservative): proceeds from *synchronous* satellite
        # sells in this same cycle land in available_bridge_capital only at
        # execution (after this planner runs), so they are not counted here and a
        # bridge-back that should follow them may under-bridge for one cycle. The
        # correct model distinguishes already-idle / sync-released-this-cycle /
        # pending-async capital — see the cross-chain reconciliation follow-up.
        available = _available_bridge_capital(state, chain_id)
        amount = min(net_flow, available)
        if amount <= 0:
            logger.info(
                "Skipping bridge-back for chain %d: net sell %s exceeds available bridge capital %s",
                chain_id,
                net_flow,
                available,
            )
            continue

        # amount > 0 implies available > 0, so a bridge position exists.
        bridge_position = state.portfolio.get_bridge_position_for_chain(chain_id)
        closing = amount >= available

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
        total_bridge_back += amount
        bridge_trades.append(trade)

    # 3. Compute the primary-chain reserve available to fund bridge-outs.
    #    Bridge-outs execute after primary/satellite sells and bridge-backs but
    #    before primary-chain buys, so the spendable amount is the current
    #    reserve plus same-cycle inflows, minus the primary buys we must still
    #    leave cash for (otherwise we would just move the underflow onto them).
    #
    #    Caveat (backtest vs live): the ``+ total_bridge_back`` term assumes the
    #    bridge-back proceeds land on the primary chain THIS cycle. That holds in
    #    the backtest (``simulate_bridge`` settles synchronously and bridge-backs
    #    sort before bridge-outs), but in live CCTP a bridge-back goes
    #    ``cctp_in_transit`` and settles on a later cycle, so this is optimistic
    #    live — the early ``NotEnoughMoney`` guard below is then less effective,
    #    but execution still enforces funding (``move_capital_from_reserves...``
    #    raises ``NotEnoughMoney`` at ``start_execution``), so it is never unsafe.
    try:
        current_primary_reserve = state.portfolio.get_reserve_position(reserve_asset).quantity
    except KeyError:
        current_primary_reserve = Decimal(0)

    fundable_primary = current_primary_reserve + primary_sells + total_bridge_back - primary_buys
    if fundable_primary < 0:
        fundable_primary = Decimal(0)

    # 4. Bridge-outs (net-buy satellites). Fund the satellite buys from capital
    #    already idle on the satellite first (tracked as available bridge
    #    capital), and only bridge out the remaining shortfall — never more than
    #    the primary chain can fund. Process chains in a deterministic order so
    #    that, under a primary-reserve shortage, it is reproducible which chain
    #    is reported as underfunded.
    for chain_id in sorted(chain_flows):
        net_flow = chain_flows[chain_id]
        if net_flow >= 0:
            continue

        bridge_pair = _find_bridge_pair(all_pairs, chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue

        required = abs(net_flow)

        # Fund the satellite buys from capital already idle on the satellite
        # first; only the remaining shortfall needs bridging.
        idle_satellite_capital = _available_bridge_capital(state, chain_id)

        shortfall = required - idle_satellite_capital
        if shortfall <= 0:
            # Capital already parked on the satellite covers the net buy; the
            # satellite buy allocates from the bridge position at execution.
            logger.info(
                "Skipping bridge-out for chain %d: idle satellite capital %s covers net buy %s",
                chain_id,
                idle_satellite_capital,
                required,
            )
            continue

        if shortfall > fundable_primary:
            raise NotEnoughMoney(
                f"Cannot fund CCTP bridge-out to chain {chain_id}: "
                f"need {shortfall} (net buy {required} - idle satellite capital "
                f"{idle_satellite_capital}), but only {fundable_primary} primary "
                f"reserve is available (reserve {current_primary_reserve} + "
                f"primary sells {primary_sells} + bridge-backs {total_bridge_back} "
                f"- primary buys {primary_buys})"
            )

        amount = shortfall
        fundable_primary -= amount

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
            "Injected bridge-out buy of %s for chain %d (idle satellite capital %s, net buy %s)",
            amount,
            chain_id,
            idle_satellite_capital,
            required,
        )
        bridge_trades.append(trade)

    return trades + bridge_trades
