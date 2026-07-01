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
from dataclasses import dataclass
from collections import defaultdict
from decimal import Decimal

from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.portfolio import NotEnoughMoney
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import CCTP_BRIDGE_ORDER_BUMP, TradeExecution, TradeType
from tradeexecutor.utils.accuracy import sum_decimal

logger = logging.getLogger(__name__)


@dataclass
class ChainLiquidity:
    """CCTP liquidity planning ledger for one satellite chain.

    All non-async satellite sells sort before CCTP bridge-backs (spot sells at
    -40M, bridge-backs at -30M), so every same-cycle sell proceed is available to
    bridge back. The ledger therefore tracks a single satellite-side available
    balance (idle bridge capital + same-cycle sells) rather than an early/late
    sell split. The ``_trade_releases_before_bridge_back`` assertion in the
    classification loop enforces that invariant.
    """

    net_flow: Decimal = Decimal(0)
    satellite_buys: Decimal = Decimal(0)
    satellite_sells_before_buy: Decimal = Decimal(0)
    available_bridge_capital: Decimal = Decimal(0)
    free_idle_bridge_capital: Decimal = Decimal(0)
    bridge_back_amount: Decimal = Decimal(0)

    @property
    def available_before_buy(self) -> Decimal:
        """Satellite-side cash available this cycle: idle bridge capital plus all
        same-cycle satellite sells (which release before bridge-backs)."""
        return self.available_bridge_capital + self.satellite_sells_before_buy

    @property
    def satellite_bridge_shortfall(self) -> Decimal:
        """How much satellite buy demand still needs a bridge-out."""
        return max(self.satellite_buys - self.available_before_buy, Decimal(0))

    def prepare(self, available_bridge_capital: Decimal) -> None:
        """Initialise free bridge-back capital after reserving buy funding.

        Withhold only the capital still needed to fund same-cycle satellite buys;
        the rest is free to bridge back to the primary hub.
        """
        self.available_bridge_capital = available_bridge_capital
        self.free_idle_bridge_capital = max(
            self.available_before_buy - self.satellite_buys,
            Decimal(0),
        )

    def reserve_bridge_back(self, amount: Decimal) -> None:
        """Reserve free idle satellite capital for a CCTP bridge-back."""
        self.bridge_back_amount += amount
        self.free_idle_bridge_capital -= amount


def _bridge_pairs_by_destination(
    pairs: list[TradingPairIdentifier],
) -> dict[int, TradingPairIdentifier]:
    """Map destination chain id to its CCTP bridge pair."""
    return {
        pair.get_destination_chain_id(): pair
        for pair in pairs
        if pair.kind == TradingPairKind.cctp_bridge
    }


def _find_bridge_pair(
    pairs: list[TradingPairIdentifier],
    destination_chain_id: int,
) -> TradingPairIdentifier | None:
    """Find the CCTP bridge pair for a satellite destination chain."""
    return _bridge_pairs_by_destination(pairs).get(destination_chain_id)


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


def _open_bridge_chain_ids(state: State) -> set[int]:
    """Which satellite chains currently have open CCTP bridge capital."""
    return {
        position.pair.get_destination_chain_id()
        for position in state.portfolio.open_positions.values()
        if position.pair.is_cctp_bridge()
    }


def _get_raw_unit(asset: AssetIdentifier) -> Decimal:
    """Smallest human-readable amount representable by a token raw unit."""
    return asset.convert_to_decimal(1)


def _floor_to_raw_units(amount: Decimal, asset: AssetIdentifier) -> Decimal:
    """Floor a positive token amount to raw-token precision."""
    assert amount >= 0
    return asset.convert_to_decimal(asset.convert_to_raw_amount(amount))


def _is_meaningful_bridge_amount(amount: Decimal, asset: AssetIdentifier) -> bool:
    """Is this amount large enough to become a CCTP trade.

    CCTP burns and mints ERC-20 raw units. Any human-unit fraction smaller than
    one raw unit would be truncated by the live routing path and is accounting
    dust, even when it is larger than Decimal calculation epsilon.
    """
    return amount >= _get_raw_unit(asset)


def _is_token_dust(amount: Decimal, asset: AssetIdentifier) -> bool:
    """Is this amount below one raw token unit."""
    return abs(amount) < _get_raw_unit(asset)


def _assert_bridge_pair_raw_units_match(bridge_pair: TradingPairIdentifier) -> None:
    """CCTP bridge pairs must use the same raw-unit precision on both sides."""
    assert bridge_pair.base.decimals == bridge_pair.quote.decimals, (
        f"CCTP bridge destination asset {bridge_pair.base} decimals "
        f"{bridge_pair.base.decimals} do not match source asset "
        f"{bridge_pair.quote} decimals {bridge_pair.quote.decimals}"
    )


def _trade_releases_before_bridge_back(trade: TradeExecution) -> bool:
    """Does this satellite sell release bridge capital before bridge-backs run.

    After the CCTP phase reorder every non-async satellite reduce sorts below
    ``-CCTP_BRIDGE_ORDER_BUMP`` (spot sells at -40M, vault withdrawals at -50M),
    so this is expected to hold for every satellite sell counted by the planner.
    It is used as an invariant assertion in the classification loop rather than
    for branching.
    """
    return trade.get_execution_sort_position() < -CCTP_BRIDGE_ORDER_BUMP


def _is_async_vault_sell_waiting_for_settlement(state: State, trade: TradeExecution) -> bool:
    """Does this sell request produce reserve only after async vault settlement?"""
    if not trade.is_sell() or not trade.is_vault():
        return False
    if trade.other_data.get("vault_async_flow"):
        return True
    if trade.pair.is_async_vault():
        return True

    position = state.portfolio.get_position_by_id(trade.position_id)
    if position is not None and position.has_async_vault_flow():
        return True

    return False


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
    order: vault redeems -> spot sells -> bridge-backs -> bridge-outs ->
    buys -> vault deposits. Bridge-backs run after satellite sells (so they
    carry same-cycle sell proceeds to the hub) and bridge-outs run before
    buys (so satellite buys are funded).

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

    # Pre-fetch bridge pairs from the universe so we only iterate once.
    bridge_pairs = _bridge_pairs_by_destination(list(strategy_universe.iterate_pairs()))
    for bridge_pair in bridge_pairs.values():
        _assert_bridge_pair_raw_units_match(bridge_pair)
    primary_bridge_asset = next(iter(bridge_pairs.values())).quote if bridge_pairs else reserve_asset

    # 1. Group trades by chain and compute net capital flow per satellite chain.
    #    Also track primary-chain reserve in/outflows so bridge-outs can be
    #    bounded by what the primary chain can actually fund.
    liquidity_by_chain: defaultdict[int, ChainLiquidity] = defaultdict(ChainLiquidity)
    primary_sells = Decimal(0)
    primary_buys = Decimal(0)

    for trade in trades:
        trade_chain_id = trade.pair.chain_id
        value = trade.planned_reserve if trade.planned_reserve else Decimal(str(trade.get_planned_value()))
        value = abs(value)
        async_vault_sell_waiting = _is_async_vault_sell_waiting_for_settlement(state, trade)

        if trade_chain_id == primary_chain_id:
            # Primary-chain sells free reserve (they execute before bridge-outs);
            # primary-chain buys consume it (they execute after bridge-outs).
            if trade.is_sell():
                if async_vault_sell_waiting:
                    logger.info(
                        "Ignoring primary-chain async vault sell #%d for same-cycle reserve planning",
                        trade.trade_id,
                    )
                    continue
                primary_sells += value
            else:
                primary_buys += value
            continue

        # HyperCore vault trades (chain_id 9999) have their own multi-phase
        # settlement mechanism and do not need CCTP bridging.
        if trade.pair.is_hyperliquid_vault():
            continue

        if trade.is_sell():
            if async_vault_sell_waiting:
                logger.info(
                    "Ignoring async vault sell #%d on chain %d for same-cycle CCTP liquidity planning",
                    trade.trade_id,
                    trade_chain_id,
                )
                continue
            # Sell frees up capital — positive flow means excess to bridge back.
            # Every counted satellite sell must sort before CCTP bridge-backs so
            # its proceeds are available for same-cycle bridge-back sizing. This
            # is the load-bearing invariant behind the single-balance ledger; the
            # assert fails loudly if a future sort change (or an unexpected sell
            # shape, e.g. a credit sell without a close/open flag) breaks it.
            assert _trade_releases_before_bridge_back(trade), (
                f"Satellite sell #{trade.trade_id} sorts at "
                f"{trade.get_execution_sort_position()}, not before CCTP bridge-backs "
                f"(< -{CCTP_BRIDGE_ORDER_BUMP}); the planner assumes every counted "
                f"satellite sell releases before bridge-backs"
            )
            liquidity = liquidity_by_chain[trade_chain_id]
            liquidity.net_flow += value
            liquidity.satellite_sells_before_buy += value
        else:
            # Buy consumes capital — negative flow means deficit to bridge out
            liquidity = liquidity_by_chain[trade_chain_id]
            liquidity.net_flow -= value
            liquidity.satellite_buys += value

    bridge_trades: list[TradeExecution] = []

    # Satellite buys spend idle satellite bridge capital before they need a
    # bridge-out. Keep that already-spoken-for capital out of primary buy
    # funding, otherwise a primary-chain buy could steal the USDC that a
    # same-cycle satellite buy is about to allocate.
    for chain_id in _open_bridge_chain_ids(state):
        liquidity_by_chain[chain_id]
    for chain_id, liquidity in liquidity_by_chain.items():
        liquidity.prepare(_available_bridge_capital(state, chain_id))

    # 2. Bridge-backs first (net-sell satellites). These free capital back onto
    #    the primary chain and execute before bridge-outs, so their proceeds are
    #    available to fund same-cycle bridge-outs.
    total_bridge_back = Decimal(0)
    for chain_id in sorted(liquidity_by_chain):
        liquidity = liquidity_by_chain[chain_id]
        net_flow = liquidity.net_flow
        if net_flow <= 0:
            continue

        bridge_pair = bridge_pairs.get(chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue

        # Never bridge back more than the per-chain free bridge capital:
        # bridging back the full net sell would move satellite USDC that an
        # in-flight deposit on the same chain still needs to settle, starving it
        # (OutOfSimulatedBalance at deposit settlement). Any excess net sell
        # stays on the satellite and bridges back once the deposits have settled.
        #
        # Satellite spot sells now sort before bridge-backs (-40M vs -30M), so
        # their proceeds are available to fund same-cycle bridge-backs; that is
        # already folded into free_idle_bridge_capital (idle + same-cycle sells,
        # minus same-cycle satellite buy needs).
        available = liquidity.free_idle_bridge_capital
        amount = _floor_to_raw_units(min(net_flow, available), bridge_pair.base)
        if not _is_meaningful_bridge_amount(amount, bridge_pair.base):
            logger.info(
                "Skipping bridge-back for chain %d: net sell %s exceeds available bridge capital %s",
                chain_id,
                net_flow,
                available,
            )
            continue

        liquidity.reserve_bridge_back(amount)
        total_bridge_back += amount

    # Primary-chain buys and bridge-outs both need primary-chain USDC. If the
    # primary reserve is short, pull only the missing amount back from genuinely
    # idle satellite bridge capital. This prevents notebook failures where
    # deployable capital existed only as idle satellite USDC, but no satellite
    # net sell was present to trigger a bridge-back.
    try:
        current_primary_reserve = state.portfolio.get_reserve_position(reserve_asset).quantity
    except KeyError:
        current_primary_reserve = Decimal(0)

    total_satellite_bridge_shortfall = sum_decimal(
        liquidity.satellite_bridge_shortfall
        for liquidity in liquidity_by_chain.values()
    )
    primary_cash_needed = primary_buys + total_satellite_bridge_shortfall
    primary_shortfall = sum_decimal([
        primary_cash_needed,
        -current_primary_reserve,
        -primary_sells,
        -total_bridge_back,
    ])
    if _is_meaningful_bridge_amount(primary_shortfall, primary_bridge_asset):
        for chain_id in sorted(liquidity_by_chain):
            bridge_pair = bridge_pairs.get(chain_id)
            if bridge_pair is None:
                logger.warning(
                    "No CCTP bridge pair found for destination chain %d, "
                    "skipping primary-shortfall bridge-back",
                    chain_id,
                )
                continue
            if not _is_meaningful_bridge_amount(primary_shortfall, bridge_pair.quote):
                break
            liquidity = liquidity_by_chain[chain_id]
            available = liquidity.free_idle_bridge_capital
            amount = _floor_to_raw_units(min(primary_shortfall, available), bridge_pair.quote)
            if not _is_meaningful_bridge_amount(amount, bridge_pair.quote):
                continue
            liquidity.reserve_bridge_back(amount)
            total_bridge_back += amount
            primary_shortfall -= amount
            logger.info(
                "Reserved bridge-back of %s from chain %d to fund primary-chain cash needs",
                amount,
                chain_id,
            )

    if _is_meaningful_bridge_amount(primary_shortfall, primary_bridge_asset):
        raise NotEnoughMoney(
            f"Cannot fund primary-chain cash needs: need {primary_cash_needed} "
            f"(primary buys {primary_buys} + satellite bridge shortfalls "
            f"{total_satellite_bridge_shortfall}), but only "
            f"{current_primary_reserve + primary_sells + total_bridge_back} primary "
            f"reserve is available after primary sells and idle satellite bridge-backs "
            f"(reserve {current_primary_reserve} + primary sells {primary_sells} + "
            f"bridge-backs {total_bridge_back})"
        )

    for chain_id in sorted(liquidity_by_chain):
        liquidity = liquidity_by_chain[chain_id]
        amount = liquidity.bridge_back_amount
        bridge_pair = bridge_pairs.get(chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue
        if not _is_meaningful_bridge_amount(amount, bridge_pair.base):
            continue

        bridge_position = state.portfolio.get_bridge_position_for_chain(chain_id)
        available = liquidity.available_before_buy
        assert amount <= available, (
            f"CCTP bridge-back amount {amount} exceeds available bridge capital "
            f"{available} for chain {chain_id}"
        )
        closing = _is_token_dust(available - amount, bridge_pair.base)

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
    fundable_primary = sum_decimal([
        current_primary_reserve,
        primary_sells,
        total_bridge_back,
        -primary_buys,
    ])
    if fundable_primary < 0:
        fundable_primary = Decimal(0)

    # 4. Bridge-outs (net-buy satellites). Fund satellite buys from capital
    #    already idle on the satellite and same-cycle synchronous sells first,
    #    and only bridge out the remaining shortfall — never more than the
    #    primary chain can fund. Process chains in a deterministic order so
    #    that, under a primary-reserve shortage, it is reproducible which chain
    #    is reported as underfunded.
    for chain_id in sorted(liquidity_by_chain):
        liquidity = liquidity_by_chain[chain_id]
        net_flow = liquidity.net_flow
        if net_flow >= 0:
            continue

        bridge_pair = bridge_pairs.get(chain_id)
        if bridge_pair is None:
            logger.warning(
                "No CCTP bridge pair found for destination chain %d, "
                "skipping bridge injection",
                chain_id,
            )
            continue

        required = abs(net_flow)

        # Fund the satellite buys from capital already idle on the satellite
        # and synchronous same-chain sells first; only the remaining shortfall
        # needs bridging.
        satellite_side_funding = liquidity.available_before_buy

        shortfall = liquidity.satellite_bridge_shortfall
        if not _is_meaningful_bridge_amount(shortfall, bridge_pair.quote):
            # Capital already parked on the satellite covers the net buy; the
            # satellite buy allocates from the bridge position at execution.
            logger.info(
                "Skipping bridge-out for chain %d: satellite-side funding %s covers satellite buys %s",
                chain_id,
                satellite_side_funding,
                liquidity.satellite_buys,
            )
            continue

        if shortfall > fundable_primary:
            if _is_token_dust(shortfall - fundable_primary, bridge_pair.quote):
                shortfall = fundable_primary
            else:
                raise NotEnoughMoney(
                    f"Cannot fund CCTP bridge-out to chain {chain_id}: "
                    f"need {shortfall} (satellite buys {liquidity.satellite_buys} - "
                    f"satellite-side funding before buys {satellite_side_funding}; "
                    f"net buy {required}), but only {fundable_primary} primary "
                    f"reserve is available (current reserve {current_primary_reserve} + "
                    f"primary sells {primary_sells} + bridge-backs {total_bridge_back} "
                    f"- primary buys {primary_buys})"
                )

        if not _is_meaningful_bridge_amount(shortfall, bridge_pair.quote):
            continue

        amount = _floor_to_raw_units(shortfall, bridge_pair.quote)
        if not _is_meaningful_bridge_amount(amount, bridge_pair.quote):
            continue
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
            "Injected bridge-out buy of %s for chain %d (satellite-side funding %s, satellite buys %s, net buy %s)",
            amount,
            chain_id,
            satellite_side_funding,
            liquidity.satellite_buys,
            required,
        )
        bridge_trades.append(trade)

    return trades + bridge_trades
