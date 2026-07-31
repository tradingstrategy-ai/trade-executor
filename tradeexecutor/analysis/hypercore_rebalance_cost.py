"""Cost analysis for successful, normal HyperCore vault rebalances.

Historic state files predate structured HyperCore bridge and close-loss
accounting.  Their report values are deliberately lower bounds: this module
only reconstructs gas from complete transaction receipts and never treats a
planned/executed difference as a fee.
"""

import datetime
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Iterable

import pandas as pd

from eth_defi.hyperliquid.session import create_hyperliquid_session

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeFlag


HYPE_WEI = Decimal(10**18)
UTC_EPOCH = datetime.datetime(1970, 1, 1)


@dataclass(frozen=True)
class HypercoreRebalanceCostReport:
    """Tabular report and coverage information for one strategy state."""

    trades: pd.DataFrame
    rebalances: pd.DataFrame
    summary: pd.DataFrame
    ignored: pd.DataFrame


class HypeHourlyPriceLookup:
    """Look up the nearest preceding HYPE hourly candle close."""

    def __init__(self, candles: list[dict]):
        samples = sorted(
            (
                UTC_EPOCH + datetime.timedelta(milliseconds=candle["t"]),
                float(candle["c"]),
            )
            for candle in candles
        )
        self.timestamps = [sample[0] for sample in samples]
        self.prices = [sample[1] for sample in samples]

    def __call__(self, timestamp: datetime.datetime | None) -> float | None:
        if timestamp is None or not self.timestamps:
            return None
        index = bisect_right(self.timestamps, timestamp) - 1
        if index < 0:
            return None
        return self.prices[index]


def fetch_hype_hourly_price_lookup(
    trades: Iterable[TradeExecution],
) -> HypeHourlyPriceLookup | None:
    """Fetch the HYPE hourly candles needed to value legacy gas receipts."""
    timestamps = [
        tx.included_at
        for trade in trades
        for tx in trade.blockchain_transactions
        if tx.included_at is not None
    ]
    if not timestamps:
        return None

    start_at = min(timestamps) - datetime.timedelta(hours=1)
    end_at = max(timestamps) + datetime.timedelta(hours=1)
    session = create_hyperliquid_session()
    response = session.post_info(
        {
            "type": "candleSnapshot",
            "req": {
                "coin": "HYPE",
                "interval": "1h",
                "startTime": int((start_at - UTC_EPOCH).total_seconds() * 1000),
                "endTime": int((end_at - UTC_EPOCH).total_seconds() * 1000),
            },
        },
        timeout=30,
    )
    response.raise_for_status()
    candles = response.json()
    if not candles:
        return None
    return HypeHourlyPriceLookup(candles)


def get_trade_gas_cost_usd(
    trade: TradeExecution,
    hype_price: Callable[[datetime.datetime | None], float | None] | None = None,
) -> float | None:
    """Return recorded or safely reconstructed HyperEVM gas cost in USD.

    A legacy trade is valued only if every recorded transaction has both gas
    receipt fields and an hourly HYPE price.  Partial data is not extrapolated.
    """
    persisted = trade.get_cost_of_gas_usd()
    if persisted is not None:
        return persisted

    if not trade.blockchain_transactions or hype_price is None:
        return None

    cost = Decimal(0)
    for tx in trade.blockchain_transactions:
        if (
            tx.realised_gas_units_consumed is None
            or tx.realised_gas_price is None
            or tx.realised_gas_price <= 0
        ):
            return None
        price = hype_price(tx.included_at)
        if price is None:
            return None
        native_cost = (
            Decimal(tx.realised_gas_units_consumed * tx.realised_gas_price) / HYPE_WEI
        )
        cost += native_cost * Decimal(str(price))
    return float(cost)


def get_ignored_trade_reason(trade: TradeExecution) -> str | None:
    """Return why a HyperCore trade is excluded from normal-rebalance costs."""
    if not trade.pair.is_hyperliquid_vault():
        return "not HyperCore vault"
    if not trade.is_rebalance():
        return f"trade type {trade.trade_type.value}"
    if not trade.is_success():
        return "not successful"
    if trade.is_repaired():
        return "repaired original"
    if trade.is_repair_trade():
        return "repair trade"
    if trade.flags and TradeFlag.test_trade in trade.flags:
        return "test trade"
    if not trade.blockchain_transactions:
        return "successful trade without transactions"
    return None


def _turnover(trade: TradeExecution) -> float | None:
    if trade.executed_reserve is None:
        return None
    return float(abs(trade.executed_reserve))


def _cost_components(
    trade: TradeExecution,
    hype_price: Callable[[datetime.datetime | None], float | None] | None,
) -> tuple[dict[str, float | None], bool]:
    """Return known components and whether the trade is fully observed."""
    components: dict[str, float | None] = {
        "gas_usd": get_trade_gas_cost_usd(trade, hype_price),
        "bridge_usd": trade.bridge_fee_usd,
        "activation_usd": trade.account_activation_fee_usd,
        "close_other_loss_usd": trade.hypercore_close_other_loss_usd,
    }
    known_total = sum(
        (float(value) for value in components.values() if value is not None), 0.0
    )
    complete = trade.hypercore_cost_data_complete
    return components | {
        "known_lower_bound_cost_usd": known_total,
        "total_cost_usd": known_total if complete else None,
    }, complete


def build_hypercore_rebalance_cost_report(
    state: State,
    hype_price: Callable[[datetime.datetime | None], float | None] | None = None,
) -> HypercoreRebalanceCostReport:
    """Build all-rebalance, whole-strategy and ignored-history tables."""
    eligible: list[TradeExecution] = []
    ignored = Counter()
    for trade in state.portfolio.get_all_trades():
        reason = get_ignored_trade_reason(trade)
        if reason is None:
            eligible.append(trade)
        elif trade.pair.is_hyperliquid_vault():
            ignored[reason] += 1

    trade_rows: list[dict] = []
    for trade in eligible:
        components, complete = _cost_components(trade, hype_price)
        turnover = _turnover(trade)
        total_cost = components["total_cost_usd"]
        trade_rows.append(
            {
                "cycle": trade.opened_at,
                "trade_id": trade.trade_id,
                "position_id": trade.position_id,
                "side": "buy" if trade.is_buy() else "sell",
                "turnover_usd": turnover,
                **components,
                "cost_bps": total_cost / turnover * 10_000
                if total_cost is not None and turnover
                else None,
                "complete": complete,
                "transactions": len(trade.blockchain_transactions),
            }
        )

    trades = pd.DataFrame(trade_rows)
    if len(trades):
        rebalance_rows: list[dict] = []
        for cycle, cycle_trades in trades.groupby("cycle", dropna=False):
            complete = bool(cycle_trades["complete"].all())
            turnover = float(cycle_trades["turnover_usd"].sum())
            known_lower_bound = float(cycle_trades["known_lower_bound_cost_usd"].sum())
            total_cost = known_lower_bound if complete else None
            gas_values = cycle_trades["gas_usd"].dropna()
            rebalance_rows.append(
                {
                    "cycle": cycle,
                    "trades": len(cycle_trades),
                    "buys": int((cycle_trades["side"] == "buy").sum()),
                    "sells": int((cycle_trades["side"] == "sell").sum()),
                    "turnover_usd": turnover,
                    "gas_usd": float(cycle_trades["gas_usd"].sum())
                    if complete
                    else None,
                    "bridge_usd": float(cycle_trades["bridge_usd"].fillna(0).sum())
                    if complete
                    else None,
                    "activation_usd": float(
                        cycle_trades["activation_usd"].fillna(0).sum()
                    )
                    if complete
                    else None,
                    "close_other_loss_usd": float(
                        cycle_trades["close_other_loss_usd"].fillna(0).sum()
                    )
                    if complete
                    else None,
                    "total_cost_usd": total_cost,
                    "cost_bps": total_cost / turnover * 10_000
                    if total_cost is not None and turnover
                    else None,
                    "known_gas_lower_bound_usd": float(gas_values.sum())
                    if len(gas_values)
                    else None,
                    "gas_observed_trades": len(gas_values),
                    "complete": complete,
                }
            )
        rebalances = pd.DataFrame(rebalance_rows)

        complete_rebalances = rebalances.loc[rebalances["complete"]]
        bps_rebalances = complete_rebalances.loc[
            complete_rebalances["turnover_usd"] > 0
        ]
        gas_observed_trades = trades.loc[trades["gas_usd"].notna()]
        exact_total_cost = (
            float(complete_rebalances["total_cost_usd"].sum())
            if len(complete_rebalances)
            else None
        )
        exact_turnover = float(bps_rebalances["turnover_usd"].sum())
        known_gas = (
            float(gas_observed_trades["gas_usd"].sum())
            if len(gas_observed_trades)
            else None
        )
        gas_observed_turnover = float(gas_observed_trades["turnover_usd"].sum())
        summary = pd.DataFrame(
            [
                {
                    "eligible_rebalances": len(rebalances),
                    "eligible_trades": len(trades),
                    "complete_rebalances": len(complete_rebalances),
                    "incomplete_rebalances": len(rebalances) - len(complete_rebalances),
                    "zero_turnover_rebalances": int(
                        (rebalances["turnover_usd"].fillna(0) <= 0).sum()
                    ),
                    "complete_turnover_usd": exact_turnover,
                    "gas_usd": float(complete_rebalances["gas_usd"].sum())
                    if len(complete_rebalances)
                    else None,
                    "bridge_usd": float(complete_rebalances["bridge_usd"].sum())
                    if len(complete_rebalances)
                    else None,
                    "activation_usd": float(complete_rebalances["activation_usd"].sum())
                    if len(complete_rebalances)
                    else None,
                    "close_other_loss_usd": float(
                        complete_rebalances["close_other_loss_usd"].sum()
                    )
                    if len(complete_rebalances)
                    else None,
                    "total_cost_usd": exact_total_cost,
                    "weighted_cost_bps": exact_total_cost / exact_turnover * 10_000
                    if exact_total_cost is not None and exact_turnover
                    else None,
                    "average_cost_usd_per_rebalance": float(
                        complete_rebalances["total_cost_usd"].mean()
                    )
                    if len(complete_rebalances)
                    else None,
                    "average_cost_bps_per_rebalance": float(
                        bps_rebalances["cost_bps"].mean()
                    )
                    if len(bps_rebalances)
                    else None,
                    "known_gas_lower_bound_usd": known_gas,
                    "gas_observed_trades": len(gas_observed_trades),
                    "gas_observed_turnover_usd": gas_observed_turnover,
                    "observed_gas_bps": known_gas / gas_observed_turnover * 10_000
                    if known_gas is not None and gas_observed_turnover
                    else None,
                    "average_gas_usd_per_observed_trade": known_gas
                    / len(gas_observed_trades)
                    if known_gas is not None and len(gas_observed_trades)
                    else None,
                }
            ]
        )
    else:
        rebalances = pd.DataFrame()
        summary = pd.DataFrame()

    ignored_table = pd.DataFrame(
        [
            {"reason": reason, "trades": count}
            for reason, count in sorted(ignored.items())
        ]
    )
    return HypercoreRebalanceCostReport(trades, rebalances, summary, ignored_table)
