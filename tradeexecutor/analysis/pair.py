"""Analytics for constructed pair universes."""
import datetime

import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import CandleSampleUnavailable
from tradingstrategy.liquidity import LiquidityDataUnavailable


def display_strategy_universe(
    strategy_universe: TradingStrategyUniverse,
    show_tvl=True,
    show_volume=True,
    show_price=True,
    now_ = None,
    tolerance=pd.Timedelta("3m"),  # Put in any number, fix later to ignore lookup errors
) -> pd.DataFrame:
    """Displays a constructed trading strategy universe in table format.

    - Designed for notebook debug
    - To be called at the end of ``create_trading_universe()``
    - Returns a human-readable table of selected trading pairs and their nature
    - Contains latest TVL, price and volume figures for the reference

    Pairs can be
    - Part of trading
    - Benchmarks
    - Benchmark pairs are detected by ``pair.other_data.get("benchmark")``

    :return:
        Human-readable DataFrame for ``display()``
    """

    pairs = []

    if not now_:
        now_ = datetime.datetime.utcnow()

    candle_now = strategy_universe.data_universe.time_bucket.floor(pd.Timestamp(now_))

    for pair in strategy_universe.iterate_pairs():
        benchmark = pair.other_data.get("benchmark")
        data = {
            "id": pair.internal_id,
            "base": pair.base.token_symbol,
            "quote": pair.quote.token_symbol,
            "exchange": pair.exchange_name,
            "fee %": pair.fee * 100,
            "type:": "benchmark/routed token" if benchmark else "traded token"
        }

        if show_price:
            try:
                data["price"], data["price_at"] = strategy_universe.data_universe.candles.get_price_with_tolerance(
                    pair=pair.internal_id,
                    when=candle_now,
                    tolerance=tolerance,
                )
            except CandleSampleUnavailable:
                data["price"] = "<not avail>"
                data["price_at"] = "-"

        if show_volume:
            try:
                data["volume"], _ = strategy_universe.data_universe.candles.get_price_with_tolerance(
                    pair=pair.internal_id,
                    when=candle_now,
                    kind="volume",
                    tolerance=tolerance,
                )
            except CandleSampleUnavailable:
                data["volume"] = "<not avail>"

        if show_tvl:
            if strategy_universe.data_universe.liquidity:
                tvl_now = strategy_universe.data_universe.liquidity_time_bucket.floor(pd.Timestamp(now_))
                try:
                    tvl, tvl_at = strategy_universe.data_universe.liquidity.get_liquidity_with_tolerance(
                        pair_id=pair.internal_id,
                        when=tvl_now,
                        tolerance=tolerance,
                    )
                except LiquidityDataUnavailable:
                    tvl = "<not avail>"
                    tvl_at = "-"
            else:
                tvl = "<not loaded>"
                tvl_at = "-"

            data["tvl"] = tvl
            data["tvl_at"] = tvl_at

        pairs.append(data)

    human_pair_universe_df = pd.DataFrame(pairs)
    return human_pair_universe_df
