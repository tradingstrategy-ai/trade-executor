"""Analytics for constructed pair universes."""
import datetime
import logging

import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.candle import CandleSampleUnavailable
from tradingstrategy.liquidity import LiquidityDataUnavailable
from tradingstrategy.timebucket import TimeBucket


logger = logging.getLogger(__name__)


def display_strategy_universe(
    strategy_universe: TradingStrategyUniverse,
    show_tvl=True,
    show_volume=True,
    show_price=True,
    show_tax=True,
    now_ = None,
    tolerance=pd.Timedelta("90D"),  # Put in any number, fix later to ignore lookup errors
    sort_key="id",
    sort_ascending=True,
    sort_numeric=True,
) -> pd.DataFrame:
    """Displays a constructed trading strategy universe in table format.

    - Designed for notebook debug and live trading startup checks
    - To be called at the end of ``create_trading_universe()``
    - Returns a human-readable table of selected trading pairs and their nature
    - Contains latest TVL, price and volume figures for the reference

    Pairs can be
    - Part of trading
    - Benchmarks
    - Benchmark pairs are detected by ``pair.other_data.get("benchmark")``

    Example:

    .. code-block:: python

        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 140):
            universe_df = display_strategy_universe(universe)
            logger.info("Universe is:\n%s", str(universe_df))
        
    Example 2:

    .. code-block:: python

        strategy_universe = create_trading_universe(
            None,
            client,
            notebook_execution_context,
            UniverseOptions.from_strategy_parameters_class(Parameters, notebook_execution_context)
        )

        display_strategy_universe(
            strategy_universe,
            sort_key="tvl",
            sort_ascending=False,
        ) 

    :return:
        Human-readable DataFrame for ``display()``.

        Empty DataFrame returned if the incoming universe is partial.
    """

    if not strategy_universe.data_universe.candles:
        return pd.DataFrame()

    pairs = []

    if not now_:
        now_ = datetime.datetime.utcnow()

    if strategy_universe.data_universe.time_bucket != TimeBucket.not_applicable:
        candle_now = strategy_universe.data_universe.time_bucket.floor(pd.Timestamp(now_))
    else:
        candle_now = now_

    if strategy_universe.data_universe.liquidity:
        if strategy_universe.data_universe.time_bucket != TimeBucket.not_applicable:
            tvl_now = strategy_universe.data_universe.liquidity_time_bucket.floor(pd.Timestamp(now_))
        else:
            tvl_now = now_
    else:
        tvl_now = None

    logger.info(
        "display_strategy_universe(): now: %s, candles at: %s, tvl at: %s, pairs: %d, candles: %d, tvl rows: %d",
        now_,
        candle_now,
        tvl_now,
        strategy_universe.data_universe.pairs.get_count(),
        strategy_universe.data_universe.candles.get_candle_count() if strategy_universe.data_universe.candles else 0,
        strategy_universe.data_universe.liquidity.get_sample_count() if strategy_universe.data_universe.liquidity else 0,
    )

    for pair in strategy_universe.iterate_pairs():
        benchmark = pair.other_data.get("benchmark")
        data = {
            "id": pair.internal_id,
            "base": pair.base.token_symbol,
            "quote": pair.quote.token_symbol,
            "exchange": pair.exchange_name,
            "fee %": pair.fee * 100,
            "type:": "benchmark" if benchmark else "traded"
        }

        if show_price:
            try:
                data["price"], data["last_price_at"] = strategy_universe.data_universe.candles.get_price_with_tolerance(
                    pair=pair.internal_id,
                    when=candle_now,
                    tolerance=tolerance,
                )
                candles = strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id)
                data["first_price_at"] = candles.index[0]
            except CandleSampleUnavailable:
                data["price"] = "<not avail>"
                data["first_price_at"] = "-"
                data["last_price_at"] = "-"

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

        if show_tax:
            buy_tax = pair.get_buy_tax()
            sell_tax = pair.get_sell_tax()
            buy_tax_fmt = f"{buy_tax:.1%}" if buy_tax is not None else "-"
            sell_tax_fmt = f"{sell_tax:.1%}" if sell_tax is not None else "-"
            data["tax"] = f"{buy_tax_fmt} / {sell_tax_fmt}"

        pairs.append(data)

    df = pd.DataFrame(pairs)

    if sort_numeric:
        df['numeric_values'] = df[sort_key].replace("-", 0).replace("<not avail>", 0).astype(int)
    else:
        df['numeric_values'] = df[sort_key]

    df = df.sort_values(
        by="numeric_values", 
        ascending=sort_ascending,
    ).set_index("id")

    df = df.drop(columns=['numeric_values'])

    # Make human-readable DF

    def _format_float_2(value):
        if isinstance(value, float):
            return f"{value:.2f}"
        return value

    def _format_float_0(value):
        if isinstance(value, float):
            return f"{value:,.0f}"
        return value
    
    df['price'] = df['price'].apply(_format_float_2)
    if "tvl" in df.columns:
        df['tvl'] = df['tvl'].apply(_format_float_0)    
    return df
