"""Analytics for constructed pair universes."""
import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


def display_strategy_universe(
    strategy_universe: TradingStrategyUniverse
) -> pd.DataFrame:
    """Displays a constructed trading strategy universe in table format.

    - Designed for notebook debug
    - To be called at the end of ``create_trading_universe()``
    - Returns a human-readable table of selected trading pairs and their nature

    Pairs can be
    - Part of trading
    - Benchmarks
    - Benchmark pairs are detected by ``pair.other_data.get("benchmark")``

    :return:
        Human-readable DataFrame for ``display()``
    """

    pairs = []

    print("Universe is (including benchmark pairs):")
    for pair in strategy_universe.iterate_pairs():
        benchmark = pair.other_data.get("benchmark")
        pairs.append({
            "id": pair.internal_id,
            "base": pair.base.token_symbol,
            "quote": pair.quote.token_symbol,
            "exchange": pair.exchange_name,
            "fee %": pair.fee * 100,
            "type:": "benchmark/routed token" if benchmark else "traded token"
        })

    human_pair_universe_df = pd.DataFrame(pairs)
    return human_pair_universe_df
