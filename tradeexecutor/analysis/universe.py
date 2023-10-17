"""Trading universe analysis."""
import pandas as pd

from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.pair import PairNotFoundError
from tradingstrategy.stablecoin import is_stablecoin_like


def analyse_long_short_universe(
    strategy_universe: TradingStrategyUniverse,
) -> pd.DataFrame:
    """Display trading pairs and their lending reserves used in long/short strategy.

    Only available fpr backtesting for now.

    :param strategy_universe:
        Constructed trading universe

    :return:
        Summary table that can be displayed in notebooks
    """

    rows = []

    data_universe = strategy_universe.data_universe

    assert len(data_universe.chains) == 1, "Only single chain strategies supported"
    assert len(strategy_universe.reserve_assets) == 1, "Only single resrve strategies supported"
    assert data_universe.lending_candles is not None, "Lending rate data missing"
    assert data_universe.lending_reserves is not None, "Lending reserve data missing"
    assert data_universe.candles is not None, "Price data missing"

    chain_id = next(iter(data_universe.chains))
    quote_token = strategy_universe.reserve_assets[0]

    for reserve in data_universe.lending_reserves.iterate_reserves():

        stablecoin = is_stablecoin_like(reserve.asset_symbol)

        lending_link = reserve.get_link()

        try:
            rate_candles = data_universe.lending_candles.variable_borrow_apr.get_rates_by_reserve(reserve)
            lending_start = rate_candles.index[0]
        except KeyError:
            # Lending not available, lendign candles missing, reserve just added
            # and data is not yet there?
            lending_start = "-"

        try:
            trading_pair = data_universe.pairs.get_pair_by_human_description((chain_id, None, reserve.asset_symbol, quote_token.token_symbol))
            exchange = data_universe.pairs.get_exchange_for_pair(trading_pair)
            trading_pair_label = f"{trading_pair.base_token_symbol}-{trading_pair.quote_token_symbol} at {trading_pair.fee} BPS fee tier on {exchange.name}"
            trading_pair_link = trading_pair.get_link()
            price_candles = data_universe.candles.get_candles_by_pair(trading_pair.pair_id)

            trading_start = "-"
            if price_candles is not None:
                trading_start = price_candles.index[0]
                if trading_start:
                    trading_start = trading_start.strftime("%Y-%m-%d")

        except PairNotFoundError as e:
            trading_pair_label = "No AMM pools found"
            trading_start = "-"
            trading_pair_link = "-"

        rows.append({
            "Lending asset": reserve.asset_symbol,
            "Stablecoin": "yes" if stablecoin else "no",
            "Best trading pair": trading_pair_label,
            "Lending available at": lending_start,
            "Trading available at": trading_start,
            "Price data page": trading_pair_link,
            "Lending rate page": lending_link,
        })

    df = pd.DataFrame(rows)
    return df
