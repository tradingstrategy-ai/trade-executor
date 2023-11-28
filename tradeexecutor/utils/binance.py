import datetime
import pandas as pd
from eth_typing import HexAddress

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.trading_strategy_universe import Dataset, TradingStrategyUniverse
from tradingstrategy.timebucket import TimeBucket

from tradingstrategy.binance.constants import (
    BINANCE_CHAIN_ID,
    BINANCE_CHAIN_SLUG,
    BINANCE_EXCHANGE_ID,
    BINANCE_EXCHANGE_SLUG,
    BINANCE_EXCHANGE_ADDRESS,
    BINANCE_EXCHANGE_TYPE,
    BINANCE_FEE,
    BINANCE_SUPPORTED_QUOTE_TOKENS,
    split_binance_symbol,
)
from tradingstrategy.exchange import Exchange, ExchangeUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.utils.format import string_to_eth_address
from tradingstrategy.lending import (
    LendingReserve,
    LendingProtocolType,
    LendingReserveAdditionalDetails,
)


def generate_pairs_for_binance(
    symbols: list[str],
) -> list[TradingPairIdentifier]:
    """Generate trading pair identifiers for Binance data.
    
    :param symbols: List of symbols to generate pairs for
    :return: List of trading pair identifiers
    """
    return [generate_pair_for_binance(symbol, i) for i, symbol in enumerate(symbols)]


def generate_pair_for_binance(
    symbol: str,
    internal_id: int,
    fee: float = BINANCE_FEE,
    base_token_decimals: int = 18,
    quote_token_decimals: int = 18,
) -> TradingPairIdentifier:
    """Generate a trading pair identifier for Binance data.

    Binance data is not on-chain, so we need to generate the identifiers
    for the trading pairs.

    .. note:: Internal exchange id is hardcoded to 129875571 and internal id to 134093847


    :param symbol: E.g. `ETHUSDT`
    :return: Trading pair identifier
    """
    assert 0 < fee < 1, f"Bad fee {fee}. Must be 0..1"

    assert symbol.endswith(BINANCE_SUPPORTED_QUOTE_TOKENS), f"Bad symbol {symbol}"

    base_token_symbol, quote_token_symbol = split_binance_symbol(symbol)

    base = AssetIdentifier(
        int(BINANCE_CHAIN_ID.value),
        string_to_eth_address(base_token_symbol),
        base_token_symbol,
        base_token_decimals,
    )

    quote = AssetIdentifier(
        int(BINANCE_CHAIN_ID.value),
        string_to_eth_address(quote_token_symbol),
        quote_token_symbol,
        quote_token_decimals,
    )

    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=string_to_eth_address(f"{base_token_symbol}{quote_token_symbol}"),
        exchange_address=BINANCE_EXCHANGE_ADDRESS,
        fee=BINANCE_FEE,
        internal_exchange_id=BINANCE_EXCHANGE_ID,
        internal_id=internal_id,
    )


def generate_exchange_for_binance(pair_count: int) -> Exchange:
    """Generate an exchange identifier for Binance data."""
    return Exchange(
        chain_id=BINANCE_CHAIN_ID,
        chain_slug=ChainId(BINANCE_CHAIN_SLUG),
        exchange_id=BINANCE_EXCHANGE_ID,
        exchange_slug=BINANCE_EXCHANGE_SLUG,
        address=BINANCE_EXCHANGE_ADDRESS,
        exchange_type=BINANCE_EXCHANGE_TYPE,
        pair_count=pair_count,
    )


def generate_exchange_universe_for_binance(pair_count: int) -> Exchange:
    """Generate an exchange universe for Binance data."""
    return ExchangeUniverse.from_collection([generate_exchange_for_binance(pair_count)])


@staticmethod
def add_info_columns_to_ohlc(df: pd.DataFrame, pairs: dict[str, TradingPairIdentifier]):
    """Add single pair informational columns to an OHLC dataframe.

    :param *args: Each argument is a dict with the format {symbol: pair}
        E.g. {'ETHUSDT': TradingPairIdentifier(...)}

    :return: The same dataframe with added columns
    """

    for symbol, pair in pairs.items():
        if symbol not in df["symbol"].values:
            raise ValueError(f"Symbol {symbol} not found in DataFrame")

        # Update the DataFrame only for the rows where 'symbol' matches
        mask = df["symbol"] == symbol
        df.loc[mask, "base_token_symbol"] = pair.base.token_symbol
        df.loc[mask, "quote_token_symbol"] = pair.quote.token_symbol
        df.loc[mask, "exchange_slug"] = "binance"
        df.loc[mask, "chain_id"] = int(pair.base.chain_id)
        df.loc[mask, "fee"] = pair.fee * 10_000
        df.loc[mask, "pair_id"] = pair.internal_id
        df.loc[mask, "buy_volume_all_time"] = 0
        df.loc[mask, "address"] = pair.pool_address
        df.loc[mask, "exchange_id"] = pair.internal_exchange_id
        df.loc[mask, "token0_address"] = pair.base.address
        df.loc[mask, "token1_address"] = pair.quote.address
        df.loc[mask, "token0_symbol"] = pair.base.token_symbol
        df.loc[mask, "token1_symbol"] = pair.quote.token_symbol
        df.loc[mask, "token0_decimals"] = pair.base.decimals
        df.loc[mask, "token1_decimals"] = pair.quote.decimals

    return df


def generate_lending_reserve_for_binance(
    asset_symbol: str,
    address: HexAddress,
    reserve_id: int,
    asset_decimals=18,
) -> LendingReserve:
    """Generate a lending reserve for Binance data.

    Binance data is not on-chain, so we need to generate the identifiers
    for the trading pairs.

    :param asset_symbol: E.g. `ETH`
    :param address: address of the reserve
    :param reserve_id: id of the reserve
    :return: LendingReserve
    """

    assert isinstance(reserve_id, int), f"Bad reserve_id {reserve_id}"

    atoken_symbol = f"a{asset_symbol.upper()}"
    vtoken_symbol = f"v{asset_symbol.upper()}"

    return LendingReserve(
        reserve_id=reserve_id,
        reserve_slug=asset_symbol.lower(),
        protocol_slug=LendingProtocolType.aave_v3,
        chain_id=BINANCE_CHAIN_ID,
        chain_slug=BINANCE_CHAIN_SLUG,
        asset_id=1,
        asset_symbol=asset_symbol,
        asset_address=address,
        asset_decimals=asset_decimals,
        atoken_id=1,
        asset_name=asset_symbol,
        atoken_symbol=atoken_symbol,
        atoken_address=string_to_eth_address(atoken_symbol),
        atoken_decimals=18,
        vtoken_id=1,
        vtoken_symbol=vtoken_symbol,
        vtoken_address=string_to_eth_address(vtoken_symbol),
        vtoken_decimals=18,
        additional_details=LendingReserveAdditionalDetails(
            ltv=0.825,
            liquidation_threshold=0.85,
        ),
    )


from tradeexecutor.strategy.pandas_trader.alternative_market_data import load_candle_universe_from_dataframe
from tradingstrategy.binance.downloader import BinanceDownloader
from tradingstrategy.lending import LendingReserveUniverse, LendingCandleUniverse

def load_binance_dataset(
    symbols: list[str] | str,
    candle_time_bucket: TimeBucket,
    stop_loss_time_bucket: TimeBucket,
    start_at: datetime.datetime | None = None,
    end_at: datetime.datetime | None = None,
) -> Dataset:
    """Load a Binance dataset.
    
    This is the one-stop shop function for loading all your Binance data. It can include
    candlestick, stop loss, lending and supply data for all valid symbols.

    If start_at and end_at are not provided, the entire dataset will be loaded.

    :param symbols: List of symbols to load
    :param candle_time_bucket: Time bucket for candle data
    :param stop_loss_time_bucket: Time bucket for stop loss data
    :param start_at: Start time for data
    :param end_at: End time for data
    :return: Dataset
    """
    if isinstance(symbols, str):
        symbols = [symbols]

    downloader = BinanceDownloader()

    pairs = generate_pairs_for_binance(symbols)

    # use stop_loss_time_bucket since, in this case, it's more granular data than the candle_time_bucket
    # we later resample to the higher time bucket for the backtest candles
    df = downloader.fetch_candlestick_data(
        symbols,
        stop_loss_time_bucket,
        start_at,
        end_at,
    )

    candle_df = add_info_columns_to_ohlc(df, {symbol: pair for symbol, pair in zip(symbols, pairs)})

    # TODO use stop_loss_candle_universe (have to fix it)
    candle_universe, stop_loss_candle_universe = load_candle_universe_from_dataframe(
        pair=pairs[0],  # TODO fix to be multipair
        df=candle_df,
        include_as_trigger_signal=True,
        resample=candle_time_bucket, 
    )

    exchange_universe = generate_exchange_universe_for_binance(pair_count=len(pairs))

    pairs_df = candle_universe.get_pairs_df()

    reserves = []
    reserve_id = 1
    for pair in pairs:
        reserves.append(generate_lending_reserve_for_binance(pair.base.token_symbol, pair.base.address, reserve_id)) 
        reserves.append(generate_lending_reserve_for_binance(pair.quote.token_symbol, pair.quote.address, reserve_id + 1))
        reserve_id += 2

    lending_reserve_universe = LendingReserveUniverse({reserve.reserve_id: reserve for reserve in reserves})

    lending_candle_type_map = downloader.load_lending_candle_type_map({reserve.reserve_id: reserve.asset_symbol for reserve in reserves}, candle_time_bucket, start_at, end_at)

    lending_candle_universe = LendingCandleUniverse(lending_candle_type_map, lending_reserve_universe)

    dataset = Dataset(
        time_bucket=candle_time_bucket,
        exchanges=exchange_universe,
        pairs=pairs_df,
        candles=candle_universe.df,
        backtest_stop_loss_time_bucket=stop_loss_time_bucket,
        backtest_stop_loss_candles=candle_universe.df,
        lending_candles=lending_candle_universe,
        lending_reserves=lending_reserve_universe,
    )

    return dataset


def create_binance_universe(
    symbols: list[str] | str,
    candle_time_bucket: TimeBucket,
    stop_loss_time_bucket: TimeBucket,
    start_at: datetime.datetime | None = None,
    end_at: datetime.datetime | None = None,
) -> TradingStrategyUniverse:
    """Create a Binance universe that can be used for backtesting.
    
    Similarly to `load_binance_dataset`, this function loads all the data needed for backtesting,
    including candlestick, stop loss, lending and supply data for all valid symbols.

    :param symbols: List of symbols to load
    :param candle_time_bucket: Time bucket for candle data
    :param stop_loss_time_bucket: Time bucket for stop loss data
    :param start_at: Start time for data
    :param end_at: End time for data
    :return: Trading strategy universe
    """
    dataset = load_binance_dataset(
        symbols,
        candle_time_bucket,
        stop_loss_time_bucket,
        start_at,
        end_at,
    )

    pair = generate_pair_for_binance("ETHUSDT", 1)
    pair_ticker = pair.identifier_to_pair_ticker('binance')

    # iterrate
    pair_tickers = []
    for index, row in dataset.pairs.iterrows():
        pair_tickers.append((row['base_token_symbol'], row['quote_token_symbol']))

    universe = TradingStrategyUniverse.create_limited_pair_universe(
        dataset=dataset,
        chain_id=BINANCE_CHAIN_ID,
        exchange_slug=BINANCE_EXCHANGE_SLUG,
        pairs=pair_tickers,
    )

    return universe