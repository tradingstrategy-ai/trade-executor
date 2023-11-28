import pandas as pd
from eth_typing import HexAddress

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier

from tradingstrategy.binance.constants import (
    BINANCE_CHAIN_ID,
    BINANCE_CHAIN_SLUG,
    BINANCE_EXCHANGE_ID,
    BINANCE_EXCHANGE_SLUG,
    BINANCE_EXCHANGE_ADDRESS,
    BINANCE_EXCHANGE_TYPE,
    BINANCE_FEE,
)
from tradingstrategy.exchange import Exchange, ExchangeUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.utils.format import string_to_eth_address
from tradingstrategy.lending import (
    LendingReserve,
    LendingProtocolType,
    LendingReserveAdditionalDetails,
)


def generate_pair_for_binance(
    base_token_symbol: str,
    quote_token_symbol: str,
    fee: float,
    internal_id: int,
    base_token_decimals: int = 18,
    quote_token_decimals: int = 18,
) -> TradingPairIdentifier:
    """Generate a trading pair identifier for Binance data.

    Binance data is not on-chain, so we need to generate the identifiers
    for the trading pairs.

    .. note:: Internal exchange id is hardcoded to 129875571 and internal id to 134093847

    :param base_token_symbol:
        E.g. ``ETH``

    :param quote_token_symbol:
        E.g. ``USDT``

    :return:
        Trading pair identifier
    """
    assert 0 < fee < 1, f"Bad fee {fee}. Must be 0..1"

    base = AssetIdentifier(
        int(ChainId.unknown.value),
        string_to_eth_address(base_token_symbol),
        base_token_symbol,
        base_token_decimals,
    )

    quote = AssetIdentifier(
        int(ChainId.unknown.value),
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
