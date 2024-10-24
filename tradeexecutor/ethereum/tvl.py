"""Real-time DEX pool market depth measurement."""
from typing import Collection

from web3 import Web3
from eth_defi.token import fetch_erc20_details
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import USDollarAmount, TokenAmount, USDollarPrice
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradingstrategy.stablecoin import is_stablecoin_like


class CurrencyConversionRateMissing(Exception):
    """We cannot get TVL because the quote token needs to have an exchange rate loaded."""



def _fetch_uni_v2_v3_quote_token_tvl(
    web3: Web3,
    pair: TradingPairIdentifier,
) -> TokenAmount:

    token_address = Web3.to_checksum_address(pair.quote.address)
    pool_address = Web3.to_checksum_address(pair.pool_address)

    erc_20 = fetch_erc20_details(
        web3,
        token_address,
    )
    tvl = erc_20.fetch_balance_of(pool_address)
    return tvl


def fetch_quote_token_tvl_with_exchange_rate(
    web3: Web3,
    pair: TradingPairIdentifier,
    exchange_rates={},
) -> USDollarAmount:
    """Fetch real-time TVL of any trading pool.

    - Get the locked in quote token amount in a bool

    - Quote token locked is a proxy for the real market depth,
      although the shape of the liquidity will vary

    - Uniswap v2 and v3 area looked depending on the pool type

    - The quote token (WETH) is translated to the US dollars.

    .. note ::

        At the moment, Uniswap v2 and v3 pairs supported only.

    :param pair:
        Trading pair

    :param exchange_rates:
        address -> conversion raet mapping for WETH, others.
    """

    quote_token_tvl = _fetch_uni_v2_v3_quote_token_tvl(web3, pair)

    # float is fine for approxs
    quote_token_tvl = float(quote_token_tvl)

    # The pair is volatile-USD, no currency conversion needed
    if is_stablecoin_like(pair.quote.token_symbol):
        return quote_token_tvl

    if pair.quote.address not in exchange_rates:
        raise CurrencyConversionRateMissing(f"TVL for {pair} is not in USD-nominated token, and exchange rate is missing. Rates are: {exchange_rates}")

    return exchange_rates[pair.quote.address] * quote_token_tvl


def find_usd_rate(
    strategy_universe: TradingStrategyUniverse,
    asset: AssetIdentifier,
) -> USDollarPrice:
    """Find WETH/USDT or WETH/USDC rate.

    - This is approximation, not the real time rate

    - We do not care about the timestamp of the conversion rate,
      because TVL / liquidity is always going to be an approximation
    """

    exchange_rate_pair = None

    for pair in strategy_universe.iterate_pairs():
        if pair.base == asset:
            exchange_rate_pair = pair
            break

    if exchange_rate_pair is None:
        raise CurrencyConversionRateMissing(f"Could not find USD exchange rate pair for the asset: {asset}")

    candles = strategy_universe.data_universe.candles.get_candles_by_pair(exchange_rate_pair.internal_id)
    if candles is None:
        raise CurrencyConversionRateMissing(f"Found a USD exchange rate pair for the asset: {asset}, pair {exchange_rate_pair}, but it does not have OHCLV data loaded in the strategy universe")

    return candles.iloc[-1]["close"]


def fetch_quote_token_tvls(
    web3: Web3,
    strategy_universe: TradingStrategyUniverse,
    pairs: Collection[TradingPairIdentifier],
) -> dict[TradingPairIdentifier, USDollarAmount]:
    """Fetch real-time TVL of any trading pool.

    - Get the real-time quote token TVLs of a trading pool in USD

    - Only works in live trading, not in backtesting, as we are using real time data

    - In theory, this logic allows you to trade arbitrary small pools and the lower limit
      becomes transaction gas efficiency, as you start to lose more money tx fees
      than you make profit from small trades

    - Quote token locked is a proxy for the real market depth,
      although the shape of the liquidity will vary

    - This measurement is a different from TVL,
      as TVL counts both sides of the pool and may be asymmetric for Uniswap v3

    - Uniswap v2 and v3 supported

    - The quote token (WETH) is translated to the US dollars,
      using an approximation rate for some pair picked from strategy universe

    .. note ::

        At the moment, Uniswap v2 and v3 pairs supported only.

    Example how to cap trade size dynamically to the available real-time onchain liquidity:

    .. code-block:: python

        from tradeexecutor.ethereum.tvl import fetch_quote_token_tvls

        class Parameters:
            min_trade_size_usd = 5.00  # If the pair does not have enough real time quote token TVL, skip trades smaller than this
            max_pool_ownership = 0.01  # Only do trades where we are less than 1% of the pool quote token TVL

        def decide_trades(input):

            # Prefetch TVLs for all pairs in a single go
            if input.execution_context.mode.is_live_trading():
                all_pairs = strategy_universe.iterate_pairs()
                pair_tvls = fetch_quote_token_tvls(
                    input.web3,
                    input.strategy_universe,
                    all_pairs,
                )
            else:
                pair_tvls = None

            #
            # ... decisions go here ...
            #
            buy_amount = cash * parameters.allocation

            if pair_tvls:
                tvl = pair_tvls[pair]
                pool_capped_usd = tvl * parameters.max_pool_ownership

                if buy_amount > pool_capped_usd:
                    logger.trade(
                        "Pool quote token TVL %f USD, our cap %f USD, capping trade amount, cap percent %f",
                        tvl,
                        pool_capped_usd,
                        parameters.max_pool_ownership,
                    )
                    buy_amount = pool_capped_usd

            if buy_amount >= parameters.min_trade_size_usd:

                logger.trade("Opening position for %s with %f USDC", pair, buy_amount)

                trades += position_manager.open_spot(
                    pair,
                    value=buy_amount,
                    stop_loss_pct=parameters.stop_loss_pct,
                )
            else:
                logger.trade(
                    "Skipping trade size %f USD, because it is below our minimum threshold %f USD",
                    buy_amount,
                    parameters.min_trade_size_usd,
                )


    :param web3:
        Web3 connection to fetch the real time TVL

    :param strategy_universe:
        Need OHCLV price data to look up exchange raets.

    :param pairs:
        List of trading pairs for which we need TVL.

    :return:
        Mapping of (trading pair -> quote token TVL in USD)
    """

    # First will WETH, etc. conversion rates
    exchange_rates = {}
    for pair in pairs:
        if not is_stablecoin_like(pair.quote.token_symbol):
            exchange_rates[pair.quote.address] = find_usd_rate(strategy_universe, pair.quote)

    # Then ge tall TVLs
    return {pair: fetch_quote_token_tvl_with_exchange_rate(web3, pair,exchange_rates) for pair in pairs}
