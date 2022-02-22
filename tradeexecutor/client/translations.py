"""Translates a trading pair presentation from Trading Strategy client Pandas format to the trade executor format.

Trade executor work with multiple different strategies, not just Trading Strategy client based.
For example, you could have a completely on-chain data based strategy.
Thus, Trade Executor has its internal asset format.

This module contains functions to translate asset presentations between Trading Strategy client
and Trade Executor.
"""
from tradingstrategy.pair import DEXPair

from tradeexecutor.state.state import TradingPairIdentifier, AssetIdentifier


def translate_trading_pair(pair: DEXPair) -> TradingPairIdentifier:
    """Translate trading pair from Pandas universe to Trade Executor universe.

    This is called when a trade is made: this is the moment when trade executor data format must be made available.
    """
    base = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.base_token_address,
        token_symbol=pair.base_token_symbol,
        decimals=None
    )
    quote = AssetIdentifier(
        chain_id=pair.chain_id.value,
        address=pair.quote_token_address,
        token_symbol=pair.quote_token_symbol,
        decimals=None
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address=pair.address,
    )