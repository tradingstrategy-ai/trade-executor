"""Reverse engineering Trading Strategy trading universe from the local EVM tester Uniswap v2 deployment."""
from typing import List

import pandas as pd
from web3 import Web3

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse, Exchange, ExchangeType
from tradingstrategy.pair import DEXPair, PandasPairUniverse


def create_pair_universe(web3: Web3, exchange: Exchange, pairs: List[TradingPairIdentifier]) -> PandasPairUniverse:
    """Creates a PairUniverse from Trade Executor test data.

    PairUniverse is used by QSTrader based tests, so we need to support it.
    """

    chain_id = ChainId(web3.eth.chain_id)

    data = []
    for p in pairs:
        assert p.exchange_address
        assert p.base.decimals
        assert p.quote.decimals
        assert p.base.address != p.quote.address
        dex_pair = DEXPair(
            pair_id=int(p.get_identifier(), 16),
            chain_id=chain_id,
            exchange_id=exchange.exchange_id if exchange else 1,
            exchange_address=p.exchange_address if exchange else None,
            address=p.pool_address,
            dex_type=ExchangeType.uniswap_v2,
            base_token_symbol=p.base.token_symbol,
            quote_token_symbol=p.quote.token_symbol,
            token0_symbol=p.base.token_symbol,
            token1_symbol=p.quote.token_symbol,
            token0_address=p.base.address,
            token1_address=p.quote.address,
            token0_decimals=p.base.decimals,
            token1_decimals=p.quote.decimals,
        )
        data.append(dex_pair.to_dict())
    df = pd.DataFrame(data)
    return PandasPairUniverse(df)


def create_exchange_universe(web3: Web3, uniswaps: List[UniswapV2Deployment]) -> ExchangeUniverse:
    """Create an exchange universe with a list of Uniswap v2 deployments."""

    exchanges = {}
    chain_id = ChainId(web3.eth.chain_id)
    for u in uniswaps:
        e = Exchange(
            chain_id=chain_id,
            chain_slug="tester",
            exchange_id=int(u.factory.address, 16),
            exchange_slug="uniswap_tester",
            address=u.factory,
            exchange_type=ExchangeType.uniswap_v2,
            pair_count=99999,
        )
        exchanges[e.exchange_id] = e
    return ExchangeUniverse(exchanges=exchanges)

