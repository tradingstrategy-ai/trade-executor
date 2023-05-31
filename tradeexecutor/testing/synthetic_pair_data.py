import random

from tradeexecutor.state.state import AssetIdentifier, TradingPairIdentifier
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address


def generate_pair(
    exchange: Exchange, 
    symbol0: str = "OSMO", 
    symbol1: str = "ATOM",
    decimals0: int = 18,
    decimals1: int = 18, 
    fee: float = 0.0005,
    internal_id = random.randint(1, 1000)
):
    token1 = AssetIdentifier(ChainId.osmosis.value, generate_random_ethereum_address(), symbol0, decimals0, 1)
    token2 = AssetIdentifier(ChainId.osmosis.value, generate_random_ethereum_address(), symbol1, decimals1, 2)
    pair = TradingPairIdentifier(
        token1,
        token2,
        generate_random_ethereum_address(),
        exchange.address,
        internal_id=internal_id,
        internal_exchange_id=exchange.exchange_id,
        fee=fee
    )

    return pair