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
) -> TradingPairIdentifier:
    """Generate a random pair.

    .. note:: Don't use this function for multipair strategies with the same tokens since the addresses are randomly generated.
    
    :param exchange:
        Exchange to use for the pair
    
    :param symbol0:
        Symbol for the first token
        
    :param symbol1:
        Symbol for the second token
    
    :param decimals0:
        Decimals for the first token
    
    :param decimals1:
        Decimals for the second token
    
    :param fee:
        Fee for the pair
    
    :param internal_id:
        Internal ID for the pair
    """


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