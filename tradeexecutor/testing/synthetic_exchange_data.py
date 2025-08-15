"""Generate synthetic exchange."""
from typing import Optional

from tradeexecutor.backtest.backtest_routing import BacktestRoutingModel
from tradeexecutor.state.types import BPS
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.pair import DEXPair


def generate_exchange(
    exchange_id: int,
    chain_id: ChainId,
    address="0x0000000000000000000000000000000000000000",
    exchange_slug="<exchange slug not filled>"
) -> Exchange:
    exchange = Exchange(
        chain_id=chain_id,
        chain_slug=chain_id.get_slug(),
        exchange_slug=exchange_slug,
        exchange_id=exchange_id,
        address=address,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=0,
    )
    return exchange


def generate_simple_routing_model(
    universe: TradingStrategyUniverse,
    trading_fee: Optional[BPS] = None,
    expected_exchages: int = 1,
) -> BacktestRoutingModel:
    """Creates a routing model for data generated synthetically.

    - Assumes there is only one exchange in the trading universe

    - Assumes all pairs in the trading universe have the same quote token and its stablecoin
    """

    assert len(universe.data_universe.exchanges) == expected_exchages, f"Expected exactly one exchange in the universe, got {len(universe.data_universe.exchanges)}"
    assert len(universe.reserve_assets) == 1

    reserve_asset = universe.reserve_assets[0]
    exchange = next(iter(universe.data_universe.exchanges))

    pair: DEXPair
    for pair in universe.data_universe.pairs.iterate_pairs():
        assert pair.exchange_id == exchange.exchange_id, f"Pair had exchange_id {pair.exchange_id}, expected {exchange.exchange_id}"
        assert pair.quote_token_symbol == reserve_asset.token_symbol
        assert pair.quote_token_address == reserve_asset.address

    # Allowed exchanges as factory -> router pairs,
    # by their smart contract addresses
    factory_router_map = {
        generate_random_ethereum_address(): (exchange.address, None)
    }

    # For three way trades, which pools we can use
    allowed_intermediary_pairs = {
    }

    return BacktestRoutingModel(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=reserve_asset.address,
        trading_fee=trading_fee,
        three_legged_resolution=False,
    )



def generate_vault_routing_model(
    universe: TradingStrategyUniverse,
    trading_fee: Optional[BPS] = None,
    expected_exchages: int = 2,
) -> BacktestRoutingModel:
    """Creates a routing model for data generated synthetically, including vaults.

    - Assumes there is only one exchange in the trading universe

    - Assumes all pairs in the trading universe have the same quote token and its stablecoin
    """

    assert len(universe.data_universe.exchanges) == expected_exchages, f"Expected exactly one exchange in the universe, got {len(universe.data_universe.exchanges)}"
    assert len(universe.reserve_assets) == 1

    reserve_asset = universe.reserve_assets[0]

    pair: DEXPair
    for pair in universe.data_universe.pairs.iterate_pairs():
        assert pair.exchange_id, f"Pair lacks exchange_id: {pair}"
        exchange = universe.data_universe.exchange_universe.get_by_id(pair.exchange_id)
        assert exchange is not None, f"Pair had exchange_id {pair.exchange_id}, but ExchangeUniverse does not have it. We have: {universe.data_universe.exchange_universe.exchanges}"
        assert exchange.address

    # Allowed exchanges as factory -> router pairs,
    # by their smart contract addresses
    factory_router_map = {
        exchange.address: (exchange.address, None) for exchange in universe.data_universe.exchange_universe.exchanges.values()
    }

    # For three way trades, which pools we can use
    allowed_intermediary_pairs = {
    }

    return BacktestRoutingModel(
        factory_router_map,
        allowed_intermediary_pairs,
        reserve_token_address=reserve_asset.address,
        trading_fee=trading_fee,
    )
