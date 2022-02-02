import datetime
from decimal import Decimal

from tradeexecutor.state.state import TradingPairIdentifier, AssetIdentifier, State, TradeExecution, TradeType
from tradingstrategy.pair import DEXPair


def resolve_pair_and_address_asset(pair: DEXPair) -> TradingPairIdentifier:
    base = AssetIdentifier(
        chain_id=pair.chain_id,
        address=pair.base_token_address,
        token_symbol=pair.base_token_symbol,
        decimals=None
        )

    quote = AssetIdentifier(
        chain_id=pair.chain_id,
        address=pair.quote_token_address,
        token_symbol=pair.quote_token_symbol,
        decimals=None
        )

    trade_pair = TradingPairIdentifier(
        base=base,
        quote=quote,
        pair_id=pair.pair_id,
        pool_address=pair.address
    )

    return trade_pair


def create_trade(clock_at: datetime.datetime, state: State, pair: DEXPair, quantity: Decimal, assumed_price: float, trade_type: TradeType=TradeType.rebalance) -> TradeExecution:
    """Helper to create a trade instructions"""

    reserve_currency, reserve_currency_price = state.portfolio.get_default_reserve_currency()

    position, trade = state.create_trade(
        ts=clock_at,
        pair=pair,
        quantity=quantity,
        assumed_price=assumed_price,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_currency,
        reserve_currency_price=reserve_currency_price)

    return trade
