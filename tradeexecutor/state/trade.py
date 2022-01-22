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


def create_trade(clock_at: datetime.datetime, state: State, pair: DEXPair, quantity: Decimal, trade_type: TradeType=TradeType.rebalance) -> TradeExecution:
    """Helper to create a trade instructions"""

    trade_id = state.allocate_trade_id()
    trading_pair = resolve_pair_and_address_asset(pair)
    execution = TradeExecution(
        trade_id=trade_id,
        trade_type=trade_type,
        clock_at=clock_at,
        trading_pair=trading_pair,
        requested_quantity=quantity,
        started_at=datetime.datetime.utcnow(),
    )
    return execution