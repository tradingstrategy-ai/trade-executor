from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.candle import GroupedCandleUniverse
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, \
    create_pair_universe_from_code
from tradingstrategy.exchange import Exchange
import pandas as pd


def create_synthetic_single_pair_universe(
    candles: pd.DataFrame,
    chain_id: ChainId, 
    exchange: Exchange,
    time_bucket: TimeBucket,
    pair: TradingPairIdentifier,
) -> TradingStrategyUniverse:

    assert type(candles) == pd.DataFrame, "candles must be a pandas DataFrame"
    assert type(chain_id) == ChainId, "chain_id must be of type ChainId"
    assert type(time_bucket) == TimeBucket, "time_bucket must be of type TimeBucket"
    assert type(pair) == TradingPairIdentifier, "pair must be of type TradingPairIdentifier"

    # Set up fake assets
    pair_universe = create_pair_universe_from_code(chain_id, [pair])

    # Load candles for backtesting
    candle_universe = GroupedCandleUniverse.create_from_single_pair_dataframe(candles)

    universe = Universe(
        time_bucket=time_bucket,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None
    )

    return TradingStrategyUniverse(universe=universe, reserve_assets=[pair.quote])