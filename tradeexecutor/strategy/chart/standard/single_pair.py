"""Single trading pair analysis."""
import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.visual.single_pair import visualise_single_pair_positions_with_duration_and_slippage, display_positions_table

from plotly.graph_objects import Figure


def trading_pair_price_and_trades(
    input: ChartInput,
) -> Figure:
    """Chart of a trades on a single trading pair.

    - Takes trading pair as an input

    :return:
        Figure visualising price and trades
    """
    state = input.state
    strategy_universe = input.strategy_universe

    assert input.pairs and len(input.pairs) >= 1, "This chart only supports a single trading pair."
    pair = input.pairs[0]

    all_trades = [t for t in state.portfolio.get_all_trades() if t.pair == pair]
    print(f"We have total {len(all_trades)} trades on {pair}")

    start = None
    end = None

    figure = visualise_single_pair_positions_with_duration_and_slippage(
        state=state,
        candles=strategy_universe.data_universe.candles.get_candles_by_pair(pair.internal_id),
        pair_id=pair.internal_id,
        execution_context=input.execution_context,
        title=f"Positions on {pair}",
        start_at=start,
        end_at=end,
    )
    return figure


def trading_pair_positions(
    input: ChartInput,
) -> pd.DataFrame:
    """Get a list of positions of a single trading pair.

    - Takes trading pair as an input

    :return:
        Human readable table
    """
    state = input.state
    strategy_universe = input.strategy_universe

    assert input.pairs and len(input.pairs) >= 1, "This chart only supports a single trading pair."
    pair = input.pairs[0]

    df = display_positions_table(state, pair, sort_by="PnL USD", ascending=True)
    return df
