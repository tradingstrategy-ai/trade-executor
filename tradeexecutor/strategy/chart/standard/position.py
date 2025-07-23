"""Position visualisation and statistics."""
import pandas as pd

from tradeexecutor.strategy.chart.definition import ChartInput


def positions_at_end(
    input: ChartInput,
) -> pd.DataFrame:
    """Open positions at the end of the backtest/currently.
    """
    state = input.state
    data = []
    for p in list(state.portfolio.open_positions.values())[0:10]:
        data.append({
            "position_id": p.id,
            "token": p.pair.base.token_symbol,
            "value": p.get_value(),
        })

    df = pd.DataFrame(data)
    df = df.set_index("position_id", drop=True)
    return df