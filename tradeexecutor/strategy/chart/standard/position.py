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
        entry = {
            "position_id": p.position_id,
            "token": p.pair.base.token_symbol,
            "value": p.get_value(),
        }
        vault_name = p.pair.get_vault_name()
        if vault_name:
            entry["vault"] = vault_name
            entry["address"] = p.pair.pool_address
        data.append(entry)

    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.set_index("position_id", drop=True)
    return df