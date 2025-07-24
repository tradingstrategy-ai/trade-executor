"""Vault charts."""
import pandas as pd

from tradeexecutor.analysis.credit import display_vault_position_table
from tradeexecutor.analysis.vault import visualise_vaults
from tradeexecutor.strategy.chart.definition import ChartInput
from plotly.graph_objects import Figure

from tradeexecutor.visual.position import visualise_position, calculate_position_timeline


def all_vaults_share_price_and_tvl(
    input: ChartInput,
    printer=print,
) -> list[Figure]:
    """Render share price and TVL for all vaults.

    - Get vault pairs from the strategy universe

    :return:
        List of figures
    """
    figures = visualise_vaults(input.strategy_universe, printer=printer)
    return figures


def vault_position_timeline(
    input: ChartInput,
    cut_off_date: pd.Timestamp = None,
    height=2000,
    width=1200,
) -> tuple[Figure, pd.DataFrame]:
    """How a single vault position evolved over time.

    - Takes vault pair as an input

    :return:
        Figure visualising the position timeline and a DataFrame with individual trades
    """

    state = input.state
    strategy_universe = input.strategy_universe

    assert input.pairs and len(input.pairs) == 1, "This chart only supports a single vault pair."
    pair = input.pairs[0]

    all_positions = list(state.portfolio.get_all_positions())
    position_id = None
    for p in reversed(all_positions):
        if p.pair == pair:
            position_id = p.position_id
            break

    assert position_id is not None, f"Position for pair {pair} not found in portfolio."

    position = state.portfolio.get_position_by_id(position_id)
    position_df = calculate_position_timeline(
        strategy_universe,
        position,
        end_at=state.backtest_data.end_at,

    )

    if cut_off_date:
        position_df = position_df[position_df.index < cut_off_date]

    fig = visualise_position(
        position,
        position_df,
        extended=True,
        autosize=False,
        height=height,
        width=width,
    )

    with pd.option_context('display.min_rows', 500):  # Show up to 50 rows
        # Assuming df is your DataFrame and condition is your boolean mask
        mask = position_df.delta != 0  # Your original condition
        extended_mask = mask | mask.shift(1).fillna(False) | mask.shift(-1).fillna(False)

        df = position_df[extended_mask]
        return fig, df


def all_vault_positions(
    chart_input: ChartInput,
) -> pd.DataFrame:
    """Display all vault positions in a table.

    """
    state = chart_input.state
    vault_df = display_vault_position_table(state)
    return vault_df
