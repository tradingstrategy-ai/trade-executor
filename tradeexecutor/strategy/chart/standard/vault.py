"""Vault charts."""

from tradeexecutor.analysis.vault import visualise_vaults
from tradeexecutor.strategy.chart.definition import ChartInput
from plotly.graph_objects import Figure


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
