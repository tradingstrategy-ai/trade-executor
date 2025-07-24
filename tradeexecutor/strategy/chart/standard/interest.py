"""Interest/vault profit calculations."""
import pandas as pd

from tradeexecutor.analysis.credit import calculate_yield_metrics, YieldType
from tradeexecutor.strategy.chart.definition import ChartInput


def lending_pool_interest_accrued(
    input: ChartInput,
) -> pd.DataFrame:
    """How much our strategy accrued in interest.

    :return: Table with statistics
    """
    state = input.state
    interest_df = calculate_yield_metrics(
        state,
        yield_type=YieldType.credit
    )
    return interest_df


def vault_statistics(
    input: ChartInput,
) -> pd.DataFrame:
    """Yield vault statistics table.

    :return: Table with statistics
    """
    state = input.state
    interest_df = calculate_yield_metrics(
        state,
        yield_type=YieldType.vault
    )
    return interest_df