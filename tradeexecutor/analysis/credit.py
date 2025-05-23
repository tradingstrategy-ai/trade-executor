"""Credit and yield position profit analysis."""

import datetime
import enum

import pandas as pd

from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionMode


class YieldType(enum.Enum):
    """Which positions to calculate yield for."""
    credit = "credit"
    vault = "vault"


def calculate_yield_metrics(
    state: State,
    yield_type: YieldType = YieldType.credit,
    execution_mode=ExecutionMode.backtesting,
) -> pd.DataFrame:
    """Calculate interest metrics for a strategy backtest.

    - Credit earned on Aave

    Example:

    .. code-block:: python

        from tradeexecutor.analysis.credit import calculate_yield_metrics
        from tradeexecutor.analysis.credit import YieldType

        interest_df = calculate_yield_metrics(
            state,
            yield_type=YieldType.credit
        )
        display(interest_df)

    :param yield_type:
        Which positions to calculate yield for

    :return:
        Human readable DataFrame
    """

    assert isinstance(execution_mode, ExecutionMode)
    assert isinstance(yield_type, YieldType), f"Invalid yield type: {yield_type}"

    if execution_mode.is_backtesting():
        assert state.backtest_data
        end_at = state.backtest_data.end_at
        assert end_at

    match yield_type:
        case YieldType.credit:
            credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
            total_interest_earned_usd = sum(p.get_total_profit_usd() for p in credit_positions)
            interest_rates = [p.get_annualised_credit_interest() for p in credit_positions]
            deposit_trades = [t for p in credit_positions for t in p.trades.values() if t.is_credit_supply()]
            durations = [p.get_duration(partial=True, execution_mode=execution_mode, end_at=end_at) for p in credit_positions]
        case YieldType.vault:
            credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_vault()]
            durations = [p.get_duration(partial=True, execution_mode=execution_mode, end_at=end_at) for p in credit_positions]
            total_interest_earned_usd = sum(p.get_total_profit_usd() for p in credit_positions)
            interest_rates = [p.calculate_annualised_profit(duration) for p, duration in zip(credit_positions, durations)]
            deposit_trades = [t for p in credit_positions for t in p.trades.values() if t.is_buy()]
        case _:
            raise NotImplementedError()

    if len(credit_positions) == 0:
        data = {
            "Credit position count": 0,
        }
        return pd.DataFrame(list(data.items()), columns=['Name', 'Value']).set_index('Name')

    durations = [d for d in durations if d is not None]

    assert len(durations) > 0, f"calculate_yield_metrics({yield_type}): No durations available for positions: {credit_positions}"

    deposits = [t.get_value() for t in deposit_trades]
    min_interest = min(interest_rates)
    max_interest = max(interest_rates)
    avg_interest = sum(interest_rates) / len(interest_rates)
    avg_duration = sum(durations, start=datetime.timedelta(0)) / len(durations)
    total_deposit_flow = sum(deposits)
    max_deposit = max(deposits)
    min_deposit = min(deposits)
    avg_deposit = sum(deposits) / len(deposits)

    data = {
        "Position count": len(credit_positions),
        "Total interest earned": f"{total_interest_earned_usd:,.2f} USD",
        "Avg interest": f"{avg_interest:.2%}",
        "Min interest": f"{min_interest:.2%}",
        "Max interest": f"{max_interest:.2%}",
        "Avg credit position duration": avg_duration,
        "Total deposit flow in": f"{total_deposit_flow:,.2f} USD",
        "Min deposit": f"{min_deposit:,.2f} USD",
        "Avg deposit": f"{avg_deposit:,.2f} USD",
        "Max deposit": f"{max_deposit:,.2f} USD",
    }

    return pd.DataFrame(list(data.items()), columns=['Name', 'Value']).set_index('Name')


#: BBB
calculate_credit_metrics = calculate_yield_metrics