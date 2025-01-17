import datetime

import pandas as pd

from tradeexecutor.state.state import State


def calculate_credit_metrics(
    state: State,
) -> pd.DataFrame:
    """Calculate interest metrics for a strategy backtest.

    - Credit earned on Aave

    :return:
        Human readable DataFrame
    """

    credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
    if len(credit_positions) == 0:
        data = {
            "Credit position count": 0,
        }
        return pd.DataFrame(list(data.items()), columns=['Name', 'Value']).set_index('Name')

    total_interest_earned_usd = sum(p.get_total_profit_usd() for p in credit_positions)
    interest_rates = [p.get_annualised_credit_interest() for p in credit_positions]
    durations = [p.get_duration() for p in credit_positions]
    durations = [d for d in durations if d is not None]
    deposit_trades = [t for p in credit_positions for t in p.trades.values() if t.is_credit_supply()]
    deposits = [t.get_value() for t in deposit_trades]
    min_interest = min(interest_rates)
    max_interest = max(interest_rates)
    avg_interest = sum(interest_rates) / len(interest_rates)
    avg_duration = sum(durations, start=datetime.timedelta(0)) / len(durations)
    total_deposit_flow = sum(deposits)
    max_deposit = max(deposits)
    min_deposit = min(deposits)
    avg_deposit = sum(deposits) / len(deposits)
    # import ipdb ; ipdb.set_trace()
    data = {
        "Credit position count": len(credit_positions),
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
