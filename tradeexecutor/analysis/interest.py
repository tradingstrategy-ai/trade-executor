import pandas as pd

from tradeexecutor.state.state import State


def calculate_credit_metrics(
    state: State,
) -> pd.DataFrame:
    """Calculate interest metrics for a strategy backtest.

    - Credit earned
    """

    credit_positions = [p for p in state.portfolio.get_all_positions() if p.is_credit_supply()]
    total_interest_earned_usd = sum(p.get_total_profit_usd() for p in credit_positions)
    interest_rates = [p.get_annualised_credit_interest() for p in credit_positions]
    min_interest = min(interest_rates)
    max_interest = max(interest_rates)
    avg_interest = sum(interest_rates) / len(interest_rates)

    data = {
        "Credit position count": len(credit_positions),
        "Total interest earned": f"{total_interest_earned_usd:.2f} USD",
        "Avg interest": f"{avg_interest:.2%}",
        "Min interest": f"{min_interest:.2%}",
        "Max interest": f"{max_interest:.2%}",
    }

    return pd.DataFrame(list(data.items()), columns=['Name', 'Value'])
