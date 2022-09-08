"""Alpha model rebalancing.

Based on the new alpha model weights, rebalance the existing portfolio.
"""
from typing import List, Dict

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution


def get_existing_portfolio_weights(portfolio: Portfolio) -> Dict[int, float]:
    """Calculate the existing portfolio weights.

    Cash is not included in the weighting.
    """

    total = portfolio.get_open_position_equity()
    result = {}
    for position in portfolio.open_positions.values():
        result[position.pair.internal_id] = position.get_value() / total
    return result


def rebalance_portfolio(
        portfolio: Portfolio,
        weights: Dict[int, float],
        portfolio_total_value: float) -> List[TradeExecution]:
    """Rebalance a portfolio based on alpha model weights.

    This will generate

    - Sells for the existing assets

    - Buys for new assetes or assets where we want to increase our position

    :param portfolio:
        Portfolio of our existing holdings

    :param weights:
        Each weight tells how much a certain trading pair
        we should hold in our portfolio.
        Pair id -> weight mappings.

    :param portfolio_total_value:
        Target portfolio value in USD

    :return:
        List of trades we need to execute to reach the target portfolio
    """

    existing_weights = get_existing_portfolio_weights(portfolio)

