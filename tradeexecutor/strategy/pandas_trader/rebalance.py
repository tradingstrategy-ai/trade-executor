"""Alpha model rebalancing.

Based on the new alpha model weights, rebalance the existing portfolio.
"""
import logging
from typing import List, Dict

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager


logger = logging.getLogger(__name__)


class BadWeightsException(Exception):
    """Sum of weights not 1."""



def check_normalised_weights(weights: Dict[int, float], epsilon=0.0001):
    """Check that the sum of weights is good."""

    total = sum(weights.values())
    if abs(total - 1) > epsilon:
        raise BadWeightsException(f"Total sum of normalised portfolio weights was not 1."
                                  f"Sum: {total}")


def clip_to_normalised(weights: Dict[int, float]) -> Dict[int, float]:
    """If the sum of the weights are not exactly 1, then decrease the largest member to make the same sum 1 precise.

    """
    total = sum(weights.values())
    diff = total - 1
    largest = max(weights.items(), key=lambda x: x[1])

    clipped = largest[1] - diff

    fixed = weights.copy()
    fixed[largest[0]] = clipped

    total = sum(fixed.values())
    assert total == 1
    return fixed


def get_existing_portfolio_weights(portfolio: Portfolio) -> Dict[int, float]:
    """Calculate the existing portfolio weights.

    Cash is not included in the weighting.
    """

    total = portfolio.get_open_position_equity()
    result = {}
    for position in portfolio.open_positions.values():
        result[position.pair.internal_id] = position.get_value() / total
    return result


def get_weight_diffs(
        existing_weights: Dict[int, float],
        new_weights: Dict[int, float],
) -> Dict[int, float]:
    """Get the weight diffs.

    The diffs include one entry for each token in existing or new portfolio.
    """

    # Check that both inputs are sane
    check_normalised_weights(new_weights)
    check_normalised_weights(existing_weights)

    diffs = {}
    for id, new_weight in new_weights.items():
        diffs[id] = new_weight - existing_weights.get(id, 0)

    # Refill gaps of old assets that did not appear
    # in the new portfolio
    for id, old_weight in existing_weights.items():
        if id not in diffs:
            # Sell all
            diffs[id] = -old_weight

    return diffs


def rebalance_portfolio(
        position_manager: PositionManager,
        new_weights: Dict[int, float],
        portfolio_total_value: float,
        min_trade_threshold=10.0,
) -> List[TradeExecution]:
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

        Each weight must be normalised in the range of 0...1
        and the total sum of the weights must be 1.

    :param portfolio_total_value:
        Target portfolio value in USD

    :param min_trade_threshold:
        If the notional value of a rebalance trade is smaller than this
        USD amount don't make a trade.

    :return:
        List of trades we need to execute to reach the target portfolio.
        The sells are sorted always before buys.
    """

    portfolio = position_manager.state.portfolio
    assert isinstance(portfolio, Portfolio)

    existing_weights = get_existing_portfolio_weights(portfolio)
    diffs = get_weight_diffs(existing_weights, new_weights)
    dollar_values = {pair_id: weight * portfolio_total_value for pair_id, weight in diffs.items()}

    # Generate trades
    trades: List[TradeExecution] = []

    for pair_id, value in dollar_values.items():
        pair = position_manager.get_trading_pair(pair_id)
        weight = new_weights.get(pair.internal_id, 0)
        dollar_diff = value

        logger.info("Rebalancing %s, old weight: %f, new weight: %f, diff: %f USD",
                    pair,
                    existing_weights.get(pair_id, 0),
                    weight,
                    dollar_diff)

        if abs(dollar_diff) < min_trade_threshold:
            logger.info("Not doing anything, value %f below trade threshold %f", value, min_trade_threshold)
        else:
            position_rebalance_trades = position_manager.adjust_position(pair, dollar_diff, weight)
            assert len(position_rebalance_trades) == 1, "Assuming always on trade for rebalacne"
            logger.info("Adjusting holdings for %s: %s", pair, position_rebalance_trades[0])
            trades += position_rebalance_trades

    trades.sort(key=lambda t: t.get_execution_sort_position())

    # Return all rebalance trades
    return trades
