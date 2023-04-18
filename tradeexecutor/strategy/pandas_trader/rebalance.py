"""Alpha model rebalancing.

Based on the new alpha model weights, rebalance the existing portfolio.
"""
import logging
from _decimal import Decimal
from typing import List, Dict

from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.weighting import check_normalised_weights

logger = logging.getLogger(__name__)


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


def rebalance_portfolio_old(
        position_manager: PositionManager,
        new_weights: Dict[int, float],
        portfolio_total_value: USDollarAmount,
        min_trade_threshold: USDollarAmount = 10.0,
) -> List[TradeExecution]:
    """Rebalance a portfolio based on alpha model weights.

    .. warning ::

        This is old deprecated method. Do not use anymore.

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

        if dollar_diff < 0:
            # Calculate token amount
            quantity_diff = Decimal(position_manager.estimate_asset_quantity(pair, dollar_diff))
        else:
            quantity_diff = None

        logger.info("Rebalancing %s, old weight: %f, new weight: %f, diff: %f USD %f %s",
                    pair,
                    existing_weights.get(pair_id, 0),
                    weight,
                    dollar_diff,
                    quantity_diff,
                    pair.base.token_symbol,
                    )

        if abs(dollar_diff) < min_trade_threshold:
            logger.info("Not doing anything, value %f below trade threshold %f", value, min_trade_threshold)
        else:
            position_rebalance_trades = position_manager.adjust_position(pair, dollar_diff, quantity_diff, weight)
            assert len(position_rebalance_trades) == 1, "Assuming always on trade for rebalance"
            logger.info("Adjusting holdings for %s: %s", pair, position_rebalance_trades[0])
            trades += position_rebalance_trades

    trades.sort(key=lambda t: t.get_execution_sort_position())

    # Return all rebalance trades
    return trades