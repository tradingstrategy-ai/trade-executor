"""Perform a test trade on a universe."""
import logging
import datetime
from decimal import Decimal

from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.hot_wallet_sync import EthereumHotWalletReserveSyncer
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

logger = logging.getLogger(__name__)


def make_test_trade(
        execution_model: ExecutionModel,
        pricing_model: PricingModel,
        reserve_syncer: EthereumHotWalletReserveSyncer,
        state: State,
        universe: TradingStrategyUniverse,
        routing_model: RoutingModel,
        routing_state: RoutingState,
        amount=Decimal("1.0"),
):
    """Perform a test trade.

    Buy and sell 1 token worth for 1 USD to check that
    our trade routing works.
    """

    ts = datetime.datetime.utcnow()\

    # Sync nonce for the hot wallet
    execution_model.initialize()

    data_universe: Universe = universe.universe

    reserve_asset = universe.get_reserve_asset()

    # TODO: Supports single pair universes only for now
    raw_pair = data_universe.pairs.get_single()
    pair = translate_trading_pair(raw_pair)

    # Get estimated price for the asset we are going to buy
    assumed_price = pricing_model.get_buy_price(
        ts,
        pair,
        amount,
    )

    logger.info("Making a test trade on pair: %s, for %f %s price is %f %s/%s",
                pair,
                amount,
                reserve_asset.token_symbol,
                assumed_price,
                pair.base.token_symbol,
                reserve_asset.token_symbol,
                )

    # Sync any incoming stablecoin transfers
    # that have not been synced yet
    reserve_syncer(
        state.portfolio,
        ts,
        universe.reserve_assets,
    )

    # Create PositionManager helper class
    # that helps open and close positions
    position_manager = PositionManager(
        ts,
        universe,
        state,
        pricing_model,

    )

    # The message left on the test positions and trades
    notes = "A test trade created with perform-test-trade command line command"

    # Open the test position only if there isn't position already open
    # on the previous run
    position = state.portfolio.get_position_by_trading_pair(pair)

    if position is None:
        # Create trades to open the position
        trades = position_manager.open_1x_long(
            pair,
            float(amount),
            notes=notes,
        )

        trade = trades[0]

        # Compose the trades as approve() + swapTokenExact(),
        # broadcast them to the blockchain network and
        # wait for the confirmation
        execution_model.execute_trades(
            ts,
            state,
            trades,
            routing_model,
            routing_state,
        )

        position_id = trade.position_id
        position = state.portfolio.get_position_by_id(position_id)

    logger.info("Position %s open. Now closing the position.", position)

    # Recreate the position manager for the new timestamp,
    # as time has passed
    ts = datetime.datetime.utcnow()
    position_manager = PositionManager(
        ts,
        universe,
        state,
        pricing_model,
    )

    trade = position_manager.close_position(
        position,
        notes=notes,
    )

    execution_model.execute_trades(
            ts,
            state,
            [trade],
            routing_model,
            routing_state,
        )

    logger.info("All ok")



