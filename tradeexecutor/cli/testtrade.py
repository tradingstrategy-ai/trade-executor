"""Perform a test trade on a universe."""
import logging
import datetime
from decimal import Decimal

from web3 import Web3

from tradingstrategy.universe import Universe
from tradingstrategy.pair import HumanReadableTradingPairDescription
from tradingstrategy.exchange import ExchangeUniverse

from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.state.types import Percent
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.statistics.statistics_table import serialise_long_short_stats_as_json_table
from tradeexecutor.strategy.execution_context import ExecutionMode
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.utils.accuracy import sum_decimal
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair

logger = logging.getLogger(__name__)


def make_test_trade(
    web3: Web3,
    execution_model: ExecutionModel,
    pricing_model: PricingModel,
    sync_model: SyncModel,
    state: State,
    universe: TradingStrategyUniverse,
    routing_model: RoutingModel,
    routing_state: RoutingState,
    max_slippage: Percent,
    amount=Decimal("1.0"),
    pair: HumanReadableTradingPairDescription | None = None,
    buy_only: bool = False,
    test_short: bool = True,
    test_credit_supply: bool = True,
):
    """Perform a test trade.

    Buy and sell 1 token worth for 1 USD to check that
    our trade routing works.

    If the pair can be shorted, open and close short position for 1 USD.
    """

    assert isinstance(sync_model, SyncModel)
    assert isinstance(universe, TradingStrategyUniverse)
    assert type(max_slippage) == float
    assert max_slippage > 0, f"max_slippage not set"

    ts = datetime.datetime.utcnow()

    # Sync nonce for the hot wallet
    logger.info("make_test_trade() at %s", ts)
    execution_model.initialize()

    data_universe: Universe = universe.data_universe

    reserve_asset = universe.get_reserve_asset()

    if data_universe.pairs.get_count() > 1 and not pair:
        raise RuntimeError(
            "You are using a multipair universe.\n"
            "Use the --pair flag to perform a test trade on a specific pair.\n"
            "Use check-universe command to get list of pair ids.\n"
            "Alternatively, use the --all-pairs flag to perform the test trade on all pairs.")
    
    if pair:
        if data_universe.exchanges:
            exchange_universe = ExchangeUniverse.from_collection(data_universe.exchanges)
        elif data_universe.exchange_universe:
            exchange_universe = data_universe.exchange_universe
        else:
            raise RuntimeError("You need to provide the exchange_universe when creating the universe")

        raw_pair = data_universe.pairs.get_pair(*pair, exchange_universe=exchange_universe)
    else:
        raw_pair = data_universe.pairs.get_single()

    logger.info("Getting price for pair %s using %s", raw_pair, pricing_model)
    pair = translate_trading_pair(raw_pair)

    # Get estimated price for the asset we are going to buy
    assumed_price_structure = pricing_model.get_buy_price(
        ts,
        pair,
        amount,
    )

    logger.info(
        "Making a test trade on pair: %s, for %f %s price is %f %s/%s",
        pair,
        amount,
        reserve_asset.token_symbol,
        assumed_price_structure.mid_price,
        pair.base.token_symbol,
        reserve_asset.token_symbol,
    )

    logger.info("Sync model is %s", sync_model)
    logger.info("Trading university reserve asset is %s", universe.get_reserve_asset())

    # Sync any incoming stablecoin transfers
    # that have not been synced yet
    balance_updates = sync_model.sync_treasury(
        ts,
        state,
        list(universe.reserve_assets),
        post_valuation=True,
    )

    logger.info("sync_treasury() received balance update events: %s", balance_updates)

    if sync_model.has_position_sync():
        balance_updates = sync_model.sync_positions(
            ts,
            state,
            universe,
            pricing_model,
        )
        logger.info("sync_positions(): received balance update events: %s", balance_updates)

    vault_address = sync_model.get_key_address()
    hot_wallet = sync_model.get_hot_wallet()
    gas_at_start = hot_wallet.get_native_currency_balance(web3)

    logger.info("Account data before test trade")
    logger.info("  Vault address: %s", vault_address)
    logger.info("  Hot wallet address: %s", hot_wallet.address)
    logger.info("  Hot wallet balance: %s", gas_at_start)

    if isinstance(sync_model, EnzymeVaultSyncModel):
        vault = sync_model.vault
        logger.info("  Comptroller address: %s", vault.comptroller.address)
        logger.info("  Vault owner: %s", vault.vault.functions.getOwner().call())
        sync_model.check_ownership()

    if len(state.portfolio.reserves) == 0:
        raise RuntimeError("No reserves detected for the strategy. Does your wallet/vault have USDC deposited for trading?")

    reserve_currency = state.portfolio.get_default_reserve_position().asset.token_symbol
    reserve_currency_at_start = state.portfolio.get_default_reserve_position().get_value()

    logger.info("  Reserve currency balance: %s %s", reserve_currency_at_start, reserve_currency)

    assert reserve_currency_at_start > 0, f"No deposits available to trade. Vault at {vault_address}"

    # Create PositionManager helper class
    # that helps open and close positions
    position_manager = PositionManager(
        ts,
        universe,
        state,
        pricing_model,
        default_slippage_tolerance=max_slippage,
    )

    # The message left on the test positions and trades
    notes = "A test trade created with perform-test-trade command line command"

    # Open the test position only if there isn't position already open
    # on the previous run

    buy_trade = open_short_trade = close_short_trade = open_credit_supply_trade = close_credit_supply_trade = None

    position = state.portfolio.get_position_by_trading_pair(pair)

    if position is None:
        # Create trades to open the position
        trades = position_manager.open_spot(
            pair,
            float(amount),
            notes=notes,
            flags={TradeFlag.test_trade},
        )

        trade = trades[0]
        buy_trade = trade

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

        assert trade.is_test()
        assert position.is_test()

        if not trade.is_success() or not position.is_open():
            # Alot of diagnostics to debug Arbitrum / WBTC issues
            trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
            direct_balance_updates = position.get_base_token_balance_update_quantity()

            logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

            logger.error("Test buy failed: %s", trade)
            logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
            logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
            logger.error("Trade dump:\n%s", trade.get_debug_dump())
            logger.error("Position dump:\n%s", position.get_debug_dump())

        if not trade.is_success():
            raise AssertionError("Test buy failed.")

        if not position.is_open():
            raise AssertionError("Test buy succeed, but the position was not opened\n"
                                 "Check for dust corrections.")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    logger.info("Position %s is open. Now closing the position.", position)

    if not buy_only:

        logger.info("Position %s is open. Now closing the position.", position)

        # Recreate the position manager for the new timestamp,
        # as time has passed
        ts = datetime.datetime.utcnow()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=max_slippage,
        )

        trades = position_manager.close_position(
            position,
            notes=notes,
            flags={TradeFlag.test_trade},
        )
        assert len(trades) == 1
        sell_trade = trades[0]

        execution_model.execute_trades(
                ts,
                state,
                [sell_trade],
                routing_model,
                routing_state,
            )

        assert sell_trade.is_test()

        if not sell_trade.is_success():
            logger.error("Test sell failed: %s", sell_trade)
            logger.error("Trade dump:\n%s", sell_trade.get_debug_dump())
            raise AssertionError("Test sell failed")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    else:
        sell_trade = None

    if universe.has_any_lending_data() and universe.can_open_short(datetime.datetime.utcnow(), pair) and test_short:
        short_pair = universe.get_shorting_pair(pair)
        position = state.portfolio.get_position_by_trading_pair(short_pair)

        if position is None:

            # Recreate the position manager for the new timestamp,
            # as time has passed
            ts = datetime.datetime.utcnow()
            position_manager = PositionManager(
                ts,
                universe,
                state,
                pricing_model,
                default_slippage_tolerance=max_slippage,
            )

            # Create trades to open the position
            trades = position_manager.open_short(
                pair,
                float(amount),
                notes=notes,
                leverage=2,
                flags={TradeFlag.test_trade},
            )

            trade = trades[0]
            open_short_trade = trade

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

            assert position.is_test()
            assert open_short_trade.is_test()

            if not trade.is_success() or not position.is_open():
                # Alot of diagnostics to debug Arbitrum / WBTC issues
                trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
                direct_balance_updates = position.get_base_token_balance_update_quantity()

                logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

                logger.error("Test open short failed: %s", trade)
                logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
                logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
                logger.error("Trade dump:\n%s", trade.get_debug_dump())
                logger.error("Position dump:\n%s", position.get_debug_dump())

            if not trade.is_success():
                raise AssertionError("Test buy failed.")

            if not position.is_open():
                raise AssertionError("Test buy succeed, but the position was not opened\n"
                                     "Check for dust corrections.")

            long_short_metrics_latest = serialise_long_short_stats_as_json_table(
                state, None
            )

            update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

        # Close the short

        # Recreate the position manager for the new timestamp,
        # as time has passed
        ts = datetime.datetime.utcnow()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=max_slippage,
        )

        # Create trades to open the position
        trades = position_manager.close_short(
            position,
            flags={TradeFlag.test_trade},
        )

        trade = trades[0]
        close_short_trade = trade

        assert close_short_trade.is_test()

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

        if not trade.is_success() or position.is_open():
            # Alot of diagnostics to debug Arbitrum / WBTC issues
            trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
            direct_balance_updates = position.get_base_token_balance_update_quantity()

            logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

            logger.error("Close short failed: %s", trade)
            logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
            logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
            logger.error("Trade dump:\n%s", trade.get_debug_dump())
            logger.error("Position dump:\n%s", position.get_debug_dump())

        if not trade.is_success():
            raise AssertionError(f"Short close failed, trade not marked as success: {trade.get_revert_reason()}")

        if not position.is_closed():
            raise AssertionError("Short close succeed, but the position was not opened\n"
                                 "Check for dust corrections.")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    # Credit supply test trade:
    # TODO: Clean up - this is written incorrectly
    if universe.has_any_lending_data() and universe.can_open_credit_supply(datetime.datetime.utcnow(), pair) and test_credit_supply:
        credit_pair = universe.get_credit_supply_pair()
        position = state.portfolio.get_position_by_trading_pair(credit_pair)

        if position is None:

            # Recreate the position manager for the new timestamp,
            # as time has passed
            ts = datetime.datetime.utcnow()
            position_manager = PositionManager(
                ts,
                universe,
                state,
                pricing_model,
                default_slippage_tolerance=max_slippage,
            )

            # Create trades to open the position
            trades = position_manager.open_credit_supply_position_for_reserves(float(amount), flags={TradeFlag.test_trade})

            trade = trades[0]
            open_credit_supply_trade = trade

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

            assert position.is_test()
            assert open_credit_supply_trade.is_test()

            if not trade.is_success() or not position.is_open():
                # Alot of diagnostics to debug Arbitrum / WBTC issues
                trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
                direct_balance_updates = position.get_base_token_balance_update_quantity()

                logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

                logger.error("Test open credit supply failed: %s", trade)
                logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
                logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
                logger.error("Trade dump:\n%s", trade.get_debug_dump())
                logger.error("Position dump:\n%s", position.get_debug_dump())

            if not trade.is_success():
                raise AssertionError("Test open credit supply failed.")

            if not position.is_open():
                raise AssertionError("Test buy succeed, but the position was not opened\n"
                                     "Check for dust corrections.")

            long_short_metrics_latest = serialise_long_short_stats_as_json_table(
                state, None
            )

            update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

        # Close credit supply

        # Recreate the position manager for the new timestamp,
        # as time has passed
        ts = datetime.datetime.utcnow()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=max_slippage,
        )

        # Create trades to open the position
        trades = position_manager.close_credit_supply_position(position, flags={TradeFlag.test_trade})

        trade = trades[0]
        close_credit_supply_trade = trade

        assert close_credit_supply_trade.is_test()

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

        if not trade.is_success() or position.is_open():
            # Alot of diagnostics to debug Arbitrum / WBTC issues
            trades = sum_decimal([t.get_position_quantity() for t in position.trades.values() if t.is_success()])
            direct_balance_updates = position.get_base_token_balance_update_quantity()

            logger.error("Trade quantity: %s, direct balance updates: %s", trades, direct_balance_updates)

            logger.error("Close credit supply failed: %s", trade)
            logger.error("Tx hash: %s", trade.blockchain_transactions[-1].tx_hash)
            logger.error("Revert reason: %s", trade.blockchain_transactions[-1].revert_reason)
            logger.error("Trade dump:\n%s", trade.get_debug_dump())
            logger.error("Position dump:\n%s", position.get_debug_dump())

        if not trade.is_success():
            raise AssertionError(f"Test close credit supply failed, trade not marked as success: {trade.get_revert_reason()}")

        if not position.is_closed():
            raise AssertionError("Short close succeed, but the position was not closed\n"
                                 "Check for dust corrections.")

        long_short_metrics_latest = serialise_long_short_stats_as_json_table(
            state, None
        )
        
        update_statistics(datetime.datetime.utcnow(), state.stats, state.portfolio, ExecutionMode.real_trading, long_short_metrics_latest=long_short_metrics_latest)

    gas_at_end = hot_wallet.get_native_currency_balance(web3)
    reserve_currency_at_end = state.portfolio.get_default_reserve_position().get_value()

    logger.info("Test trade report")
    logger.info("  Gas spent: %s", gas_at_start - gas_at_end)
    logger.info("  Trades done currently: %d", len(list(state.portfolio.get_all_trades())))
    logger.info("  Reserves currently: %s %s", reserve_currency_at_end, reserve_currency)
    logger.info("  Reserve currency spent: %s %s", reserve_currency_at_start - reserve_currency_at_end, reserve_currency)
    if buy_trade:
        logger.info("  Buy trade price, expected: %s, actual: %s (%s)", buy_trade.planned_price, buy_trade.executed_price, pair.get_ticker())
    if sell_trade:
        logger.info("  Sell trade price, expected: %s, actual: %s (%s)", sell_trade.planned_price, sell_trade.executed_price, pair.get_ticker())
    if open_short_trade:
        logger.info("  Open short, expected: %s, actual: %s (%s)", open_short_trade.planned_price, open_short_trade.executed_price, short_pair.get_ticker())
    if close_short_trade:
        logger.info("  Close short, expected: %s, actual: %s (%s)", close_short_trade.planned_price, close_short_trade.executed_price, short_pair.get_ticker())
    if open_credit_supply_trade:
        logger.info("  Open credit supply, expected: %s, actual: %s (%s)", open_credit_supply_trade.planned_price, open_credit_supply_trade.executed_price, credit_pair.get_ticker())
    if close_credit_supply_trade:
        logger.info("  Close credit supply, expected: %s, actual: %s (%s)", close_credit_supply_trade.planned_price, close_credit_supply_trade.executed_price, credit_pair.get_ticker())
