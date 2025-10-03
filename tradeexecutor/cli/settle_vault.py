"""Vault settle utilities for CLI.

"""
import logging
import datetime

from tabulate import tabulate
from web3 import Web3

from tradeexecutor.analysis.position import display_positions
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.sync_model import SyncModel
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.strategy.valuation_update import update_position_valuations
from tradingstrategy.types import Percent
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)


def settle_vault(
    web3: Web3,
    execution_model: ExecutionModel,
    execution_context: ExecutionContext,
    pricing_model: PricingModel,
    sync_model: SyncModel,
    state: State,
    universe: TradingStrategyUniverse,
    routing_model: RoutingModel,
    routing_state: RoutingState,
    valuation_model: ValuationModel,
    unit_testing=False,
):
    """Settle vault logic"""

    assert isinstance(sync_model, SyncModel)
    assert isinstance(universe, TradingStrategyUniverse)
    assert isinstance(valuation_model, ValuationModel)
    assert isinstance(execution_context, ExecutionContext)

    assert isinstance(sync_model, LagoonVaultSyncModel), f"Only Lagoon vaults supported, got {sync_model}"

    ts = datetime.datetime.utcnow()

    # Sync nonce for the hot wallet
    execution_model.initialize()

    logger.info("Sync model is %s", sync_model)
    logger.info("Trading university reserve asset is %s", universe.get_reserve_asset())

    vault_address = sync_model.get_key_address()
    hot_wallet = sync_model.get_hot_wallet()
    gas_at_start = hot_wallet.get_native_currency_balance(web3)

    logger.info("Account data before starting to close all")
    logger.info("  Vault address: %s", vault_address)
    logger.info("  Hot wallet address: %s", hot_wallet.address)
    logger.info("  Hot wallet balance: %s", gas_at_start)

    if isinstance(sync_model, EnzymeVaultSyncModel):
        vault = sync_model.vault
        logger.info("  Comptroller address: %s", vault.comptroller.address)
        logger.info("  Vault owner: %s", vault.vault.functions.getOwner().call())
        sync_model.check_ownership()
    elif isinstance(sync_model, LagoonVaultSyncModel):
        vault = sync_model.vault
        logger.info("  Trading Strategy module address: %s", vault.trading_strategy_module_address)
        logger.info("  Safe address: %s", vault.safe_address)

    # Use unit_testing flag so this code path is easier to check
    if sync_model.has_async_deposits() or unit_testing:
        logger.info("Vault must be revalued before proceeding, using: %s", sync_model.__class__.__name__)
        update_position_valuations(
            timestamp=ts,
            state=state,
            universe=universe,
            execution_context=execution_context,
            routing_state=routing_state,
            valuation_model=valuation_model,
            long_short_metrics_latest=None,
        )
        valuation = state.portfolio.get_net_asset_value()

    # Sync any incoming stablecoin transfers
    # that have not been synced yet
    balance_updates = sync_model.sync_treasury(
        ts,
        state,
        list(universe.reserve_assets),
        post_valuation=True,
    )

    logger.info("We received balance update events: %s", balance_updates)

    # Velvet capital code path
    if sync_model.has_position_sync():
        sync_model.sync_positions(
            ts,
            state,
            universe,
            pricing_model
        )

    if len(state.portfolio.reserves) == 0:
        raise RuntimeError("No reserves detected for the strategy. Does your wallet/vault have USDC deposited for trading?")

    reserve_currency = state.portfolio.get_default_reserve_position().asset.token_symbol
    reserve_currency_at_start = state.portfolio.get_default_reserve_position().get_value()

    logger.info("  Reserve currency balance: %s %s", reserve_currency_at_start, reserve_currency)

    assert reserve_currency_at_start > 0, f"No deposits available to trade. Vault at {vault_address}"

    gas_at_end = hot_wallet.get_native_currency_balance(web3)
    reserve_currency_at_end = state.portfolio.get_default_reserve_position().get_value()

    logger.info("Sync report")
    logger.info("  Balance update events: %d", len(balance_updates))
    logger.info("  Current valuation: %f %s", valuation, reserve_currency)
    logger.info("  Gas spent: %s", gas_at_start - gas_at_end)
    logger.info("  Trades done currently: %d", len(list(state.portfolio.get_all_trades())))
    logger.info("  Reserves currently: %s %s", reserve_currency_at_end, reserve_currency)
    logger.info("  Reserve currency spent: %s %s", reserve_currency_at_start - reserve_currency_at_end, reserve_currency)

    df = display_positions(state.portfolio.frozen_positions.values())
    position_info = tabulate(df, headers='keys', tablefmt='rounded_outline')

    logger.info("Position data for positions that were closed:\n%s", position_info)
