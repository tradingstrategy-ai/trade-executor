"""Sync exchange account positions with external perp DEX APIs.

This command syncs the internal state with the actual account values
from external perp DEXes like Derive or Hyperliquid.
"""

import datetime
import enum
import sys
from pathlib import Path
from typing import Optional

from typer import Option


class DeriveNetwork(enum.Enum):
    """Derive network selection."""
    mainnet = "mainnet"
    testnet = "testnet"

from .app import app
from ..bootstrap import prepare_executor_id, backup_state
from ..log import setup_logging
from . import shared_options
from ...state.state import State
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.strategy_module import read_strategy_module
from ...strategy.universe_model import UniverseOptions


@app.command()
def sync_exchange_account(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    log_level: str = shared_options.log_level,

    unit_testing: bool = shared_options.unit_testing,
    skip_save: bool = Option(False, "--skip-save", envvar="SKIP_SAVE", help="Do not update state file after sync. Only used in testing."),

    # Derive-specific options
    derive_owner_private_key: Optional[str] = Option(None, envvar="DERIVE_OWNER_PRIVATE_KEY", help="Derive owner wallet private key"),
    derive_session_private_key: Optional[str] = Option(None, envvar="DERIVE_SESSION_PRIVATE_KEY", help="Derive session key private key"),
    derive_wallet_address: Optional[str] = Option(None, envvar="DERIVE_WALLET_ADDRESS", help="Derive wallet address (auto-derived if not provided)"),
    derive_network: DeriveNetwork = Option(DeriveNetwork.mainnet, envvar="DERIVE_NETWORK", help="Derive network: mainnet or testnet"),
):
    """Sync exchange account positions with external perp DEX APIs.

    This command reads account values from external perp DEXes (Derive, Hyperliquid)
    and updates the internal state to reflect any PnL changes.

    Exchange account positions represent capital deployed to external exchanges
    where actual trading happens. This command syncs the tracked value with
    the actual account value from the exchange API.

    Example:

        trade-executor sync-exchange-account \\
            --strategy-file my_strategy.py \\
            --state-file state/my_strategy.json
    """
    global logger

    id = prepare_executor_id(id, strategy_file)
    logger = setup_logging(log_level)

    if not state_file:
        state_file = Path(f"state/{id}.json")

    store, state = backup_state(state_file, unit_testing=unit_testing)

    # Read strategy module
    mod = read_strategy_module(strategy_file)

    # Create execution context
    execution_context = ExecutionContext(
        mode=ExecutionMode.one_off,
        engine_version=mod.trading_strategy_engine_version,
    )

    # Create universe
    universe_options = UniverseOptions()
    strategy_universe = mod.create_trading_universe(
        ts=datetime.datetime.utcnow(),
        client=None,
        execution_context=execution_context,
        universe_options=universe_options,
    )

    logger.info("Strategy universe contains %d pairs", strategy_universe.data_universe.pairs.get_count())

    # Find exchange account positions
    exchange_account_positions = [
        p for p in state.portfolio.get_open_and_frozen_positions()
        if p.is_exchange_account()
    ]

    if not exchange_account_positions:
        logger.info("No exchange account positions found")
        sys.exit(0)

    logger.info("Found %d exchange account position(s)", len(exchange_account_positions))

    # Set up exchange API clients based on position protocols
    from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel

    account_value_func = _create_account_value_func(
        exchange_account_positions,
        derive_owner_private_key=derive_owner_private_key,
        derive_session_private_key=derive_session_private_key,
        derive_wallet_address=derive_wallet_address,
        derive_network=derive_network,
        logger=logger,
    )

    if account_value_func is None:
        logger.error("Failed to create account value function - check API credentials")
        sys.exit(1)

    # Create sync model and run sync
    sync_model = ExchangeAccountSyncModel(account_value_func)

    timestamp = datetime.datetime.utcnow()
    events = sync_model.sync_positions(
        timestamp=timestamp,
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=None,
    )

    logger.info("Sync completed with %d balance update(s)", len(events))
    for evt in events:
        logger.info("  Position %d: %s (change: %s)", evt.position_id, evt.notes, evt.quantity)

    if not skip_save:
        logger.info("Saving state to %s", store.path)
        store.sync(state)
    else:
        logger.info("Saving skipped (--skip-save)")

    sys.exit(0)


def _create_account_value_func(
    positions,
    derive_owner_private_key: str | None,
    derive_session_private_key: str | None,
    derive_wallet_address: str | None,
    derive_network: DeriveNetwork,
    logger,
):
    """Create account value function based on position protocols."""
    from decimal import Decimal

    # Check which protocols are needed
    protocols = set()
    for p in positions:
        protocol = p.pair.get_exchange_account_protocol()
        if protocol:
            protocols.add(protocol)

    logger.info("Exchange account protocols needed: %s", protocols)

    # Set up Derive if needed
    derive_clients = {}
    if "derive" in protocols:
        if not derive_owner_private_key or not derive_session_private_key:
            logger.error("Derive credentials required: DERIVE_OWNER_PRIVATE_KEY and DERIVE_SESSION_PRIVATE_KEY")
            return None

        from eth_account import Account
        from eth_defi.derive.authentication import DeriveApiClient
        from eth_defi.derive.account import fetch_subaccount_ids
        from eth_defi.derive.onboarding import fetch_derive_wallet_address

        is_testnet = (derive_network == DeriveNetwork.testnet)
        owner_account = Account.from_key(derive_owner_private_key)

        if not derive_wallet_address:
            derive_wallet_address = fetch_derive_wallet_address(
                owner_account.address,
                is_testnet=is_testnet,
            )

        client = DeriveApiClient(
            owner_account=owner_account,
            derive_wallet_address=derive_wallet_address,
            is_testnet=is_testnet,
            session_key_private=derive_session_private_key,
        )

        # Get all subaccounts
        subaccount_ids = fetch_subaccount_ids(client)
        logger.info("Found %d Derive subaccount(s)", len(subaccount_ids))

        # Create client for each subaccount needed by positions
        for p in positions:
            if p.pair.get_exchange_account_protocol() == "derive":
                subaccount_id = p.pair.get_exchange_account_id()
                if subaccount_id in subaccount_ids:
                    # Create a separate client for this subaccount
                    subaccount_client = DeriveApiClient(
                        owner_account=owner_account,
                        derive_wallet_address=derive_wallet_address,
                        is_testnet=is_testnet,
                        session_key_private=derive_session_private_key,
                    )
                    subaccount_client.subaccount_id = subaccount_id
                    derive_clients[subaccount_id] = subaccount_client
                else:
                    logger.warning("Subaccount %d not found in Derive account", subaccount_id)

    # Create unified account value function
    from tradeexecutor.exchange_account.derive import create_derive_account_value_func

    if derive_clients:
        return create_derive_account_value_func(derive_clients)

    # TODO: Add support for other protocols (Hyperliquid, etc.)

    logger.error("No valid exchange API clients created")
    return None
