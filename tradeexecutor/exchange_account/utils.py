"""Shared utilities for exchange account setup.

Extracted from correct_accounts command so that both ``correct_accounts``
and ``start`` commands can reuse the same logic.
"""

import logging
from typing import Callable

from tradeexecutor.exchange_account.derive import DeriveNetwork

logger = logging.getLogger(__name__)


def create_exchange_account_value_func(
    positions,
    derive_owner_private_key: str | None,
    derive_session_private_key: str | None,
    derive_wallet_address: str | None,
    derive_network: DeriveNetwork,
    ccxt_exchange_id: str | None,
    ccxt_options: str | None,
    ccxt_sandbox: bool,
    logger,
):
    """Create account value function for exchange account positions.

    Dispatches to the appropriate protocol (Derive, CCXT) based on
    ``exchange_protocol`` in each position's ``other_data``.

    :param positions: List of exchange account positions
    :param derive_owner_private_key: Derive owner wallet private key
    :param derive_session_private_key: Derive session key private key
    :param derive_wallet_address: Derive wallet address (auto-derived if not provided)
    :param derive_network: Derive network (mainnet or testnet)
    :param ccxt_exchange_id: CCXT exchange identifier (e.g. "aster")
    :param ccxt_options: CCXT exchange constructor options as JSON string
    :param ccxt_sandbox: Whether to use CCXT sandbox mode
    :param logger: Logger instance
    :return: Account value function or None if credentials not provided
    """
    from decimal import Decimal
    from tradeexecutor.state.identifier import TradingPairIdentifier

    # Check which protocols are needed
    protocols = set()
    for p in positions:
        protocol = p.pair.get_exchange_account_protocol()
        if protocol:
            protocols.add(protocol)

    logger.info("Exchange account protocols needed: %s", protocols)

    # Set up Derive if needed
    derive_value_func = None
    if "derive" in protocols:
        if not derive_session_private_key:
            logger.error("Derive credentials required: DERIVE_SESSION_PRIVATE_KEY")
        elif not derive_owner_private_key and not derive_wallet_address:
            logger.error("Derive credentials required: either DERIVE_OWNER_PRIVATE_KEY or DERIVE_WALLET_ADDRESS must be provided")
        else:
            from eth_defi.derive.authentication import DeriveApiClient
            from eth_defi.derive.account import fetch_subaccount_ids

            is_testnet = (derive_network == DeriveNetwork.testnet)

            owner_account = None
            if derive_owner_private_key:
                from eth_account import Account
                owner_account = Account.from_key(derive_owner_private_key)

            if not derive_wallet_address:
                from eth_defi.derive.onboarding import fetch_derive_wallet_address
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

            derive_clients = {}
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

            if derive_clients:
                from tradeexecutor.exchange_account.derive import create_derive_account_value_func
                derive_value_func = create_derive_account_value_func(derive_clients)

    # Set up CCXT if needed
    ccxt_value_func = None
    if "ccxt" in protocols:
        if not ccxt_exchange_id:
            logger.error("CCXT exchange ID required: set CCXT_EXCHANGE_ID")
        elif not ccxt_options:
            logger.error("CCXT options required: set CCXT_OPTIONS (JSON string with apiKey, secret, etc.)")
        else:
            import json
            from tradeexecutor.exchange_account.ccxt_exchange import (
                create_ccxt_exchange,
                create_ccxt_account_value_func,
            )

            try:
                options = json.loads(ccxt_options)
            except json.JSONDecodeError as e:
                logger.error("Failed to parse CCXT_OPTIONS as JSON: %s", e)
                options = None

            if options:
                exchange = create_ccxt_exchange(
                    exchange_id=ccxt_exchange_id,
                    options=options,
                    sandbox=ccxt_sandbox,
                )
                logger.info("Created CCXT exchange: %s", ccxt_exchange_id)

                # Map all CCXT positions to this exchange instance
                exchanges = {}
                for p in positions:
                    if p.pair.get_exchange_account_protocol() == "ccxt":
                        account_id = p.pair.other_data.get("ccxt_account_id")
                        if account_id:
                            exchanges[account_id] = exchange

                if exchanges:
                    ccxt_value_func = create_ccxt_account_value_func(exchanges)
                    logger.info("Created CCXT account value function for %d account(s)", len(exchanges))

    # Create unified dispatcher
    if derive_value_func or ccxt_value_func:
        def unified_account_value_func(pair: TradingPairIdentifier) -> Decimal:
            protocol = pair.get_exchange_account_protocol()
            if protocol == "derive" and derive_value_func:
                return derive_value_func(pair)
            elif protocol == "ccxt" and ccxt_value_func:
                return ccxt_value_func(pair)
            else:
                raise ValueError(f"No account value function for protocol: {protocol}")

        return unified_account_value_func

    logger.error("No valid exchange API clients created")
    return None


def create_derive_value_func_from_credentials(
    derive_session_private_key: str,
    derive_owner_private_key: str | None,
    derive_wallet_address: str | None,
    derive_network: DeriveNetwork,
) -> Callable:
    """Create a Derive account value func from raw credentials.

    Unlike :py:func:`create_exchange_account_value_func`, this does not
    require positions to be known upfront. It creates a single client that
    dynamically handles any subaccount_id it encounters.

    Used by the ``start`` command where positions may not yet exist at
    startup time.

    :return:
        Function that takes a TradingPairIdentifier and returns account value in USD.
    """
    from decimal import Decimal
    from eth_defi.derive.authentication import DeriveApiClient
    from eth_defi.derive.account import fetch_account_summary
    from tradeexecutor.state.identifier import TradingPairIdentifier

    is_testnet = (derive_network == DeriveNetwork.testnet)

    owner_account = None
    if derive_owner_private_key:
        from eth_account import Account
        owner_account = Account.from_key(derive_owner_private_key)

    if not derive_wallet_address:
        from eth_defi.derive.onboarding import fetch_derive_wallet_address
        assert owner_account is not None, \
            "Either DERIVE_OWNER_PRIVATE_KEY or DERIVE_WALLET_ADDRESS must be provided"
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

    logger.info(
        "Created Derive client for %s (testnet=%s, wallet=%s)",
        "start command",
        is_testnet,
        derive_wallet_address,
    )

    def get_derive_account_value(pair: TradingPairIdentifier) -> Decimal:
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert pair.get_exchange_account_protocol() == "derive", \
            f"Not a Derive pair: {pair.get_exchange_account_protocol()}"

        subaccount_id = pair.get_exchange_account_id()
        if subaccount_id is None:
            raise ValueError(f"No exchange_subaccount_id in pair other_data: {pair}")

        try:
            summary = fetch_account_summary(client, subaccount_id)
            logger.debug(
                "Derive subaccount %d value: $%.2f",
                subaccount_id,
                summary.total_value_usd,
            )
            return summary.total_value_usd
        except Exception as e:
            logger.error(
                "Failed to get Derive account value for subaccount %d: %s",
                subaccount_id,
                e,
            )
            raise

    return get_derive_account_value
