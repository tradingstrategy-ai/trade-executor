"""Helper functions for creating Freqtrade trading universe."""

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier, TradingPairKind


def create_freqtrade_pair(
    freqtrade_id: str,
    api_url: str,
    exchange_name: str,
    reserve_currency: AssetIdentifier,
    transfer_method: str,
    recipient_address: str | None = None,
    vault_address: str | None = None,
    broker_id: int | None = None,
    orderly_account_id: str | None = None,
    token_id: str | None = None,
    is_mainnet: bool = True,
    fee_tolerance: str = "1.0",
    confirmation_timeout: int = 600,
    poll_interval: int = 10,
    internal_id: int | None = None,
    internal_exchange_id: int | None = None,
    info_url: str | None = None,
) -> TradingPairIdentifier:
    """Create a Freqtrade trading pair identifier.

    Args:
        freqtrade_id: Unique identifier for the Freqtrade bot instance
        api_url: Freqtrade API URL (e.g., "http://localhost:8080")
        exchange_name: Exchange name (e.g., "binance", "aster")
        reserve_currency: Reserve currency asset (e.g., USDC, USDT)
        transfer_method: Capital transfer method. One of: "on_chain_transfer", "aster", "hyperliquid", "orderly_vault"
        recipient_address: Wallet address for on_chain_transfer
        vault_address: Vault contract address for vault-based transfers
        broker_id: Broker ID for Aster/Orderly
        orderly_account_id: Orderly account ID (32 bytes hex)
        token_id: Token ID for Orderly
        is_mainnet: Mainnet (True) or testnet (False) for Hyperliquid
        fee_tolerance: Maximum fee variance in reserve currency units
        confirmation_timeout: Seconds to wait for balance confirmation
        poll_interval: Seconds between balance checks
        internal_id: Internal pair ID
        internal_exchange_id: Internal exchange ID
        info_url: URL to info page

    Returns:
        TradingPairIdentifier configured for Freqtrade

    Example:
        >>> from tradingstrategy.chain import ChainId
        >>> usdc = AssetIdentifier(
        ...     chain_id=ChainId.polygon.value,
        ...     address="0x2791bca1f2de4661ed88a30c99a7a9449aa84174",
        ...     token_symbol="USDC",
        ...     decimals=6,
        ... )
        >>> pair = create_freqtrade_pair(
        ...     freqtrade_id="momentum-bot",
        ...     api_url="http://localhost:8080",
        ...     exchange_name="binance",
        ...     reserve_currency=usdc,
        ...     transfer_method="on_chain_transfer",
        ...     recipient_address="0xabcdef...",
        ... )
    """
    # Create a synthetic "Freqtrade asset" for the base token
    # Base token symbol is the freqtrade_id
    # Use hash of freqtrade_id as synthetic address (must start with 0x)
    import hashlib
    synthetic_address = "0x" + hashlib.sha256(f"freqtrade-{freqtrade_id}".encode()).hexdigest()[:40]

    freqtrade_asset = AssetIdentifier(
        chain_id=reserve_currency.chain_id,
        address=synthetic_address,
        token_symbol=freqtrade_id.upper(),
        decimals=reserve_currency.decimals,  # Same decimals as reserve
    )

    # Build other_data with Freqtrade configuration
    other_data = {
        "freqtrade_id": freqtrade_id,
        "freqtrade_api_url": api_url,
        "freqtrade_exchange": exchange_name,
        "freqtrade_transfer_method": transfer_method,
    }

    # Add deposit-specific fields
    if recipient_address:
        other_data["freqtrade_recipient_address"] = recipient_address
    if vault_address:
        other_data["freqtrade_vault_address"] = vault_address
    if broker_id is not None:
        other_data["freqtrade_broker_id"] = broker_id
    if orderly_account_id:
        other_data["freqtrade_orderly_account_id"] = orderly_account_id
    if token_id:
        other_data["freqtrade_token_id"] = token_id
    if transfer_method == "hyperliquid":
        other_data["freqtrade_is_mainnet"] = is_mainnet

    # Add timeout/polling config
    other_data["freqtrade_fee_tolerance"] = fee_tolerance
    other_data["freqtrade_confirmation_timeout"] = confirmation_timeout
    other_data["freqtrade_poll_interval"] = poll_interval

    # Use synthetic addresses for pool and exchange (must start with 0x)
    pool_address = "0x" + hashlib.sha256(f"freqtrade-pool-{freqtrade_id}".encode()).hexdigest()[:40]
    exchange_address = "0x" + hashlib.sha256(f"freqtrade-exchange-{exchange_name}".encode()).hexdigest()[:40]

    return TradingPairIdentifier(
        base=freqtrade_asset,
        quote=reserve_currency,
        pool_address=pool_address,
        exchange_address=exchange_address,
        internal_id=internal_id,
        internal_exchange_id=internal_exchange_id,
        info_url=info_url,
        fee=0.0,  # No LP fees for Freqtrade
        reverse_token_order=False,
        kind=TradingPairKind.freqtrade,
        exchange_name=exchange_name,
        other_data=other_data,
    )


def load_freqtrade_bots(
    freqtrade_bots: list[dict],
    reserve_currency: AssetIdentifier,
) -> list[TradingPairIdentifier]:
    """Load Freqtrade bot configurations and create trading pairs.

    Args:
        freqtrade_bots: List of Freqtrade bot config dicts from strategy file
        reserve_currency: Reserve currency asset (e.g., USDC, USDT)

    Returns:
        List of TradingPairIdentifier objects for Freqtrade bots

    Example strategy configuration:

        >>> FREQTRADE_BOTS = [
        ...     {
        ...         "freqtrade_id": "momentum-bot",
        ...         "api_url": "http://localhost:8080",
        ...         "exchange_name": "binance",
        ...         "transfer_method": "on_chain_transfer",
        ...         "recipient_address": "0xabcdef...",
        ...     },
        ...     {
        ...         "freqtrade_id": "aster-bot",
        ...         "api_url": "http://localhost:8081",
        ...         "exchange_name": "aster",
        ...         "transfer_method": "aster",
        ...         "vault_address": "0x123456...",
        ...         "broker_id": 0,
        ...     },
        ... ]
        >>> pairs = load_freqtrade_bots(FREQTRADE_BOTS, usdc)
    """
    pairs = []

    for bot_config in freqtrade_bots:
        pair = create_freqtrade_pair(
            freqtrade_id=bot_config["freqtrade_id"],
            api_url=bot_config["api_url"],
            exchange_name=bot_config["exchange_name"],
            reserve_currency=reserve_currency,
            transfer_method=bot_config["transfer_method"],
            recipient_address=bot_config.get("recipient_address"),
            vault_address=bot_config.get("vault_address"),
            broker_id=bot_config.get("broker_id"),
            orderly_account_id=bot_config.get("orderly_account_id"),
            token_id=bot_config.get("token_id"),
            is_mainnet=bot_config.get("is_mainnet", True),
            fee_tolerance=bot_config.get("fee_tolerance", "1.0"),
            confirmation_timeout=bot_config.get("confirmation_timeout", 600),
            poll_interval=bot_config.get("poll_interval", 10),
        )
        pairs.append(pair)

    return pairs
