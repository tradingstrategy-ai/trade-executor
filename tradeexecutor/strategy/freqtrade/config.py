"""Configuration for Freqtrade integration."""

import enum
from dataclasses import dataclass
from decimal import Decimal

# Hyperliquid constants
HYPERLIQUID_BRIDGE_MAINNET = "0x2Df1c51E09aECF9cacB7bc98cB1742757f163dF7"
HYPERLIQUID_BRIDGE_TESTNET = "0x..."  # Add testnet bridge if needed
USDC_ARBITRUM_MAINNET = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"
USDC_ARBITRUM_TESTNET = "0x..."  # Add testnet USDC if needed

class FreqtradeDepositMethod(enum.StrEnum):
    """How capital is deposited to Freqtrade."""

    #: Simple ERC20 transfer to a wallet address (for CEX or Lagoon vault integration)
    on_chain_transfer = "on_chain_transfer"

    #: Aster vault deposit on BSC (approve + vault.deposit)
    aster_vault = "aster_vault"

    #: Hyperliquid deposit (bridge transfer on Arbitrum + SDK vault deposit)
    hyperliquid = "hyperliquid"

    #: Orderly vault deposit (approve + vault.deposit with hashed params)
    orderly_vault = "orderly_vault"


class FreqtradeWithdrawalMethod(enum.StrEnum):
    """How capital is withdrawn from Freqtrade."""

    #: Simple ERC20 transfer from a wallet address (for CEX or Lagoon vault integration)
    on_chain_transfer = "on_chain_transfer"

    #: Aster vault withdrawal on BSC
    aster_vault = "aster_vault"

    #: Hyperliquid withdrawal (SDK vault withdrawal + bridge transfer on Arbitrum)
    hyperliquid = "hyperliquid"

    #: Orderly vault withdrawal (vault.withdraw with hashed params)
    orderly_vault = "orderly_vault"


@dataclass(slots=True)
class OnChainTransferExchangeConfig:
    """Configuration for exchanges using simple on-chain transfers.

    Typically used for Lagoon vault integration where the vault cannot sign
    transactions directly, requiring a wallet-in-the-middle to delegate to.

    Deposit flow:
    1. ERC20.transfer(recipient_address, amount)

    Withdrawal flow:
    1. Withdrawal initiated externally (via CEX/exchange API)
    2. ERC20.transfer arrives from recipient_address
    """

    #: Wallet address for transfers (e.g., CEX deposit address or delegate wallet)
    recipient_address: str

    #: Maximum fee variance allowed when confirming deposit/withdrawal (in reserve currency units)
    fee_tolerance: Decimal = Decimal("1.0")

    #: Seconds to wait for balance update after on-chain tx confirms
    confirmation_timeout: int = 600

    #: Seconds between Freqtrade balance checks
    poll_interval: int = 10


@dataclass(slots=True)
class AsterExchangeConfig:
    """Configuration for Aster vault deposits and withdrawals on BSC.

    Deposit flow:
    1. ERC20.approve(vault_address, amount)
    2. AstherusVault.deposit(token_address, amount, broker_id)

    Withdrawal flow:
    1. AstherusVault.withdraw() - requires signed message or validator signatures
    Note: Withdrawal implementation deferred (requires signature infrastructure)
    """

    #: AstherusVault contract address on BSC
    vault_address: str

    #: Broker identifier for Aster (default 0)
    broker_id: int = 0

    #: Maximum fee variance allowed when confirming deposit/withdrawal (in reserve currency units)
    fee_tolerance: Decimal = Decimal("1.0")

    #: Seconds to wait for balance update after on-chain tx confirms
    confirmation_timeout: int = 600

    #: Seconds between Freqtrade balance checks
    poll_interval: int = 10


@dataclass(slots=True)
class HyperliquidExchangeConfig:
    """Configuration for Hyperliquid vault deposits and withdrawals.

    Deposit flow:
    1. On-chain: ERC20 transfer to Hyperliquid bridge on Arbitrum
    2. Off-chain: SDK vault_usd_transfer(is_deposit=True) to deposit into vault

    Withdrawal flow:
    1. Off-chain: SDK vault_usd_transfer(is_deposit=False) to withdraw from vault
    2. On-chain: funds arrive via bridge transfer

    Note: USDC only. Bridge address is hardcoded per network.
    """

    #: Hyperliquid vault address
    vault_address: str

    #: Use mainnet (True) or testnet (False)
    is_mainnet: bool = True

    #: Maximum fee variance allowed when confirming deposit/withdrawal (in reserve currency units)
    fee_tolerance: Decimal = Decimal("1.0")

    #: Seconds to wait for balance update after on-chain tx confirms
    confirmation_timeout: int = 600

    #: Seconds between Freqtrade balance checks
    poll_interval: int = 10


@dataclass(slots=True)
class OrderlyExchangeConfig:
    """Configuration for Orderly vault deposits and withdrawals.

    Deposit flow:
    1. ERC20.approve(vault_address, amount)
    2. Vault.deposit((account_id, broker_hash, token_hash, amount))

    Withdrawal flow:
    1. Vault.withdraw((account_id, broker_hash, token_hash, amount))

    broker_hash = keccak256(broker_id)
    token_hash = keccak256(token_id)
    """

    #: Orderly vault contract address
    vault_address: str

    #: Orderly account ID (32 bytes hex)
    orderly_account_id: str

    #: Broker ID string (will be keccak256 hashed)
    broker_id: str

    #: Token ID string (will be keccak256 hashed)
    token_id: str | None = None

    #: Maximum fee variance allowed when confirming deposit/withdrawal (in reserve currency units)
    fee_tolerance: Decimal = Decimal("1.0")

    #: Seconds to wait for balance update after on-chain tx confirms
    confirmation_timeout: int = 600

    #: Seconds between Freqtrade balance checks
    poll_interval: int = 10


#: Type alias for all exchange configurations
FreqtradeExchangeConfig = (
    OnChainTransferExchangeConfig
    | AsterExchangeConfig
    | HyperliquidExchangeConfig
    | OrderlyExchangeConfig
)


@dataclass(slots=True)
class FreqtradeConfig:
    """Configuration for one Freqtrade instance.

    Each Freqtrade bot instance gets its own configuration specifying
    how to connect and interact with it.

    Freqtrade uses JWT-based authentication.
    See: https://www.freqtrade.io/en/stable/rest-api/#advanced-api-usage-using-jwt-tokens
    """

    #: Unique identifier for this Freqtrade instance
    freqtrade_id: str

    #: Base URL of Freqtrade API (e.g., "http://localhost:8080")
    api_url: str

    #: Username for JWT authentication
    api_username: str

    #: Password for JWT authentication
    api_password: str

    #: Exchange name (e.g., "aster", "modetrade")
    exchange_name: str

    #: Reserve currency token address (USDT, USDC contract address)
    #: Used for on-chain deposits and withdrawals
    reserve_currency: str

    #: Exchange configuration for deposits and withdrawals
    exchange: FreqtradeExchangeConfig | None = None
