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

    #: Aster vault deposit on BSC (approve + vault.deposit)
    aster_vault = "aster_vault"

    #: Hyperliquid deposit (bridge transfer on Arbitrum + SDK vault deposit)
    hyperliquid = "hyperliquid"

    #: Orderly vault deposit (approve + vault.deposit with hashed params)
    orderly_vault = "orderly_vault"


@dataclass(slots=True)
class FreqtradeDepositConfig:
    """Base configuration for depositing to a Freqtrade instance.

    Use method-specific subclasses for actual deposits:
    - AsterDepositConfig for Aster vault
    - HyperliquidDepositConfig for Hyperliquid
    - OrderlyDepositConfig for Orderly
    """

    #: How deposits are made to Freqtrade
    method: FreqtradeDepositMethod

    #: Maximum fee variance allowed when confirming deposit (in reserve currency units)
    fee_tolerance: Decimal = Decimal("1.0")

    #: Seconds to wait for balance update after on-chain tx confirms
    confirmation_timeout: int = 600

    #: Seconds between Freqtrade balance checks
    poll_interval: int = 10


@dataclass(slots=True)
class AsterDepositConfig(FreqtradeDepositConfig):
    """Aster vault deposit configuration on BSC.

    Flow:
    1. ERC20.approve(vault_address, amount)
    2. AstherusVault.deposit(token_address, amount, broker_id)
    """

    #: Deposit method
    method: FreqtradeDepositMethod = FreqtradeDepositMethod.aster_vault

    #: AstherusVault contract address on BSC
    vault_address: str | None = None

    #: Broker identifier for Aster (default 0)
    broker_id: int = 0


@dataclass(slots=True)
class HyperliquidDepositConfig(FreqtradeDepositConfig):
    """Hyperliquid deposit configuration.

    Two-step flow:
    1. On-chain: ERC20 transfer to Hyperliquid bridge on Arbitrum
    2. Off-chain: SDK vault_usd_transfer() to deposit into vault

    Note: USDC only. Bridge address is hardcoded per network.
    """

    #: Deposit method
    method: FreqtradeDepositMethod = FreqtradeDepositMethod.hyperliquid

    #: Hyperliquid vault address
    vault_address: str | None = None

    #: Use mainnet (True) or testnet (False)
    is_mainnet: bool = True


@dataclass(slots=True)
class OrderlyDepositConfig(FreqtradeDepositConfig):
    """Orderly vault deposit configuration.

    Flow:
    1. ERC20.approve(vault_address, amount)
    2. Vault.deposit((account_id, broker_hash, token_hash, amount))

    broker_hash = keccak256(broker_id)
    token_hash = keccak256(token_id)
    """

    #: Deposit method
    method: FreqtradeDepositMethod = FreqtradeDepositMethod.orderly_vault

    #: Orderly vault contract address
    vault_address: str | None = None

    #: Orderly account ID (32 bytes hex)
    orderly_account_id: str | None = None

    #: Broker ID string (will be keccak256 hashed)
    broker_id: str | None = None

    #: Token ID string (will be keccak256 hashed)
    token_id: str | None = None


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
    exchange: str

    #: Reserve currency token address (USDT, USDC contract address)
    #: Used for on-chain deposits
    reserve_currency: str

    #: Deposit configuration (use method-specific config classes)
    deposit: FreqtradeDepositConfig | None = None
