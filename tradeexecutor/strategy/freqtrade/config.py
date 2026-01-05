"""Configuration for Freqtrade integration."""

import enum
from dataclasses import dataclass


class FreqtradeDepositMethod(enum.StrEnum):
    """How capital is deposited to Freqtrade."""

    #: Manual deposits - user deposits manually, executor just tracks
    manual = "manual"

    # TODO


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

    #: Exchange name (e.g., "binance", "modetrade")
    exchange: str

    #: Reserve currency (USDT, USDC, etc.)
    reserve_currency: str

    #: How deposits are made to Freqtrade
    deposit_method: FreqtradeDepositMethod
