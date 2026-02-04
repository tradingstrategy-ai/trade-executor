"""Exchange account position integration for external perp DEXes.

This module provides support for tracking positions on external exchanges
like Derive, Hyperliquid, etc. The position value is the total account
value in USD from the exchange API.

Key components:

- :py:class:`ExchangeAccountPricingModel` - Pricing model (always 1:1 USD)
- :py:class:`ExchangeAccountValuator` - Valuation using configurable account value function
- :py:func:`create_derive_account_value_func` - Derive-specific account value function
"""
