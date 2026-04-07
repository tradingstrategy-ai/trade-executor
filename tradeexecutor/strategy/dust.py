"""Dust amounts and epsilon rounding.

Because of very small and very accurate token units,
a lot of trades may end up having rounding artifacts.
We need to deal with these rounding artifacts by checking for "dust".

"""
from _decimal import Decimal
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import Percent
from tradeexecutor.utils.accuracy import COLLATERAL_EPSILON

#: The absolute number of tokens we consider the value to be zero
#:
#:
#: Because of funny %s of values divided near zero,
#: we cannot use relative comparison near zero values.
#:
#:
DEFAULT_DUST_EPSILON = Decimal(10 ** -10)


#: The default % we allow the balance to drift before we consider it a mismatch.
#:
#: Set to 5 BPS
#:
DEFAULT_RELATIVE_EPSILON = 5 * 10 ** -4


#: When to close 1delta positions
ONE_DELTA_CLOSE_EPSILON = 1 * 10**-4


#: If position value is less than 10c consider it to be zero
DEFAULT_USD_LOW_VALUE_THRESHOLD = 0.10

#: Set by maxRedeem() issue on Spark USDC on Morpho
DEFAULT_VAULT_EPSILON = Decimal(10 ** -6)

#: Hypercore vault withdrawal leaves ~$0.10 dust due to the safety margin
#: subtracted from live equity (HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW = 100_000
#: raw = $0.10 in 6-decimal USDC).  This epsilon must exceed that margin so
#: can_be_closed() recognises the residual as dust.
HYPERLIQUID_VAULT_CLOSE_EPSILON = Decimal("0.20")

#: Hypercore vault equities fluctuate every block due to active trading
#: inside the vault, so we allow 100 BPS (1%) relative drift before
#: flagging a mismatch.
HYPERLIQUID_VAULT_RELATIVE_EPSILON = 0.01


def get_dust_epsilon_for_pair(pair: TradingPairIdentifier) -> Decimal:
    """Get the dust threshold for a trading pair.

    See also :py:func:`get_close_epsilon_for_pair`.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """

    if pair.is_hyperliquid_vault():
        return HYPERLIQUID_VAULT_CLOSE_EPSILON

    if pair.is_vault():
        return DEFAULT_VAULT_EPSILON

    return get_dust_epsilon_for_asset(pair.base)


def get_close_epsilon_for_pair(pair: TradingPairIdentifier) -> Decimal:
    """Get the close threshold for a trading pair.

    - Currently same as dust epsilon

    See also :py:func:`get_dust_epsilon_for_pair`.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """

    # Credit positions we have larger tolerance
    # │ 10 │          │ aBasUSDC-USDC │ 2025-03-18 12:00 │          │ -0.0012        │ Initial supply
    # Relea                      │            │          │                  │
    # │ 10 │ T, B     │ ‎ ‎ ‎ ‎ ‎ ┗        │ 2025-03-18 12:00 │          │ 20,282.0378    │ Initial supply       │ 40         │ 1.000000 │ 2025-03-18 12:09 │
    # │ 10 │ T, S     │ ‎ ‎ ‎ ‎ ‎ ┗        │ 2025-03-18 16:00 │          │ -20,282.3650   │ Releasing all funds  │ 42         │ 1.000000 │ 2025-03-18 16:09 │
    # │ 11 │          │ aBasUSDC-USDC │ 2025-03-18 16:00 │          │ 20,172.9335    │ Redepositing remaini │            │          │                  │
    # │ 11 │ T, B     │ ‎ ‎ ‎ ‎ ‎ ┗        │ 2025-03-18 16:00 │          │ 20,172.9335    │ Redepositing remaini │ 43         │ 1.000000 │ 2025-03-18 16:09 │
    # ╰────┴──────────┴───────────────┴──────────────────┴──────────┴────────────────┴──────────────────────┴────────────┴──────────┴──────────────────╯
    #
    # Frozen positions
    if pair.is_credit_supply():
        return COLLATERAL_EPSILON
    elif pair.is_hyperliquid_vault():
        return HYPERLIQUID_VAULT_CLOSE_EPSILON
    elif pair.is_vault():
        return DEFAULT_VAULT_EPSILON

    return get_dust_epsilon_for_asset(pair.base)


def get_dust_epsilon_for_asset(asset: AssetIdentifier) -> Decimal:
    """Get the dust threshold for a trading pair.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """

    # Hardcoded rules for now.
    # Some practical problems we have run across in backtesting.
    # We have wrapped and non-wrapped token symbols as we are backtesting both on DEX and CEX data
    if asset.token_symbol in ("WBTC", "BTC"):
        return Decimal(10 ** -7)
    elif asset.token_symbol in ("ETH", "WETH"):
        return Decimal(10 ** -7)
    elif asset.token_symbol in ("USDC", "USDC.e"):
        return Decimal(0.1)
    elif asset.token_symbol in ("aPolUSDC", "aEthUSDC"):
        return Decimal(0.1)
    elif "variableDebt" in asset.token_symbol:
        # 1delta closing epsilon higher than default
        return ONE_DELTA_CLOSE_EPSILON
    else:
        return DEFAULT_DUST_EPSILON


def get_relative_epsilon_for_asset(asset: AssetIdentifier) -> Percent:
    """Get the relative threshold for a trading pair.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """
    if asset.token_symbol in ("aPolUSDC", "USDC"):
        # Temp allow 0.5% tolerance
        return 0.005
    
    # 5 BPS
    return DEFAULT_RELATIVE_EPSILON


def get_relative_epsilon_for_pair(pair: TradingPairIdentifier) -> Percent:
    if pair.is_hyperliquid_vault():
        return HYPERLIQUID_VAULT_RELATIVE_EPSILON
    return get_relative_epsilon_for_asset(pair.base)

