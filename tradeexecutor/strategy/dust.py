"""Dust amounts and epsilon rounding.

Because of very small and very accurate token units,
a lot of trades may end up having rounding artifacts.
We need to deal with these rounding artifacts by checking for "dust".

"""
from collections.abc import Iterable
from _decimal import Decimal
from decimal import Decimal

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.types import Percent, USDollarPrice
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

#: Hypercore vault withdrawal leaves dust due to the safety margin subtracted
#: from live equity.
#:
#: Incident reference:
#:
#: - HyperAI trade #326, Super Moon, on 2026-04-15 withdrew successfully.
#: - Routing used the live full-close safety margin of 1.5 USDC.
#: - The position then remained open with 1.500000 quantity because this close
#:   epsilon was still 0.20 USDC from an older safety-margin setting.
#:
#: Keep this above HYPERCORE_WITHDRAWAL_SAFETY_MARGIN_RAW = 1_500_000 raw
#: ($1.50 in 6-decimal USDC) so can_be_closed() recognises the intentional
#: residual as dust and the runner can auto-close it before account checks.
#:
#: .. warning::
#:
#:     This threshold is denominated in **US dollars**, not in vault share units.
#:     Callers comparing it against a position quantity must convert it first with
#:     :py:func:`convert_usd_close_epsilon_to_quantity`. For vaults whose share price is
#:     near 1 USD the two happen to coincide, which is why the distinction went unnoticed;
#:     for a share priced at 12.78 USD, treating "2.00" as share units makes the dust
#:     threshold 25.56 USD instead of 2.00 USD.
HYPERLIQUID_VAULT_CLOSE_EPSILON = Decimal("2.00")

#: A HyperCore small-position cleanup residual above this amount can still
#: produce verifiable movement through every withdrawal phase.
HYPERCORE_SMALL_POSITION_CLEANUP_CLOSE_EPSILON = Decimal("0.30")

#: Position marker that keeps a verifiable cleanup residual open despite the
#: normal 2 USDC HyperCore close epsilon.
HYPERCORE_SMALL_POSITION_CLEANUP_PENDING_REDEEM = (
    "hypercore_small_position_cleanup_pending_redeem"
)

HYPERLIQUID_VAULT_CLOSE_EPSILON_CAPITAL_PCT = Decimal("0.005")

#: Absolute ceiling, in US dollars, for the capital-scaled Hypercore close epsilon.
#:
#: Hypercore withdrawal dust is caused by an *absolute* safety margin of ~1.50 USDC
#: subtracted from live equity, so it does not grow with the strategy. Scaling the
#: threshold by ``initial_cash`` was intended to give larger strategies a little more
#: slack, but without a ceiling a 150,000 USD strategy gets a 750 USD "dust" threshold -
#: five hundred times the residual it is supposed to absorb, and comfortably large enough
#: to swallow a real position.
#:
#: 50 USD is ~33x the observed withdrawal margin and far below any position this strategy
#: family opens, so it keeps the intended slack without letting the threshold reach
#: economically meaningful sizes.
HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD = Decimal("50.00")

#: Hypercore vault equities fluctuate every block due to active trading
#: inside the vault, and live cycles can spend a long time in sequential
#: settlement before the final accounting read happens.
#:
#: This is problematic because even small absolute NAV moves on tiny vault
#: positions can become multi-percent relative drift by the time we compare the
#: fresh API equity against our state mark. We now refresh Hypercore marks right
#: before post-trade checks, but still keep a wider 2% tolerance here because
#: the live API and the state read are not truly atomic.
#:
#: This must stay a compromise, not the main fix: widening the tolerance too
#: much would hide genuine settlement failures and stale-state bugs.
HYPERLIQUID_VAULT_RELATIVE_EPSILON = 0.02


def get_dust_epsilon_for_pair(pair: TradingPairIdentifier) -> Decimal:
    """Get the dust threshold for a trading pair.

    See also :py:func:`get_close_epsilon_for_pair`.

    :param pair:
        Trading pair identifier.

    :return:
        Maximum amount of units we consider "zero".

    """

    if pair.is_cctp_bridge():
        return DEFAULT_DUST_EPSILON
    elif pair.is_vault():
        return DEFAULT_VAULT_EPSILON

    return get_dust_epsilon_for_asset(pair.base)


def get_hyperliquid_vault_close_epsilon(initial_cash: float | None = None) -> Decimal:
    """Get a Hypercore vault close epsilon, in **US dollars**, from a strategy's initial cash.

    A strategy with initial cash uses 0.5% of that value, clamped between the default
    safety-margin floor and :py:data:`HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD`. Strategies
    without configured initial cash retain the default close epsilon.

    ``initial_cash`` is a strategy-module input and can differ materially from live
    strategy equity, so the clamp is what stops a large configured bankroll turning the
    dust threshold into a real position size.

    :return:
        Dust threshold in US dollars. Convert with
        :py:func:`convert_usd_close_epsilon_to_quantity` before comparing to a quantity.
    """
    if initial_cash is None or initial_cash <= 0:
        return HYPERLIQUID_VAULT_CLOSE_EPSILON

    scaled = Decimal(str(initial_cash)) * HYPERLIQUID_VAULT_CLOSE_EPSILON_CAPITAL_PCT
    return min(max(scaled, HYPERLIQUID_VAULT_CLOSE_EPSILON), HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD)


def convert_usd_close_epsilon_to_quantity(
    epsilon_usd: Decimal,
    price: USDollarPrice | None,
) -> Decimal:
    """Convert a USD-denominated close epsilon into base token units.

    :py:meth:`TradingPosition.can_be_closed` compares the close epsilon against a position
    *quantity*, so a USD threshold has to be divided by the share price first. Skipping the
    conversion silently scales the threshold by the share price: at a 12.78 USD share price
    a 750 USD threshold becomes 750 *shares* worth 9,585 USD, and any position below that
    is written off as dust the moment it is opened.

    :param epsilon_usd:
        Dust threshold in US dollars.

    :param price:
        Current mark price of one base token, in US dollars.

    :return:
        Dust threshold in base token units.

        When the price is unknown or non-positive the conversion is impossible, and this
        falls back to :py:data:`DEFAULT_VAULT_EPSILON` rather than to the unconverted USD
        figure. Leaving a dust position open is recoverable on a later cycle; closing a
        funded position is not.
    """
    if not price or price <= 0:
        return DEFAULT_VAULT_EPSILON
    return epsilon_usd / Decimal(str(price))


def configure_hyperliquid_vault_close_epsilon(
    positions: Iterable,
    initial_cash: float | None = None,
) -> Decimal:
    """Apply the strategy-specific close epsilon to Hypercore positions."""
    epsilon = get_hyperliquid_vault_close_epsilon(initial_cash)
    for position in positions:
        if position.pair.is_hyperliquid_vault():
            position.hyperliquid_vault_close_epsilon = epsilon
    return epsilon


def get_close_epsilon_for_pair(
    pair: TradingPairIdentifier,
    hyperliquid_vault_close_epsilon: Decimal | None = None,
) -> Decimal:
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
        return hyperliquid_vault_close_epsilon or HYPERLIQUID_VAULT_CLOSE_EPSILON
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
