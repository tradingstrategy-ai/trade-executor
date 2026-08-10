"""Hypercore dust close epsilon must be denominated in USD, not in vault share units.

Regression tests for a defect where a funded Hypercore vault position was silently
destroyed the moment it was opened.

``get_hyperliquid_vault_close_epsilon()`` returns ``initial_cash * 0.005``, which is a US
dollar amount, and ``TradingPosition.can_be_closed()`` compared it against
``get_quantity()``, which is a count of vault *shares*. For vaults whose share price is
near 1 USD the two coincide and nothing is noticed. For a share priced at 12.78 USD the
threshold is multiplied by 12.78, and at a 150,000 USD bankroll it becomes 750 shares =
9,585 USD, so essentially every position the strategy opened in that vault qualified as
"dust" on arrival.

``Portfolio.close_position()`` is bookkeeping only - it moves the position between
dictionaries and sells nothing - so the shares vanished with no offsetting cash and no
realised loss. A 150,000 USD backtest ended at 10,645 USD while its own positions reported
-1,333 USD realised and +558 USD unrealised: 138,580 USD unaccounted.

The tests drive the real trade path (``State.create_trade`` / ``State.mark_trade_success``),
because ``mark_trade_success()`` is where the engine consults ``can_be_closed()`` and calls
``Portfolio.close_position()``. That is exactly the code path that destroyed the positions.
"""

import datetime
from decimal import Decimal

import pytest

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.portfolio import (
    CLOSE_POSITION_VALUE_DESTRUCTION_LIMIT_USD,
    PositionValueDestroyedError,
)
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.dust import (
    HYPERLIQUID_VAULT_CLOSE_EPSILON,
    HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD,
    configure_hyperliquid_vault_close_epsilon,
    convert_usd_close_epsilon_to_quantity,
    get_hyperliquid_vault_close_epsilon,
)


#: Share price of the vault that exposed the defect. pmalt traded around 12.78 USD/share,
#: far enough from 1.00 that the unit confusion stops being invisible.
PMALT_SHARE_PRICE = 12.78

#: The position that was destroyed: 701.15 shares bought for 8,963.78 USD.
PMALT_QUANTITY = Decimal("701.151323")


def make_hypercore_pair() -> TradingPairIdentifier:
    """A Hypercore vault pair.

    ``is_hyperliquid_vault()`` keys off chain 9999, the synthetic chain ID for native
    Hyperliquid vaults, so the assets must be homed there.
    """
    base = AssetIdentifier(
        chain_id=9999,
        address="0x4dec0a851849056e259128464ef28ce78afa27f6",
        token_symbol="pmalt",
        decimals=6,
    )
    quote = AssetIdentifier(
        chain_id=9999,
        address="0x0000000000000000000000000000000000000002",
        token_symbol="USDC",
        decimals=6,
    )
    return TradingPairIdentifier(
        base=base,
        quote=quote,
        pool_address="0x4dec0a851849056e259128464ef28ce78afa27f6",
        exchange_address="0x0000000000000000000000000000000000000004",
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.vault,
    )


def buy_vault_position(
    quantity: Decimal,
    share_price: float = PMALT_SHARE_PRICE,
    initial_cash: float = 150_000,
):
    """Open a Hypercore vault position through the real trade path.

    :return:
        ``(state, position)``. The position is whichever one the buy landed in - open or,
        if the engine decided it was dust, closed.
    """
    pair = make_hypercore_pair()
    assert pair.is_hyperliquid_vault(), "Test pair must be recognised as a Hypercore vault"

    state = State()
    state.portfolio.initialise_reserves(pair.quote, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(pair.quote, Decimal(str(initial_cash)))

    opened_at = datetime.datetime(2026, 3, 8)
    position, trade, _created = state.create_trade(
        strategy_cycle_at=opened_at,
        pair=pair,
        quantity=quantity,
        reserve=None,
        assumed_price=share_price,
        trade_type=TradeType.rebalance,
        reserve_currency=pair.quote,
        reserve_currency_price=1.0,
    )

    # The runner stamps the strategy's dust threshold onto open positions every cycle.
    configure_hyperliquid_vault_close_epsilon([position], initial_cash)

    state.start_execution(opened_at, trade)
    trade.mark_broadcasted(opened_at)
    state.mark_trade_success(
        executed_at=opened_at,
        trade=trade,
        executed_price=share_price,
        executed_amount=quantity,
        executed_reserve=quantity * Decimal(str(share_price)),
        lp_fees=0.0,
        native_token_price=1.0,
    )
    position.last_token_price = share_price
    return state, position


def test_epsilon_is_clamped_to_a_usd_ceiling():
    """A large configured bankroll must not scale the dust threshold into position sizes."""
    # Unconfigured strategies keep the default safety margin
    assert get_hyperliquid_vault_close_epsilon(None) == HYPERLIQUID_VAULT_CLOSE_EPSILON
    assert get_hyperliquid_vault_close_epsilon(0) == HYPERLIQUID_VAULT_CLOSE_EPSILON

    # Small bankrolls scale but never below the floor
    assert get_hyperliquid_vault_close_epsilon(100) == HYPERLIQUID_VAULT_CLOSE_EPSILON

    # 5,000 * 0.5% = 25 USD, under the ceiling, so the percentage rule still applies
    assert get_hyperliquid_vault_close_epsilon(5_000) == Decimal("25.000")

    # 150,000 would be 750 USD unclamped - the value that destroyed real positions
    assert get_hyperliquid_vault_close_epsilon(150_000) == HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD
    assert get_hyperliquid_vault_close_epsilon(10_000_000) == HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD


def test_destruction_limit_sits_above_the_largest_legitimate_dust():
    """The two thresholds must not contradict each other.

    ``can_be_closed()`` permits closing anything the dust rules call negligible - up to
    :py:data:`HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD` of value. If the value-destruction
    guard sat below that, it would raise on closes the engine is supposed to perform, which
    is exactly what happened in a 150,000 USD backtest: a 31.36 USD residual was dust by the
    50 USD ceiling but blocked by a 25 USD guard.
    """
    assert CLOSE_POSITION_VALUE_DESTRUCTION_LIMIT_USD > float(HYPERLIQUID_VAULT_CLOSE_EPSILON_MAX_USD)


def test_usd_epsilon_converts_to_share_units():
    """The threshold must be divided by the share price before meeting a quantity."""
    # 50 USD of a 12.78 USD share is ~3.9 shares, not 50
    converted = convert_usd_close_epsilon_to_quantity(Decimal("50"), PMALT_SHARE_PRICE)
    assert converted == pytest.approx(Decimal("3.912"), abs=Decimal("0.001"))

    # A ~1 USD share price is where the units coincide, which is why this went unnoticed
    assert convert_usd_close_epsilon_to_quantity(Decimal("2"), 1.0) == Decimal("2")

    # An unknown price cannot be converted. Falling back to the raw USD number would
    # reintroduce the defect, so it must collapse to a dust-sized threshold instead.
    assert convert_usd_close_epsilon_to_quantity(Decimal("750"), None) < Decimal("1")
    assert convert_usd_close_epsilon_to_quantity(Decimal("750"), 0.0) < Decimal("1")


@pytest.mark.parametrize("initial_cash", [25_000, 75_000, 150_000, 1_000_000])
def test_funded_position_survives_its_own_opening_trade(initial_cash):
    """The regression: buying a real position must not immediately destroy it.

    701.15 pmalt shares at 12.78 USD is 8,963 USD. Before the fix, at a 150,000 bankroll
    the dust threshold was 750 *shares*, so ``mark_trade_success()`` closed this position
    on the same timestamp it was opened, leaving the shares orphaned at zero value.
    """
    state, position = buy_vault_position(PMALT_QUANTITY, initial_cash=initial_cash)

    expected_value = float(PMALT_QUANTITY) * PMALT_SHARE_PRICE  # ~8,960 USD
    assert position.get_quantity() == PMALT_QUANTITY
    assert position.get_value() == pytest.approx(expected_value, abs=1)

    assert not position.is_closed(), (
        f"A position worth {position.get_value():,.0f} USD was closed as dust "
        f"at initial_cash={initial_cash}"
    )
    assert not position.can_be_closed()
    assert position.position_id in state.portfolio.open_positions

    # Equity must reflect the holding, not have silently lost it
    assert state.portfolio.get_total_equity() == pytest.approx(initial_cash, rel=0.001)


def test_genuine_dust_still_closes():
    """The fix must not strand the withdrawal residual the epsilon exists to absorb.

    ``mark_trade_success()`` auto-closes a dust-sized position on the spot, so the proof is
    that the position ends up closed. A closed position values at zero, which is why the
    value is asserted on the trade rather than on the position.
    """
    # ~1.50 USD of unredeemable residual, the documented Hypercore withdrawal margin
    dust_quantity = Decimal("0.117")  # 0.117 * 12.78 = ~1.50 USD
    _state, position = buy_vault_position(dust_quantity, initial_cash=150_000)

    bought_value = float(dust_quantity) * PMALT_SHARE_PRICE
    assert bought_value == pytest.approx(1.5, abs=0.05)
    assert position.is_closed(), "Genuine sub-margin dust should still be closed automatically"


def test_close_epsilon_is_expressed_in_share_units():
    """``get_close_epsilon()`` feeds a quantity comparison, so it must return units."""
    _state, position = buy_vault_position(PMALT_QUANTITY, initial_cash=150_000)

    epsilon = position.get_close_epsilon()

    # 50 USD ceiling / 12.78 USD per share
    assert epsilon == pytest.approx(Decimal("3.912"), abs=Decimal("0.001"))
    # Before the fix this was 750: the raw USD figure used as a share count
    assert epsilon < Decimal("10")


def test_close_position_refuses_to_destroy_value():
    """A bookkeeping close must not make a funded position disappear."""
    state, position = buy_vault_position(PMALT_QUANTITY, initial_cash=150_000)
    equity_before = state.portfolio.get_total_equity()

    with pytest.raises(PositionValueDestroyedError) as exc_info:
        state.portfolio.close_position(position, datetime.datetime(2026, 3, 8))

    assert "bookkeeping only" in str(exc_info.value)
    # The position and the equity must survive the refusal
    assert position.position_id in state.portfolio.open_positions
    assert not position.is_closed()
    assert state.portfolio.get_total_equity() == pytest.approx(equity_before)


def test_close_position_allows_deliberate_value_destruction():
    """Accounting corrections, repair tooling and manual operations still force a close."""
    state, position = buy_vault_position(PMALT_QUANTITY, initial_cash=150_000)

    state.portfolio.close_position(
        position,
        datetime.datetime(2026, 3, 8),
        allow_value_destruction=True,
    )

    assert position.is_closed()
    assert position.position_id in state.portfolio.closed_positions


def test_close_position_allows_dust():
    """A position below the value limit closes without opting in."""
    dust_quantity = Decimal(str(CLOSE_POSITION_VALUE_DESTRUCTION_LIMIT_USD / 2 / PMALT_SHARE_PRICE))
    state, position = buy_vault_position(dust_quantity, initial_cash=150_000)

    if position.is_closed():
        # mark_trade_success() already auto-closed it as dust, which is the intended path
        return

    assert position.get_value() < CLOSE_POSITION_VALUE_DESTRUCTION_LIMIT_USD
    state.portfolio.close_position(position, datetime.datetime(2026, 3, 8))
    assert position.is_closed()
