"""Lighter exchange-account adapter and NAV safety tests."""

import datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.exchange_account.lighter import (
    NegativeLighterEquityError,
    create_lighter_account_value_func,
    create_lighter_exchange_account_pair,
    create_lighter_vault_valuation_func,
    has_lighter_exchange_account_pairs,
)
from tradeexecutor.exchange_account.pricing import ExchangeAccountPricingModel
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.exchange_account.sync_model import ExchangeAccountSyncModel
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from eth_defi.compat import native_datetime_utc_now


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )


def test_lighter_pair_and_universe_detection(usdc: AssetIdentifier):
    """Create a public Lighter pair and detect it from a strategy universe.

    1. Create a synthetic Lighter exchange-account pair.
    2. Verify public protocol, account and deployment metadata.
    3. Verify universe detection ignores an unrelated spot pair.
    """
    # 1. Create a synthetic Lighter exchange-account pair.
    pair = create_lighter_exchange_account_pair(usdc, account_index=123)

    # 2. Verify public protocol, account and deployment metadata.
    assert pair.kind == TradingPairKind.exchange_account
    assert pair.get_exchange_account_protocol() == "lighter"
    assert pair.get_exchange_account_id() == 123
    assert pair.other_data["lighter_deployment"] == "ethereum"
    assert pair.base.token_symbol == "LIGHTER-ACCOUNT"

    # 3. Verify universe detection ignores an unrelated spot pair.
    universe = SimpleNamespace(iterate_pairs=lambda: [pair])
    assert has_lighter_exchange_account_pairs(universe)
    assert not has_lighter_exchange_account_pairs(SimpleNamespace(iterate_pairs=lambda: []))


def test_lighter_account_value_uses_public_reader(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
):
    """Read canonical total asset value without requesting a private key.

    1. Patch the public equity reader with a deterministic response.
    2. Call the account value function for a Lighter pair.
    3. Verify the session and public account index were forwarded.
    """
    # 1. Patch the public equity reader with a deterministic response.
    session = object()
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=SimpleNamespace(get_total=lambda: Decimal("12.50")),
    )
    pair = create_lighter_exchange_account_pair(usdc, account_index=456)

    # 2. Call the account value function for a Lighter pair.
    value = create_lighter_account_value_func(session)(pair, block_identifier=99)

    # 3. Verify the session and public account index were forwarded.
    assert value == Decimal("12.50")
    reader.assert_called_once_with(session, 456)


def test_negative_lighter_equity_does_not_mutate_state(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
):
    """Reject negative equity before creating valuation or balance updates.

    1. Create an exchange-account position with a non-negative starting value.
    2. Return negative Lighter equity from the injected public reader.
    3. Verify valuation fails closed and state remains unchanged.
    """
    # 1. Create an exchange-account position with a non-negative starting value.
    pair = create_lighter_exchange_account_pair(usdc, account_index=789)
    state = State()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=Decimal("10"),
    )
    position = list(state.portfolio.open_positions.values())[0]
    previous_value = position.get_value()
    reader = mocker.Mock(return_value=Decimal("-1"))

    # 2. Return negative Lighter equity from the injected public reader.
    valuator = ExchangeAccountValuator(
        ExchangeAccountPricingModel(reader),
    )

    # 3. Verify valuation fails closed and state remains unchanged.
    with pytest.raises(NegativeLighterEquityError):
        valuator(native_datetime_utc_now(), position)
    assert position.get_value() == previous_value
    assert not position.balance_updates
    assert not position.valuation_updates


def test_negative_lighter_equity_aborts_sync_without_state_mutation(usdc: AssetIdentifier):
    """Reject negative equity in the periodic sync path as well.

    1. Create an exchange-account position with a non-negative starting value.
    2. Return negative equity from the injected Lighter reader.
    3. Verify sync raises before recording a balance update or advancing accounting.
    """
    # 1. Create an exchange-account position with a non-negative starting value.
    pair = create_lighter_exchange_account_pair(usdc, account_index=790)
    state = State()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=Decimal("10"),
    )
    position = list(state.portfolio.open_positions.values())[0]
    previous_value = position.get_value()
    sync_model = ExchangeAccountSyncModel(lambda _pair, **_kwargs: Decimal("-1"))

    # 2. Return negative equity from the injected Lighter reader.
    # 3. Verify sync raises before recording a balance update or advancing accounting.
    with pytest.raises(NegativeLighterEquityError):
        sync_model.sync_positions(
            timestamp=native_datetime_utc_now(),
            state=state,
            strategy_universe=None,
            pricing_model=None,
        )
    assert position.get_value() == previous_value
    assert not position.balance_updates
    assert not state.sync.accounting.balance_update_refs


def test_lighter_api_error_does_not_create_stale_valuation(usdc: AssetIdentifier):
    """Propagate Lighter API failures instead of caching the old value.

    1. Create an exchange-account position with an existing value.
    2. Inject an account reader that raises an API error.
    3. Verify no fallback valuation event is appended.
    """
    # 1. Create an exchange-account position with an existing value.
    pair = create_lighter_exchange_account_pair(usdc, account_index=791)
    state = State()
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=usdc,
        reserve_amount=Decimal("10"),
    )
    position = list(state.portfolio.open_positions.values())[0]
    reader = lambda _pair, **_kwargs: (_ for _ in ()).throw(RuntimeError("API unavailable"))
    valuator = ExchangeAccountValuator(ExchangeAccountPricingModel(reader))

    # 2. Inject an account reader that raises an API error.
    # 3. Verify no fallback valuation event is appended.
    with pytest.raises(RuntimeError, match="API unavailable"):
        valuator(native_datetime_utc_now(), position)
    assert not position.valuation_updates
    assert not position.balance_updates


def test_lighter_vault_valuation_adds_safe_balance_and_external_equity(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
):
    """Return float NAV from Safe reserve plus current public Lighter equity.

    1. Mock the ERC-20 Safe balance and public Lighter account response.
    2. Evaluate the custom Lagoon valuation function at a fixed block.
    3. Verify Decimal components are summed before the float boundary.
    """
    # 1. Mock the ERC-20 Safe balance and public Lighter account response.
    token = SimpleNamespace(fetch_balance_of=mocker.Mock(return_value=Decimal("3.25")))
    mocker.patch("tradeexecutor.exchange_account.lighter.fetch_erc20_details", return_value=token)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=SimpleNamespace(get_total=lambda: Decimal("8.75")),
    )
    web3 = object()

    # 2. Evaluate the custom Lagoon valuation function at a fixed block.
    valuation = create_lighter_vault_valuation_func(
        web3=web3,
        safe_address="0x0000000000000000000000000000000000000002",
        reserve_asset=usdc,
        account_index=123,
        session=object(),
    )
    result = valuation(None, block_number=42)

    # 3. Verify Decimal components are summed before the float boundary.
    assert result == pytest.approx(Decimal("12.00"))
    token.fetch_balance_of.assert_called_once_with(
        "0x0000000000000000000000000000000000000002",
        block_identifier=42,
    )


def test_negative_lighter_vault_valuation_aborts_before_nav(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
):
    """Reject negative external equity before a Lagoon NAV can be posted.

    1. Mock a valid Safe balance and negative public Lighter equity.
    2. Evaluate the custom NAV function.
    3. Verify the invariant aborts before returning a value to Lagoon.
    """
    # 1. Mock a valid Safe balance and negative public Lighter equity.
    token = SimpleNamespace(fetch_balance_of=mocker.Mock(return_value=Decimal("3")))
    mocker.patch("tradeexecutor.exchange_account.lighter.fetch_erc20_details", return_value=token)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=SimpleNamespace(get_total=lambda: Decimal("-0.01")),
    )
    valuation = create_lighter_vault_valuation_func(
        web3=object(),
        safe_address="0x0000000000000000000000000000000000000002",
        reserve_asset=usdc,
        account_index=124,
        session=object(),
    )

    # 2. Evaluate the custom NAV function.
    # 3. Verify the invariant aborts before returning a value to Lagoon.
    with pytest.raises(NegativeLighterEquityError):
        valuation(None, block_number=7)


def test_lagoon_nav_adds_pending_settlement_float():
    """Keep the Lighter NAV return compatible with Lagoon pending settlement.

    1. Use a minimal sync-model instance with a custom base valuation.
    2. Add a non-zero pending settlement value.
    3. Verify the result is the float sum passed to Lagoon.
    """
    # 1. Use a minimal sync-model instance with a custom base valuation.
    sync_model = object.__new__(LagoonVaultSyncModel)
    sync_model.valuation_data_freshness = datetime.timedelta(hours=1)
    sync_model.calculate_valuation_func = lambda _state, **_kwargs: 12.5
    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            get_open_and_frozen_positions=lambda: [],
            get_vault_settlement_pending_value=lambda: 0.5,
        )
    )

    # 2. Add a non-zero pending settlement value.
    # 3. Verify the result is the float sum passed to Lagoon.
    assert sync_model.calculate_valuation(state, block_number=10) == pytest.approx(13.0)
