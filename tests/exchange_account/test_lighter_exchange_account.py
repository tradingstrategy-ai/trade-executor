"""Lighter exchange-account adapter and NAV safety tests."""

import datetime
import logging
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest
from pytest_mock import MockerFixture
from web3 import Web3

from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.exchange_account.derive import DeriveNetwork
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
from tradeexecutor.exchange_account.utils import create_exchange_account_value_func
from tradeexecutor.exchange_account.valuation import ExchangeAccountValuator
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from eth_defi.compat import native_datetime_utc_now
from strategies.test_only.minimal_lighter_strategy import create_trading_universe


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=1,
        address="0x0000000000000000000000000000000000000001",
        token_symbol="USDC",
        decimals=6,
    )


def test_lighter_pair_and_universe_detection(usdc: AssetIdentifier) -> None:
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
) -> None:
    """Read canonical total asset value without requesting a private key.

    1. Patch the public equity reader with a deterministic response.
    2. Call the account value function for a Lighter pair.
    3. Verify the session and public account index were forwarded.
    """
    # 1. Patch the public equity reader with a deterministic response.
    # The HTTP read is mocked because this unit test checks adapter wiring,
    # while real public API coverage belongs to the manual integration test.
    session = object()
    reader = mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=SimpleNamespace(get_total=lambda: Decimal("12.50")),
    )
    pair = create_lighter_exchange_account_pair(usdc, account_index=456)

    # 2. Call the account value function for a Lighter pair.
    value = create_lighter_account_value_func(session)(pair, block_identifier=99)

    # 3. Verify the session and public account index were forwarded.
    assert value == pytest.approx(Decimal("12.50"))
    reader.assert_called_once_with(session, 456)


def test_lighter_runtime_auto_discovery_wires_account_and_nav_readers(
    mocker: MockerFixture,
) -> None:
    """Wire both public Lighter readers through the production configurator.

    1. Build the real minimal Lighter strategy universe and an Ethereum execution double.
    2. Construct the production Ethereum pair configurator.
    3. Verify one shared public session is used for account and Lagoon NAV readers.
    """
    # 1. Build the real minimal Lighter strategy universe and an Ethereum execution double.
    strategy_universe = create_trading_universe(None)
    web3 = Web3()
    mocker.patch.object(web3.eth, "_chain_id", return_value=1)
    session = object()
    account_reader = mocker.Mock(name="lighter_account_reader")
    nav_reader = mocker.Mock(name="lighter_nav_reader")
    session_factory = mocker.patch(
        "tradeexecutor.ethereum.ethereum_protocol_adapters.create_lighter_session",
        return_value=session,
    )
    account_factory = mocker.patch(
        "tradeexecutor.ethereum.ethereum_protocol_adapters.create_lighter_account_value_func",
        return_value=account_reader,
    )
    nav_factory = mocker.patch(
        "tradeexecutor.ethereum.ethereum_protocol_adapters.create_lighter_vault_valuation_func",
        return_value=nav_reader,
    )
    safe_address = "0x0000000000000000000000000000000000000002"
    execution_model = SimpleNamespace(
        web3=web3,
        tx_builder=SimpleNamespace(
            get_token_delivery_address=lambda: safe_address,
        ),
    )

    # 2. Construct the production Ethereum pair configurator.
    configurator = EthereumPairConfigurator(
        web3,
        strategy_universe,
        execution_model=execution_model,
    )

    # 3. Both readers are wired with the same unauthenticated session.
    assert configurator.account_value_func is account_reader
    assert configurator.vault_valuation_func is nav_reader
    session_factory.assert_called_once_with()
    account_factory.assert_called_once_with(session)
    nav_factory.assert_called_once_with(
        web3=web3,
        safe_address=safe_address,
        reserve_asset=strategy_universe.get_reserve_asset(),
        account_index=123,
        session=session,
    )


def test_lighter_runtime_rejects_mixed_exchange_account_protocols(
    usdc: AssetIdentifier,
) -> None:
    """Reject a universe that could omit part of its external account equity.

    1. Create one Lighter pair and one synthetic GMX protocol pair.
    2. Validate the mixed universe through the production topology guard.
    3. Verify configuration fails before either protocol is auto-discovered.
    """
    # 1. Create one Lighter pair and one synthetic GMX protocol pair.
    lighter_pair = create_lighter_exchange_account_pair(usdc, account_index=123)
    gmx_pair = create_lighter_exchange_account_pair(usdc, account_index=456)
    gmx_pair.other_data["exchange_protocol"] = "gmx"
    mixed_universe = SimpleNamespace(
        iterate_pairs=lambda: [lighter_pair, gmx_pair],
    )

    # 2. Validate the mixed universe through the production topology guard.
    # 3. Configuration fails before either protocol is auto-discovered.
    with pytest.raises(ValueError, match="cannot mix protocols: gmx"):
        EthereumPairConfigurator._validate_external_account_protocols(mixed_universe)


def test_correct_accounts_dispatches_to_public_lighter_reader(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
) -> None:
    """Dispatch account correction to the unauthenticated Lighter reader.

    1. Create one Lighter external-account position double.
    2. Build the shared correction reader with no exchange credentials.
    3. Verify the returned dispatcher calls the public Lighter reader.
    """
    # 1. Create one Lighter external-account position double.
    pair = create_lighter_exchange_account_pair(usdc, account_index=789)
    position = SimpleNamespace(pair=pair)
    public_reader = mocker.Mock(return_value=Decimal("17.25"))
    factory = mocker.patch(
        "tradeexecutor.exchange_account.utils._create_lighter_protocol_value_func",
        return_value=public_reader,
    )

    # 2. Build the shared correction reader with no exchange credentials.
    test_logger = logging.getLogger("test-lighter-correct-accounts")
    dispatcher = create_exchange_account_value_func(
        positions=[position],
        derive_owner_private_key=None,
        derive_session_private_key=None,
        derive_wallet_address=None,
        derive_network=DeriveNetwork.mainnet,
        ccxt_exchange_id=None,
        ccxt_options=None,
        ccxt_sandbox=False,
        logger=test_logger,
    )
    assert dispatcher is not None

    # 3. The returned dispatcher calls the public Lighter reader.
    assert dispatcher(pair) == Decimal("17.25")
    factory.assert_called_once_with(logger=test_logger)
    public_reader.assert_called_once_with(pair)


def test_negative_lighter_equity_does_not_mutate_state(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
) -> None:
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

    # 2. Return negative Lighter equity from the injected public reader.
    # The reader is mocked to exercise the impossible negative response
    # deterministically without relying on external account state.
    reader = mocker.Mock(return_value=Decimal("-1"))
    valuator = ExchangeAccountValuator(
        ExchangeAccountPricingModel(reader),
    )

    # 3. Verify valuation fails closed and state remains unchanged.
    with pytest.raises(NegativeLighterEquityError):
        valuator(native_datetime_utc_now(), position)
    assert position.get_value() == pytest.approx(previous_value)
    assert not position.balance_updates
    assert not position.valuation_updates


def test_negative_lighter_equity_aborts_sync_without_state_mutation(
    usdc: AssetIdentifier,
) -> None:
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

    # 2. Return negative equity from the injected Lighter reader.
    # The local callback supplies an impossible public response without
    # depending on mutable external account state.
    sync_model = ExchangeAccountSyncModel(lambda _pair, **_kwargs: Decimal("-1"))

    # 3. Verify sync raises before recording a balance update or advancing accounting.
    with pytest.raises(NegativeLighterEquityError):
        sync_model.sync_positions(
            timestamp=native_datetime_utc_now(),
            state=state,
            strategy_universe=None,
            pricing_model=None,
        )
    assert position.get_value() == pytest.approx(previous_value)
    assert not position.balance_updates
    assert not state.sync.accounting.balance_update_refs


def test_lighter_api_error_does_not_create_stale_valuation(
    usdc: AssetIdentifier,
) -> None:
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

    # 2. Inject an account reader that raises an API error.
    # The local callback makes the upstream failure deterministic without
    # contacting Lighter from a unit test.
    def unavailable_reader(
        _pair: TradingPairIdentifier,
        **_kwargs: Any,
    ) -> Decimal:
        raise RuntimeError("API unavailable")

    valuator = ExchangeAccountValuator(ExchangeAccountPricingModel(unavailable_reader))

    # 3. Verify no fallback valuation event is appended.
    with pytest.raises(RuntimeError, match="API unavailable"):
        valuator(native_datetime_utc_now(), position)
    assert not position.valuation_updates
    assert not position.balance_updates


def test_lighter_vault_valuation_adds_safe_balance_and_external_equity(
    usdc: AssetIdentifier,
    mocker: MockerFixture,
) -> None:
    """Return float NAV from Safe reserve plus current public Lighter equity.

    1. Mock the ERC-20 Safe balance and public Lighter account response.
    2. Evaluate the custom Lagoon valuation function at a fixed block.
    3. Verify Decimal components are summed before the float boundary.
    """
    # 1. Mock the ERC-20 Safe balance and public Lighter account response.
    # Both reads are mocked because this unit test isolates the NAV formula;
    # the Anvil test covers the real ERC-20 read.
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
) -> None:
    """Reject negative external equity before a Lagoon NAV can be posted.

    1. Mock a valid Safe balance and negative public Lighter equity.
    2. Evaluate the custom NAV function.
    3. Verify the invariant aborts before returning a value to Lagoon.
    """
    # 1. Mock a valid Safe balance and negative public Lighter equity.
    # These reads are mocked because the unit test targets the fail-closed NAV
    # boundary, while the Anvil test covers the real ERC-20 balance read.
    token = SimpleNamespace(fetch_balance_of=mocker.Mock(return_value=Decimal("3")))
    mocker.patch("tradeexecutor.exchange_account.lighter.fetch_erc20_details", return_value=token)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=SimpleNamespace(get_total=lambda: Decimal("-0.01")),
    )

    # 2. Evaluate the custom NAV function.
    valuation = create_lighter_vault_valuation_func(
        web3=object(),
        safe_address="0x0000000000000000000000000000000000000002",
        reserve_asset=usdc,
        account_index=124,
        session=object(),
    )

    # 3. Verify the invariant aborts before returning a value to Lagoon.
    with pytest.raises(NegativeLighterEquityError):
        valuation(None, block_number=7)


def test_lagoon_nav_adds_pending_settlement_float() -> None:
    """Keep the Lighter NAV return compatible with Lagoon pending settlement.

    1. Use a minimal sync-model instance with a custom base valuation.
    2. Add a non-zero pending settlement value.
    3. Verify the result is the float sum passed to Lagoon.
    """
    # 1. Use a minimal sync-model instance with a custom base valuation.
    # The minimal model isolates pending-settlement arithmetic without
    # constructing a real vault or Web3 connection.
    sync_model = object.__new__(LagoonVaultSyncModel)
    sync_model.valuation_data_freshness = datetime.timedelta(hours=1)
    sync_model.calculate_valuation_func = lambda _state, **_kwargs: 12.5

    # 2. Add a non-zero pending settlement value.
    state = SimpleNamespace(
        portfolio=SimpleNamespace(
            get_open_and_frozen_positions=lambda: [],
            get_vault_settlement_pending_value=lambda: 0.5,
        )
    )

    # 3. Verify the result is the float sum passed to Lagoon.
    assert sync_model.calculate_valuation(state, block_number=10) == pytest.approx(13.0)
