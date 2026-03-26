"""Test close-position CLI path with Hypercore vault positions.

Hypercore vaults have a 1-day redemption cycle and cannot be tested live,
so all Hyperliquid API calls are mocked.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from eth_defi.hyperliquid.api import UserVaultEquity
from eth_defi.token import USDC_NATIVE_TOKEN

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.ethereum.multichain_balance import (
    fetch_onchain_balances_multichain,
)
from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.sync_model import OnChainBalance, SyncModel
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.strategy.trading_strategy_universe import (
    TradingStrategyUniverse,
    create_pair_universe_from_code,
    translate_token,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_SAFE_ADDRESS = "0x000000000000000000000000000000000000dEaD"
VAULT_ADDRESS = HLP_VAULT_ADDRESS["mainnet"]


@pytest.fixture()
def hypercore_usdc_asset() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=999,
        address=USDC_NATIVE_TOKEN[999],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def hypercore_vault_pair(hypercore_usdc_asset: AssetIdentifier) -> TradingPairIdentifier:
    return create_hypercore_vault_pair(
        quote=hypercore_usdc_asset,
        vault_address=VAULT_ADDRESS,
    )


@pytest.fixture()
def hypercore_strategy_universe(
    hypercore_vault_pair: TradingPairIdentifier,
) -> TradingStrategyUniverse:
    exchange_universe = ExchangeUniverse(
        exchanges={
            1: Exchange(
                chain_id=ChainId(999),
                chain_slug="hyperliquid",
                exchange_id=1,
                exchange_slug="hypercore",
                address=hypercore_vault_pair.exchange_address,
                exchange_type=ExchangeType.erc_4626_vault,
                pair_count=1,
            ),
        }
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.hypercore,
        [hypercore_vault_pair],
    )
    pair_universe.exchange_universe = exchange_universe

    universe = Universe(
        chains={ChainId(999)},
        time_bucket=TimeBucket.not_applicable,
        exchange_universe=pair_universe.exchange_universe,
        pairs=pair_universe,
    )

    reserve_token = pair_universe.get_token(hypercore_vault_pair.quote.address)
    return TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets={translate_token(reserve_token)},
    )


def _create_state_with_hypercore_position(
    pair: TradingPairIdentifier,
    usdc: AssetIdentifier,
    reserve_amount: Decimal = Decimal(100),
    position_amount: Decimal = Decimal(50),
) -> State:
    """Build a State with reserves and one open Hypercore vault position."""
    state = State()

    # Initialise reserves (reserve_token_price=1.0 for USDC)
    state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(usdc, reserve_amount, "Initial deposit")

    # Create a buy trade for the Hypercore vault and spoof it as executed
    position, trade, _created = state.create_trade(
        strategy_cycle_at=datetime.datetime(2026, 3, 24),
        pair=pair,
        quantity=None,
        reserve=position_amount,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc,
        reserve_currency_price=1.0,
        notes="Test Hypercore vault position",
    )
    trade.mark_success(
        executed_at=datetime.datetime(2026, 3, 24, 0, 2),
        executed_price=1.0,
        executed_quantity=position_amount,
        executed_reserve=position_amount,
        lp_fees=0,
        native_token_price=0,
        force=True,
    )

    return state


def _make_mock_sync_model() -> MagicMock:
    """Build a mock SyncModel with the methods close_single_or_all_positions needs."""
    mock = MagicMock(spec=SyncModel)
    mock.get_token_storage_address.return_value = FAKE_SAFE_ADDRESS
    mock.get_key_address.return_value = FAKE_SAFE_ADDRESS
    mock.has_async_deposits.return_value = False
    mock.has_position_sync.return_value = False
    mock.sync_treasury.return_value = []

    # Hot wallet mock
    hw = MagicMock()
    hw.address = "0x0000000000000000000000000000000000001234"
    hw.get_native_currency_balance.return_value = Decimal(1)
    mock.get_hot_wallet.return_value = hw

    return mock


# ---------------------------------------------------------------------------
# Tests for fetch_onchain_balances_multichain
# ---------------------------------------------------------------------------


def test_fetch_onchain_balances_multichain_routes_hypercore(
    monkeypatch,
    hypercore_vault_pair: TradingPairIdentifier,
):
    """Verify the multichain helper routes Hypercore pairs to the Hyperliquid API.

    1. Mock the Hyperliquid session and equity API.
    2. Call fetch_onchain_balances_multichain in pair-aware mode.
    3. Assert the result is an OnChainBalance with the mocked equity.
    """

    # 1. Mock session and API
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.create_hyperliquid_session",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.fetch_user_vault_equity",
        lambda session, user, vault_address: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=datetime.datetime(2026, 3, 25),
        ),
    )

    web3 = MagicMock()

    # 2. Call the helper in pair-aware mode
    results = list(fetch_onchain_balances_multichain(
        web3,
        FAKE_SAFE_ADDRESS,
        [hypercore_vault_pair.base],
        pairs=[hypercore_vault_pair],
        filter_zero=False,
    ))

    # 3. Check result
    assert len(results) == 1
    assert isinstance(results[0], OnChainBalance)
    assert results[0].amount == Decimal("50")
    assert results[0].asset == hypercore_vault_pair.base
    assert results[0].block_number is None


def test_fetch_onchain_balances_multichain_handles_none_equity(
    monkeypatch,
    hypercore_vault_pair: TradingPairIdentifier,
):
    """Verify the helper returns amount=0 when the API reports no vault position.

    1. Mock fetch_user_vault_equity to return None.
    2. Call fetch_onchain_balances_multichain in pair-aware mode.
    3. Assert result.amount == Decimal(0).
    """

    # 1. Mock API returning None (no position)
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.create_hyperliquid_session",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.fetch_user_vault_equity",
        lambda session, user, vault_address: None,
    )

    web3 = MagicMock()

    # 2. Call the helper
    results = list(fetch_onchain_balances_multichain(
        web3,
        FAKE_SAFE_ADDRESS,
        [hypercore_vault_pair.base],
        pairs=[hypercore_vault_pair],
        filter_zero=False,
    ))

    # 3. Check zero balance
    assert len(results) == 1
    assert results[0].amount == Decimal(0)
    assert results[0].asset == hypercore_vault_pair.base


def test_fetch_onchain_balances_multichain_routes_erc20(
    monkeypatch,
    hypercore_usdc_asset: AssetIdentifier,
):
    """Verify that non-Hypercore pairs batch ERC-20 assets via fetch_address_balances.

    1. Create a regular spot pair (not Hypercore).
    2. Call fetch_onchain_balances_multichain in pair-aware mode.
    3. Assert fetch_address_balances was called with the base asset.
    """

    # 1. Regular ERC-20 spot pair
    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=999,
            address="0x1111111111111111111111111111111111111111",
            token_symbol="WETH",
            decimals=18,
        ),
        quote=hypercore_usdc_asset,
        pool_address="0x2222222222222222222222222222222222222222",
        exchange_address="0x3333333333333333333333333333333333333333",
        internal_id=999,
        internal_exchange_id=1,
    )

    expected_balance = OnChainBalance(
        block_number=100,
        timestamp=datetime.datetime(2026, 3, 26),
        asset=spot_pair.base,
        amount=Decimal("1.5"),
    )

    web3 = MagicMock()

    # 2. Mock fetch_address_balances at the module level
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.fetch_address_balances",
        lambda *args, **kwargs: iter([expected_balance]),
    )

    results = list(fetch_onchain_balances_multichain(
        web3,
        FAKE_SAFE_ADDRESS,
        [spot_pair.base],
        pairs=[spot_pair],
        filter_zero=False,
    ))

    # 3. Check delegation
    assert len(results) == 1
    assert results[0].amount == Decimal("1.5")


# ---------------------------------------------------------------------------
# End-to-end: close-position with mark-down
# ---------------------------------------------------------------------------


def test_close_position_hypercore_vault_mark_down(
    monkeypatch,
    hypercore_vault_pair: TradingPairIdentifier,
    hypercore_usdc_asset: AssetIdentifier,
    hypercore_strategy_universe: TradingStrategyUniverse,
):
    """Verify close-position mark-down path succeeds for a Hypercore vault position.

    This is the code path that crashed in production because
    fetch_onchain_balances tried to call ERC-20 balanceOf on a Hypercore vault.

    1. Create state with an open Hypercore vault position and reserve cash.
    2. Mock the Hyperliquid API to return deterministic vault equity.
    3. Mock the sync model and execution model dependencies.
    4. Call close_single_or_all_positions with close_by_sell=False.
    5. Assert the position was closed and moved to closed_positions.
    """
    from tradeexecutor.cli.close_position import close_single_or_all_positions

    # 1. Build state
    state = _create_state_with_hypercore_position(
        pair=hypercore_vault_pair,
        usdc=hypercore_usdc_asset,
        reserve_amount=Decimal(100),
        position_amount=Decimal(50),
    )
    assert len(state.portfolio.open_positions) == 1

    # 2. Mock Hyperliquid API
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.create_hyperliquid_session",
        lambda **kwargs: MagicMock(),
    )
    monkeypatch.setattr(
        "tradeexecutor.ethereum.multichain_balance.fetch_user_vault_equity",
        lambda session, user, vault_address: UserVaultEquity(
            vault_address=vault_address,
            equity=Decimal("50"),
            locked_until=datetime.datetime(2026, 3, 25),
        ),
    )

    # 3. Mock dependencies (spec= makes isinstance checks pass)
    sync_model = _make_mock_sync_model()
    execution_model = MagicMock(spec=ExecutionModel)
    pricing_model = MagicMock(spec=PricingModel)
    valuation_model = MagicMock(spec=ValuationModel)
    routing_model = MagicMock(spec=RoutingModel)
    routing_state = MagicMock(spec=RoutingState)
    web3 = MagicMock()
    execution_context = MagicMock(spec=ExecutionContext)

    # 4. Close with mark-down (no sell)
    position = next(iter(state.portfolio.open_positions.values()))
    close_single_or_all_positions(
        web3=web3,
        execution_model=execution_model,
        execution_context=execution_context,
        pricing_model=pricing_model,
        sync_model=sync_model,
        state=state,
        universe=hypercore_strategy_universe,
        routing_model=routing_model,
        routing_state=routing_state,
        valuation_model=valuation_model,
        slippage_tolerance=0.20,
        interactive=False,
        position_id=position.position_id,
        unit_testing=False,
        close_by_sell=False,
    )

    # 5. Verify position was closed
    assert len(state.portfolio.open_positions) == 0
    assert len(state.portfolio.closed_positions) == 1
    closed = next(iter(state.portfolio.closed_positions.values()))
    assert closed.is_closed()
