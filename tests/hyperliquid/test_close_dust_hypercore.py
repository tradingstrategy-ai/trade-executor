"""Test closing dusty Hypercore vault positions and verify account correction ignores them.

Uses a downloaded production state (hyper-ai) that contains dust positions
(positions #1, #2, #3 with quantities below HYPERLIQUID_VAULT_CLOSE_EPSILON = 0.02).

1. Load production state and identify dust Hypercore vault positions.
2. Verify can_be_closed() returns True for dust positions.
3. Close dust positions via close_single_or_all_positions() with close_by_sell=True (auto-detect).
4. Verify positions moved to closed_positions with repair trades.
5. Verify no sell execution was attempted.
6. Verify _build_hypercore_vault_account_checks() does not re-pick closed dust positions.
7. Verify remaining non-dust Hypercore positions are still checked.
"""

import datetime
import shutil
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import UserVaultEquity

from tradeexecutor.cli.close_position import close_single_or_all_positions
from tradeexecutor.analysis.redemption_audit import audit_redemption_state
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.account_correction import _build_hypercore_vault_account_checks
from tradeexecutor.strategy.dust import HYPERLIQUID_VAULT_CLOSE_EPSILON
from tradeexecutor.strategy.execution_context import ExecutionContext
from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.sync_model import OnChainBalance, SyncModel
from tradeexecutor.strategy.valuation import ValuationModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


SAMPLE_STATE_FILE = Path(__file__).parent / "state" / "hyper-ai-dust-close.json"

pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.skipif(
        not SAMPLE_STATE_FILE.exists(),
        reason=f"Sample state file not found: {SAMPLE_STATE_FILE}",
    ),
]


def _make_mock_sync_model() -> MagicMock:
    """Build a mock SyncModel with the methods close_single_or_all_positions needs."""
    mock = MagicMock(spec=SyncModel)
    mock.get_token_storage_address.return_value = "0x000000000000000000000000000000000000dEaD"
    mock.get_key_address.return_value = "0x000000000000000000000000000000000000dEaD"
    mock.has_async_deposits.return_value = False
    mock.has_position_sync.return_value = False
    mock.sync_treasury.return_value = []

    hw = MagicMock()
    hw.address = "0x0000000000000000000000000000000000001234"
    hw.get_native_currency_balance.return_value = Decimal(1)
    mock.get_hot_wallet.return_value = hw

    return mock


def test_close_dust_hypercore_and_account_correction(
    tmp_path: Path,
):
    """Close dusty Hypercore vault positions from production state and verify account correction safety.

    1. Load production state and identify dust positions.
    2. Verify can_be_closed() returns True for dust positions.
    3. Record initial count of open Hypercore vault positions.
    4. Mock dependencies and close each dust position with auto-detect (close_by_sell=True, close_dust=None).
    5. Verify dust positions moved to closed_positions with repair trades.
    6. Verify no sell execution was attempted.
    7. Verify _build_hypercore_vault_account_checks() excludes closed dust positions.
    8. Verify remaining non-dust Hypercore positions are still present and checked.
    """

    # 1. Load production state and identify dust positions.
    state_file = tmp_path / "state.json"
    shutil.copy2(SAMPLE_STATE_FILE, state_file)
    state = State.read_json_file(state_file)

    all_hypercore_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.is_hyperliquid_vault()
    ]
    dust_positions = [p for p in all_hypercore_positions if p.can_be_closed()]
    non_dust_positions = [p for p in all_hypercore_positions if not p.can_be_closed()]

    assert len(dust_positions) >= 2, f"Expected at least 2 dust positions, got {len(dust_positions)}"
    assert len(non_dust_positions) >= 1, f"Expected at least 1 non-dust position, got {len(non_dust_positions)}"

    # 2. Verify can_be_closed() returns True for dust positions
    #    and their quantities are below the Hypercore dust epsilon.
    for p in dust_positions:
        assert p.can_be_closed(), f"Position #{p.position_id} should be closeable as dust"
        assert p.get_quantity() < HYPERLIQUID_VAULT_CLOSE_EPSILON, \
            f"Position #{p.position_id} qty {p.get_quantity()} should be below epsilon {HYPERLIQUID_VAULT_CLOSE_EPSILON}"

    # 3. Record initial count.
    initial_open_hypercore_count = len(all_hypercore_positions)
    initial_total_open = len(state.portfolio.open_positions)
    dust_position_ids = {p.position_id for p in dust_positions}

    # 4. Mock dependencies and close each dust position.
    #    We mock fetch_onchain_balances_multichain to return small equity values.
    def fake_fetch_balances(web3, address, assets, pairs=None, filter_zero=False, block_number=None):
        for pair in (pairs or []):
            yield OnChainBalance(
                block_number=None,
                timestamp=native_datetime_utc_now(),
                asset=pair.base,
                amount=Decimal("0.01"),
            )

    sync_model = _make_mock_sync_model()
    execution_model = MagicMock(spec=ExecutionModel)
    pricing_model = MagicMock(spec=PricingModel)
    valuation_model = MagicMock(spec=ValuationModel)
    routing_model = MagicMock(spec=RoutingModel)
    routing_state = MagicMock(spec=RoutingState)
    execution_context = MagicMock(spec=ExecutionContext)
    web3 = MagicMock()

    # We need a minimal universe mock with reserve_assets and get_reserve_asset
    universe = MagicMock(spec=TradingStrategyUniverse)
    universe.reserve_assets = set()
    universe.get_reserve_asset.return_value = MagicMock()

    with patch(
        "tradeexecutor.cli.close_position.fetch_onchain_balances_multichain",
        side_effect=fake_fetch_balances,
    ):
        for p in dust_positions:
            close_single_or_all_positions(
                web3=web3,
                execution_model=execution_model,
                execution_context=execution_context,
                pricing_model=pricing_model,
                sync_model=sync_model,
                state=state,
                universe=universe,
                routing_model=routing_model,
                routing_state=routing_state,
                valuation_model=valuation_model,
                slippage_tolerance=0.20,
                interactive=False,
                position_id=p.position_id,
                close_by_sell=True,
                close_dust=None,  # auto-detect
            )

    # 5. Verify dust positions moved to closed_positions with repair trades.
    for pid in dust_position_ids:
        assert pid not in state.portfolio.open_positions, \
            f"Dust position #{pid} should not be in open_positions"
        assert pid in state.portfolio.closed_positions, \
            f"Dust position #{pid} should be in closed_positions"

        closed_p = state.portfolio.closed_positions[pid]
        assert closed_p.is_closed(), f"Position #{pid} should be closed"

        # Should have a repair trade as the closing trade
        trades = list(closed_p.trades.values())
        closing_trade = trades[-1]
        assert closing_trade.trade_type == TradeType.repair, \
            f"Position #{pid} closing trade should be repair type, got {closing_trade.trade_type}"

        # Notes should mention dust closure
        assert "dust" in (closed_p.notes or "").lower(), \
            f"Position #{pid} notes should mention dust: {closed_p.notes}"

    # 6. Verify no sell execution was attempted.
    execution_model.execute_trades.assert_not_called()

    # 7. Verify _build_hypercore_vault_account_checks() excludes closed dust positions.
    #    Mock the balance fetcher for the account check path.
    remaining_vault_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.is_hyperliquid_vault()
    ]
    assert len(remaining_vault_positions) == initial_open_hypercore_count - len(dust_positions)

    def fake_multichain_balances_for_check(web3, address, assets, pairs=None, filter_zero=False, block_number=None):
        for pair in (pairs or []):
            yield OnChainBalance(
                block_number=None,
                timestamp=native_datetime_utc_now(),
                asset=pair.base,
                amount=Decimal("50.0"),
            )

    account_check_sync = MagicMock(spec=SyncModel)
    account_check_sync.web3 = web3
    account_check_sync.get_token_storage_address.return_value = "0x000000000000000000000000000000000000dEaD"

    with patch(
        "tradeexecutor.strategy.account_correction.fetch_onchain_balances_multichain",
        side_effect=fake_multichain_balances_for_check,
    ):
        corrections = _build_hypercore_vault_account_checks(state, account_check_sync)

    # 8. Verify remaining non-dust Hypercore positions are still checked
    #    and closed dust positions are excluded.
    checked_position_ids = set()
    for c in corrections:
        for pos in c.positions:
            checked_position_ids.add(pos.position_id)

    assert len(corrections) == len(remaining_vault_positions), \
        f"Expected {len(remaining_vault_positions)} account checks, got {len(corrections)}"

    for pid in dust_position_ids:
        assert pid not in checked_position_ids, \
            f"Closed dust position #{pid} should not appear in account corrections"

    for p in remaining_vault_positions:
        assert p.position_id in checked_position_ids, \
            f"Open position #{p.position_id} should appear in account corrections"


def test_sample_state_audit_reports_blocked_redemption_rows() -> None:
    """Audit the sample Hyper AI state for blocked redemption rows.

    1. Load the sample Hyper AI state fixture that already contains cannot_redeem signals.
    2. Run the read-only audit helper against a far-future timestamp so expired recorded lockups are easy to detect.
    3. Verify the helper returns blocked rows with readable identifiers and at least one mismatch candidate.
    """
    # 1. Load the sample Hyper AI state fixture that already contains cannot_redeem signals.
    state = State.read_json_file(SAMPLE_STATE_FILE)

    # 2. Run the read-only audit helper against a far-future timestamp so expired recorded lockups are easy to detect.
    rows, mismatch_count = audit_redemption_state(
        state,
        now=datetime.datetime(2026, 12, 31),
    )

    # 3. Verify the helper returns blocked rows with readable identifiers and at least one mismatch candidate.
    assert rows, "Expected at least one blocked redemption row from the sample state"
    assert mismatch_count >= 1
    assert rows[0].pair_ticker is not None
    assert rows[0].vault_address is not None
