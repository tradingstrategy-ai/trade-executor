"""Manual native deposit closures must not inherit assumed-open admission."""

import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradeexecutor.backtest.backtest_pricing import BacktestPricing
from tradeexecutor.ethereum.vault.hypercore_valuation import HypercoreVaultPricing
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.strategy.redemption import DepositBlockReason, DepositCheckResult, DepositCheckStage
from tradeexecutor.strategy.vault_closure import (
    VaultClosureOutcome, check_live_deposit, get_manual_deposit_closure,
    get_vault_closure, can_interact_with_vault,
)
from tradeexecutor.testing.vault_fee_sample import load_fee_sample


GROWI_HF_2_ADDRESS = "0x9b5b1931fa5838a891973f9877c7648c93ee80cc"


def test_growi_manual_closure_beats_assumed_open_and_provider_routes() -> None:
    """Prevent both early-history purchases and live/simulated provider bypasses.

    1. Supply the exact native identity and the NB04 purchase timestamps.
    2. Check historical, window and captured-open assumptions cannot admit it.
    3. Check live/native simulation routes refuse before requesting any API data.
    """
    # 1. Minimal identity data substitutes metadata; admission logic remains real.
    pair = SimpleNamespace(chain_id=9999, pool_address=GROWI_HF_2_ADDRESS.upper(),
        is_hyperliquid_vault=lambda: True, get_ticker=lambda: "Growi HF-2-USDC")
    # 2. These are the actual original NB04 deposit request dates, plus a later date.
    for ts in (datetime.datetime(2025, 9, 22), datetime.datetime(2025, 10, 17), datetime.datetime(2026, 11, 10)):
        assessment = get_vault_closure(pair, timestamp=ts,
            historical_state={"deposits_open": True, "max_deposit": 1_000_000},
            unknown_is_closed=False, include_strategy_policy=False, apply_hypercore_cutoff=False)
        assert assessment.status is VaultClosureOutcome.closed
        assert assessment.blocked and assessment.source == "manual_override"
        assert assessment.reason == "permanently_closed"
        check = assessment.as_deposit_check(pair, ts, DepositCheckStage.buy_rebalance)
        assert not check.can_deposit and check.max_deposit == 0
        assert check.reason_code is DepositBlockReason.vault_deposits_closed
        assert check.evidence_source == "manual_override"
    window = SimpleNamespace(is_deposit_open=lambda ts: True)
    assert not can_interact_with_vault(pair, timestamp=ts, window_override=window)
    assert not can_interact_with_vault(pair, deposit_check=DepositCheckResult(can_deposit=True))
    with pytest.raises(ValueError, match="captured deposit check"):
        get_vault_closure(pair, timestamp=ts, deposit_check=DepositCheckResult(can_deposit=True))
    with pytest.raises(ValueError, match="exactly one"):
        get_vault_closure(pair, pricing_model=Mock(), historical_state={})
    with pytest.raises(ValueError, match="Historical-only controls"):
        get_vault_closure(pair, pricing_model=Mock(), trust_protocol_state=True)
    # 3. Provider mocks make unintended network access fail the test.
    provider = Mock()
    assert not can_interact_with_vault(pair, timestamp=ts, pricing_model=provider)
    provider.check_deposit.assert_not_called()
    adapter = Mock()
    assert not check_live_deposit(timestamp=ts, pair=pair, vault=adapter, owner=None,
        stage=DepositCheckStage.buy_rebalance).can_deposit
    adapter.get_deposit_manager.assert_not_called()
    for simulate in (False, True):
        pricing = HypercoreVaultPricing(value_func=None, simulate=simulate)
        pricing._get_vault_info = Mock(side_effect=AssertionError("Unexpected API request"))
        assert not pricing.check_deposit(ts, pair).can_deposit
        pricing._get_vault_info.assert_not_called()
        # Final execution must also block a buy constructed without pricing.
        routing = object.__new__(HypercoreVaultRouting)
        routing.simulate = simulate
        routing._get_session = Mock(side_effect=AssertionError("Unexpected API session"))
        buy = SimpleNamespace(pair=pair, is_buy=lambda: True)
        assert "permanently closed" in routing.check_trade_before_execution(buy)
        routing._get_session.assert_not_called()
        # Redemptions retain the existing preflight result and simulation path.
        routing._check_full_close_before_execution = Mock(return_value="Original redemption preflight")
        sell = SimpleNamespace(pair=pair, is_buy=lambda: False)
        assert routing.check_trade_before_execution(sell) == (None if simulate else "Original redemption preflight")


def test_manual_closure_preserves_loaded_history_redemptions_and_other_vaults() -> None:
    """Keep a closed vault data-only for deposits without narrowing other histories.

    1. Load the real native fixture and apply Growi's identity to its translated pair.
    2. Check the public backtest API refuses the original early purchase dates.
    3. Verify history, redemption permission and unrelated native assumptions survive.
    """
    # 1. Fixture loading exercises production translation without downloads.
    universe = load_fee_sample(Path(__file__).resolve().parents[1] / "backtest/fixtures/vault_externalised_fees", "hyperliquid")
    pair = next(universe.iterate_pairs())
    pair.pool_address = GROWI_HF_2_ADDRESS
    candle_count = len(universe.data_universe.candles.df)
    pricing = BacktestPricing(universe, None, allow_missing_fees=True)
    # 2. Public deposit diagnostics and the boolean decision must agree.
    for ts in (datetime.datetime(2025, 9, 22), datetime.datetime(2025, 10, 17)):
        check = pricing.check_deposit(ts, pair)
        assert not pricing.can_deposit(ts, pair) and not check.can_deposit
        assert check.evidence_status is VaultClosureOutcome.closed
        assert check.evidence_source == "manual_override"
    # 3. Only deposits for this exact venue/address acquire the override.
    assert len(universe.data_universe.candles.df) == candle_count
    assert pair.get_ignore_reason() is None
    assert can_interact_with_vault(pair, operation="redemption", timestamp=ts,
        historical_state={"redemption_open": True, "max_redeem": 100})
    other = SimpleNamespace(chain_id=9999, pool_address="0x" + "1" * 40,
        is_hyperliquid_vault=lambda: True)
    assessment = get_vault_closure(other, timestamp=ts)
    assert assessment.status is VaultClosureOutcome.unknown and not assessment.blocked
    other.pool_address = GROWI_HF_2_ADDRESS
    other.chain_id = 999  # HyperEVM is a different venue from native HyperCore.
    assert get_manual_deposit_closure(other) is None
