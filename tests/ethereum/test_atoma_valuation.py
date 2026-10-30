"""Atoma entitlement arithmetic and zero-loss accounting on existing positions.

Only the manager/RPC boundary is stubbed here to isolate arithmetic. Real ABI,
custody and receipt behaviour is exercised by test_atoma_async_cli.py on Anvil.
"""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from eth_defi.erc_4626.vault_protocol.atoma.deposit_redeem import AtomaDepositManager, AtomaFlowAnalysis
from eth_defi.vault.deposit_redeem import AsyncVaultRequestStatus, VaultFlowUnavailable
from hexbytes import HexBytes

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_routing import convert_vault_flow_analysis
from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus, TradeType


@pytest.fixture()
def pending_atoma() -> tuple[State, TradingPosition, VaultValuator, Mock]:
    """Create real executor accounting with a stubbed settled four-share claim."""
    ts = datetime.datetime(2026, 10, 30)
    usdc = AssetIdentifier(42161, "0x" + "11" * 20, "USDC", 6)
    share = AssetIdentifier(42161, "0x" + "22" * 20, "ATOMA", 12)
    pair = TradingPairIdentifier(share, usdc, share.address, share.address, kind=TradingPairKind.vault, fee=0)
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc)
    reserve.quantity = Decimal(200)
    reserve.reserve_token_price = 1.0
    position, buy, _ = state.create_trade(ts, pair, Decimal(10), None, 10.0, TradeType.rebalance, usdc, 1.0)
    state.start_execution(ts, buy)
    buy.mark_broadcasted(ts)
    state.mark_trade_success(ts, buy, 10.0, Decimal(10), Decimal(100), 0, 0)
    _, sell, _ = state.create_trade(ts, pair, Decimal(-4), None, 10.0, TradeType.rebalance, usdc, 1.0)
    state.start_execution(ts, sell)
    sell.mark_broadcasted(ts)
    sell.other_data["vault_async_flow"] = True
    state.mark_vault_settlement_pending(ts, sell, {"vault_raw_amount": "4"})
    # Stub only the protocol boundary; arithmetic/state transitions remain real.
    manager = Mock(spec=AtomaDepositManager)
    manager.vault = Mock()
    manager.web3.eth.block_number = 100
    manager.reconstruct_redemption_ticket.return_value = SimpleNamespace(raw_shares=4, reconciliation_required=False)
    manager.get_redemption_request_status.return_value = AsyncVaultRequestStatus.claimable
    manager.fetch_claim_amounts.return_value = (20, 0, 20)
    manager.vault.share_token.convert_to_decimals.return_value = Decimal(4)
    manager.vault.denomination_token.convert_to_decimals.return_value = Decimal(20)
    manager.estimate_redeem.side_effect = lambda *, owner, shares, block_identifier: shares * Decimal(10)
    pricing = Mock(spec=VaultPricing)
    pricing.get_vault.return_value.get_deposit_manager.return_value = manager
    pricing.get_owner_address.return_value = "0x" + "33" * 20
    return state, position, VaultValuator(pricing), manager


def test_atoma_pending_entitlement_valuation(pending_atoma: tuple[State, TradingPosition, VaultValuator, Mock]) -> None:
    """Value a partial illiquid claim once and fail without refreshing invalid data.

    1. Value remaining shares plus the settled entitlement without liquidity checks.
    2. Refuse an aggregate mismatch without changing the last successful valuation.
    3. Refuse a consumed ticket so receipt recovery, rather than zero valuation, runs.
    """
    state, position, valuator, manager = pending_atoma
    ts = datetime.datetime(2026, 10, 30, 1)
    # 1. Value remaining shares plus the settled entitlement without liquidity checks.
    manager.can_finish_redeem.return_value = False
    event = valuator(ts, position)
    assert event.new_value == pytest.approx(80)
    assert state.portfolio.get_net_asset_value() == pytest.approx(180)
    assert state.portfolio.get_cash() == pytest.approx(100)
    assert position.get_quantity() == Decimal(10)
    manager.can_finish_redeem.assert_not_called()
    manager.fetch_claim_amounts.assert_called_once_with(manager.reconstruct_redemption_ticket.return_value, block_identifier=100)
    # 2. Refuse an aggregate mismatch without changing the last successful valuation.
    manager.fetch_claim_amounts.side_effect = VaultFlowUnavailable("request_reconciliation_required")
    with pytest.raises(VaultFlowUnavailable, match="reconciliation"):
        valuator(ts + datetime.timedelta(minutes=1), position)
    assert position.last_pricing_at == ts
    assert position.get_value() == pytest.approx(80)
    # 3. Refuse a consumed ticket so receipt recovery, rather than zero valuation, runs.
    manager.get_redemption_request_status.return_value = AsyncVaultRequestStatus.none
    with pytest.raises(VaultFlowUnavailable, match="Recover"):
        valuator(ts + datetime.timedelta(minutes=2), position)
    assert position.last_pricing_at == ts


def test_atoma_zero_redemption_accounting(pending_atoma: tuple[State, TradingPosition, VaultValuator, Mock]) -> None:
    """A validated total-loss receipt realises cost basis without crediting cash.

    1. Convert a manager-validated zero payout for four consumed shares.
    2. Verify ordinary trade accounting still refuses zero proceeds.
    3. Apply the explicit vault-loss success and verify quantity, cash and PnL.
    """
    state, position, _, _ = pending_atoma
    trade = position.get_last_trade()
    ts = datetime.datetime(2026, 10, 30, 1)
    # 1. Convert a manager-validated zero payout for four consumed shares.
    analysis = AtomaFlowAnalysis(
        from_="0x" + "33" * 20, to="0x" + "33" * 20,
        denomination_amount=Decimal(0), share_count=Decimal(4),
        tx_hash=HexBytes("0x" + "aa" * 32), block_number=100,
        block_timestamp=ts, gross_assets=Decimal(0), withdrawal_fee=Decimal(0),
        implementation="0x" + "44" * 20,
    )
    reserve, quantity, price = convert_vault_flow_analysis(analysis, direction="redeem")
    assert (reserve, quantity, price) == (Decimal(0), Decimal(-4), Decimal(0))
    # 2. Verify ordinary trade accounting still refuses zero proceeds.
    trade.vault_settlement_pending_at = None
    with pytest.raises(AssertionError, match="reserve must be positive"):
        state.mark_trade_success(ts, trade, float(price), quantity, reserve, 0, 0)
    # 3. Apply the explicit vault-loss success and verify quantity, cash and PnL.
    state.mark_trade_success(ts, trade, float(price), quantity, reserve, 0, 0, allow_zero_proceeds=True)
    assert trade.get_status() is TradeStatus.success
    assert position.get_quantity() == Decimal(6)
    assert state.portfolio.get_cash() == pytest.approx(100)
    assert position.get_realised_profit_usd() == pytest.approx(-40)
