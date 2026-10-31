"""Keep Upshift guard and payout handling within existing execution rules."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.ethereum.vault import vault_routing
from tradeexecutor.ethereum.vault.settlement_retry import has_pending_vault_claims
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


@pytest.mark.parametrize("target_chain", [1, 42161], ids=("home", "satellite"))
@pytest.mark.parametrize("buy", [True, False], ids=("deposit", "redeem"))
def test_routing_rejects_payout_different_from_funded_reserve(monkeypatch: pytest.MonkeyPatch, target_chain: int, buy: bool) -> None:
    """Never trade a USDx payout vault as a USDC reserve position.

    1. Supply an observed USDx payout vault behind mocked RPC discovery.
    2. Plan a deposit or redemption using the real routing entry point.
    3. Refuse before constructing or signing any manager transaction.
    """
    # 1. Supply an observed USDx payout vault behind mocked RPC discovery.
    reserve = SimpleNamespace(chain_id=1, address=USDC_NATIVE_TOKEN[1].lower())
    manager = MagicMock()
    vault = SimpleNamespace(chain_id=target_chain, denomination_token=SimpleNamespace(address_lower="0x1111111111111111111111111111111111111111"), get_deposit_manager=manager)
    monkeypatch.setattr(vault_routing, "get_vault_for_pair", lambda *args, **kwargs: vault)
    builder = MagicMock()
    routing_state = vault_routing.VaultRoutingState(builder, SimpleNamespace(get_reserve_asset=lambda: reserve))
    trade = MagicMock(spec=TradeExecution)
    trade.is_vault.return_value = True
    trade.slippage_tolerance = 0.01
    trade.is_buy.return_value = buy
    trade.pair = SimpleNamespace(quote=SimpleNamespace(chain_id=target_chain, address=USDC_NATIVE_TOKEN[target_chain].lower()))
    routing = vault_routing.VaultRouting(reserve.address)
    # 2. Plan a deposit or redemption using the real routing entry point.
    with pytest.raises(vault_routing.IncompatibleDepositAsset, match="withdrawal asset differs") as raised:
        routing.deposit_or_redeem(State(), routing_state, trade)
    # 3. Refuse before constructing or signing any manager transaction.
    assert raised.value.selected_asset.lower() == USDC_NATIVE_TOKEN[target_chain].lower()
    manager.assert_not_called()
    builder.sign_transaction.assert_not_called()


def test_routing_rejects_accepted_input_without_funding_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """An accepted input override cannot spend a different reserve token.

    1. Supply USDC payout and a different accepted input behind mocked RPC reads.
    2. Select that input for a USDC-funded deposit through actual routing.
    3. Refuse before constructing or signing a deposit transaction.
    """
    # 1. Supply USDC payout and a different accepted input behind mocked RPC reads.
    reserve = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    accepted = SimpleNamespace(symbol="other", address="0x1111111111111111111111111111111111111111")
    manager = SimpleNamespace(fetch_accepted_assets=lambda: (accepted,), create_deposit_request=MagicMock())
    vault = SimpleNamespace(chain_id=1, denomination_token=SimpleNamespace(address_lower=reserve.address), get_deposit_manager=lambda: manager)
    monkeypatch.setattr(vault_routing, "get_vault_for_pair", lambda *args, **kwargs: vault)
    builder = MagicMock()
    routing_state = vault_routing.VaultRoutingState(builder, SimpleNamespace(get_reserve_asset=lambda: reserve))
    trade = MagicMock(spec=TradeExecution)
    trade.is_vault.return_value = trade.is_buy.return_value = True
    trade.slippage_tolerance = 0.01
    trade.get_planned_reserve.return_value = Decimal(10)
    trade.pair = SimpleNamespace(quote=reserve, base=reserve)
    routing = vault_routing.VaultRouting(reserve.address, deposit_asset_override=accepted.address)
    # 2. Select that input for a USDC-funded deposit through actual routing.
    with pytest.raises(vault_routing.IncompatibleDepositAsset, match="differs from the trade input"):
        routing.deposit_or_redeem(State(), routing_state, trade)
    # 3. Refuse before constructing or signing a deposit transaction.
    manager.create_deposit_request.assert_not_called()
    builder.sign_transaction.assert_not_called()


@pytest.mark.parametrize(
    "features,expected",
    [
        ({ERC4626Feature.upshift_like}, True),
        ({ERC4626Feature.lagoon_like}, True),
        ({ERC4626Feature.atoma_like}, True),
        (set(), True),
        ({ERC4626Feature.gains_like}, False),
        ({ERC4626Feature.d2_like}, True),
    ],
)
def test_delayed_redemption_uses_existing_protocol_classification(features: set[ERC4626Feature], expected: bool) -> None:
    """Use the same conservative allocation rule for queued vault protocols.

    1. Construct ordinary protocol metadata on a real pair.
    2. Evaluate the allocation cash-settlement rule.
    3. Retain conservative unknown/window rules and existing protocol behaviour.
    """
    # 1. Construct ordinary protocol metadata on a real pair.
    quote = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    base = AssetIdentifier(1, "0x1111111111111111111111111111111111111111", "shares", 18)
    pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0, kind=TradingPairKind.vault, other_data={"vault_features": features, "vault_protocol": "custom"})
    # 2. Evaluate the allocation cash-settlement rule.
    delayed = pair.has_delayed_vault_redemption()
    # 3. Retain conservative unknown/window rules and existing protocol behaviour.
    assert delayed is expected


@pytest.mark.parametrize("metadata,expected", [({"vault_requires_ticket_valuation": True}, True), ({"upshift_year": 2026}, False)])
def test_pending_claim_recovery_uses_generic_trade_metadata(metadata: dict, expected: bool) -> None:
    """Use the persisted generic recovery flag without interpreting ticket fields.

    1. Supply an in-flight sell in the existing portfolio state interface.
    2. Check whether valuation needs prior claim recovery.
    3. Require the generic flag rather than protocol-specific date fields.

    Lightweight state stand-ins isolate this predicate from RPC and execution.
    """
    # 1. Supply an in-flight sell in the existing portfolio state interface.
    trade = SimpleNamespace(other_data=metadata, is_sell=lambda: True, is_vault_settlement_in_flight=lambda: True)
    position = SimpleNamespace(pair=SimpleNamespace(is_vault=lambda: True, get_vault_features=lambda: {ERC4626Feature.upshift_like}), has_pending_vault_settlement=lambda: True, trades={1: trade})
    state = SimpleNamespace(portfolio=SimpleNamespace(get_all_positions=lambda **kwargs: [position]))
    # 2. Check whether valuation needs prior claim recovery.
    recovery_needed = has_pending_vault_claims(state)
    # 3. Require the generic flag rather than protocol-specific date fields.
    assert recovery_needed is expected
