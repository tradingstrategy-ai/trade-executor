"""Keep Upshift guard and payout handling within existing execution rules."""

from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from eth_defi.erc_4626.core import ERC4626Feature
from eth_defi.erc_4626.vault import ERC4626Vault
from eth_defi.token import TokenDiskCache, USDC_NATIVE_TOKEN
from eth_typing import HexAddress

from tradeexecutor.cli.commands import lagoon_deploy_vault as deployment
from tradeexecutor.ethereum.vault import vault_routing
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


@pytest.mark.parametrize("features", [set(), {ERC4626Feature.mellow_like}, {ERC4626Feature.atoma_like}, {ERC4626Feature.upshift_like, ERC4626Feature.upshift_multi_asset_like}])
def test_explicit_guard_vault_preserves_generic_protocols(monkeypatch: pytest.MonkeyPatch, features: set[ERC4626Feature]) -> None:
    """Specialise only guard policies supported by the deployment option.

    1. Replace RPC-owned detection and construction with observable boundaries.
    2. Resolve an explicit address through the deployment helper.
    3. Preserve generic support and forward complete Atoma/Upshift features.
    """
    # 1. Replace RPC-owned detection and construction with observable boundaries.
    web3 = MagicMock()
    cache = MagicMock(spec=TokenDiskCache)
    address = HexAddress("0x1111111111111111111111111111111111111111")
    descriptor = MagicMock(spec=ERC4626Vault)
    factory = MagicMock(return_value=descriptor)
    monkeypatch.setattr(deployment, "detect_vault_features", lambda *args, **kwargs: features)
    monkeypatch.setattr(deployment, "create_vault_instance", factory)
    # 2. Resolve an explicit address through the deployment helper.
    assert deployment._fetch_erc_4626_guard_vault(web3, address, token_cache=cache) is descriptor
    # 3. Preserve generic support and forward complete Atoma/Upshift features.
    expected = features if features & {ERC4626Feature.atoma_like, ERC4626Feature.upshift_like} else None
    factory.assert_called_once_with(web3, address, features=expected, token_cache=cache)


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
    "features,flow,expected",
    [
        ({ERC4626Feature.upshift_like}, "synchronous", False),
        ({ERC4626Feature.upshift_like}, "asynchronous", True),
        (set(), "synchronous", True),
        ({ERC4626Feature.gains_like}, "asynchronous", False),
        ({ERC4626Feature.d2_like}, "synchronous", True),
    ],
)
def test_live_flow_preserves_other_protocol_settlement_rules(features: set[ERC4626Feature], flow: str, expected: bool) -> None:
    """Refresh configurable Upshift delays without reclassifying other vaults.

    1. Construct persisted feature and live-flow metadata on a real pair.
    2. Evaluate the allocation cash-settlement rule.
    3. Retain conservative unknown/window rules and existing protocol behaviour.
    """
    # 1. Construct persisted feature and live-flow metadata on a real pair.
    quote = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    base = AssetIdentifier(1, "0x1111111111111111111111111111111111111111", "shares", 18)
    pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0, kind=TradingPairKind.vault, other_data={"vault_features": features, "vault_protocol": "custom", "vault_redemption_flow": flow})
    # 2. Evaluate the allocation cash-settlement rule.
    delayed = pair.has_delayed_vault_redemption()
    # 3. Retain conservative unknown/window rules and existing protocol behaviour.
    assert delayed is expected
