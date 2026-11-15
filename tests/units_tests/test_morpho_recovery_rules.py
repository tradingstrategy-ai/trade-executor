"""Keep a reviewed execution recovery separate from historical vault admission."""

from dataclasses import asdict
from unittest.mock import MagicMock, patch
from web3 import Web3

from tradeexecutor.ethereum.vault import vault_routing
from tradeexecutor.strategy.vault_protocol import morpho

from eth_defi.erc_4626.core import ERC4626Feature
from tradingstrategy.vault import VaultMetadata, VaultProtocolRules

from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.vault_protocol import enrich_vault_pair, redemption_recovery_fee_limit


def test_morpho_recovery_scope_and_pair_snapshot() -> None:
    """Retain the explicit cap through the production pair snapshot.

    1. Enrich both reviewed identities using ordinary loaded vault metadata.
    2. Round-trip the pair, checking the cap without changing fees or queues.
    3. Check wrong-chain/unreviewed identities and older snapshots stay disabled.
    """
    # 1. Enrich both reviewed identities through the same dispatcher as trading.
    for address in ("0x36cfe1568461e499391ef0a555300f1ae2da2439", "0xc92a37fd0250f4eecf092960a2f70a1334217528"):
        pair = TradingPairIdentifier(
            AssetIdentifier(1, address, "vault", 18),
            AssetIdentifier(1, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "USDC", 6),
            pool_address=address, exchange_address=address, fee=0,
            kind=TradingPairKind.vault,
            other_data={"token_metadata": VaultMetadata(vault_name="reviewed", protocol_name="Morpho", protocol_slug="morpho", features=[ERC4626Feature.morpho_v2_like], deposit_fee=0, withdrawal_fee=0)},
        )
        enrich_vault_pair(pair)
        # 2. The saved execution cap is not a blanket backtest fee or queue.
        restored = TradingPairIdentifier.from_json(pair.to_json())
        rules = restored.get_vault_protocol_rules()
        assert rules == pair.get_vault_protocol_rules()
        assert rules.redemption_recovery_max_fee_bps == 1
        assert rules.redemption_fee == 0 and rules.redemption_fee_supported
        assert not rules.deposit_queued and not rules.redemption_queued
        assert rules.simulation_max_holding_seconds is None
        # 3. Preserve the disabled default and exact identity scope.
        old = asdict(rules)
        old.pop("redemption_recovery_max_fee_bps")
        assert VaultProtocolRules(**old).redemption_recovery_max_fee_bps is None
        assert redemption_recovery_fee_limit(42161, address) is None
        # Old open-position pairs gain today's reviewed cap; revocation clears
        # it on the cached adapter as well. The fake factory avoids RPC: this
        # assertion concerns execution policy selection, not contract mechanics.
        restored.other_data.pop("vault_protocol_rules")
        restored.other_data["vault_features"] = [ERC4626Feature.morpho_v2_like]
        connection = Web3()
        adapter = MagicMock()
        with patch.dict(vault_routing._vault_cache, {}, clear=True), patch.object(vault_routing, "create_vault_instance", return_value=adapter):
            assert vault_routing.get_vault_for_pair(connection, restored).redemption_recovery_max_fee_bps == 1
            with patch.dict(morpho.REDEMPTION_RECOVERY_MAX_FEE_BPS, {}, clear=True):
                assert vault_routing.get_vault_for_pair(connection, restored) is adapter
                assert adapter.redemption_recovery_max_fee_bps is None
    assert redemption_recovery_fee_limit(1, "0x" + "1" * 40) is None
