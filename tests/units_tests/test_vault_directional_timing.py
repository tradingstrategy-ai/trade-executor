"""Directional timing for vaults whose withdrawal interface is asynchronous."""

import datetime
from dataclasses import asdict

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.chart.standard.vault import _get_no_data_settlement_delay, _has_backtest_settlement_logic
from tradeexecutor.strategy.vault_protocol import resolve_vault_protocol_rules


def test_vault_directional_timing() -> None:
    """Keep immediate deposits separate from queued withdrawal funding.

    1. Build metadata for each affected manager, including overlapping ERC-7540 flags.
    2. Check planning and backtest directions agree without historical deposit gates.
    3. Check an explicit simulation override still takes precedence.
    4. Check Upshift estimates affect delayed exits without delaying deposits.
    """
    # 1. Build metadata for each affected manager, including overlapping ERC-7540 flags.
    quote = AssetIdentifier(1, "0x" + "11" * 20, "USDC", 6)
    base = AssetIdentifier(1, "0x" + "22" * 20, "SHARE", 18)
    execution = BacktestExecution(SimulatedWallet())
    cases = [
        ({ERC4626Feature.gains_like}, True),
        ({ERC4626Feature.nova_markets_like}, True),
        ({ERC4626Feature.ember_like}, True),
        ({ERC4626Feature.accountable_like, ERC4626Feature.erc_7540_like, ERC4626Feature.erc_7575_like}, True),
        ({ERC4626Feature.plutus_like, ERC4626Feature.erc_7540_like}, True),
        ({ERC4626Feature.plutus_like}, False),
        ({ERC4626Feature.atoma_like}, True),
        ({ERC4626Feature.upshift_like}, True),
    ]
    for features, delayed in cases:
        pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0, kind=TradingPairKind.vault, other_data={"vault_features": features})
        # 2. Check planning and backtest directions agree without historical deposit gates.
        assert not pair.is_async_vault(), features
        assert pair.has_delayed_vault_redemption() is delayed, features
        assert not execution._is_async_vault(pair, is_buy=True), features
        assert execution._is_async_vault(pair, is_buy=False) is delayed, features
        assert not execution._requires_historical_deposit_settlement(pair), features
        assert _has_backtest_settlement_logic(pair) is delayed, features
        if ERC4626Feature.plutus_like in features and delayed:
            assert _get_no_data_settlement_delay(pair, []) == datetime.timedelta(days=14)

        # 3. Check an explicit simulation override still takes precedence.
        overridden = BacktestExecution(SimulatedWallet(), vault_settlement_delay_overrides={pair.pool_address: datetime.timedelta(0)})
        assert overridden._is_async_vault(pair, is_buy=True)
        assert overridden._is_async_vault(pair, is_buy=False)

    # 4. Check Upshift estimates affect delayed exits without delaying deposits.
    pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0,
        kind=TradingPairKind.vault, other_data={"vault_features": {ERC4626Feature.upshift_like}})
    pair.other_data["vault_estimated_settlement"] = datetime.timedelta(days=7)
    timestamp = datetime.datetime(2026, 1, 1)
    assert not execution._is_async_vault(pair, is_buy=True)
    assert execution._get_settlement_due(pair, timestamp) == timestamp + datetime.timedelta(days=7)
    pair.other_data["vault_estimated_settlement"] = 0
    # Zero metadata falls back to the global two-day delay; only an explicit override permits zero.
    assert execution._get_settlement_due(pair, timestamp) == timestamp + datetime.timedelta(days=2)


def test_event_gated_vaults_keep_current_classification_with_old_snapshots() -> None:
    """Keep historical deposit gates when saved rules lack the current capability.

    1. Create Lagoon, D2 and T3tris pairs using only their protocol feature.
    2. Restore old rules with a missing or false event flag.
    3. Check the current deposit gate and T3tris timing still use protocol rules.
    """
    # 1. Create Lagoon, D2 and T3tris pairs using only their protocol feature.
    quote = AssetIdentifier(1, "0x" + "11" * 20, "USDC", 6)
    base = AssetIdentifier(1, "0x" + "22" * 20, "SHARE", 18)
    execution = BacktestExecution(SimulatedWallet())
    timestamp = datetime.datetime(2026, 1, 1)
    for feature in (ERC4626Feature.lagoon_like, ERC4626Feature.d2_like, ERC4626Feature.t3tris_like):
        pair = TradingPairIdentifier(base, quote, base.address, base.address, fee=0,
            kind=TradingPairKind.vault, other_data={"vault_features": [feature]})
        # 2. Restore old rules with a missing or false event flag.
        snapshot = asdict(resolve_vault_protocol_rules(pair))
        for flag in (None, False):
            snapshot["deposit_requires_settlement_event"] = flag
            if flag is None:
                snapshot.pop("deposit_requires_settlement_event")
            pair.other_data["vault_protocol_rules"] = snapshot.copy()
            restored = TradingPairIdentifier.from_json(pair.to_json())
            # 3. Check the current gate, rather than the saved flag, selects events.
            assert not restored.get_vault_protocol_rules().deposit_requires_settlement_event
            assert execution._requires_historical_deposit_settlement(restored)
        if feature == ERC4626Feature.t3tris_like:
            assert pair.is_async_vault() and pair.has_delayed_vault_redemption()
            assert _has_backtest_settlement_logic(pair)
            assert execution._get_settlement_due(pair, timestamp) == timestamp + datetime.timedelta(days=2)
            pair.other_data["vault_estimated_settlement"] = datetime.timedelta(days=5)
            for is_buy in (True, False):
                assert execution._is_async_vault(pair, is_buy=is_buy)
                assert execution._get_settlement_due(pair, timestamp, is_buy=is_buy) == timestamp + datetime.timedelta(days=5)
