"""Directional timing for vaults whose withdrawal interface is asynchronous."""

import datetime

from eth_defi.erc_4626.core import ERC4626Feature

from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.strategy.chart.standard.vault import _get_no_data_settlement_delay, _has_backtest_settlement_logic


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
    pair.other_data["vault_estimated_settlement"] = datetime.timedelta(days=7)
    timestamp = datetime.datetime(2026, 1, 1)
    assert not execution._is_async_vault(pair, is_buy=True)
    assert execution._get_settlement_due(pair, timestamp) == timestamp + datetime.timedelta(days=7)
    pair.other_data["vault_estimated_settlement"] = 0
    assert execution._get_settlement_due(pair, timestamp) == timestamp + datetime.timedelta(days=2)
