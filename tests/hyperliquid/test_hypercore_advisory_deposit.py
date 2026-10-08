"""Account for submitted deposits without waiting for market equity to recover.

The protocol boundaries are mocked because these tests must never transfer
funds. Accounting, sequential execution, freezing and JSON persistence remain
real so a warning cannot conceal a reserve refund or stop the next trade.
"""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
import requests
from eth_defi.hyperliquid.api import HypercoreDepositVerificationError, UserVaultEquity
from hexbytes import HexBytes

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.vault.hypercore_routing import HypercoreVaultRouting
from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS, create_hypercore_vault_pair
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.trade import TradeExecution, TradeStatus


@pytest.mark.parametrize("observation", ["falling_equity", "missing", "request_error", "malformed", "confirmed", "settlement_error"])
def test_advisory_deposit_preserves_accounting_and_sequential_progress(observation: str, caplog: pytest.LogCaptureFixture) -> None:
    """Continue the basket with the submitted fill and a durable warning.

    1. Prepare a real existing holding, two top-ups and mocked protocol receipts.
    2. Execute them sequentially with final observation failures or normal confirmation.
    3. Check fills, cash, capital cleanup, warning persistence and settlement error propagation.
    """
    # 1. Seed an existing holding so the observation can reproduce #1765's
    # market movement independently of the newly submitted capital.
    ts = datetime.datetime(2026, 10, 8)
    usdc = AssetIdentifier(999, "0x0000000000000000000000000000000000000002", "USDC", 6)
    pair = create_hypercore_vault_pair(quote=usdc, vault_address=HLP_VAULT_ADDRESS["mainnet"], internal_id=1)
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    reserve.quantity = Decimal(500)
    position, opening, _ = state.create_trade(ts, pair, quantity=None, reserve=Decimal("11144.875437"), assumed_price=1.0, trade_type=TradeType.rebalance, reserve_currency=usdc, reserve_currency_price=1.0)
    opening.mark_success(ts, 1.0, Decimal("11144.875437"), Decimal("11144.875437"), 0, 0, force=True)
    trades: list[TradeExecution] = []
    for amount in [Decimal("28.175784"), Decimal("10")]:
        _, trade, _ = state.create_trade(ts, pair, quantity=None, reserve=amount, assumed_price=1.0, trade_type=TradeType.rebalance, reserve_currency=usdc, reserve_currency_price=1.0)
        trades.append(trade)

    routing = object.__new__(HypercoreVaultRouting)
    routing.web3 = MagicMock()
    routing.deployer = MagicMock()
    routing.lagoon_vault = MagicMock()
    routing.lagoon_vault.safe_address = "0xSAFE"
    routing.chain_id = 999
    routing.simulate = False
    routing._session = MagicMock()
    phase2 = BlockchainTransaction(tx_hash="0xbbb", status=True)
    phase3 = BlockchainTransaction(tx_hash="0xccc", status=True)
    baseline = UserVaultEquity(pair.pool_address, Decimal("11144.875437"), datetime.datetime(2030, 1, 1))
    confirmed = UserVaultEquity(pair.pool_address, Decimal("11173.051221"), datetime.datetime(2030, 1, 1))
    outcome = {
        "falling_equity": HypercoreDepositVerificationError("could not be verified within 60.0s; existing equity: 11144.875437; last queried equity: 11146.946569"),
        "missing": None,
        "request_error": requests.Timeout("response exceeded the settlement deadline"),
        "malformed": ValueError("invalid equity response"),
        "confirmed": confirmed,
        "settlement_error": requests.Timeout("response exceeded the settlement deadline"),
    }[observation]
    verification = MagicMock(side_effect=[outcome, confirmed])

    # 2. Real sequential execution controls reserve allocation and freezing;
    # only signing/broadcast/provider reads are replaced by protocol fixtures.
    execution = MagicMock(spec=EthereumExecution)
    execution.max_slippage = None
    router = MagicMock()
    router.check_trade_before_execution.return_value = None

    def prepare(**kwargs: object) -> None:
        """Model the existing durable bridge marker before any broadcast."""
        trade = kwargs["trades"][0]
        trade.blockchain_transactions = [BlockchainTransaction(tx_hash="0xaaa", status=True)]
        trade.other_data = {
            "hypercore_phase1_spot_baseline_usdc": "0.008129",
            "hypercore_deposit_capital_at_risk": {"phase": "phase1_broadcast_pending"},
            "retain_reserve_allocation_on_failure": True,
        }

    def settle(router: MagicMock, state: State, batch: list[TradeExecution], **kwargs: object) -> None:
        """Replace the chain broadcast while retaining production accounting."""
        trade = batch[0]
        state.mark_broadcasted(ts, trade)
        routing._settle_deposit(routing.web3, state, trade, {HexBytes("0xaaa"): {"status": 1, "blockNumber": 100}}, True)

    router.setup_trades.side_effect = prepare
    execution._execute_trade_batch.side_effect = settle
    with (
        patch("tradeexecutor.ethereum.vault.hypercore_routing.get_block_timestamp", return_value=ts),
        patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear"),
        patch("tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity", return_value=baseline),
        patch("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation", verification),
        patch.object(routing, "_fetch_safe_spot_free_usdc_balance", return_value=Decimal("28.183913")),
        patch.object(routing, "_fetch_safe_perp_withdrawable_balance", return_value=Decimal("3.498677")),
        patch.object(routing, "_wait_for_deposit_spot_to_perp_transfer"),
        patch.object(routing, "_broadcast_deposit_spot_to_perp", return_value=(phase2, {"status": 1, "blockNumber": 101})),
        patch.object(routing, "_broadcast_deposit_perp_to_vault", return_value=(phase3, {"status": 1, "blockNumber": 102})),
        patch.object(routing, "_fetch_hype_usd_price", return_value=30.0) as price_read,
    ):
        if observation == "settlement_error":
            # A read-only failure may be accepted; an accounting failure may
            # not. This guard prevents a later refactor widening the catch.
            with patch.object(State, "mark_trade_success", side_effect=RuntimeError("accounting error")), pytest.raises(RuntimeError, match="accounting error"):
                EthereumExecution._execute_trades_sequentially(execution, ts, state, trades, router, MagicMock(), False, False, False)
            return
        EthereumExecution._execute_trades_sequentially(execution, ts, state, trades, router, MagicMock(), False, False, False)

    # 3. The unverified top-up is spent once, never refunded as failed-buy
    # cash. Its warning and exact principal survive the authoritative JSON.
    assert all(trade.get_status() == TradeStatus.success for trade in trades)
    assert execution._execute_trade_batch.call_count == 2
    assert not state.portfolio.frozen_positions
    assert reserve.quantity == Decimal("461.824216")
    assert trades[0].executed_quantity == Decimal("28.175784")
    assert trades[0].executed_reserve == Decimal("28.175784")
    assert trades[0].executed_price == 1.0
    assert verification.call_args_list[0].kwargs["timeout"] == 60.0
    for trade in trades:
        assert trade.reserve_currency_allocated == 0
        assert "hypercore_deposit_capital_at_risk" not in trade.other_data
        assert "retain_reserve_allocation_on_failure" not in trade.other_data
        assert "hypercore_stranded_usdc" not in trade.other_data
    restored = State.from_json(state.to_json())
    saved = restored.portfolio.open_positions[position.position_id].trades[trades[0].trade_id]
    assert saved.get_status() == TradeStatus.success
    assert saved.executed_quantity == Decimal("28.175784")
    warnings = [record.message for record in caplog.records if record.message.startswith("UNVERIFIED HyperCore deposit")]
    if observation == "confirmed":
        assert not warnings
        assert "UNVERIFIED" not in (saved.notes or "")
        assert price_read.call_count == 2
    else:
        assert len(warnings) == 1
        assert "28.175784" in warnings[0]
        assert "0xSAFE" in warnings[0] and "0xccc" in warnings[0]
        assert warnings[0] in saved.notes
        assert price_read.call_count == 1
