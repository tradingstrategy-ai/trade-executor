"""Resume a paid Morpho recovery through the existing executor State lifecycle."""

import datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

from hexbytes import HexBytes
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.morpho.recovery import MorphoRecoveryAnalysis, MorphoRecoveryTicket
from eth_defi.erc_4626.vault_protocol.morpho.vault_v2 import MorphoV2Vault
from eth_defi.vault.base import VaultSpec

from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus, TradeType


def test_morpho_recovery_state_resume_accounts_fee_once() -> None:
    """Preserve the full disposal and actual net cash after a process restart.

    1. Fill a deposit and persist a pending exit whose first transaction paid a fee.
    2. Reload State JSON and reconstruct its real manager ticket.
    3. Mock only chain execution/receipt analysis, covered by the real fork tests,
       and run the shared settlement resolver.
    4. Verify the original quantity closes, net cash is credited once and another
       sweep neither signs another transaction nor applies another fee.
    """
    # 1. The accounting position still includes shares burned by the paid request.
    address = Web3.to_checksum_address("0x36cfe1568461e499391ef0a555300f1ae2da2439")
    owner = "0x" + "44" * 20
    asset = AssetIdentifier(1, "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "USDC", 6)
    pair = TradingPairIdentifier(AssetIdentifier(1, address, "vault", 18), asset, address, address, fee=0, kind=TradingPairKind.vault)
    state = State()
    ts = datetime.datetime(2026, 11, 15)
    state.portfolio.initialise_reserves(asset)
    reserve = state.portfolio.get_default_reserve_position()
    reserve.quantity = Decimal(10_000)
    reserve.reserve_token_price = 1.0
    _, buy, _ = state.create_trade(ts, pair, Decimal(1000), None, 1.0, TradeType.rebalance, asset, 1.0)
    state.start_execution(ts, buy)
    buy.mark_broadcasted(ts)
    state.mark_trade_success(ts, buy, 1.0, Decimal(1000), Decimal(1000), 0, 0)
    _, sell, _ = state.create_trade(ts, pair, Decimal(-1000), None, 1.0, TradeType.rebalance, asset, 1.0)
    state.start_execution(ts, sell)
    sell.mark_broadcasted(ts)
    request = BlockchainTransaction()
    request.tx_hash = "0x" + "11" * 32
    sell.blockchain_transactions = [request]
    ticket = MorphoRecoveryTicket(address, owner, owner, 1000 * 10**18, HexBytes(request.tx_hash), 10**17, 100_000)
    manager = MorphoV2Vault(Web3(), VaultSpec(1, address)).get_deposit_manager()
    state.mark_vault_settlement_pending(ts, sell, {**manager.serialize_redemption_ticket(ticket), "vault_chain_id": 1, "vault_direction": "redeem", "vault_initial_tx_count": 1})
    assert reserve.quantity == 9000

    # 2. Use the actual State and manager serializers, not a parallel checkpoint.
    state = State.from_json(state.to_json())
    sell = state.portfolio.get_trade_by_id(sell.trade_id)
    assert manager.reconstruct_redemption_ticket(sell.other_data) == ticket
    assert sell.get_status() is TradeStatus.vault_settlement_pending
    pending_position = state.portfolio.find_position_for_trade(sell)
    assert list(pending_position.get_held_assets())[0][1] == Decimal("999.9")
    assert pending_position.get_available_trading_quantity() == 0

    # 3. Chain behaviour is proven separately; isolate executor cash/share accounting.
    claim = BlockchainTransaction()
    claim.tx_hash = "0x" + "22" * 32
    claim.signed_bytes = b"\x22" * 32
    claim.other = {}
    receipt = {"status": 1, "transactionHash": HexBytes(claim.tx_hash), "blockNumber": 100, "blockHash": HexBytes("0x" + "33" * 32), "gasUsed": 21000}
    execution = MagicMock()
    execution.web3.eth.chain_id = 1
    execution.web3.block_tip_latency = 0
    execution.tx_builder.sign_transaction.return_value = claim
    vault = MagicMock()
    vault.get_deposit_manager.return_value = manager
    analysis = MorphoRecoveryAnalysis(owner, owner, Decimal("1000.9"), Decimal(1000), HexBytes(claim.tx_hash), 100, ts, Decimal("0.1"))
    with (
        patch("tradeexecutor.ethereum.vault.settlement_retry.get_vault_for_pair", return_value=vault),
        patch("tradeexecutor.ethereum.vault.settlement_retry.wait_for_transaction_receipt_robust", return_value=receipt),
        patch.object(manager, "finish_redemption", return_value=MagicMock()) as finish,
        patch.object(manager, "analyse_redemption", return_value=analysis),
    ):
        assert check_and_resolve_vault_settlements(state, execution) == [sell]
        # 4. Paid fee shares plus final shares consume the original position once.
        assert sell.executed_quantity == -1000
        assert sell.executed_reserve == Decimal("1000.9")
        assert sell.other_data["vault_withdrawal_fee"] == "0.1"
        assert state.portfolio.get_default_reserve_position().quantity == Decimal("10000.9")
        assert state.portfolio.find_position_for_trade(sell).get_quantity() == 0
        assert check_and_resolve_vault_settlements(state, execution) == []
        finish.assert_called_once()
        execution.tx_builder.sign_transaction.assert_called_once()
        assert len(sell.blockchain_transactions) == 2
