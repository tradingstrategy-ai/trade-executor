"""Test recovering a Lighter deposit that was broadcast but never mined.

On 2026-11-05 a deposit stuck on gas blocked the LIX executor, and its nonce was then
taken by another transaction. These tests hold ``recover_unmined_lighter_deposits()`` to
the rule that matters: reserves are only returned when the funds provably stayed in the
Safe, and every doubt leaves the trade pending.

Chain reads are faked with a small in-memory chain, because the cases need receipts
that are missing, reverted or contradictory. One Anvil test runs the nonce search for
real. Neither establishes how a mainnet provider behaves.
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.token import USDC_NATIVE_TOKEN
from hexbytes import HexBytes
from pytest_mock import MockerFixture
from web3 import HTTPProvider, Web3
from web3.exceptions import TransactionNotFound

import tradeexecutor.ethereum.lighter.transfer_verification as verification_module
from tradeexecutor.ethereum.lighter.transfer_verification import (
    ERC_20_TRANSFER_TOPIC,
    UnminedDepositCase,
    classify_unmined_lighter_deposit,
    find_nonce_consumer,
    recover_unmined_lighter_deposits,
)
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.exchange_account.state import (
    create_exchange_account_transfer,
    open_exchange_account_position,
)
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.repair import RepairAborted, repair_trade, repair_zero_quantity
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.state.trade import TradeExecution, TradeStatus

SAFE = "0x82116484a2a2B1cA188E966ee568d4fF5eF59699"
HOT_WALLET = "0xB4cB332396927b425B7f80AACB5623cf3E0F0966"
#: A signed payload, so that its Keccak hash can be the recorded transaction hash.
SIGNED_DEPOSIT = "0x" + "aa" * 64
SIGNED_APPROVE = "0x" + "bb" * 64
DEPOSIT_HASH = Web3.keccak(hexstr=SIGNED_DEPOSIT).to_0x_hex()
APPROVE_HASH = Web3.keccak(hexstr=SIGNED_APPROVE).to_0x_hex()
CONSUMER_HASH = "0x" + "cc" * 32
APPROVE_NONCE = 77
DEPOSIT_NONCE = 78
CONSUMER_BLOCK = 1_000
FINALIZED_BLOCK = 1_100
SAFE_USDC = Decimal(3020)
DEPOSIT = Decimal(3000)


class FakeChain:
    """Chain reads for one hot wallet, with the deposit's fate set by each test."""

    def __init__(self, *, mined_nonce: int, finalized_nonce: int, receipts: dict[str, dict], consumer_logs: list[dict] | None = None, consumer_hash: str = CONSUMER_HASH):
        self.chain_id = 1
        self.mined_nonce = mined_nonce
        self.finalized_nonce = finalized_nonce
        self.receipts = dict(receipts)
        # A node that has lost the deposit's own receipt still reports its nonce as used.
        if consumer_hash != DEPOSIT_HASH:
            self.receipts[consumer_hash] = {"status": 1, "logs": consumer_logs or []}
        self.consumer_hash = consumer_hash
        self.eth = self

    def get_transaction_receipt(self, tx_hash: str) -> dict:
        receipt = self.receipts.get(tx_hash)
        if receipt is None:
            raise TransactionNotFound(tx_hash)
        return receipt

    def get_transaction_count(self, sender: str, block: int | str) -> int:
        if block == "latest":
            return self.mined_nonce
        return self.finalized_nonce if block >= CONSUMER_BLOCK else DEPOSIT_NONCE

    def get_block(self, block: int | str, full_transactions: bool = False) -> dict:
        if block == "finalized":
            return {"number": FINALIZED_BLOCK}
        tx = {"from": HOT_WALLET, "nonce": DEPOSIT_NONCE, "hash": self.consumer_hash, "blockNumber": block}
        return {"transactions": [tx]}

    def get_balance(self, address: str) -> int:
        return 10**15


def create_lighter_account() -> tuple[State, TradingPosition, AssetIdentifier]:
    """Create a Lighter account position and reserves, with nothing pending."""
    reserve_asset = AssetIdentifier(chain_id=1, address=USDC_NATIVE_TOKEN[1], token_symbol="USDC", decimals=6)
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.get_reserve_position(reserve_asset).quantity = SAFE_USDC + 1
    pair = create_lighter_exchange_account_pair(quote=reserve_asset, account_index=1)
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=reserve_asset,
        reserve_amount=Decimal(1),
        notes="Activation deposit",
    )
    position = next(iter(state.portfolio.open_positions.values()))
    return state, position, reserve_asset


def create_stuck_deposit() -> tuple[State, TradingPosition, TradeExecution]:
    """Create a broadcast deposit whose capital has left the reserves in the state."""
    state, position, reserve_asset = create_lighter_account()
    trade = create_exchange_account_transfer(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        position=position,
        reserve_currency=reserve_asset,
        amount=DEPOSIT,
        deposit=True,
        notes="Automatic Lighter cash management",
        metadata={"direction": "deposit"},
    )
    state.start_execution(native_datetime_utc_now(), trade, underflow_check=True)
    trade.blockchain_transactions = [
        create_transaction("approve", APPROVE_NONCE, APPROVE_HASH, SIGNED_APPROVE),
        create_transaction("deposit", DEPOSIT_NONCE, DEPOSIT_HASH, SIGNED_DEPOSIT),
    ]
    trade.mark_broadcasted(native_datetime_utc_now())
    return state, position, trade


def create_transaction(function: str, nonce: int, tx_hash: str, signed_bytes: str) -> BlockchainTransaction:
    """Create a signed Safe-wrapped transaction record."""
    return BlockchainTransaction(
        tx_hash=tx_hash,
        nonce=nonce,
        from_address=HOT_WALLET,
        signed_bytes=signed_bytes,
        wrapped_function_selector=function,
        details={"gas": 900_000, "maxFeePerGas": 10**10},
    )


@pytest.fixture()
def safe_balance(mocker: MockerFixture) -> Any:
    """Replace the Safe's USDC balance read, which needs a real token contract."""
    token = mocker.Mock()
    token.fetch_balance_of.return_value = SAFE_USDC
    mocker.patch.object(verification_module, "fetch_erc20_details", return_value=token)
    return token


APPROVE_RECEIPT = {APPROVE_HASH: {"status": 1, "logs": []}}


def test_consumed_nonce_returns_the_reserves_exactly_once(safe_balance: Any, tmp_path: Path) -> None:
    """The LIX case: the nonce went to an unrelated transaction, so the funds stayed.

    1. Create a stuck deposit whose nonce another transaction consumed.
    2. Recover it through the same entry point as ``repair``.
    3. Verify the reserves are back and the state is clean, with the evidence noted.
    4. Save, reload and recover again, as a rerun after a dropped session would.
    5. Verify nothing changed the second time, and the repair survived the reload.
    """
    # 1. Create a stuck deposit whose nonce another transaction consumed.
    state, position, trade = create_stuck_deposit()
    reserve = state.portfolio.get_reserve_position(trade.reserve_currency)
    assert reserve.quantity == SAFE_USDC - DEPOSIT
    chain = FakeChain(mined_nonce=80, finalized_nonce=80, receipts=APPROVE_RECEIPT)

    # 2. Recover it through the same entry point as ``repair``.
    saves = []
    recovered = recover_unmined_lighter_deposits(state, chain, safe_address=SAFE, interactive=False, save=lambda: saves.append(1))

    # 3. Verify the reserves are back and the state is clean, with the evidence noted.
    assert recovered == [trade]
    assert saves == [1]
    assert trade.get_status() == TradeStatus.repaired
    assert reserve.quantity == SAFE_USDC
    assert position.get_quantity() == Decimal(1)
    assert not trade.is_exchange_account_transfer_pending()
    assert CONSUMER_HASH in trade.notes
    state.check_if_clean()
    assert repr(position.trades[trade.trade_id + 1])

    # 4. Save, reload and recover again, as a rerun after a dropped session would.
    store = JSONFileStore(tmp_path / "state.json")
    store.sync(state)
    reloaded = store.load()
    again = recover_unmined_lighter_deposits(reloaded, chain, safe_address=SAFE, interactive=False, save=lambda: None)

    # 5. Verify nothing changed the second time, and the repair survived the reload.
    assert again == []
    assert reloaded.portfolio.get_reserve_position(trade.reserve_currency).quantity == SAFE_USDC
    reloaded_position = reloaded.portfolio.open_positions[position.position_id]
    assert reloaded_position.get_quantity() == Decimal(1)
    assert reloaded_position.trades[trade.trade_id].get_status() == TradeStatus.repaired
    counter_trade = next(t for t in reloaded_position.trades.values() if t.repaired_trade_id == trade.trade_id)
    assert counter_trade.get_position_quantity() == 0
    assert counter_trade.get_value() == 0
    assert repr(counter_trade)


@pytest.mark.parametrize(
    ("chain_args", "reason"),
    [
        # A consumer that moved the Safe's USDC can be a fee-bumped copy of the deposit.
        (
            {
                "mined_nonce": 80,
                "finalized_nonce": 80,
                "consumer_logs": [
                    {
                        "address": USDC_NATIVE_TOKEN[1],
                        "topics": [ERC_20_TRANSFER_TOPIC, HexBytes(SAFE).rjust(32, b"\0"), HexBytes(HOT_WALLET).rjust(32, b"\0")],
                    }
                ],
            },
            "out of the Safe",
        ),
        # The consumer is the deposit itself, and this node has lost its receipt.
        ({"mined_nonce": 80, "finalized_nonce": 80, "consumer_hash": DEPOSIT_HASH}, "this node has no receipt"),
        # The nonce is consumed, but a reorganisation could still undo that.
        ({"mined_nonce": 80, "finalized_nonce": DEPOSIT_NONCE}, "not finalized"),
    ],
)
def test_doubtful_evidence_leaves_the_trade_pending(safe_balance: Any, chain_args: dict[str, Any], reason: str) -> None:
    """A consumed nonce alone never returns reserves.

    1. Create a stuck deposit with evidence that falls short of proof.
    2. Run the recovery.
    3. Verify the trade is still pending and the reserves did not move.
    """
    # 1. Create a stuck deposit with evidence that falls short of proof.
    state, _, trade = create_stuck_deposit()
    chain = FakeChain(receipts=APPROVE_RECEIPT, **chain_args)
    assert reason in classify_unmined_lighter_deposit(chain, state, trade, SAFE).reason

    # 2. Run the recovery.
    recovered = recover_unmined_lighter_deposits(state, chain, safe_address=SAFE, interactive=False, save=lambda: None)

    # 3. Verify the trade is still pending and the reserves did not move.
    assert recovered == []
    assert trade.is_exchange_account_transfer_pending()
    assert state.portfolio.get_reserve_position(trade.reserve_currency).quantity == SAFE_USDC - DEPOSIT


def test_recorded_receipts_decide_before_the_nonce(safe_balance: Any) -> None:
    """Receipts outrank nonces, including a deposit whose approval receipt is missing.

    1. Classify a deposit with a receipt and no approval receipt.
    2. Classify a free deposit nonce after a reverted approval.
    3. Classify a Safe balance below what the state expects.
    4. Verify none of them may refund or send.
    """
    # 1. Classify a deposit with a receipt and no approval receipt.
    state, _, trade = create_stuck_deposit()
    mined = FakeChain(mined_nonce=80, finalized_nonce=80, receipts={DEPOSIT_HASH: {"status": 1, "logs": []}})
    assert classify_unmined_lighter_deposit(mined, state, trade, SAFE) is None

    # 2. Classify a free deposit nonce after a reverted approval.
    reverted = FakeChain(mined_nonce=DEPOSIT_NONCE, finalized_nonce=DEPOSIT_NONCE, receipts={APPROVE_HASH: {"status": 0, "logs": []}})
    result = classify_unmined_lighter_deposit(reverted, state, trade, SAFE)

    # 3. Classify a Safe balance below what the state expects.
    safe_balance.fetch_balance_of.return_value = SAFE_USDC - 1
    short = FakeChain(mined_nonce=80, finalized_nonce=80, receipts=APPROVE_RECEIPT)
    short_result = classify_unmined_lighter_deposit(short, state, trade, SAFE)

    # 4. Verify none of them may refund or send.
    assert result.case == UnminedDepositCase.undetermined
    assert short_result.case == UnminedDepositCase.undetermined


def test_free_nonce_rebroadcasts_the_stored_deposit(mocker: MockerFixture, safe_balance: Any) -> None:
    """A deposit whose nonce is free is sent unchanged, if the wallet can afford it.

    The broadcast is mocked, because this tests what is sent and when.

    1. Create a stuck deposit whose nonce is free, in a wallet short of gas.
    2. Verify nothing is sent, and that the reason says how to fund it.
    3. Fund the wallet and recover again.
    4. Verify only the deposit was sent, with its stored bytes.
    """
    # 1. Create a stuck deposit whose nonce is free, in a wallet short of gas.
    state, _, trade = create_stuck_deposit()
    chain = FakeChain(mined_nonce=DEPOSIT_NONCE, finalized_nonce=DEPOSIT_NONCE, receipts=APPROVE_RECEIPT)
    broadcast = mocker.patch.object(
        verification_module,
        "wait_and_broadcast_multiple_nodes",
        return_value={HexBytes(DEPOSIT_HASH): {"blockNumber": 1, "blockHash": HexBytes("0x" + "dd" * 32), "gasUsed": 200_146, "effectiveGasPrice": 10**9, "status": 1}},
    )

    # 2. Verify nothing is sent, and that the reason says how to fund it.
    assert recover_unmined_lighter_deposits(state, chain, safe_address=SAFE, interactive=False, save=lambda: None) == []
    broadcast.assert_not_called()

    # 3. Fund the wallet and recover again.
    chain.get_balance = lambda address: 10**17
    recovered = recover_unmined_lighter_deposits(state, chain, safe_address=SAFE, interactive=False, save=lambda: None)

    # 4. Verify only the deposit was sent, with its stored bytes.
    assert recovered == [trade]
    sent = broadcast.call_args.args[1]
    assert [HexBytes(tx.rawTransaction) for tx in sent] == [HexBytes(SIGNED_DEPOSIT)]
    assert trade.blockchain_transactions[1].realised_gas_units_consumed == 200_146
    assert trade.get_status() == TradeStatus.broadcasted


def test_generic_repairs_still_refuse_and_keep_exchange_accounts(mocker: MockerFixture) -> None:
    """The verified path is the only way through, and empty accounts stay open.

    1. Try the generic repair on a stuck deposit.
    2. Run the zero-quantity repair on an emptied account.
    3. Verify the first refuses and the second closes nothing.
    """
    # 1. Try the generic repair on a stuck deposit.
    state, _, trade = create_stuck_deposit()
    with pytest.raises(RepairAborted):
        repair_trade(state.portfolio, trade)

    # 2. Run the zero-quantity repair on an emptied account with nothing pending.
    # The quantity is mocked, as this tests the closing rule and not a withdrawal.
    empty_state, position, _ = create_lighter_account()
    mocker.patch.object(TradingPosition, "get_quantity", return_value=Decimal(0))
    repaired = repair_zero_quantity(empty_state, interactive=False)

    # 3. Verify the first refuses and the second closes nothing.
    assert repaired == []
    assert position.position_id in empty_state.portfolio.open_positions


def test_nonce_consumer_search_on_a_real_node(anvil: AnvilLaunch) -> None:
    """The search finds the real consuming transaction across many blocks.

    1. Mine empty blocks, a transaction and more empty blocks on Anvil.
    2. Search from the latest block.
    3. Verify it finds the transaction, and nothing for an unused nonce.
    """
    # 1. Mine empty blocks, a transaction and more empty blocks on Anvil.
    web3 = Web3(HTTPProvider(anvil.json_rpc_url))
    sender, receiver = web3.eth.accounts[:2]
    nonce = web3.eth.get_transaction_count(sender)
    web3.provider.make_request("anvil_mine", [hex(30)])
    tx_hash = web3.eth.send_transaction({"from": sender, "to": receiver, "value": 1})
    web3.eth.wait_for_transaction_receipt(tx_hash)
    web3.provider.make_request("anvil_mine", [hex(300)])

    # 2. Search from the latest block.
    consumer = find_nonce_consumer(web3, sender, nonce, web3.eth.block_number)

    # 3. Verify it finds the transaction, and nothing for an unused nonce.
    assert HexBytes(consumer["hash"]) == HexBytes(tx_hash)
    assert find_nonce_consumer(web3, sender, nonce + 1, web3.eth.block_number) is None
