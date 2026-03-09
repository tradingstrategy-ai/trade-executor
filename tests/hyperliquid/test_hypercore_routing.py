"""Integration tests for Hypercore vault BlockchainTransaction and TradeExecution flow.

Tests the BlockchainTransaction <-> TradeExecution mapping using a plain EOA wallet
on HyperEVM testnet, without requiring a Lagoon vault deployment.

Environment variables:
    HYPERCORE_WRITER_TEST_PRIVATE_KEY: EOA private key with HYPE gas and USDC on testnet
"""

import datetime
import os
from decimal import Decimal

import pytest
from eth_account import Account
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.api import fetch_spot_clearinghouse_state
from eth_defi.hyperliquid.core_writer import (
    CORE_DEPOSIT_WALLET,
    SPOT_DEX,
    get_core_deposit_wallet_contract,
)
from eth_defi.hyperliquid.evm_escrow import (
    is_account_activated,
    wait_for_evm_escrow_clear,
)
from eth_defi.hyperliquid.session import (
    HYPERLIQUID_TESTNET_API_URL,
    create_hyperliquid_session,
)
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details

from tradeexecutor.ethereum.vault.hypercore_vault import (
    HLP_VAULT_ADDRESS,
    create_hypercore_vault_pair,
)
from tradeexecutor.state.blockhain_transaction import (
    BlockchainTransaction,
    BlockchainTransactionType,
)
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.pickle_over_json import encode_pickle_over_json
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus, TradeType
from tradeexecutor.utils.hex import hexbytes_to_hex_str

#: Skip all tests if no private key is set
pytestmark = pytest.mark.skipif(
    not os.environ.get("HYPERCORE_WRITER_TEST_PRIVATE_KEY"),
    reason="HYPERCORE_WRITER_TEST_PRIVATE_KEY not set",
)

CHAIN_ID = 998
USDC_ADDRESS = USDC_NATIVE_TOKEN[CHAIN_ID]
CDW_ADDRESS = CORE_DEPOSIT_WALLET[CHAIN_ID]


@pytest.fixture
def web3():
    """HyperEVM testnet Web3 connection."""
    return create_multi_provider_web3("https://rpc.hyperliquid-testnet.xyz/evm")


@pytest.fixture
def hot_wallet(web3):
    """EOA wallet from test private key."""
    key = os.environ["HYPERCORE_WRITER_TEST_PRIVATE_KEY"]
    wallet = HotWallet(Account.from_key(key))
    wallet.sync_nonce(web3)
    return wallet


@pytest.fixture
def session():
    """Hyperliquid testnet API session."""
    return create_hyperliquid_session(api_url=HYPERLIQUID_TESTNET_API_URL)


@pytest.fixture
def usdc(web3):
    """USDC token details on testnet."""
    return fetch_erc20_details(web3, USDC_ADDRESS)


@pytest.fixture
def usdc_asset():
    """USDC AssetIdentifier for testnet."""
    return AssetIdentifier(
        chain_id=CHAIN_ID,
        address=USDC_ADDRESS,
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture
def hypercore_vault_pair(usdc_asset):
    """Hypercore vault TradingPairIdentifier for HLP testnet vault."""
    return create_hypercore_vault_pair(
        quote=usdc_asset,
        vault_address=HLP_VAULT_ADDRESS["testnet"],
        is_testnet=True,
    )


def _sign_and_create_blockchain_tx(
    web3: Web3,
    hot_wallet: HotWallet,
    fn,
    function_name: str,
    notes: str = "",
) -> BlockchainTransaction:
    """Helper: sign a ContractFunction and create a BlockchainTransaction."""
    chain_id = web3.eth.chain_id
    tx_data = fn.build_transaction({
        "chainId": chain_id,
        "from": hot_wallet.address,
        "gas": 200_000,
        "gasPrice": web3.eth.gas_price,
    })
    signed = hot_wallet.sign_transaction_with_new_nonce(tx_data)
    tx_data["function"] = function_name
    return BlockchainTransaction(
        type=BlockchainTransactionType.hot_wallet,
        chain_id=chain_id,
        from_address=hot_wallet.address,
        contract_address=fn.address,
        function_selector=function_name,
        transaction_args=None,
        args=None,
        wrapped_args=None,
        signed_bytes=hexbytes_to_hex_str(signed.rawTransaction),
        signed_tx_object=encode_pickle_over_json(signed),
        tx_hash=hexbytes_to_hex_str(signed.hash),
        nonce=signed.nonce,
        details=tx_data,
        asset_deltas=[],
        notes=notes,
    )


def test_eoa_blockchain_transaction_creation(web3, hot_wallet, usdc):
    """Test that signing EOA approve + CDW.deposit produces valid BlockchainTransaction objects."""
    cdw = get_core_deposit_wallet_contract(web3, CDW_ADDRESS)
    raw_amount = 1_000_000  # 1 USDC

    # Build approve
    approve_fn = usdc.contract.functions.approve(
        Web3.to_checksum_address(CDW_ADDRESS),
        raw_amount,
    )
    approve_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, approve_fn, "approve", "USDC approve for CDW",
    )

    assert approve_tx.chain_id == CHAIN_ID
    assert approve_tx.tx_hash is not None
    assert approve_tx.signed_bytes is not None
    assert approve_tx.nonce >= 0
    assert approve_tx.function_selector == "approve"
    assert approve_tx.from_address == hot_wallet.address

    # Build CDW.deposit
    deposit_fn = cdw.functions.deposit(raw_amount, SPOT_DEX)
    deposit_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, deposit_fn, "deposit", "CDW deposit",
    )

    assert deposit_tx.chain_id == CHAIN_ID
    assert deposit_tx.tx_hash is not None
    assert deposit_tx.signed_bytes is not None
    assert deposit_tx.nonce == approve_tx.nonce + 1
    assert deposit_tx.function_selector == "deposit"


def test_trade_execution_state_transitions(web3, hot_wallet, usdc, usdc_asset, hypercore_vault_pair):
    """Test the full TradeExecution state machine with BlockchainTransaction objects."""
    ts = datetime.datetime(2026, 3, 9, tzinfo=None)
    state = State()
    state.portfolio.initialise_reserves(usdc_asset, reserve_token_price=1.0)

    # Create a vault deposit trade
    position, trade, created = state.create_trade(
        strategy_cycle_at=ts,
        pair=hypercore_vault_pair,
        quantity=None,
        reserve=Decimal("5"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_asset,
        reserve_currency_price=1.0,
        notes="Test Hypercore vault deposit",
        pair_fee=0.0,
        lp_fees_estimated=0,
    )

    assert trade.get_status() == TradeStatus.planned
    assert trade.is_vault()

    # Start execution
    state.start_execution_all(ts, [trade], max_slippage=0.01)
    assert trade.get_status() == TradeStatus.started

    # Create a dummy blockchain transaction
    cdw = get_core_deposit_wallet_contract(web3, CDW_ADDRESS)
    deposit_fn = cdw.functions.deposit(5_000_000, SPOT_DEX)
    blockchain_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, deposit_fn, "deposit", "Test deposit",
    )
    trade.blockchain_transactions = [blockchain_tx]

    # Mark broadcasted
    state.mark_broadcasted(ts, trade)
    assert trade.get_status() == TradeStatus.broadcasted

    # Mark success
    state.mark_trade_success(
        ts,
        trade,
        executed_price=1.0,
        executed_amount=Decimal("5"),
        executed_reserve=Decimal("5"),
        lp_fees=0,
        native_token_price=0,
    )
    assert trade.get_status() == TradeStatus.success
    assert len(trade.blockchain_transactions) == 1
    assert trade.executed_reserve == pytest.approx(Decimal("5"))


def test_eoa_deposit_and_escrow_clear(web3, hot_wallet, usdc, session):
    """Test actual EOA deposit on HyperEVM testnet with escrow wait."""
    # Pre-checks
    activated = is_account_activated(web3, hot_wallet.address)
    if not activated:
        pytest.skip("Account not activated on HyperCore — run test-hypercore-escrow.py first")

    usdc_balance = usdc.contract.functions.balanceOf(hot_wallet.address).call()
    usdc_human = Decimal(usdc_balance) / Decimal(10**6)
    if usdc_human < 3:
        pytest.skip(f"Insufficient USDC balance: {usdc_human} (need 3)")

    raw_amount = 3_000_000  # 3 USDC
    cdw = get_core_deposit_wallet_contract(web3, CDW_ADDRESS)

    # Approve
    approve_fn = usdc.contract.functions.approve(
        Web3.to_checksum_address(CDW_ADDRESS),
        raw_amount,
    )
    approve_tx = approve_fn.build_transaction({
        "chainId": CHAIN_ID,
        "from": hot_wallet.address,
        "gas": 200_000,
        "gasPrice": web3.eth.gas_price,
    })
    signed_approve = hot_wallet.sign_transaction_with_new_nonce(approve_tx)
    approve_hash = web3.eth.send_raw_transaction(signed_approve.rawTransaction)
    approve_receipt = web3.eth.wait_for_transaction_receipt(approve_hash)
    assert approve_receipt["status"] == 1

    # Deposit
    deposit_fn = cdw.functions.deposit(raw_amount, SPOT_DEX)
    deposit_tx = deposit_fn.build_transaction({
        "chainId": CHAIN_ID,
        "from": hot_wallet.address,
        "gas": 200_000,
        "gasPrice": web3.eth.gas_price,
    })
    signed_deposit = hot_wallet.sign_transaction_with_new_nonce(deposit_tx)
    deposit_hash = web3.eth.send_raw_transaction(signed_deposit.rawTransaction)
    deposit_receipt = web3.eth.wait_for_transaction_receipt(deposit_hash)
    assert deposit_receipt["status"] == 1

    # Wait for escrow to clear
    wait_for_evm_escrow_clear(session, user=hot_wallet.address, timeout=60.0)

    # Verify spot balance
    spot_state = fetch_spot_clearinghouse_state(session, hot_wallet.address)
    assert not spot_state.evm_escrows, f"Escrow not cleared: {spot_state.evm_escrows}"

    # USDC should be in spot balance
    usdc_spot = None
    for b in spot_state.balances:
        if b.coin == "USDC":
            usdc_spot = Decimal(b.total)
            break
    assert usdc_spot is not None, "No USDC in spot balance after deposit"
    assert usdc_spot > 0


def test_trade_execution_with_eoa_deposit(web3, hot_wallet, usdc, usdc_asset, hypercore_vault_pair, session):
    """End-to-end: create TradeExecution, sign/broadcast real txs, settle with API verification."""
    # Pre-checks
    activated = is_account_activated(web3, hot_wallet.address)
    if not activated:
        pytest.skip("Account not activated on HyperCore")

    usdc_balance = usdc.contract.functions.balanceOf(hot_wallet.address).call()
    if usdc_balance < 3_000_000:
        pytest.skip(f"Insufficient USDC: {usdc_balance / 1e6}")

    ts = datetime.datetime(2026, 3, 9, tzinfo=None)
    raw_amount = 3_000_000
    state = State()
    state.portfolio.initialise_reserves(usdc_asset, reserve_token_price=1.0)

    # Create trade
    position, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=hypercore_vault_pair,
        quantity=None,
        reserve=Decimal("3"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=usdc_asset,
        reserve_currency_price=1.0,
        notes="E2E Hypercore deposit test",
        pair_fee=0.0,
        lp_fees_estimated=0,
    )

    state.start_execution_all(ts, [trade], max_slippage=0.01)

    # Build and sign approve + CDW.deposit
    cdw = get_core_deposit_wallet_contract(web3, CDW_ADDRESS)

    approve_fn = usdc.contract.functions.approve(
        Web3.to_checksum_address(CDW_ADDRESS),
        raw_amount,
    )
    approve_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, approve_fn, "approve", "E2E approve",
    )

    deposit_fn = cdw.functions.deposit(raw_amount, SPOT_DEX)
    deposit_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, deposit_fn, "deposit", "E2E deposit",
    )

    trade.blockchain_transactions = [approve_tx, deposit_tx]

    # Broadcast both
    from hexbytes import HexBytes

    for tx in trade.blockchain_transactions:
        tx_hash = web3.eth.send_raw_transaction(HexBytes(tx.signed_bytes))
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
        assert receipt["status"] == 1, f"Tx {tx.function_selector} reverted"

    state.mark_broadcasted(ts, trade)

    # Wait for escrow to clear
    wait_for_evm_escrow_clear(session, user=hot_wallet.address, timeout=60.0)

    # Verify spot balance via API
    spot_state = fetch_spot_clearinghouse_state(session, hot_wallet.address)
    assert not spot_state.evm_escrows

    # Settle trade
    state.mark_trade_success(
        ts,
        trade,
        executed_price=1.0,
        executed_amount=Decimal("3"),
        executed_reserve=Decimal("3"),
        lp_fees=0,
        native_token_price=0,
    )

    assert trade.is_success()
    assert trade.executed_reserve == pytest.approx(Decimal("3"))
    assert len(trade.blockchain_transactions) == 2


def test_depositfor_blockchain_transaction(web3, hot_wallet, usdc):
    """Test that a depositFor BlockchainTransaction can be built and signed.

    Verifies the transaction metadata (chain ID, function selector, nonce)
    is correct for a CDW.depositFor call, which is used during Safe activation.
    """
    cdw = get_core_deposit_wallet_contract(web3, CDW_ADDRESS)
    raw_amount = 2_000_000  # 2 USDC

    deposit_for_fn = cdw.functions.depositFor(
        Web3.to_checksum_address(hot_wallet.address),
        raw_amount,
        SPOT_DEX,
    )
    deposit_for_tx = _sign_and_create_blockchain_tx(
        web3, hot_wallet, deposit_for_fn, "depositFor",
        "Test depositFor tx",
    )

    assert deposit_for_tx.chain_id == CHAIN_ID
    assert deposit_for_tx.function_selector == "depositFor"
    assert deposit_for_tx.tx_hash is not None
    assert deposit_for_tx.signed_bytes is not None
    assert deposit_for_tx.nonce >= 0


def test_is_account_activated_precompile(web3, hot_wallet):
    """Test that ``is_account_activated()`` correctly reports activation status.

    - The existing test hot_wallet is known to be activated on HyperCore testnet.
    - A random address that has never interacted with HyperCore is not activated.

    Safe (smart contract) activation via ``build_activate_account_multicall``
    requires a Lagoon vault deployment and is tested end-to-end by the manual
    lifecycle script (``manual-trade-executor-hyperliquid.py``).
    """
    # Known activated account
    assert is_account_activated(web3, hot_wallet.address), \
        "Test hot_wallet should be activated on HyperCore testnet"

    # Random address — never interacted with HyperCore
    random_address = Account.create().address
    assert not is_account_activated(web3, random_address), \
        "Random address should not be activated"
