"""Unit tests for GMX exchange account pair creation and metadata."""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from eth_defi.gmx.contracts import get_contract_addresses
from eth_defi.gmx.funding import ClaimableFundingFee, ClaimedFundingFee
from eth_defi.token import USDC_NATIVE_TOKEN

from tradeexecutor.exchange_account.gmx import (
    claim_gmx_funding_fees_if_due,
    create_gmx_exchange_account_pair,
    has_gmx_exchange_account_pairs,
    is_gmx_funding_claim_enabled,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.identifier import (
    AssetIdentifier,
    TradingPairIdentifier,
    TradingPairKind,
)
from tradeexecutor.state.state import State

#: Arbitrum mainnet chain ID
ARBITRUM_CHAIN_ID = 42161


@pytest.fixture()
def usdc() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


def test_create_gmx_exchange_account_pair(usdc):
    """Verify pair fields, kind, and is_exchange_account()."""
    pair = create_gmx_exchange_account_pair(quote=usdc)

    exchange_router = get_contract_addresses("arbitrum").exchangerouter
    assert pair.kind == TradingPairKind.exchange_account
    assert pair.is_exchange_account()
    assert pair.base.token_symbol == "GMX-ACCOUNT"
    assert pair.base.decimals == 6
    assert pair.base.chain_id == ARBITRUM_CHAIN_ID
    assert pair.base.address == exchange_router.lower()  # AssetIdentifier lowercases
    assert pair.quote.token_symbol == "USDC"
    assert pair.pool_address == exchange_router
    assert pair.exchange_address == exchange_router
    assert pair.exchange_name == "GMX"
    assert pair.fee == 0.0


def test_gmx_pair_protocol_detection(usdc):
    """Verify get_exchange_account_protocol() and other_data."""
    pair = create_gmx_exchange_account_pair(
        quote=usdc,
        is_testnet=True,
    )

    assert pair.get_exchange_account_protocol() == "gmx"

    config = pair.get_exchange_account_config()
    assert config["exchange_protocol"] == "gmx"
    assert config["exchange_is_testnet"] is True

    # Mainnet variant
    pair_mainnet = create_gmx_exchange_account_pair(
        quote=usdc,
        is_testnet=False,
    )
    assert pair_mainnet.get_exchange_account_config()["exchange_is_testnet"] is False


def test_gmx_funding_claim_is_explicitly_opt_in(usdc):
    """Only a pair created with the claim flag enables the scheduled transaction."""
    disabled_pair = create_gmx_exchange_account_pair(quote=usdc)
    enabled_pair = create_gmx_exchange_account_pair(quote=usdc, claim_funding_fees=True)

    assert (
        is_gmx_funding_claim_enabled(
            SimpleNamespace(iterate_pairs=lambda: [disabled_pair])
        )
        is False
    )
    assert (
        is_gmx_funding_claim_enabled(
            SimpleNamespace(iterate_pairs=lambda: [enabled_pair])
        )
        is True
    )


def test_has_gmx_exchange_account_pairs(usdc: AssetIdentifier):
    """Universe-driven GMX detection finds GMX exchange account pairs and nothing else.

    EthereumPairConfigurator uses this detection to auto-wire the GMX value
    functions without any environment variable, so the check must be positive
    exactly when the strategy universe trades GMX through an exchange account
    pair.

    1. Build a universe stub containing a GMX exchange account pair and verify detection is positive.
    2. Build a universe stub with only a spot pair and verify detection is negative.
    3. Verify a universe object without iterate_pairs() (non-trading stub) is negative.
    """
    # 1. Build a universe stub containing a GMX exchange account pair and verify detection is positive.
    gmx_pair = create_gmx_exchange_account_pair(quote=usdc)
    spot_pair = TradingPairIdentifier(
        base=AssetIdentifier(
            chain_id=ARBITRUM_CHAIN_ID,
            address="0x0000000000000000000000000000000000000005",
            token_symbol="WETH",
            decimals=18,
        ),
        quote=usdc,
        pool_address="0x0000000000000000000000000000000000000006",
        exchange_address="0x0000000000000000000000000000000000000007",
        internal_id=2,
        internal_exchange_id=2,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
        exchange_name="uniswap-v3",
    )
    gmx_universe = SimpleNamespace(iterate_pairs=lambda: [spot_pair, gmx_pair])
    assert has_gmx_exchange_account_pairs(gmx_universe) is True

    # 2. Build a universe stub with only a spot pair and verify detection is negative.
    spot_universe = SimpleNamespace(iterate_pairs=lambda: [spot_pair])
    assert has_gmx_exchange_account_pairs(spot_universe) is False

    # 3. Verify a universe object without iterate_pairs() (non-trading stub) is negative.
    assert has_gmx_exchange_account_pairs(SimpleNamespace(reserve_assets=[])) is False


def test_claim_gmx_funding_fees_updates_reserves(monkeypatch, usdc):
    """A mined funding receipt is credited once and throttled on the next hourly refresh.

    1. Create an opted-in GMX position and mock one claimable USDC tuple.
    2. Execute the low-frequency task and verify exact receipt accounting.
    3. Call again within one day and verify that no second transaction is built.
    """
    timestamp = datetime.datetime(2026, 9, 18, 12, 0)
    state = State()
    reserve = state.portfolio.initialise_reserves(usdc, reserve_token_price=1.0)
    reserve.quantity = Decimal("100")
    pair = create_gmx_exchange_account_pair(quote=usdc, claim_funding_fees=True)
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=timestamp,
        pair=pair,
        reserve_currency=usdc,
    )
    universe = SimpleNamespace(iterate_pairs=lambda: [pair])

    market = "0x1000000000000000000000000000000000000001"
    safe = "0x3000000000000000000000000000000000000001"
    claimable = ClaimableFundingFee(
        market=market, token=usdc.checksum_address, amount=33_523_756
    )
    claimed = ClaimedFundingFee(
        market=market,
        token=usdc.checksum_address,
        account=safe,
        receiver=safe,
        amount=33_523_756,
    )
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.fetch_claimable_funding_fees",
        lambda *_args, **_kwargs: [claimable],
    )
    build_call = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.build_claim_funding_fees_call", build_call
    )
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.extract_claimed_funding_fees",
        lambda *_args, **_kwargs: [claimed],
    )
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.get_exchange_router_contract",
        lambda *_args: MagicMock(),
    )
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.GMXConfig",
        lambda _web3: SimpleNamespace(chain="arbitrum"),
    )
    monkeypatch.setattr(
        "tradeexecutor.exchange_account.gmx.get_block_timestamp",
        lambda *_args: timestamp,
    )

    tx = SimpleNamespace(tx_hash="0xabc")
    tx_builder = SimpleNamespace(
        get_token_delivery_address=lambda: safe,
        sign_transaction=MagicMock(return_value=tx),
        broadcast_and_wait_transactions_to_complete=MagicMock(),
    )
    web3 = MagicMock()
    web3.eth.get_transaction_receipt.return_value = {
        "status": 1,
        "blockNumber": 123_456,
    }
    execution_model = SimpleNamespace(
        tx_builder=tx_builder,
        web3=web3,
        disable_broadcast=False,
        confirmation_block_count=0,
        confirmation_timeout=datetime.timedelta(minutes=1),
    )
    store = SimpleNamespace(sync=MagicMock())

    events = claim_gmx_funding_fees_if_due(
        timestamp,
        state,
        universe,
        execution_model,
        store,
    )

    assert len(events) == 1
    assert events[0].cause == BalanceUpdateCause.interest
    assert events[0].quantity == Decimal("33.523756")
    assert reserve.quantity == Decimal("133.523756")
    assert len(state.sync.accounting.balance_update_refs) == 1
    store.sync.assert_called_once_with(state)

    second_events = claim_gmx_funding_fees_if_due(
        timestamp + datetime.timedelta(hours=1),
        state,
        universe,
        execution_model,
        store,
    )
    assert second_events == []
    build_call.assert_called_once()
