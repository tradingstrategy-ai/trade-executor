"""Test Safe-compatible CoW orders without external API calls."""

import pytest
from hexbytes import HexBytes
from web3 import Web3

from tradeexecutor.ethereum.cowswap import safe_proposal
from tradeexecutor.ethereum.cowswap.safe_proposal import (
    CowSwapAPIError,
    calculate_order_digest,
    calculate_order_uid,
    post_presigned_order,
    prepare_cowswap_order,
)


SAFE = "0x7838A4E4ecD438c1BdD13b014675c7e877b8b490"
SETTLEMENT = "0x9008D19f58AAbD9eD0D60971565AA8510560ab41"
WBTC = "0x2f2a2543B76A4166549F7aaB2e75Bef0aefC5B0f"
USDC = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"


def test_prepare_cowswap_order_uid():
    """Test that a quote becomes a correctly packed pre-sign order UID.

    1. Prepare a deterministic Safe-compatible CoW quote.
    2. Apply one per cent price protection and the current zero signed-fee rule.
    3. Verify the owner and expiry packed into the prepared order UID.
    4. Check hashing against a real fulfilled Arbitrum CoW order UID.
    """
    # 1. Prepare a deterministic Safe-compatible CoW quote.
    quote_response = {
        "quote": {
            "sellToken": WBTC,
            "buyToken": USDC,
            "receiver": SAFE,
            "sellAmount": "184800",
            "buyAmount": "146000000",
            "validTo": 1_800_000_000,
            "appData": "0x" + "00" * 32,
            "feeAmount": "17",
            "kind": "sell",
            "partiallyFillable": False,
            "sellTokenBalance": "erc20",
            "buyTokenBalance": "erc20",
            "signingScheme": "presign",
        }
    }

    # 2. Apply one per cent price protection and the current zero signed-fee rule.
    prepared = prepare_cowswap_order(
        symbol="WBTC",
        balance=184817,
        sell_token=WBTC,
        buy_token=USDC,
        quote_response=quote_response,
        chain_id=42161,
        settlement=SETTLEMENT,
        owner=SAFE,
        slippage_bps=100,
    )
    assert prepared.order["buyAmount"] == "144540000"
    assert prepared.order["feeAmount"] == "0"
    assert prepared.order["sellAmount"] == "184817"

    # 3. Verify the owner and expiry packed into the prepared order UID.
    digest = calculate_order_digest(
        chain_id=42161, settlement=SETTLEMENT, order=prepared.order
    )
    assert len(digest) == 32
    assert len(prepared.uid) == 56
    assert prepared.uid[:32] == digest
    assert prepared.uid[32:52] == HexBytes(SAFE)
    assert int.from_bytes(prepared.uid[52:], byteorder="big") == 1_800_000_000

    # 4. Check hashing against a real fulfilled Arbitrum CoW order UID.
    # Source: CoW Order Book API, auction 8657295, fetched 2026-08-28.
    settled_order = {
        "sellToken": "0xe50fa9b3c56ffb159cb0fca61f5c9d750e8128c8",
        "buyToken": "0x724dc807b04555b71ed48a6896b6f41593b8c637",
        "receiver": "0xcb46281bee2dfa0af6753576bb6b11923243582c",
        "sellAmount": "36653551915617423412",
        "buyAmount": "91019171318",
        "validTo": 1_787_922_985,
        "appData": "0xa3465c7c262898375e108c2bee18f39e29e9894807d5f724aba044a14a33ff3d",
        "feeAmount": "0",
        "kind": "sell",
        "partiallyFillable": False,
        "sellTokenBalance": "erc20",
        "buyTokenBalance": "erc20",
    }
    settled_uid = calculate_order_uid(
        chain_id=42161,
        settlement=SETTLEMENT,
        owner=settled_order["receiver"],
        order=settled_order,
    )
    assert Web3.to_hex(settled_uid) == (
        "0xa2b42c3bd124e5ba6e8b56399e5b98d807b940ab60e1e6c840dfa380ea9ed125"
        "cb46281bee2dfa0af6753576bb6b11923243582c6a918a29"
    )


def test_cowswap_order_rejects_unsafe_api_data(monkeypatch: pytest.MonkeyPatch):
    """Test that unsafe quote and order-book responses are rejected.

    1. Prepare a quote whose receiver is not the expected Safe.
    2. Attempt to prepare a pre-signable order.
    3. Verify preparation rejects the unsafe receiver before transaction creation.
    4. Stub the CoW API because this unit test must not make network requests.
    5. Verify a mismatched UID returned during order creation is rejected.
    """
    # 1. Prepare a quote whose receiver is not the expected Safe.
    quote_response = {
        "quote": {
            "sellToken": WBTC,
            "buyToken": USDC,
            "receiver": "0x0000000000000000000000000000000000000001",
            "sellAmount": "184800",
            "buyAmount": "146000000",
            "validTo": 1_800_000_000,
            "appData": "0x" + "00" * 32,
            "feeAmount": "17",
            "kind": "sell",
            "partiallyFillable": False,
            "sellTokenBalance": "erc20",
            "buyTokenBalance": "erc20",
            "signingScheme": "presign",
        }
    }

    # 2. Attempt to prepare a pre-signable order.
    # 3. Verify preparation rejects the unsafe receiver before transaction creation.
    with pytest.raises(AssertionError, match="Unexpected quote receiver"):
        prepare_cowswap_order(
            symbol="WBTC",
            balance=184817,
            sell_token=WBTC,
            buy_token=USDC,
            quote_response=quote_response,
            chain_id=42161,
            settlement=SETTLEMENT,
            owner=SAFE,
            slippage_bps=100,
        )

    # 4. Stub the CoW API because this unit test must not make network requests.
    quote_response["quote"]["receiver"] = SAFE
    prepared = prepare_cowswap_order(
        symbol="WBTC",
        balance=184817,
        sell_token=WBTC,
        buy_token=USDC,
        quote_response=quote_response,
        chain_id=42161,
        settlement=SETTLEMENT,
        owner=SAFE,
        slippage_bps=100,
    )
    responses = iter(((404, {}), (201, "0x" + "00" * 56)))
    monkeypatch.setattr(
        safe_proposal, "_request_json", lambda *args, **kwargs: next(responses)
    )

    # 5. Verify a mismatched UID returned during order creation is rejected.
    with pytest.raises(CowSwapAPIError, match="unexpected order UID"):
        post_presigned_order(
            api_base_url="https://api.cow.fi/arbitrum_one",
            owner=SAFE,
            prepared_order=prepared,
        )
