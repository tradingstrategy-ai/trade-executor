"""Build CoW Protocol pre-signed orders for Gnosis Safe proposals."""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from eth_account.messages import encode_typed_data
from hexbytes import HexBytes
from web3 import Web3


COWSWAP_ORDER_TYPES = {
    "EIP712Domain": [
        {"name": "name", "type": "string"},
        {"name": "version", "type": "string"},
        {"name": "chainId", "type": "uint256"},
        {"name": "verifyingContract", "type": "address"},
    ],
    "Order": [
        {"name": "sellToken", "type": "address"},
        {"name": "buyToken", "type": "address"},
        {"name": "receiver", "type": "address"},
        {"name": "sellAmount", "type": "uint256"},
        {"name": "buyAmount", "type": "uint256"},
        {"name": "validTo", "type": "uint32"},
        {"name": "appData", "type": "bytes32"},
        {"name": "feeAmount", "type": "uint256"},
        {"name": "kind", "type": "string"},
        {"name": "partiallyFillable", "type": "bool"},
        {"name": "sellTokenBalance", "type": "string"},
        {"name": "buyTokenBalance", "type": "string"},
    ],
}


@dataclass(slots=True, frozen=True)
class PreparedCowSwapOrder:
    """A CoW order and the metadata needed to submit it after Safe execution."""

    symbol: str
    balance: int
    quoted_buy_amount: int
    order: dict
    uid: HexBytes

    def to_json(self) -> dict:
        """Serialise the prepared order without any private material."""
        return {
            "symbol": self.symbol,
            "balance": str(self.balance),
            "quoted_buy_amount": str(self.quoted_buy_amount),
            "uid": Web3.to_hex(self.uid),
            "order": self.order,
        }


class CowSwapAPIError(RuntimeError):
    """A CoW Order Book API request failed."""


def _request_json(
    url: str, method: str = "GET", payload: dict | None = None, timeout: float = 30
) -> tuple[int, dict | str]:
    """Make a JSON request using the standard library.

    ``requests`` is currently blocked by CoW's CloudFront configuration while
    urllib and curl are accepted, so production scripts use urllib here.
    """
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "trade-executor-cowswap-cleanup/1",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            decoded_body = json.loads(body)
        except json.JSONDecodeError:
            decoded_body = body
        return exc.code, decoded_body


def fetch_cowswap_quote(
    *,
    api_base_url: str,
    safe_address: str,
    sell_token: str,
    buy_token: str,
    sell_amount_before_fee: int,
    valid_for_seconds: int,
) -> dict:
    """Fetch a long-lived, Safe-compatible exact-sell quote."""
    payload = {
        "sellToken": sell_token,
        "buyToken": buy_token,
        "receiver": safe_address,
        "from": safe_address,
        "kind": "sell",
        "sellAmountBeforeFee": str(sell_amount_before_fee),
        "priceQuality": "optimal",
        "signingScheme": "presign",
        "onchainOrder": True,
        "validFor": valid_for_seconds,
    }
    status, response = _request_json(
        f"{api_base_url}/api/v1/quote", method="POST", payload=payload
    )
    if status != 200 or not isinstance(response, dict):
        raise CowSwapAPIError(f"CoW quote failed with HTTP {status}: {response}")
    return response


def calculate_order_digest(*, chain_id: int, settlement: str, order: dict) -> HexBytes:
    """Calculate the CoW EIP-712 order digest."""
    typed_message = {
        "types": COWSWAP_ORDER_TYPES,
        "primaryType": "Order",
        "domain": {
            "name": "Gnosis Protocol",
            "version": "v2",
            "chainId": chain_id,
            "verifyingContract": settlement,
        },
        "message": {
            "sellToken": order["sellToken"],
            "buyToken": order["buyToken"],
            "receiver": order["receiver"],
            "sellAmount": int(order["sellAmount"]),
            "buyAmount": int(order["buyAmount"]),
            "validTo": int(order["validTo"]),
            "appData": HexBytes(order["appData"]),
            "feeAmount": int(order["feeAmount"]),
            "kind": order["kind"],
            "partiallyFillable": order["partiallyFillable"],
            "sellTokenBalance": order["sellTokenBalance"],
            "buyTokenBalance": order["buyTokenBalance"],
        },
    }
    signable_message = encode_typed_data(full_message=typed_message)
    return HexBytes(
        Web3.keccak(
            b"\x19"
            + signable_message.version
            + signable_message.header
            + signable_message.body
        )
    )


def calculate_order_uid(
    *, chain_id: int, settlement: str, owner: str, order: dict
) -> HexBytes:
    """Calculate the 56-byte CoW order UID."""
    digest = calculate_order_digest(
        chain_id=chain_id, settlement=settlement, order=order
    )
    owner_bytes = HexBytes(Web3.to_checksum_address(owner))
    valid_to = int(order["validTo"]).to_bytes(4, byteorder="big")
    return HexBytes(digest + owner_bytes + valid_to)


def prepare_cowswap_order(
    *,
    symbol: str,
    balance: int,
    sell_token: str,
    buy_token: str,
    quote_response: dict,
    chain_id: int,
    settlement: str,
    owner: str,
    slippage_bps: int,
) -> PreparedCowSwapOrder:
    """Turn a CoW quote into a pre-signable order with explicit price protection."""
    assert 0 <= slippage_bps < 10_000, f"Invalid slippage: {slippage_bps} bps"
    quote = quote_response["quote"]
    assert quote["signingScheme"] == "presign", (
        f"Unexpected signing scheme: {quote['signingScheme']}"
    )
    assert quote["kind"] == "sell", f"Unexpected order kind: {quote['kind']}"
    assert quote["sellToken"].lower() == sell_token.lower(), (
        f"Unexpected quote sell token: {quote['sellToken']}"
    )
    assert quote["buyToken"].lower() == buy_token.lower(), (
        f"Unexpected quote buy token: {quote['buyToken']}"
    )
    assert quote["receiver"].lower() == owner.lower(), (
        f"Unexpected quote receiver: {quote['receiver']}"
    )
    assert quote["appData"] == "0x" + "00" * 32, (
        f"Unexpected quote appData: {quote['appData']}"
    )
    assert quote["partiallyFillable"] is False, "Partial fills are not allowed"
    assert quote["sellTokenBalance"] == "erc20", (
        f"Unexpected sell token balance: {quote['sellTokenBalance']}"
    )
    assert quote["buyTokenBalance"] == "erc20", (
        f"Unexpected buy token balance: {quote['buyTokenBalance']}"
    )
    quote_sell_amount = int(quote["sellAmount"])
    quote_fee_amount = int(quote["feeAmount"])
    assert quote_sell_amount + quote_fee_amount == balance, (
        f"CoW quote does not consume the full {symbol} balance: "
        f"sell {quote_sell_amount} + fee {quote_fee_amount} != balance {balance}"
    )
    quoted_buy_amount = int(quote["buyAmount"])
    minimum_buy_amount = quoted_buy_amount * (10_000 - slippage_bps) // 10_000
    assert minimum_buy_amount > 0, f"Minimum buy amount is zero for {symbol}"

    order = {
        "sellToken": Web3.to_checksum_address(quote["sellToken"]),
        "buyToken": Web3.to_checksum_address(quote["buyToken"]),
        "receiver": Web3.to_checksum_address(owner),
        # The quote splits the requested input into sellAmount + feeAmount.
        # New orders sign a zero legacy fee, so the signed sell amount must be
        # the original before-fee input for the full Safe balance to be sold.
        "sellAmount": str(balance),
        "buyAmount": str(minimum_buy_amount),
        "validTo": int(quote["validTo"]),
        "appData": quote["appData"],
        # CoW's current API requires newly created orders to sign a zero legacy fee.
        # Solvers compute the actual fee dynamically within the order's limit price.
        "feeAmount": "0",
        "kind": "sell",
        "partiallyFillable": False,
        "sellTokenBalance": "erc20",
        "buyTokenBalance": "erc20",
    }
    uid = calculate_order_uid(
        chain_id=chain_id, settlement=settlement, owner=owner, order=order
    )
    return PreparedCowSwapOrder(
        symbol=symbol,
        balance=balance,
        quoted_buy_amount=quoted_buy_amount,
        order=order,
        uid=uid,
    )


def post_presigned_order(
    *, api_base_url: str, owner: str, prepared_order: PreparedCowSwapOrder
) -> str:
    """Post a Safe-pre-signed order, returning its UID.

    The operation is idempotent: an order already visible through the API is
    accepted as success after its UID is checked.
    """
    uid = Web3.to_hex(prepared_order.uid)
    status, existing = _request_json(f"{api_base_url}/api/v1/orders/{uid}")
    if status == 200:
        assert (
            isinstance(existing, dict)
            and existing.get("uid", "").lower() == uid.lower()
        ), f"Unexpected existing order: {existing}"
        return uid
    if status != 404:
        raise CowSwapAPIError(
            f"Could not check CoW order {uid}: HTTP {status}: {existing}"
        )

    payload = prepared_order.order.copy()
    payload.update(
        {
            "signature": "0x",
            "signingScheme": "presign",
            "from": Web3.to_checksum_address(owner),
            # CoW's OrderCreation API uses this to validate the full fill-or-kill
            # balance and allowance rather than accepting a partial balance.
            "fullBalanceCheck": True,
        }
    )
    status, response = _request_json(
        f"{api_base_url}/api/v1/orders", method="POST", payload=payload
    )
    if status not in {200, 201}:
        raise CowSwapAPIError(
            f"Posting CoW order {uid} failed with HTTP {status}: {response}"
        )
    posted_uid = response if isinstance(response, str) else response.get("uid")
    if not isinstance(posted_uid, str) or posted_uid.lower() != uid.lower():
        raise CowSwapAPIError(
            f"CoW returned unexpected order UID {posted_uid}, expected {uid}"
        )
    return uid
