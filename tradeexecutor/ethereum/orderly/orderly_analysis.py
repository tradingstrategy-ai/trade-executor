"""Orderly vault deposit/withdrawal transaction analysis."""

from decimal import Decimal
from typing import Literal

from web3.logs import DISCARD

from tradeexecutor.ethereum.orderly.orderly_vault import OrderlyVault
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.trade import TradeSuccess, TradeFail


def analyse_orderly_flow_transaction(
    vault: OrderlyVault,
    tx_hash: str | bytes,
    tx_receipt: dict,
    direction: Literal["deposit", "withdraw"],
    hot_wallet=True,
) -> TradeSuccess | TradeFail:
    """Analyse an Orderly vault deposit/withdraw transaction.

    Parses transaction logs to extract:
    - Actual deposited/withdrawn amounts
    - Fees paid
    - Success/failure status
    - Gas costs

    :param vault:
        OrderlyVault instance

    :param tx_hash:
        Transaction hash

    :param tx_receipt:
        Transaction receipt from web3

    :param direction:
        Either "deposit" or "withdraw"

    :param hot_wallet:
        Is this a hot wallet originated transaction.
        We can perform additional checks with hot wallet transactions.

    :return:
        TradeSuccess with actual amounts or TradeFail with revert reason
    """

    web3 = vault.web3

    if hot_wallet:
        assert tx_receipt["to"] == vault.address, (
            f"Transaction receipt 'to' address {tx_receipt['to']} does not match vault address {vault.address}.\n"
            f"Vault is: {vault}"
        )

    assert direction in ("deposit", "withdraw"), f"Invalid direction: {direction}"

    effective_gas_price = tx_receipt.get("effectiveGasPrice", 0)
    gas_used = tx_receipt["gasUsed"]

    # Check if transaction reverted
    if tx_receipt["status"] != 1:
        reason = fetch_transaction_revert_reason(web3, tx_hash)
        return TradeFail(gas_used, effective_gas_price, revert_reason=reason)

    contract = vault.vault_contract

    # Parse Orderly vault events
    # Note: Event names depend on actual Orderly vault contract ABI
    # TODO: Update event names based on actual Orderly vault implementation
    if direction == "deposit":
        # Look for deposit events - adjust event name as needed
        try:
            deposit_events = contract.events.Deposit().process_receipt(tx_receipt, errors=DISCARD)
        except Exception:
            # Fallback: try alternative event names
            deposit_events = []

        # Filter events from this vault only
        deposit_events = [
            event for event in deposit_events
            if event["address"].lower() == vault.address.lower()
        ]

        if len(deposit_events) == 0:
            # TODO: Remove this fallback once proper event parsing is implemented
            # For now, return a placeholder indicating we couldn't parse logs
            raise AssertionError(
                f"No deposit events detected for Orderly vault {vault.address}. "
                f"Transaction: {tx_hash}. "
                f"This likely means event parsing needs to be updated for Orderly vault ABI."
            )

        # Extract deposit amount from event
        first_event = deposit_events[0]
        # TODO: Update field names based on actual Orderly event structure
        # Common patterns: amount, assets, value
        amount_in = first_event["args"].get("amount") or first_event["args"].get("assets")

        # Orderly deposits: USDC in -> vault shares out
        denomination_token = vault.fetch_denomination_token()
        in_token = denomination_token
        out_token = denomination_token  # Orderly uses underlying token as accounting unit

        path = [in_token.address.lower(), out_token.address.lower()]

    else:  # withdraw
        # Look for withdrawal events
        try:
            withdraw_events = contract.events.Withdraw().process_receipt(tx_receipt, errors=DISCARD)
        except Exception:
            withdraw_events = []

        withdraw_events = [
            event for event in withdraw_events
            if event["address"].lower() == vault.address.lower()
        ]

        if len(withdraw_events) == 0:
            raise AssertionError(
                f"No withdraw events detected for Orderly vault {vault.address}. "
                f"Transaction: {tx_hash}. "
                f"This likely means event parsing needs to be updated for Orderly vault ABI."
            )

        first_event = withdraw_events[0]
        amount_out = first_event["args"].get("amount") or first_event["args"].get("assets")

        # Orderly withdrawals: vault shares in -> USDC out
        denomination_token = vault.fetch_denomination_token()
        in_token = denomination_token
        out_token = denomination_token

        path = [in_token.address.lower(), out_token.address.lower()]
        amount_in = amount_out  # For withdrawals, we measure by output amount

    # Convert to human-readable decimals
    amount_out_cleaned = Decimal(abs(amount_in)) / Decimal(10**denomination_token.decimals)
    amount_in_cleaned = Decimal(abs(amount_in)) / Decimal(10**denomination_token.decimals)

    # For Orderly, price is typically 1:1 with underlying token
    price = Decimal(1)

    # TODO: Extract actual fees from transaction logs
    lp_fee_paid = 0

    return TradeSuccess(
        gas_used=gas_used,
        gas_price=effective_gas_price,
        path=path,
        amount_in=int(amount_in),
        amount_out=int(amount_in),  # 1:1 for Orderly vault accounting
        amount_out_min=0,
        price=float(price),
        token0=in_token.address.lower(),
        token1=out_token.address.lower(),
        lp_fee_paid=lp_fee_paid,
    )
