"""Display trading positions as Pandas notebook items."""
import binascii
import datetime
from typing import Iterable

import numpy as np
import pandas as pd
from eth_abi import decode

from tradeexecutor.ethereum.revert import clean_revert_reason_message
from tradeexecutor.state.trade import TradeExecution


def _ftime(v: datetime.datetime) -> str:
    """Format times"""
    if not v:
        return ""
    return v.strftime('%Y-%m-%d %H:%M')


def _decode_generic_adapter_execute_calls_args(data: bytes) -> dict:
    """Decode arguments for a generic adapter call.

    Needs to bulldoze through various warpped ABI encodings.
    """

    #   return encodeArgs(
    #     ['address[]', 'uint256[]', 'address[]', 'uint256[]', 'bytes'],
    #     [incomingAssets, minIncomingAssetAmounts, spendAssets, spendAssetAmounts, encodedExternalCallsData],
    #   );

    #all_args_encoded = encode(
    #    ["address[]", "uint256[]", "address[]", "uint256[]", "bytes"],
    #    [_addressify_collection(incoming_assets), min_incoming_asset_amounts, _addressify_collection(spend_assets), spend_asset_amounts, encoded_external_calls_data],
    #)

    integration_manager_abi = ["address", "bytes4", "bytes"]
    generic_adapter_address, selector, generic_adapter_payload = decode(integration_manager_abi, data)

    # incoming assets, min_incoming, spend assets, spendasset amounnts, encoded_external_calls_data
    generic_adapter_abi = ["address[]", "uint256[]", "address[]", "uint256[]", "bytes"]
    decoded_2 = decode(generic_adapter_abi, generic_adapter_payload)

    return {
        "incoming_assets": decoded_2[0],
        "min_incoming": decoded_2[1],
        "spend_assets": decoded_2[2],
        "spend_assets_amounts": decoded_2[3],
        "encoded_external_calls_data": decoded_2[4],
    }


def display_slippage(trades: Iterable[TradeExecution]) -> pd.DataFrame:
    """Format trade slippage details for Jupyter Notebook table output.

    Display in one table
    :return:
        DataFrame containing positions and trades, values as string formatted
    """

    items = []
    idx = []
    t: TradeExecution
    for t in trades:
        idx.append(t.trade_id)
        flags = []

        if t.is_failed():
            flags.append("FAIL")

        if t.is_repaired():
            flags.append("REP")

        if t.is_repair_trade():
            flags.append("FIX")

        lag = t.get_execution_lag()

        reason = t.get_revert_reason()
        if reason:
            reason = clean_revert_reason_message(reason)

        tx_link = None
        block_number = None

        input = t.get_input_asset()
        output = t.get_output_asset()

        # Start with all columns NaN
        amount_in = np.NaN
        amount_out = np.NaN
        uniswap_price = np.NaN
        enzyme_expected_amount = np.NaN

        # Swap is always the last transaction
        if len(t.blockchain_transactions) > 0:
            swap_tx = t.blockchain_transactions[-1]
            block_number = swap_tx.block_number
            import ipdb ; ipdb.set_trace()
            if swap_tx.function_selector == "callOnExtension":
                # Enzyme vault tx + underlying GenericAdapter wrapper
                # Assume Uniswap v3 always
                #  wrapped args:[['2791bca1f2de4661ed88a30c99a7a9449aa841740001f47ceb23fd6bc0add59e62ac25578270cff1b9f619', '0x07f7eB451DfeeA0367965646660E85680800E352', 9223372036854775808, 3582781, 1896263219612875]]
                uni_arg_list = swap_tx.wrapped_args[0]
                uniswap_amount_in = uni_arg_list[-1]
                uniswap_amount_out = uni_arg_list[-2]

                amount_in = input.convert_to_decimal(uniswap_amount_in)
                amount_out = output.convert_to_decimal(uniswap_amount_out)
                uniswap_price = amount_in / amount_out

                generic_adapter_data = binascii.unhexlify(swap_tx.args[2])
                enzyme_args = _decode_generic_adapter_execute_calls_args(generic_adapter_data)

                import ipdb ; ipdb.set_trace()

                # Check we did not pass wrong token address to enzyme
                enzyme_incoming_token = enzyme_args["incoming_assets"][0]
                assert enzyme_incoming_token == output.address

                enzyme_min_incoming_raw = enzyme_args["min_incoming"][0]
                enzyme_expected_amount = output.convert_to_decimal(enzyme_min_incoming_raw)

            tx_hash = swap_tx.tx_hash
            # TODO: Does not work in all notebook run times
            # tx_link = f"""<a href="https://polygonscan.io/tx/{tx_hash}>{tx_hash}</a>"""
            tx_link = tx_hash

        items.append({
            "Flags": ", ".join(flags),
            "Position": f"#{t.position_id}",
            "Trade": f"{input.token_symbol}->{output.token_symbol}",
            # "Started": _ftime(t.started_at),
            #"Executed": _ftime(t.executed_at),
            # "Block": f"{block_number:,}" if block_number else "",
            "Lag": lag.total_seconds() if lag else np.NaN,
            "Slippage": int(t.slippage_tolerance * 10000) if t.slippage_tolerance else np.NaN,
            "Amount in": amount_in,
            "Amount out": amount_out,
            "Enzyme expected amount": enzyme_expected_amount,
            "Assumed price": t.planned_price,
            "Uniswap price": uniswap_price,
            #"Tx": tx_link,
            # "Notes": t.notes,
            "Failure reason": reason,
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df


