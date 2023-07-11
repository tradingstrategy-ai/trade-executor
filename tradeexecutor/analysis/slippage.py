"""Display trading positions as Pandas notebook items."""
import datetime
from typing import Iterable

import numpy as np
import pandas as pd

from tradeexecutor.ethereum.revert import clean_revert_reason_message

from tradeexecutor.state.trade import TradeExecution


def _ftime(v: datetime.datetime) -> str:
    """Format times"""
    if not v:
        return ""
    return v.strftime('%Y-%m-%d %H:%M')


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

        if t.is_buy():
            trade_type = "BUY"
        else:
            trade_type = "SELL"

        lag = t.get_execution_lag()

        reason = t.get_revert_reason()
        if reason:
            reason = clean_revert_reason_message(reason)

        tx_link = None

        input = t.get_input_asset()
        output = t.get_output_asset()

        # Swap is always the last transaction
        if len(t.blockchain_transactions) > 0:
            swap_tx = t.blockchain_transactions[-1]
            if swap_tx.function_selector == "callOnExtension":
                # Enzyme vault tx + underlying GenericAdapter wrapper
                # Assume Uniswap v3 always
                #  wrapped args:[['2791bca1f2de4661ed88a30c99a7a9449aa841740001f47ceb23fd6bc0add59e62ac25578270cff1b9f619', '0x07f7eB451DfeeA0367965646660E85680800E352', 9223372036854775808, 3582781, 1896263219612875]]
                uni_arg_list = swap_tx.wrapped_args[0]
                uniswap_amount_in = uni_arg_list[-1]
                uniswap_amount_out = uni_arg_list[-2]
            else:
                uniswap_amount_in = np.NaN
                uniswap_amount_out = np.NaN

            tx_hash = swap_tx.tx_hash
            # TODO: Does not work in all notebook run times
            # tx_link = f"""<a href="https://polygonscan.io/tx/{tx_hash}>{tx_hash}</a>"""
            tx_link = tx_hash

        items.append({
            "Flags": ", ".join(flags),
            "Position": f"#{t.position_id}",
            "Trade": f"{input.token_symbol}->{output.token_symbol}",
            "Started": _ftime(t.started_at),
            "Executed": _ftime(t.executed_at),
            "Lag": lag.total_seconds() if lag else np.NaN,
            "Slippage tol (BPS)": int(t.slippage_tolerance * 10000) if t.slippage_tolerance else np.NaN,
            "Tx": tx_link,
            "Notes": t.notes,
            "Failure reason": reason,
        })

    df = pd.DataFrame(items, index=idx)
    df = df.fillna("")
    df = df.replace({pd.NaT: ""})
    return df


