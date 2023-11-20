from typing import TypedDict

from web3 import Web3
from web3.logs import DISCARD

from eth_defi.abi import get_transaction_data_field
from eth_defi.one_delta.deployment import OneDeltaDeployment
from eth_defi.one_delta.constants import TradeOperation
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeFail, TradeSuccess
from eth_defi.uniswap_v3.pool import fetch_pool_details


def decode_path(encoded_path: bytes) -> list:
    """Decodes the 1delta path. 

    :param encoded_path: 
        1delta encoded path

    :return:
        Fully decoded path array including addresses, fees and others
    """
    assert isinstance(encoded_path, bytes), "encoded path must be provided as bytes"

    # drop the flag from last byte since we don't use it
    encoded_path = encoded_path[:-1]

    current_position = 0
    decoded = []
    index = 0
    byte_order = {
        0: 20,
        1: 3,
        2: 1,
        3: 1,
    }

    while True:
        # stop at the end
        if current_position == len(encoded_path):
            break

        chunk_lenth = byte_order[index]
        chunk_position = current_position + chunk_lenth
        chunk = encoded_path[current_position : chunk_position]

        if chunk_lenth == 20:
            decoded.append(Web3.to_checksum_address(chunk.hex()))
        else:
            fee = int.from_bytes(chunk, "big", signed=False)
            decoded.append(fee)
        
        current_position += chunk_lenth
        index += 1
        if index > 3:
            index = 0

    return decoded


def analyse_trade_by_receipt(
    web3: Web3,
    one_delta: OneDeltaDeployment,
    uniswap: OneDeltaDeployment,
    tx: dict,
    tx_hash: str | bytes,
    tx_receipt: dict,
    input_args: tuple | None = None,
    trade_operation: TradeOperation = TradeOperation.OPEN,
) -> TradeSuccess | TradeFail:
    """Analyse a 1delta trade.

    Figure out

    - The success of the trade
    - Output amount

    :param tx_receipt:
        Transaction receipt

    :param input_args:
        The swap input arguments.

        If not given automatically decode from `tx`.
        You need to pass this for Enzyme transactions, because transaction payload 
        is too complex to decode.
    """
    effective_gas_price = tx_receipt.get("effectiveGasPrice", 0)
    gas_used = tx_receipt["gasUsed"]

    # tx reverted
    if tx_receipt["status"] != 1:
        reason = fetch_transaction_revert_reason(web3, tx_hash)
        return TradeFail(gas_used, effective_gas_price, revert_reason=reason)

    input_args = input_args[0]
    if len(input_args) == 3:
        if trade_operation == TradeOperation.OPEN:
            encoded_multicall_args = input_args[-1]
        else:
            encoded_multicall_args = input_args[0]
    elif len(input_args) == 1:
        encoded_multicall_args = input_args[0]
    else:
        raise ValueError("Should not happen")

    _, multicall_args = one_delta.flash_aggregator.decode_function_input(encoded_multicall_args)

    # amount_in = multicall_args["amountIn"]
    amount_out_min = multicall_args.get("amountOutMinimum", 0)
    path = decode_path(multicall_args["path"])

    # there should be only 1 swap event
    swap_event = uniswap.PoolContract.events.Swap().process_receipt(tx_receipt, errors=DISCARD)[0]

    props = swap_event["args"]
    amount0 = props["amount0"]
    amount1 = props["amount1"]
    tick = props["tick"]

    pool_address = swap_event["address"]
    pool = fetch_pool_details(web3, pool_address)

    # Depending on the path, the out token can pop up as amount0Out or amount1Out
    # For complex swaps (unspported) we can have both
    assert (amount0 > 0 and amount1 < 0) or (amount0 < 0 and amount1 > 0), "Unsupported swap type"

    if amount0 > 0:
        amount_in = amount0
        amount_out = amount1
    else:
        amount_in = amount1
        amount_out = amount0
    assert amount_out < 0, "amount out should be negative for uniswap v3"

    in_token_details = fetch_erc20_details(web3, path[0])
    out_token_details = fetch_erc20_details(web3, path[-1])
    price = pool.convert_price_to_human(tick)

    # TODO: this doesn't look quite correct
    lp_fee_paid = float(amount_in * pool.fee / 10**in_token_details.decimals)
    
    return TradeSuccess(
        gas_used,
        effective_gas_price,
        path=[pool.token1.address, pool.token0.address],
        amount_in=amount_in,
        amount_out_min=amount_out_min,
        amount_out=abs(amount_out),
        price=price,
        amount_in_decimals=in_token_details.decimals,
        amount_out_decimals=out_token_details.decimals,
        token0=pool.token0,
        token1=pool.token1,
        lp_fee_paid=lp_fee_paid,
    )
