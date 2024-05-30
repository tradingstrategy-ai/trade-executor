from web3 import Web3
from web3.logs import DISCARD
from eth_abi import decode
from decimal import Decimal

from eth_defi.abi import (
    get_transaction_data_field,
    decode_function_args,
    humanise_decoded_arg_data,
    get_deployed_contract,
)
from eth_defi.one_delta.deployment import OneDeltaDeployment
from eth_defi.one_delta.constants import TradeOperation, Exchange
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeFail, TradeSuccess
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.pool import fetch_pool_details
from eth_defi.aave_v3.deployment import AaveV3Deployment


def analyse_credit_trade_by_receipt(
    web3: Web3,
    *,
    aave_v3_deployment: AaveV3Deployment,
    tx: dict,
    tx_hash: str | bytes,
    tx_receipt: dict,
    input_args: tuple | None = None,
    trade_operation: TradeOperation = TradeOperation.OPEN,
) -> TradeSuccess | TradeFail:
    """Analyse a 1delta credit supply trade.

    Figure out

    - The success of the trade

    :param tx_receipt:
        Transaction receipt

    :param input_args:
        The swap input arguments.

        If not given automatically decode from `tx`.
        You need to pass this for Enzyme transactions, because transaction payload 
        is too complex to decode.

    :return:
        Trade result
    """
    effective_gas_price = tx_receipt.get("effectiveGasPrice", 0)
    gas_used = tx_receipt["gasUsed"]

    # tx reverted
    if tx_receipt["status"] != 1:
        reason = fetch_transaction_revert_reason(web3, tx_hash)
        return TradeFail(gas_used, effective_gas_price, revert_reason=reason), None

    # transferERC20In -> deposit
    # transferERC20AllIn -> withdraw
    print(input_args)
    print(tx)
    args = input_args[0][0]
    print(args)
    pool = aave_v3_deployment.pool
    # _, multicall_args = one_delta.flash_aggregator.decode_function_input(args)

    if trade_operation == TradeOperation.OPEN:
        supply_event = pool.events.Supply().process_receipt(tx_receipt, errors=DISCARD)[0]
        amount_in = supply_event["args"]["amount"]

        in_token = input_args[0]
        in_token_decimals = fetch_erc20_details(web3, in_token).decimals
        out_token_decimals = 0

        amount_out = 0

    else:
        withdraw_event = pool.events.Withdraw().process_receipt(tx_receipt, errors=DISCARD)[0]
        amount_out = withdraw_event["args"]["amount"]

        out_token = input_args[0]
        out_token_decimals = fetch_erc20_details(web3, out_token).decimals
        in_token_decimals = 0

        amount_in = 0
    
    return TradeSuccess(
        gas_used,
        effective_gas_price,
        path=None,
        amount_in=amount_in,
        amount_out_min=None,
        amount_out=amount_out,
        price=Decimal(0),
        amount_in_decimals=in_token_decimals,
        amount_out_decimals=out_token_decimals,
        token0=None,
        token1=None,
        lp_fee_paid=None,
    )



def analyse_one_delta_trade(
    web3: Web3,
    *,
    tx_hash: str,
) -> None:
    """Analyse a 1delta trade.
    """
    flash_aggregator = get_deployed_contract(
        web3,
        "1delta/FlashAggregator.json",
        "0x74E95F3Ec71372756a01eB9317864e3fdde1AC53",
    )

    tx = web3.eth.get_transaction(tx_hash)
    # print(tx)
    input_args = get_transaction_data_field(tx)

    if isinstance(input_args, str) and input_args.startswith("0x"):
        data = bytes.fromhex(input_args[2:])
    else:
        data = input_args
    (multicall_payload,) = decode(("bytes[]",), data[4:])

    print("------- Trade details -------")
    for call in multicall_payload:
        selector, params = call[:4], call[4:]

        function = flash_aggregator.get_function_by_selector(selector)
        args = decode_function_args(function, params)
        human_args = humanise_decoded_arg_data(args)
        symbolic_args = []
        for k, v in human_args.items():
            if k == "path":
                path = decode_path(bytes.fromhex(v))
                symbolic_args.append(f"    {k} = {v}")
                symbolic_args.append(f"    decoded path =")
                for i, part in enumerate(path):
                    if i in (0, len(path) - 1):
                        token = fetch_erc20_details(web3, part)
                        symbolic_args.append(f"        {part} ({token.symbol})")
                    elif i == 1:
                        symbolic_args.append(f"        {part} (pool fee: {part/10000}%)")
                    elif i == 2:
                        symbolic_args.append(f"        {part} (exchange: {Exchange(part).name})")
                    elif i == 3:
                        symbolic_args.append(f"        {part} (trade operation)")
            elif k == "asset":
                token = fetch_erc20_details(web3, v)
                symbolic_args.append(f"    {k} = {v} ({token.symbol})")
            else:
                symbolic_args.append(f"    {k} = {v}")
            
        symbolic_args = "\n".join(symbolic_args)
        
        print(f"\n{function.fn_name}:\n{symbolic_args}")

    tx_receipt = web3.eth.get_transaction_receipt(tx_hash)
    if tx_receipt["status"] != 0:
        print("\nThis transaction didn't fail so only debug info is printed")
        return

    # build a new transaction to replay:
    print("\n------- Trying to replay the tx -------")
    replay_tx = {
        "to": tx["to"],
        "from": tx["from"],
        "value": tx["value"],
        "data": input_args,
    }

    try:
        result = web3.eth.call(replay_tx)
        print(f"Replayed result: {result}")
    except Exception as e:
        print(f"Possible reason: {type(e)} {e.args[0]}")

    
