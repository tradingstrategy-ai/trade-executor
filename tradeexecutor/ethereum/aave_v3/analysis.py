from web3 import Web3
from web3.logs import DISCARD
from decimal import Decimal

from eth_defi.one_delta.constants import TradeOperation, Exchange
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeFail, TradeSuccess
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
    """Analyse an Aave v3 credit supply trade.

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

    pool = aave_v3_deployment.pool

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
