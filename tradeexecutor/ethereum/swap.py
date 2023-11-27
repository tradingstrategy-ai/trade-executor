"""Post-trade execution analysis for Uniswap v2 and v3"""

from typing import Callable

from hexbytes import HexBytes
from web3 import Web3

from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from tradeexecutor.ethereum.ethereum_execution_model import get_swap_transactions
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution


def analyse_swap(
    web3: Web3,
    state: State,
    trade: TradeExecution,
    uniswap: UniswapV2Deployment | UniswapV3Deployment,
    analyse_trade_by_receipt: Callable,
):
    base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
    quote_token_details = fetch_erc20_details(web3, trade.pair.quote.checksum_address)
    reserve = trade.reserve_currency

    swap_tx = get_swap_transactions(trade)

    tx_dict = swap_tx.get_transaction()
    receipt = receipts[HexBytes(swap_tx.tx_hash)]

    input_args = swap_tx.get_actual_function_input_args()

    result = .analyse_trade_by_receipt(
        web3,
        deployment=uniswap,
        tx=tx_dict,
        tx_hash=swap_tx.tx_hash,
        tx_receipt=receipt,
        input_args=input_args,
        pair_fee=trade.pair.fee,
    )

    if isinstance(result, TradeSuccess):

        # v3 path includes fee (int) as well
        path = [a.lower() for a in result.path if type(a) == str]

        if trade.is_buy():
            assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"

            if self.is_v3():
                price = result.get_human_price(quote_token_details.address == result.token0.address)
            else:
                price = 1 / result.price

            executed_reserve = result.amount_in / Decimal(10 ** reserve.decimals)
            executed_amount = result.amount_out / Decimal(10 ** base_token_details.decimals)

            # lp fee is already in terms of quote token
            lp_fee_paid = result.lp_fee_paid
        else:
            # Ordered other way around
            assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
            assert path[-1] == reserve.address

            if self.is_v3():
                price = result.get_human_price(quote_token_details.address == result.token0.address)
            else:
                price = result.price

            executed_amount = -result.amount_in / Decimal(10 ** base_token_details.decimals)
            executed_reserve = result.amount_out / Decimal(10 ** reserve.decimals)

            # convert lp fee to be in terms of quote token
            lp_fee_paid = result.lp_fee_paid * float(price) if result.lp_fee_paid else None

        assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}, tx info {trade.tx_info}"

        # Mark as success
        state.mark_trade_success(
            ts,
            trade,
            executed_price=float(price),
            executed_amount=executed_amount,
            executed_reserve=executed_reserve,
            lp_fees=lp_fee_paid,
            native_token_price=0,  # won't fix
            cost_of_gas=result.get_cost_of_gas(),
        )
    else:
        # Trade failed
        report_failure(ts, state, trade, stop_on_execution_failure)
