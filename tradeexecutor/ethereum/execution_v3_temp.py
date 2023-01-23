import logging
import datetime
from collections import Counter
from decimal import Decimal
from typing import List, Dict, Set, Tuple

from eth_account.datastructures import SignedTransaction
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.contract import Contract
from web3.logs import DISCARD

from eth_defi.abi import get_deployed_contract
from eth_defi.gas import GasPriceSuggestion, apply_gas, estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.confirmation import wait_transactions_to_complete, \
    broadcast_and_wait_transactions_to_complete, broadcast_transactions
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v2.analysis import TradeSuccess, TradeFail # ok to do this?
from eth_defi.abi import get_deployed_contract, get_contract, get_transaction_data_field
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details

from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.ethereum.execution import get_swap_transactions, TradeExecutionFailed, broadcast, wait_trades_to_complete

logger = logging.getLogger(__name__)


def resolve_trades(
    web3: Web3,
    ts: datetime.datetime,
    state: State,
    tx_map: Dict[HexBytes, Tuple[TradeExecution, BlockchainTransaction]],
    receipts: Dict[HexBytes, dict],
    stop_on_execution_failure=True
):
    """Resolve trade outcome for uniswap_v3 like exchanges.

    Read on-chain Uniswap swap data from the transaction receipt and record how it went.

    Mutates the trade objects in-place.

    :param tx_map:
        tx hash -> (trade, transaction) mapping

    :param receipts:
        tx hash -> receipt object mapping

    :param stop_on_execution_failure:
        Raise an exception if any of the trades failed
    """

    trades: set[TradeExecution] = set()

    # First update the state of all transactions,
    # as we now have receipt for them
    for tx_hash, receipt in receipts.items():
        trade, tx = tx_map[tx_hash.hex()]
        logger.info("Resolved trade %s", trade)
        # Update the transaction confirmation status
        status = receipt["status"] == 1
        reason = None
        if status == 0:
            reason = fetch_transaction_revert_reason(web3, tx_hash)
        tx.set_confirmation_information(
            ts,
            receipt["blockNumber"],
            receipt["blockHash"].hex(),
            receipt.get("effectiveGasPrice", 0),
            receipt["gasUsed"],
            status,
            revert_reason=reason,
        )
        trades.add(trade)

    # Then resolve trade status by analysis the tx receipt
    # if the blockchain transaction was successsful.
    # Also get the actual executed token counts.
    for trade in trades:
        base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, trade.pair.quote.checksum_address)
        reserve = trade.reserve_currency
        swap_tx = get_swap_transactions(trade)
        uniswap = mock_partial_deployment_for_analysis_uniswap_v3(web3, swap_tx.contract_address)

        tx_dict = swap_tx.get_transaction()
        receipt = receipts[HexBytes(swap_tx.tx_hash)]
        result = analyse_trade_by_receipt_uniswap_v3(web3, uniswap, tx_dict, swap_tx.tx_hash, receipt)

        if isinstance(result, TradeSuccess):
            path = [a.lower() for a in result.path]
            if trade.is_buy():
                assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"
                price = 1 / result.price
                executed_reserve = result.amount_in / Decimal(10**quote_token_details.decimals)
                executed_amount = result.amount_out / Decimal(10**base_token_details.decimals)
            else:
                # Ordered other way around
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == reserve.address
                price = result.price
                executed_amount = -result.amount_in / Decimal(10**base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10**quote_token_details.decimals)

            assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}, tx info {trade.tx_info}"

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=0,
                native_token_price=1.0,
            )
        else:
            logger.error("Trade failed %s: %s", ts, trade)
            state.mark_trade_failed(
                ts,
                trade,
            )
            if stop_on_execution_failure:
                success_txs = []
                for tx in trade.blockchain_transactions:
                    if not tx.is_success():
                        raise TradeExecutionFailed(f"Could not execute a trade: {trade}, transaction failed: {tx}, had other transactions {success_txs}")
                    else:
                        success_txs.append(tx)

def broadcast_and_resolve(
        web3: Web3,
        state: State,
        trades: List[TradeExecution],
        confirmation_timeout: datetime.timedelta = datetime.timedelta(minutes=1),
        confirmation_block_count: int=0,
        stop_on_execution_failure=False,
):
    """Do the live trade execution.

    - Push trades to a live blockchain

    - Wait transactions to be mined

    - Based on the transaction result, update the state of the trade if it was success or not

    :param confirmation_block_count:
        How many blocks to wait until marking transaction as confirmed

    :confirmation_timeout:
        Max time to wait for a confirmation.

        We can use zero or negative values to simulate unconfirmed trades.
        See `test_broadcast_failed_and_repair_state`.

    :param stop_on_execution_failure:
        If any of the transactions fail, then raise an exception.
        Set for unit test.
    """

    assert isinstance(confirmation_timeout, datetime.timedelta)

    broadcasted = broadcast(web3, datetime.datetime.utcnow(), trades)

    if confirmation_timeout > datetime.timedelta(0):

        receipts = wait_trades_to_complete(
            web3,
            trades,
            max_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
        )

        resolve_trades(
            web3,
            datetime.datetime.now(),
            state,
            broadcasted,
            receipts,
            stop_on_execution_failure=stop_on_execution_failure)


# move below to eth_defi/uniswap_v3/analysis.py

def mock_partial_deployment_for_analysis_uniswap_v3(web3: Web3, router_address: str):
    """Only need swap_router and PoolContract?"""
    
    factory = None
    swap_router = get_deployed_contract(web3, "uniswap_v3/SwapRouter.json", router_address)
    weth = None
    position_manager = None
    quoter = None
    PoolContract = get_contract(web3, "uniswap_v3/UniswapV3Pool.json")
    return UniswapV3Deployment(
        web3,
        factory,
        weth,
        swap_router,
        position_manager,
        quoter,
        PoolContract,
    )
    
    
def analyse_trade_by_receipt_uniswap_v3(web3: Web3, uniswap: UniswapV3Deployment, tx: dict, tx_hash: str, tx_receipt: dict) -> TradeSuccess | TradeFail:
    """
    """

    pool = uniswap.PoolContract

    # Example tx https://etherscan.io/tx/0xa8e6d47fb1429c7aec9d30332eafaeb515c8dfa73ab413c48560d8d6060c3193#eventlog
    # swapExactTokensForTokens

    router = uniswap.swap_router
    assert tx_receipt["to"] == router.address, f"For now, we can only analyze naive trades to the router. This tx was to {tx_receipt['to']}, router is {router.address}"

    effective_gas_price = tx_receipt.get("effectiveGasPrice", 0)
    gas_used = tx_receipt["gasUsed"]

    # TODO: Unit test this code path
    # Tx reverted
    if tx_receipt["status"] != 1:
        reason = fetch_transaction_revert_reason(web3, tx_hash)
        return TradeFail(gas_used, effective_gas_price, revert_reason=reason)

    # Decode inputs going to the Uniswap swap
    # https://stackoverflow.com/a/70737448/315168
    function, input_args = router.decode_function_input(get_transaction_data_field(tx))
    path = input_args["path"]

    assert function.fn_name == "exactInput", f"Unsupported Uniswap v3 trade function {function}"
    assert len(path), f"Seeing a bad path Uniswap routing {path}"

    amount_in = input_args["amountIn"]
    amount_out_min = input_args["amountOutMin"]

    # Decode the last output.
    # Assume Swap events go in the same chain as path
    swap = pool.events.Swap()

    # The tranasction logs are likely to contain several events like Transfer,
    # Sync, etc. We are only interested in Swap events.
    events = swap.processReceipt(tx_receipt, errors=DISCARD)

    # (AttributeDict({'args': AttributeDict({'sender': '0xDe09E74d4888Bc4e65F589e8c13Bce9F71DdF4c7', 'to': '0x2B5AD5c4795c026514f8317c7a215E218DcCD6cF', 'amount0In': 0, 'amount1In': 500000000000000000000, 'amount0Out': 284881561276680858, 'amount1Out': 0}), 'event': 'Swap', 'logIndex': 4, 'transactionIndex': 0, 'transactionHash': HexBytes('0x58312ff98147ca16c3a81019c8bca390cd78963175e4c0a30643d45d274df947'), 'address': '0x68931307eDCB44c3389C507dAb8D5D64D242e58f', 'blockHash': HexBytes('0x1222012923c7024b1d49e1a3e58552b89e230f8317ac1b031f070c4845d55db1'), 'blockNumber': 12}),)
    amount0_out = events[-1]["args"]["amount0Out"]
    amount1_out = events[-1]["args"]["amount1Out"]

    # Depending on the path, the out token can pop up as amount0Out or amount1Out
    # For complex swaps (unspported) we can have both
    assert amount0_out == 0 or amount1_out == 0, "Unsupported swap type"

    amount_out = amount0_out if amount0_out > 0 else amount1_out

    in_token_details = fetch_erc20_details(web3, path[0])
    out_token_details = fetch_erc20_details(web3, path[-1])

    amount_out_cleaned = Decimal(amount_out) / Decimal(10**out_token_details.decimals)
    amount_in_cleaned = Decimal(amount_in) / Decimal(10**in_token_details.decimals)

    price = amount_out_cleaned / amount_in_cleaned

    return TradeSuccess(
        gas_used,
        effective_gas_price,
        path,
        amount_in,
        amount_out_min,
        amount_out,
        price,
        in_token_details.decimals,
        out_token_details.decimals,
    )