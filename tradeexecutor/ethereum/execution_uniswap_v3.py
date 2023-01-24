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
        uniswap = mock_partial_deployment_for_analysis(web3, swap_tx.contract_address)

        tx_dict = swap_tx.get_transaction()
        receipt = receipts[HexBytes(swap_tx.tx_hash)]
        result = analyse_trade_by_receipt(web3, uniswap, tx_dict, swap_tx.tx_hash, receipt)

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

def mock_partial_deployment_for_analysis(web3: Web3, router_address: str):
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

def get_input_args(params: tuple) -> dict:
    """Names and decodes input arguments from router.decode_function_input()
    Note there is no support yet for SwapRouter02, it does not accept a deadline parameter
    See: https://docs.uniswap.org/contracts/v3/reference/periphery/interfaces/ISwapRouter#exactinputparams
    
    :params:
    params from router.decode_function_input
    
    :returns:
    Dict of exactInputParams as specified in the link above
    """
    
    full_path_decoded = decode_path(params[0])
    
    # TODO: add support for SwapRouter02 which does not accept deadline parameter
    return {
        "path": full_path_decoded,
        "recipient": params[1],
        "deadline": params[2],
        "amountIn": params[3],
        "amountOutMinimum": params[4]
    }
    
def decode_path(full_path_encoded: str) -> list:
    """Decodes the path. A bit tricky. Thanks to https://degencode.substack.com/p/project-uniswapv3-mempool-watcher
    
    :param full_path_encoded:
    Encoded path as returned from router.decode_function_input
    
    :returns:
    fully decoded path including addresses and fees
    """
    
    path_pos = 0
    full_path_decoded = []
    # read alternating 20 and 3 byte chunks from the encoded path,
    # store each address (hex) and fee (int)
    
    byte_length = 20
    while True:
        # stop at the end
        if path_pos == len(full_path_encoded):
            break
        elif (
            byte_length == 20
            and len(full_path_encoded)
            >= path_pos + byte_length
        ):
            address = full_path_encoded[
                path_pos : path_pos + byte_length
            ].hex()
            full_path_decoded.append(f"0x{Web3.toChecksumAddress(address)}")
        elif (
            byte_length == 3
            and len(full_path_encoded)
            >= path_pos + byte_length
        ):
            fee = int(
                full_path_encoded[
                    path_pos : path_pos + byte_length
                ].hex(),
                16,
            )
            full_path_decoded.append(fee)
        else:
            raise IndexError("Bad path")
        
        path_pos += byte_length
        byte_length = 3 if byte_length == 20 else 20
        
    return full_path_decoded

# TODO: delete  
# def hex_to_path(path_str: str):
#     """Ethereum Wallet Address is a distinct alphanumeric crypto identifier that contains 42 hexadecimal characters that start with 0x and is followed by a series of 40 random characters which can send transactions and has a balance in it. 
    
#     From: geeksforgeeks.org/how-to-create-an-ethereum-wallet-address-from-a-private-key/"""
    
#     path = []
#     start = 0

#     # should always be 42, but just in case
#     start, stop = (
#         (2, 42) 
#         if path_str.startswith('0x') 
#         else (0, 40)
#     )

#     while (stop <= len(path_str)):
#         address = f"0x{path_str[start:stop]}"

#         path.append(address)
#         start += 40
#         stop += 40
    
#     return path
    
    
def analyse_trade_by_receipt(web3: Web3, uniswap: UniswapV3Deployment, tx: dict, tx_hash: str, tx_receipt: dict) -> TradeSuccess | TradeFail:
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
    function, params_struct = router.decode_function_input(get_transaction_data_field(tx))
    input_args = get_input_args(params_struct["params"])
    
    path = input_args["path"]

    assert function.fn_name == "exactInput", f"Unsupported Uniswap v3 trade function {function}"
    assert len(path), f"Seeing a bad path Uniswap routing {path}"

    amount_in = input_args["amountIn"]
    amount_out_min = input_args["amountOutMinimum"]

    # Decode the last output.
    # Assume Swap events go in the same chain as path
    swap = pool.events.Swap()

    # The tranasction logs are likely to contain several events like Transfer,
    # Sync, etc. We are only interested in Swap events.
    events = swap.processReceipt(tx_receipt, errors=DISCARD)

    # AttributeDict({'args': AttributeDict({'sender': '0x6D411e0A54382eD43F02410Ce1c7a7c122afA6E1', 'recipient': '0xC2c2C1C8871C189829d3CCD169010F430275BC70', 'amount0': -292184487391376249, 'amount1': 498353865, 'sqrtPriceX96': 3267615572280113943555521, 'liquidity': 41231056256176602, 'tick': -201931}), 'event': 'Swap', 'logIndex': 3, 'transactionIndex': 0, 'transactionHash': HexBytes('0xe7fff8231effe313010aed7d973fdbe75f58dc4a59c187b230e3fc101c58ec97'), 'address': '0x4529B3F2578Bf95c1604942fe1fCDeB93F1bb7b6', 'blockHash': HexBytes('0xe06feb724020c57c6a0392faf7db29fedf4246ce5126a5b743b2627b7dc69230'), 'blockNumber': 24})
    
    # See https://docs.uniswap.org/contracts/v3/reference/core/interfaces/pool/IUniswapV3PoolEvents#swap
    
    temp = events[-1]["args"] # TODO delete
    amount0 = events[-1]["args"]["amount0"]
    amount1 = events[-1]["args"]["amount1"]

    # Depending on the path, the out token can pop up as amount0Out or amount1Out
    # For complex swaps (unspported) we can have both
    assert (amount0 > 0 and amount1 < 0) or (amount0 < 0 and amount1 > 0), "Unsupported swap type"

    amount_out = amount0 if amount0 > 0 else amount1

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