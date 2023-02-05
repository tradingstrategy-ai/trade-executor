"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
from typing import List, Dict, Tuple
import logging

from web3 import Web3
from hexbytes import HexBytes


from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.analysis import TradeSuccess
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from eth_defi.uniswap_v3.analysis import mock_partial_deployment_for_analysis, analyse_trade_by_receipt


from tradeexecutor.ethereum.execution import TradeExecutionFailed, get_swap_transactions
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.execution_model import ExecutionModel


logger = logging.getLogger(__name__)


class UniswapV3ExecutionModel(ExecutionModel):
    """Run order execution on a single Uniswap v2 style exchanges."""

    def __init__(self,
                 web3: Web3,
                 hot_wallet: HotWallet,
                 min_balance_threshold=Decimal("0.5"),
                 confirmation_block_count=6,
                 confirmation_timeout=datetime.timedelta(minutes=5),
                 max_slippage: float = 0.01,
                 stop_on_execution_failure=True,
                 swap_gas_fee_limit=2_000_000):
        """
        :param web3:
            Web3 connection used for this instance

        :param hot_wallet:
            Hot wallet instance used for this execution

        :param min_balance_threshold:
            Abort execution if our hot wallet gas fee balance drops below this

        :param confirmation_block_count:
            How many blocks to wait for the receipt confirmations to mitigate unstable chain tip issues

        :param confirmation_timeout:
            How long we wait transactions to clear

        :param stop_on_execution_failure:
            Raise an exception if any of the trades fail top execute

        :param max_slippage:
            Max slippage tolerance per trade. 0.01 is 1%.
        """
        super().__init__(
            web3,
            hot_wallet,
            min_balance_threshold,
            confirmation_block_count,
            confirmation_timeout,
            max_slippage,
            stop_on_execution_failure,
            swap_gas_fee_limit
        )

    def repair_unconfirmed_trades(self, state: State) -> List[TradeExecution]:
        """Repair unconfirmed trades.

        Repair trades that failed to properly broadcast or confirm due to
        blockchain node issues.
        """
        return super().repair_unconfirmed_trades(state, resolve_trades)

def get_current_price(web3: Web3, uniswap: UniswapV3Deployment, pair: TradingPairIdentifier, quantity=Decimal(1)) -> float:
    """Get a price from Uniswap v3 pool, assuming you are selling 1 unit of base token.
    See see eth_defi.uniswap_v2.fees.estimate_sell_price_decimals
    
    Does decimal adjustment.
    :return: Price in quote token.
    """
    
    quantity_raw = pair.base.convert_to_raw_amount(quantity)
    
    path = [pair.base.checksum_address,  pair.quote.checksum_address] 
    fees = [pair.fee]
    assert fees, "no fees in pair"        
        
    price_helper = UniswapV3PriceHelper(uniswap)
    out_raw = price_helper.get_amount_out(
        amount_in=quantity_raw,
        path=path,
        fees=fees
    )
    
    return float(pair.quote.convert_to_decimal(out_raw))

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
            path = [a.lower() for a in result.path if type(a) == str]
            
            if trade.is_buy():
                assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"
                executed_reserve = result.amount_in / Decimal(10**quote_token_details.decimals)
                executed_amount = result.amount_out / Decimal(10**base_token_details.decimals)
            else:
                # Ordered other way around
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == reserve.address
                executed_amount = -result.amount_in / Decimal(10**base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10**quote_token_details.decimals)

            price = 1/(1/Decimal(result.price) * Decimal(10**quote_token_details.decimals) / Decimal(10**base_token_details.decimals))
            
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

