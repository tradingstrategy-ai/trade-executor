"""Execution model where trade happens on 1delta, utilizing Aave v3 and Uniswap v3."""

import datetime
from decimal import Decimal
import logging

from web3 import Web3
from eth_typing import HexAddress, HexStr
from hexbytes import HexBytes

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.analysis import TradeSuccess, TradeFail
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper, estimate_sell_received_amount
from eth_defi.uniswap_v3.deployment import mock_partial_deployment_for_analysis
from eth_defi.token import fetch_erc20_details, TokenDetails
from eth_defi.one_delta.deployment import OneDeltaDeployment, fetch_deployment
from eth_defi.one_delta.constants import TradeOperation

from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction, BlockchainTransactionType
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.ethereum.execution import EthereumExecutionModel, get_swap_transactions

from tradeexecutor.ethereum.one_delta.analysis import analyse_trade_by_receipt
from tradeexecutor.strategy.lending_protocol_leverage import create_short_loan, plan_loan_update_for_short, create_credit_supply_loan, update_credit_supply_loan

logger = logging.getLogger(__name__)


class OneDeltaExecutionModel(EthereumExecutionModel):
    """Run order execution on 1delta."""

    def __init__(
        self,
        tx_builder: TransactionBuilder,
        min_balance_threshold=Decimal("0.5"),
        confirmation_block_count=6,
        confirmation_timeout=datetime.timedelta(minutes=5),
        max_slippage: float = 0.01,
        stop_on_execution_failure=True,
        swap_gas_fee_limit=2_000_000,
        mainnet_fork=False,
    ):
        """
        :param tx_builder:
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
        assert isinstance(tx_builder, TransactionBuilder), f"Got: {tx_builder}"
        super().__init__(
            tx_builder,
            min_balance_threshold,
            confirmation_block_count,
            confirmation_timeout,
            max_slippage,
            stop_on_execution_failure,
            mainnet_fork=mainnet_fork,
        )

    def analyse_trade_by_receipt(self, *args, **kwargs) -> (TradeSuccess | TradeFail):
        pass

    def is_v3(self) -> bool:
        return True

    def mock_partial_deployment_for_analysis(
        self,
        web3: Web3, 
        router_address: str,
    ) -> UniswapV3Deployment:
        pass

    def resolve_trades(
        self,
        ts: datetime.datetime,
        state: State,
        tx_map: dict[HexStr, tuple[TradeExecution, BlockchainTransaction]],
        receipts: dict[HexBytes, dict],
        stop_on_execution_failure=True,
    ):
        """Resolve trade outcome.

        Read on-chain Uniswap swap data from the transaction receipt and record how it went.

        Mutates the trade objects in-place.

        :param tx_map:
            tx hash -> (trade, transaction) mapping

        :param receipts:
            tx hash -> receipt object mapping

        :param stop_on_execution_failure:
            Raise an exception if any of the trades failed"""

        web3 = self.web3

        trades = self.update_confirmation_status(ts, tx_map, receipts)

        # Then resolve trade status by analysis the tx receipt
        # if the blockchain transaction was successsful.
        # Also get the actual executed token counts.
        for trade in trades:
            pricing_pair = trade.pair.get_pricing_pair()
            base_token_details = fetch_erc20_details(web3, pricing_pair.base.checksum_address)
            quote_token_details = fetch_erc20_details(web3, pricing_pair.quote.checksum_address)
            reserve = trade.reserve_currency
            tx = get_swap_transactions(trade)
            one_delta = fetch_deployment(web3, tx.contract_address, tx.contract_address)
            # TODO: this router address is wrong, but it doesn't matter since we don't use it here
            uniswap = mock_partial_deployment_for_analysis(web3, tx.contract_address)

            tx_dict = tx.get_transaction()
            receipt = receipts[HexBytes(tx.tx_hash)]

            input_args = tx.get_actual_function_input_args()

            result = analyse_trade_by_receipt(
                web3,
                one_delta=one_delta,
                uniswap=uniswap,
                tx=tx_dict,
                tx_hash=tx.tx_hash,
                tx_receipt=receipt,
                input_args=input_args,
                trade_operation=TradeOperation.OPEN if trade.is_sell() else TradeOperation.CLOSE,
            )

            if isinstance(result, TradeSuccess):
                # path in 1delta is always from base -> quote
                assert result.path[0].lower() == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert result.path[-1].lower() == reserve.address.lower()
                
                price = result.get_human_price(quote_token_details.address == result.token0.address)

                # TODO: verify these numbers
                if trade.is_buy():
                    executed_amount = -result.amount_in / Decimal(10**base_token_details.decimals)
                    executed_reserve = result.amount_out / Decimal(10**reserve.decimals)
                else:
                    executed_amount = result.amount_in / Decimal(10**base_token_details.decimals)
                    executed_reserve = result.amount_out / Decimal(10**reserve.decimals)

                lp_fee_paid = result.lp_fee_paid

                assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}, tx info {trade.tx_info}"

                # update the executed loan
                # TODO: check if this is the right spot for this
                trade.executed_loan_update = trade.planned_loan_update

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
