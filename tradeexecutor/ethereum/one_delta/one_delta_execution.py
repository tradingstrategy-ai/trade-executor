"""Execution model where trade happens on 1delta, utilizing Aave v3 and Uniswap v3."""

import datetime
from decimal import Decimal
import logging

from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.analysis import TradeSuccess, TradeFail
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper, estimate_sell_received_amount
from eth_defi.uniswap_v3.analysis import analyse_trade_by_receipt
from eth_defi.uniswap_v3.deployment import mock_partial_deployment_for_analysis
from tradeexecutor.ethereum.tx import TransactionBuilder

from tradeexecutor.state.identifier import TradingPairIdentifier
#from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.ethereum.execution import EthereumExecutionModel

logger = logging.getLogger(__name__)


class OneDeltaExecutionModel(EthereumExecutionModel):
    """Run order execution on a single Uniswap v3 style exchanges."""

    def __init__(self,
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

    def analyse_trade_by_receipt(
        self,
        web3: Web3, 
        uniswap: UniswapV3Deployment, 
        tx: dict, 
        tx_hash: str,
        tx_receipt: dict,
        input_args: tuple,
    ) -> (TradeSuccess | TradeFail):
        # assert type(input_args) == tuple and len(input_args) > 0, "Uniswap v3 trade success analysis needs input args"
        # return analyse_trade_by_receipt(web3, uniswap, tx, tx_hash, tx_receipt, input_args)
        return None

    # def mock_partial_deployment_for_analysis(
    #     self,
    #     web3: Web3, 
    #     router_address: str
    # ) -> UniswapV3Deployment:
    #     return mock_partial_deployment_for_analysis(web3, router_address)
    

def get_current_price(web3: Web3, uniswap: UniswapV3Deployment, pair: TradingPairIdentifier, quantity=Decimal(1)) -> float:
    """Get a price from Uniswap v3 pool, assuming you are selling 1 unit of base token.
    
    Does decimal adjustment.
    
    :return: Price in quote token.
    """
    
    quantity_raw = pair.base.convert_to_raw_amount(quantity)
    
    out_raw = estimate_sell_received_amount(
        uniswap=uniswap,
        base_token_address=pair.base.checksum_address,
        quote_token_address=pair.quote.checksum_address,
        quantity=quantity_raw,
        target_pair_fee=int(pair.fee * 1_000_000),        
    )

    return float(pair.quote.convert_to_decimal(out_raw))

