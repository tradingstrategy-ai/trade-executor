"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
import logging

from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.analysis import TradeSuccess, TradeFail
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper
from eth_defi.uniswap_v3.analysis import analyse_trade_by_receipt
from eth_defi.uniswap_v3.deployment import mock_partial_deployment_for_analysis
from tradeexecutor.ethereum.tx import TransactionBuilder

from tradeexecutor.state.identifier import TradingPairIdentifier
#from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.ethereum.execution import EthereumExecutionModel

logger = logging.getLogger(__name__)


class UniswapV3ExecutionModel(EthereumExecutionModel):
    """Run order execution on a single Uniswap v3 style exchanges."""

    def __init__(self,
                 tx_builder: TransactionBuilder,
                 min_balance_threshold=Decimal("0.5"),
                 confirmation_block_count=6,
                 confirmation_timeout=datetime.timedelta(minutes=5),
                 max_slippage: float = 0.01,
                 stop_on_execution_failure=True,
                 swap_gas_fee_limit=2_000_000):
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
        )

    @staticmethod
    def analyse_trade_by_receipt(
        web3: Web3, 
        uniswap: UniswapV3Deployment, 
        tx: dict, 
        tx_hash: str,
        tx_receipt: dict
    ) -> (TradeSuccess | TradeFail):
        return analyse_trade_by_receipt(web3, uniswap, tx, tx_hash, tx_receipt)
    
    @staticmethod
    def mock_partial_deployment_for_analysis(
        web3: Web3, 
        router_address: str
    ) -> UniswapV3Deployment:
        return mock_partial_deployment_for_analysis(web3, router_address)
    
    @staticmethod
    def is_v3() -> bool:
        """Returns true if instance is related to Uniswap V3, else false. 
        Kind of a hack to be able to share resolve trades function amongst v2 and v3."""
        return True
    

def get_current_price(web3: Web3, uniswap: UniswapV3Deployment, pair: TradingPairIdentifier, quantity=Decimal(1)) -> float:
    """Get a price from Uniswap v3 pool, assuming you are selling 1 unit of base token.
    See see eth_defi.uniswap_v2.fees.estimate_sell_price_decimals
    
    Does decimal adjustment.
    :return: Price in quote token.
    """
    
    quantity_raw = pair.base.convert_to_raw_amount(quantity)
    
    path = [pair.base.checksum_address,  pair.quote.checksum_address] 
    raw_fees = [int(pair.fee * 1_000_000)]
    assert raw_fees, "no fees in pair"        
        
    price_helper = UniswapV3PriceHelper(uniswap)
    out_raw = price_helper.get_amount_out(
        amount_in=quantity_raw,
        path=path,
        fees=raw_fees
    )
    
    return float(pair.quote.convert_to_decimal(out_raw))

