"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import datetime
from decimal import Decimal
import logging

from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.trade import TradeSuccess, TradeFail

#from tradeexecutor.strategy.execution_model import ExecutionModel
from tradeexecutor.ethereum.execution import EthereumExecutionModel


import logging
import datetime
from decimal import Decimal
from typing import List, Dict, Set, Tuple

from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3


from eth_defi.hotwallet import HotWallet
from eth_defi.revert_reason import fetch_transaction_revert_reason
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.fees import estimate_sell_price_decimals
from eth_defi.uniswap_v2.analysis import TradeSuccess, analyse_trade_by_receipt
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, mock_partial_deployment_for_analysis
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier


logger = logging.getLogger(__name__)


class UniswapV2ExecutionModel(EthereumExecutionModel):
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

    @staticmethod
    def analyse_trade_by_receipt(
        web3: Web3, 
        uniswap: UniswapV2Deployment, 
        tx: dict, 
        tx_hash: str,
        tx_receipt: dict
    ) -> (TradeSuccess | TradeFail):
        return analyse_trade_by_receipt(web3, uniswap, tx, tx_hash, tx_receipt)
    
    @staticmethod
    def mock_partial_deployment_for_analysis(
        web3: Web3,
        router_address: str
    ) -> UniswapV2Deployment:
        return mock_partial_deployment_for_analysis(web3, router_address)
    

def get_current_price(web3: Web3, uniswap: UniswapV2Deployment, pair: TradingPairIdentifier, quantity=Decimal(1)) -> float:
    """Get a price from Uniswap v2 pool, assuming you are selling 1 unit of base token.

    Does decimal adjustment.

    :return: Price in quote token.
    """
    price = estimate_sell_price_decimals(uniswap, pair.base.checksum_address, pair.quote.checksum_address, quantity)
    return float(price)

