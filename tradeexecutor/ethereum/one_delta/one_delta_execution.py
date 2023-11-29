"""Execution model where trade happens on 1delta, utilizing Aave v3 and Uniswap v3."""

import datetime
from decimal import Decimal
import logging

from web3 import Web3
from eth_defi.uniswap_v2.analysis import TradeSuccess, TradeFail
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment


from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.execution import EthereumExecutionModel


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

