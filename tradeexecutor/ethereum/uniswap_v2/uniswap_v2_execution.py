"""Execution model where trade happens directly on Uniswap v2 style exchange."""

import logging
import datetime
from decimal import Decimal

from web3 import Web3

from eth_defi.trade import TradeFail
from eth_defi.uniswap_v2.fees import estimate_sell_price_decimals
from eth_defi.uniswap_v2.analysis import TradeSuccess, analyse_trade_by_receipt
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, mock_partial_deployment_for_analysis

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier

logger = logging.getLogger(__name__)


class UniswapV2Execution(EthereumExecution):
    """Run order execution on a single Uniswap v2 style exchanges."""

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
        :param web3:
            Web3 connection used for this instance

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
        super().__init__(
            tx_builder=tx_builder,
            min_balance_threshold=min_balance_threshold,
            confirmation_block_count=confirmation_block_count,
            confirmation_timeout=confirmation_timeout,
            max_slippage=max_slippage,
            stop_on_execution_failure=stop_on_execution_failure,
            swap_gas_fee_limit=swap_gas_fee_limit,
            mainnet_fork=mainnet_fork,
        )

    def analyse_trade_by_receipt(
        self,
        web3: Web3,
        *,
        deployment: UniswapV2Deployment, 
        tx: dict, 
        tx_hash: str,
        tx_receipt: dict,
        input_args: tuple | None = None,
        pair_fee: float | None = None,
    ) -> (TradeSuccess | TradeFail):
        return analyse_trade_by_receipt(web3, deployment, tx, tx_hash, tx_receipt, pair_fee)

    def mock_partial_deployment_for_analysis(
        self,
        web3: Web3,
        router_address: str
    ) -> UniswapV2Deployment:
        return mock_partial_deployment_for_analysis(web3, router_address)

    def is_v3(self) -> bool:
        """Returns true if instance is related to Uniswap V3, else false. 
        Kind of a hack to be able to share resolve trades function amongst v2 and v3."""
        return False


def get_current_price(web3: Web3, uniswap: UniswapV2Deployment, pair: TradingPairIdentifier, quantity=Decimal(1)) -> float:
    """Get a price from Uniswap v2 pool, assuming you are selling 1 unit of base token.

    Does decimal adjustment.

    :return: Price in quote token.
    """
    price = estimate_sell_price_decimals(uniswap, pair.base.checksum_address, pair.quote.checksum_address, quantity)
    return float(price)

