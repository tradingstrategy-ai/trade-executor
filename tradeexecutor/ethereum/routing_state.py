"""Route trades to different Uniswap v2 like exchanges."""

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Set, List, Optional, Tuple

from eth_typing import HexAddress, ChecksumAddress

from tradeexecutor.state.types import BPS
from tradingstrategy.chain import ChainId
from web3 import Web3
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.gas import estimate_gas_fees
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment as fetch_deployment_v3
from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from eth_defi.uniswap_v3.swap import swap_with_slippage_protection as swap_with_slippage_protection_v3
from web3.exceptions import ContractLogicError

from tradeexecutor.ethereum.execution import get_token_for_asset
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState, CannotRouteTrade
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair, \
    translate_token
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


logger = logging.getLogger(__name__)


class OutOfBalance(Exception):
    """Did not have enough tokens"""


class EthereumRoutingStateBase(RoutingState):
    """Manage transaction building for multiple Uniswap trades.

    - Lifespan is one rebalance - remembers already made approvals

    - Web3 connection and hot wallet

    - Approval tx creation

    - Swap tx creation

    Manage the state of already given approvals here,
    so that we do not do duplicates.

    The approvals are not persistent in the executor state,
    but are specific for each cycle.
    """

    def __init__(self,
                 pair_universe: PandasPairUniverse,
                 tx_builder: Optional[TransactionBuilder]=None,
                 swap_gas_limit=2_000_000):
        """

        :param pair_universe:
            Pairs we trade

        :param tx_builder:
            For creating trade transactions.

            Can be set to None on DummyExecutionModel.

        :param swap_gas_limit:
            What is the max gas we are willing to pay for a swap.

        """
        self.pair_universe = pair_universe
        self.tx_builder = tx_builder
        self.hot_wallet = tx_builder.hot_wallet
        self.web3 = self.tx_builder.web3
        # router -> erc-20 mappings
        self.approved_routes = defaultdict(set)
        self.swap_gas_limit = swap_gas_limit

    def is_route_approved(self, router_address: str):
        return router_address in self.approved_routes

    def mark_router_approved(self, token_address, router_address):
        self.approved_routes[router_address].add(token_address)

    def is_approved_on_chain(self, token_address: str, router_address: str) -> bool:
        erc_20 = get_deployed_contract(self.web3, "ERC20MockDecimals.json", token_address)
        # Assume allowance is always infinity
        return erc_20.functions.allowance.call(self.hot_wallet.address, router_address) > 0

    def check_has_enough_tokens(
            self,
            erc_20: Contract,
            amount: int,
    ):
        """Check we have enough buy side tokens to do a trade.

        This might not be the case if we are preparing transactions ahead of time and
        sell might have not happened yet.
        """
        balance = erc_20.functions.balanceOf(self.hot_wallet.address).call()
        if balance < amount:
            token_details = fetch_erc20_details(
                erc_20.web3,
                erc_20.address,
            )
            d_balance = token_details.convert_to_decimals(balance)
            d_amount = token_details.convert_to_decimals(amount)
            raise OutOfBalance(f"Address {self.hot_wallet.address} does not have enough {token_details} tokens to trade. Need {d_amount}, has {d_balance}")

    def ensure_token_approved(self,
                              token_address: str,
                              router_address: str) -> List[BlockchainTransaction]:
        """Make sure we have ERC-20 approve() for the trade

        - Infinite approval on-chain

        - ...or previous approval in this state,

        :param token_address:
        :param router_address:

        :return: Create 0 or 1 transactions if needs to be approved
        """

        if token_address in self.approved_routes[router_address]:
            # Already approved for this cycle in previous trade
            return []

        erc_20 = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(token_address))

        # Set internal state we are approved
        self.mark_router_approved(token_address, router_address)

        hot_wallet = self.tx_builder.hot_wallet

        if erc_20.functions.allowance(hot_wallet.address, router_address).call() > 0:
            # already approved in previous execution cycle
            return []
        
        # Create infinite approval
        tx = self.tx_builder.create_transaction(
            erc_20,
            "approve",
            (router_address, 2**256-1),
            100_000,  # For approve, assume it cannot take more than 100k gas
        )

        return [tx]
    
    def get_base_quote_intermediary(self, target_pair, intermediary_pair, reserve_asset):
        
        web3 = self.web3
        
        if reserve_asset == intermediary_pair.quote:
            # Buy BUSD -> BNB -> Cake
            base_token = get_token_for_asset(web3, target_pair.base)
            quote_token = get_token_for_asset(web3, intermediary_pair.quote)
            intermediary_token = get_token_for_asset(web3, intermediary_pair.base)
        elif reserve_asset == target_pair.base:
            # Sell, Cake -> BNB -> BUSD
            base_token = get_token_for_asset(web3, intermediary_pair.quote)  # BUSD
            quote_token = get_token_for_asset(web3, target_pair.base)  # Cake
            intermediary_token = get_token_for_asset(web3, intermediary_pair.base)  # BNB
        else:
            raise RuntimeError(f"Cannot trade {target_pair} through {intermediary_pair}")
        return base_token,quote_token,intermediary_token
    
    def get_base_and_quote(self, target_pair, reserve_asset):
        """Get base and quote token from the pair and reserve asset. 
        
        See: https://tradingstrategy.ai/docs/programming/market-data/trading-pairs.html
        
        :param target_pair: Pair to be traded
        :param reserver_asset: Asset to be kept as reserves
        :returns: (base_token: Contract, quote_token: Contract)
        """
        web3 = self.web3
        
        if reserve_asset == target_pair.quote:
            # Buy with e.g. BUSD
            base_token = get_token_for_asset(web3, target_pair.base)
            quote_token = get_token_for_asset(web3, target_pair.quote)
        elif reserve_asset == target_pair.base:
            # Sell, flip the direction
            base_token = get_token_for_asset(web3, target_pair.quote)
            quote_token = get_token_for_asset(web3, target_pair.base)
        else:
            raise RuntimeError(f"Cannot trade {target_pair}")
        return base_token,quote_token
    
    def get_signed_tx(self, swap_func, gas_limit):
        signed_tx = self.tx_builder.sign_transaction(
            swap_func, gas_limit
        )
        return [signed_tx]
    
    @staticmethod
    def validate_pairs(target_pair, intermediary_pair):
        """Check we can chain two pairs
        """
        assert intermediary_pair.base == target_pair.quote, f"Could not hop from intermediary {intermediary_pair} -> destination {target_pair}"

        assert target_pair.exchange_address, f"Target pair {target_pair} missing exchange information"
        assert intermediary_pair.exchange_address, f"Intermediary pair {intermediary_pair} missing exchange information"

        # Check routing happens on the same exchange
        assert intermediary_pair.exchange_address.lower() == target_pair.exchange_address.lower()
        
    @staticmethod
    def validate_exchange(target_pair, intermediary_pair):
        """Check routing happens on the same exchange"""
        assert intermediary_pair.exchange_address.lower() == target_pair.exchange_address.lower()
    
    
def route_tokens(
        trading_pair: TradingPairIdentifier,
        intermediate_pair: Optional[TradingPairIdentifier],
)-> Tuple[ChecksumAddress, ChecksumAddress, Optional[ChecksumAddress]]:
    """Convert trading pair route to physical token addresses.
    """

    if intermediate_pair is None:
        return (Web3.toChecksumAddress(trading_pair.base.address),
            Web3.toChecksumAddress(trading_pair.quote.address),
            None)

    return (Web3.toChecksumAddress(trading_pair.base.address),
        Web3.toChecksumAddress(intermediate_pair.quote.address),
        Web3.toChecksumAddress(trading_pair.quote.address))
