"""Route trades to different Uniswap v2 like exchanges."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Tuple

from eth_typing import HexAddress

from eth_defi.abi import get_deployed_contract
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from tradeexecutor.ethereum.execution import get_token_for_asset
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.routing import RoutingModel


class UniswapV2RoutingState:
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

    def __init__(self, tx_builder: TransactionBuilder, swap_gas_limit=2_000_000):
        self.tx_builder = tx_builder
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

    def get_uniswap_for_pair(self, factory_router_map: dict, target_pair: TradingPairIdentifier) -> UniswapV2Deployment:
        """Get a router for a trading pair."""
        assert target_pair.exchange_address, f"Exchange address missing for {target_pair}"
        factory_address = target_pair.exchange_address
        router_address, init_code_hash = factory_router_map[factory_address.lower()]
        return fetch_deployment(
            self.web3,
            factory_address,
            router_address,
            init_code_hash=init_code_hash,
        )

    def ensure_token_approved(self, token_address: str, router_address: str) -> List[BlockchainTransaction]:
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

        erc_20 = get_deployed_contract(self.web3, "ERC20MockDecimals.json", token_address)

        # Set internal state we are approved
        self.mark_router_approved(token_address, router_address)

        hot_wallet = self.tx_builder.hot_wallet

        if erc_20.functions.allowance(hot_wallet.address, router_address).call() > 0:
            # already approved in previous execution cycle
            return []
        else:
            # Create infinite approval
            tx = self.tx_builder.create_transaction(
                erc_20,
                "approve",
                (router_address, 2**256-1),
                100_000,  # For approve, assume it cannot take more than 100k gas
            )

            return [tx]

    def trade_on_router_two_way(self,
            uniswap: UniswapV2Deployment,
            target_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: int,
            max_slippage: float):
        """Prepare the actual swap"""

        web3 = self.web3
        hot_wallet = self.tx_builder.hot_wallet

        if reserve_asset == target_pair.quote:
            # Buy with e.g. BUSD
            base_token = get_token_for_asset(web3, target_pair.base)
            quote_token = get_token_for_asset(web3, target_pair.quote)
        else:
            raise RuntimeError(f"Cannot trade {target_pair}")

        bound_swap_func = swap_with_slippage_protection(
            uniswap,
            recipient_address=hot_wallet.address,
            base_token=base_token,
            quote_token=quote_token,
            amount_in=reserve_amount,
            max_slippage=max_slippage,
        )

        # Create infinite approval
        tx = self.tx_builder.sign_transaction(bound_swap_func, self.swap_gas_limit)
        return [tx]


class UniswapV2SimpleRoutingModel(RoutingModel):
    """A simple router that does not optimise the trade execution cost.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(self,
                 factory_router_map: Dict[str, Tuple[str, Optional[str]]],
                 allowed_intermediary_tokens: Set[HexAddress],
                 ):
        """
        :param factory_router_map:
            Defines router smart contracts to be used with each DEX.
            Each Uniswap v2 is uniquely identified by its factory contract.
            Addresses always lowercase.

        :param allowed_intermediary_tokens:
            Because we hold our reserves only in one currecy e.g. BUSD
            and we want to trade XXX/BNB pairs, we need to whitelist
            BNB as an allowed intermediary token.

        :param allowed_slippage:
            Maximum allowed slippage in trades
        """
        self.factory_router_map = {k.lower(): v for k, v in factory_router_map.items()}
        self.allowed_intermediary_tokens = allowed_intermediary_tokens

    def make_direct_trade(self,
                          routing_state: UniswapV2RoutingState,
                          target_pair: TradingPairIdentifier,
                          reserve_asset: AssetIdentifier,
                          reserve_amount: int,
                          max_slippage: float,
                          ) -> List[BlockchainTransaction]:
        """Prepare a trade where target pair has out reserve asset as a quote token.

        :return:
            List of approval transactions (if any needed)
        """
        uniswap = routing_state.get_uniswap_for_pair(self.factory_router_map, target_pair)
        token_address = reserve_asset.address
        txs = routing_state.ensure_token_approved(token_address, uniswap.router.address)
        txs += routing_state.trade_on_router_two_way(
            uniswap,
            target_pair,
            reserve_asset,
            reserve_amount,
            max_slippage,
            )
        return txs

    def trade(self,
              routing_state: UniswapV2RoutingState,
              target_pair: TradingPairIdentifier,
              reserve_asset: AssetIdentifier,
              reserve_asset_amount: int,  # Raw amount of the reserve asset
              max_slippage: float,
              ) -> List[BlockchainTransaction]:

        # Our reserves match directly the asset on trading pair
        # -> we can do one leg trade
        if target_pair.quote == reserve_asset or target_pair.base == reserve_asset:
            return self.make_direct_trade(
                routing_state,
                target_pair,
                reserve_asset,
                reserve_asset_amount,
                max_slippage,
            )