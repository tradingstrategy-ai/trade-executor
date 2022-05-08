"""Route trades to different Uniswap v2 like exchanges."""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Set, List

from eth_typing import HexAddress

from eth_defi.abi import get_deployed_contract
from eth_defi.hotwallet import HotWallet
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.strategy.routing import RoutingModel


class UniswapV2RoutingCycleState:
    """Each cycle needs a set of approvals.

    Manage the state of already given approvals here,
    so that we do not do duplicates.

    The approvals are not persistent in the executor state,
    but are specific for each cycle.
    """

    def __init__(self, tx_builder: TransactionBuilder):
        self.tx_builder = tx_builder
        # router -> erc-20 mappings
        self.approved_routes = defaultdict(set)

    def is_route_approved(self, router_address: str):
        return router_address in self.approved_routes

    def mark_router_approved(self, token_address, router_address):
        self.approved_routes.add(router_address)

    def is_approved_on_chain(self, token_address: str, router_address: str) -> bool:
        erc_20 = get_deployed_contract(self.web3, "ERC20MockDecimals.json", token_address)
        # Assume allowance is always infinity
        return erc_20.functions.allowance.call(self.hot_wallet.address, router_address) > 0

    def ensure_token_approved(self, token_address: str, router_address: str) -> List[BlockchainTransaction]:
        """Make sure we have ERC-20 approve() for the trade

        - Infinite approval on-chain

        - ...or previous approval in this state,

        :param token_address:
        :param router_address:
        :return: Create 0 or 1 transactions if needs to be approved
        """

        if token_address in self.approved_routes[router_address]:
            return []

        erc_20 = get_deployed_contract(self.web3, "ERC20MockDecimals.json", token_address)

        # Set internal state we are approved
        self.mark_router_approved(token_address, router_address)

        if erc_20.functions.allowance.call(self.hot_wallet.address, router_address) > 0:
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


class UniswapV2SimpleRoutingModel(RoutingModel):
    """A simple router that does not optimise the trade execution cost.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(self,
                 factory_router_map: Dict[HexAddress, HexAddress],
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
        """
        self.factory_router_map = {k.lower(): v.lower() for k, v in factory_router_map.items()}
        self.allowed_intermediary_tokens = allowed_intermediary_tokens

    def get_router(self, target_pair: TradingPairIdentifier):
        """Get a router for a trading pair."""
        assert target_pair.exchange_address, f"Exchange address missing for {target_pair}"
        return self.factory_router_map[target_pair.exchange_address.lower()]

    def is_token_approved(self,
                          hot_wallet: HotWallet,
                          router_address: str):


    def create_router_approve(self,
                              hot_wallet: HotWallet,
                              router_address,
                              router_state: UniswapV2RoutingCycleState) -> BlockchainTransaction:

        router_state.mark_router_approved(router_address)

    def make_direct_trade(self,
              target_pair: TradingPairIdentifier,
              reserve_asset: AssetIdentifier,
              hot_wallet: HotWallet,
              routing_state: UniswapV2RoutingCycleState,
              ):
        """Prepare a trade where target pair has out reserve asset as a quote token."""

        if routing_state.is_route_approved():
            pass


    def route(self,
              target_pair: TradingPairIdentifier,
              reserve_asset: AssetIdentifier,
              hot_wallet: HotWallet,
              routing_state: UniswapV2RoutingCycleState,
              ):

        if target_pair.quote == reserve_asset:
            return self.make_direct_trade(
                target_pair,
                reserve_asset,
                hot_wallet,
                routing_state
            )