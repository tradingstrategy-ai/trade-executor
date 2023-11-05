"""Route trades for 1delta."""

import logging
from typing import Dict, Optional, List

from eth_typing import HexAddress
from web3 import Web3
from web3.exceptions import ContractLogicError

from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse
from eth_defi.tx import AssetDelta
from eth_defi.gas import estimate_gas_fees
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.aave_v3.deployment import fetch_deployment as fetch_aave_v3_deployment
from eth_defi.one_delta.deployment import OneDeltaDeployment, fetch_deployment
from eth_defi.one_delta.position import (
    approve,
    close_short_position,
    open_short_position,
)

from tradeexecutor.state.types import Percent
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.ethereum.routing_state import (
    EthereumRoutingState, 
    route_tokens, # don't remove, forwarded import
    OutOfBalance, # don't remove, forwarded import
    get_base_quote,
    get_base_quote_intermediary
)
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import get_uniswap_for_pair

logger = logging.getLogger(__name__)


class OneDeltaRoutingState(EthereumRoutingState):
    def __init__(
        self,
        pair_universe: PandasPairUniverse,
        tx_builder: Optional[HotWalletTransactionBuilder]=None,
        swap_gas_limit=2_000_000
    ):
        super().__init__(pair_universe, tx_builder, swap_gas_limit)
    
    def __repr__(self):
        return f"<OneDeltaRoutingState Tx builder: {self.tx_builder} web3: {self.web3}>"
    
    def get_uniswap_for_pair(self, address_map: dict, target_pair: TradingPairIdentifier) -> UniswapV3Deployment:
        """Get a router for a trading pair."""
        return get_uniswap_for_pair(self.web3, address_map, target_pair)

    def get_one_delta_for_pair(self, address_map: dict, target_pair: TradingPairIdentifier) -> OneDeltaDeployment:    
        broker_proxy_address = Web3.to_checksum_address(address_map["one_delta_broker_proxy"])
        aave_pool_address = Web3.to_checksum_address(address_map["aave_v3_pool"])

        try:
            aave_v3 = fetch_aave_v3_deployment(
                self.web3,
                aave_pool_address,
                # TODO
                aave_pool_address,
                aave_pool_address,
            )
            return fetch_deployment(
                self.web3,
                aave_v3,
                broker_proxy_address,
                broker_proxy_address,
            )
        except ContractLogicError as e:
            raise RuntimeError(f"Could not fetch deployment data for router address {router_address} (factory {factory_address}) - data is likely wrong") from e


    def trade_on_one_delta(
        self,
        *,
        one_delta: OneDeltaDeployment,
        uniswap: UniswapV3Deployment,
        target_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        collateral_amount: int,
        borrow_amount: int,
        max_slippage: Percent,
        check_balances: False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ):
        assert one_delta
        assert target_pair.kind.is_leverage()

        base_token, quote_token = get_base_quote(self.web3, target_pair, reserve_asset)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        logger.info(
            "Creating a trade for %s, slippage tolerance %f, trade reserve %s, amount in %d",
            target_pair,
            max_slippage,
            reserve_asset,
            collateral_amount,
        )

        pool_fee_raw = int(target_pair.get_pricing_pair().fee * 1_000_000)

        # TODO: differentiate open and close
        bound_swap_func = open_short_position(
            one_delta_deployment=one_delta,
            collateral_token=quote_token,
            borrow_token=base_token,
            pool_fee=pool_fee_raw,
            collateral_amount=collateral_amount,
            borrow_amount=borrow_amount,
            wallet_address=self.tx_builder.get_token_delivery_address(),
        )

        return self.create_signed_transaction(
            one_delta.broker_proxy,
            bound_swap_func,
            self.swap_gas_limit,
            asset_deltas,
            notes=notes,
        )
    
    def trade_on_router_two_way(
        self,
        uniswap: UniswapV3Deployment,
        target_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        max_slippage: Percent,
        check_balances: False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
        one_delta: OneDeltaDeployment | None = None,
    ):
        """Prepare the actual swap. Same for Uniswap V2 and V3.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """
        pass
        
    def trade_on_router_three_way(
        self,
        uniswap: UniswapV3Deployment,
        target_pair: TradingPairIdentifier,
        intermediary_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        max_slippage: float,
        check_balances: False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
        one_delta: OneDeltaDeployment | None = None,
    ):
        """Prepare the actual swap for three way trade.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """
        # TODO
        pass

    def ensure_multiple_tokens_approved(self, one_delta: OneDeltaDeployment) -> List[BlockchainTransaction]:
        """Make sure we have ERC-20 approve() for the 1delta

        - Infinite approval on-chain

        - ...or previous approval in this state,

        :param token_address:
        :param router_address:

        :return: Create 0 or 1 transactions if needs to be approved
        """
        txs = []

        # TODO: this is not the right place for approve, move this later
        for fn in approve(
            one_delta_deployment=one_delta,
            collateral_token=usdc.contract,
            borrow_token=weth.contract,
            atoken=ausdc.contract,
            vtoken=vweth.contract,
        ):
            _execute_tx(web3, hot_wallet, fn)

        return [tx]


class OneDeltaSimpleRoutingModel(EthereumRoutingModel):
    """A simple router that does not optimise the trade execution cost.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(
        self,
        address_map: Dict[str, HexAddress],
        allowed_intermediary_pairs: Dict[str, str],
        reserve_token_address: str,
        chain_id: Optional[ChainId] = None,
    ):
        """
        :param address_map:
            Defines router smart contracts to be used with each DEX.
            Address map is a dict of factory, router, position_manager,
            and quoter addresses

        :param allowed_intermediary_pairs:

            Quote token address -> pair smart contract address mapping.

            Because we hold our reserves only in one currecy e.g. BUSD
            and we want to trade e.g. Cake/BNB pairs, we need to whitelist
            BNB as an allowed intermediary token.
            This makes it possible to do BUSD -> BNB -> Cake trade.
            This set is the list of pair smart contract addresses that
            are allowed to be used as a hop.

        :param chain_id:
            Store the chain id for which these routes were generated for.

        :param reserve_token_address:
            Token address of our reserve currency.
            Relevent for buy/sell routing.
            Lowercase.
        """

        super().__init__(allowed_intermediary_pairs, reserve_token_address, chain_id)
        
        assert type(address_map) == dict
        self.address_map = self.convert_address_dict_to_lower(address_map)

    def perform_preflight_checks_and_logging(
        self,
        pair_universe: PandasPairUniverse,
    ):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """

        logger.info("Routing details")
        logger.info("  Factory: %s", self.address_map["factory"])
        logger.info("  Router: %s", self.address_map["router"])
        logger.info("  Position Manager: %s", self.address_map["position_manager"])
        logger.info("  Quoter: %s", self.address_map["quoter"])

        self.reserve_asset_logging(pair_universe)
        
    def make_leverage_trade(
        self, 
        routing_state: EthereumRoutingState,
        target_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        max_slippage: float,
        check_balances=False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ) -> list[BlockchainTransaction]:
        
        return super().make_leverage_trade(
            routing_state,
            target_pair,
            reserve_asset,
            reserve_amount,
            max_slippage,
            self.address_map,
            check_balances,
            asset_deltas=asset_deltas,
            notes="",
        )

