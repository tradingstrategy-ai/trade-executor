"""Route trades for 1delta."""

import logging
from typing import Dict, Optional, List

from eth_typing import HexAddress
from web3 import Web3
from web3.exceptions import ContractLogicError

from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse
from eth_defi.abi import get_deployed_contract
from eth_defi.tx import AssetDelta
from eth_defi.gas import estimate_gas_fees
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.aave_v3.deployment import AaveV3Deployment, fetch_deployment as fetch_aave_v3_deployment
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
    get_base_quote_intermediary,
    get_token_for_asset,
    DEFAULT_APPROVE_GAS_LIMIT,
    APPROVE_GAS_LIMITS,
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
        return get_uniswap_for_pair(self.web3, address_map, target_pair.get_pricing_pair())

    def get_aave_v3_for_pair(self, address_map: dict, target_pair: TradingPairIdentifier) -> AaveV3Deployment:
        try:
            return fetch_aave_v3_deployment(
                self.web3,
                pool_address=Web3.to_checksum_address(address_map["aave_v3_pool"]),
                data_provider_address=Web3.to_checksum_address(address_map["aave_v3_data_provider"]),
                oracle_address=Web3.to_checksum_address(address_map["aave_v3_oracle"]),
            )
        except ContractLogicError as e:
            raise RuntimeError(f"Could not fetch deployment data for router address {router_address} (factory {factory_address}) - data is likely wrong") from e

    def get_one_delta_for_pair(self, address_map: dict, target_pair: TradingPairIdentifier) -> OneDeltaDeployment:
        broker_proxy_address = Web3.to_checksum_address(address_map["one_delta_broker_proxy"])

        try:
            return fetch_deployment(
                self.web3,
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
        collateral_amount: int,
        borrow_amount: int,
        max_slippage: Percent,
        check_balances: False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ):
        assert one_delta
        assert target_pair.kind.is_leverage()

        base_token, quote_token = get_base_quote(self.web3, target_pair.get_pricing_pair(), target_pair.get_pricing_pair().quote)
        atoken = get_token_for_asset(self.web3, target_pair.quote)

        if check_balances:
            self.check_has_enough_tokens(quote_token, collateral_amount)

        logger.info(
            "Creating a trade for %s, slippage tolerance %f, borrow amount in %d",
            target_pair,
            max_slippage,
            borrow_amount,
        )

        pool_fee_raw = int(target_pair.get_pricing_pair().fee * 1_000_000)

        if borrow_amount < 0:
            bound_func = open_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                pool_fee=pool_fee_raw,
                collateral_amount=collateral_amount,
                borrow_amount=-borrow_amount,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )
        else:
            # TODO: this is currently fully close the position
            # we should support reducing the position later
            bound_func = close_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                atoken=atoken,
                pool_fee=pool_fee_raw,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )

        return self.create_signed_transaction(
            one_delta.broker_proxy,
            bound_func,
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

    def ensure_multiple_tokens_approved(
        self,
        *,
        one_delta: OneDeltaDeployment,
        aave_v3: AaveV3Deployment,
        uniswap_v3: UniswapV3Deployment,
        collateral_token_address: str,
        borrow_token_address: str,
        atoken_address: str,
        vtoken_address: str,
    ) -> list[BlockchainTransaction]:
        """Make sure we have ERC-20 approve() for the 1delta

        - Infinite approval on-chain

        - ...or previous approval in this state,

        :param token_address:

        :return: Create 0 or 1 transactions if needs to be approved
        """
        txs = []

        broker_proxy_address = one_delta.broker_proxy.address
        aave_v3_pool_address = aave_v3.pool.address
        uniswap_router_address = uniswap_v3.swap_router.address

        for token_address in [
            collateral_token_address,
            borrow_token_address,
            atoken_address,
        ]:
            txs += self.ensure_token_approved(token_address, broker_proxy_address)
            txs += self.ensure_token_approved(token_address, aave_v3_pool_address)
            txs += self.ensure_token_approved(token_address, uniswap_router_address)

        txs += self.ensure_vtoken_delegation_approved(vtoken_address, broker_proxy_address)

        return txs

    def ensure_vtoken_delegation_approved(
        self,
        token_address: str,
        destination_address: str,
        amount: int = 2**256-1,
    ) -> list[BlockchainTransaction]:
        """Make sure we have approveDelegation() for the trade

        - Infinite approval on-chain
        - ...or previous approval in this state,

        :param token_address:
        :param destination_address:
        :param amount: How much to approve, default to approve infinite amount

        :return: Create 0 or 1 transactions if needs to be approved
        """

        assert self.tx_builder is not None

        if token_address in self.approved_routes[destination_address]:
            # Already approved for this cycle in previous trade
            return []

        token_contract = get_deployed_contract(
            self.web3,
            "aave_v3/VariableDebtToken.json",
            Web3.to_checksum_address(token_address),
        )

        # Set internal state we are approved
        self.mark_router_approved(token_address, destination_address)

        approve_address = self.tx_builder.get_token_delivery_address()

        # TODO
        # if token_contract.functions.allowance(approve_address, destination_address).call() > 0:
        #     # already approved in previous execution cycle
        #     return []

        # Gas limit for ERC-20 approve() may vary per chain,
        # see Arbitrum
        gas_limit = APPROVE_GAS_LIMITS.get(self.tx_builder.chain_id, DEFAULT_APPROVE_GAS_LIMIT)
        
        # Create infinite approval
        tx = self.tx_builder.sign_transaction(
            token_contract,
            token_contract.functions.approveDelegation(destination_address, amount),
            gas_limit=gas_limit,
            gas_price_suggestion=None,
            asset_deltas=[],
        )

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

    def create_routing_state(
        self,
        universe: StrategyExecutionUniverse,
        execution_details: dict
    ) -> OneDeltaRoutingState:
        """Create a new routing state for this cycle.

        - Connect routing to web3 and hot wallet

        - Read on-chain data on what gas fee we are going to use

        - Setup transaction builder based on this information
        """

        return super().create_routing_state(universe, execution_details, OneDeltaRoutingState)

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
        *,
        borrow_amount: int,
        collateral_amount: int,
        max_slippage: float,
        check_balances=False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ) -> list[BlockchainTransaction]:
        
        return super().make_leverage_trade(
            routing_state,
            target_pair,
            borrow_amount=borrow_amount,
            collateral_amount=collateral_amount,
            max_slippage=max_slippage,
            address_map=self.address_map,
            check_balances=check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
        )

