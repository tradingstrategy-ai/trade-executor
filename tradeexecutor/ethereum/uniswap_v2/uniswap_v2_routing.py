"""Route trades to different Uniswap v2 like exchanges."""
import logging
from typing import Dict, Set, List, Optional, Tuple

from eth_defi.tx import AssetDelta
from tradeexecutor.state.types import BPS
from tradingstrategy.chain import ChainId
from web3 import Web3
from web3.exceptions import ContractLogicError
from web3.exceptions import ContractLogicError

from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.uniswap_v2.swap import swap_with_slippage_protection

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.ethereum.routing_state import (
    EthereumRoutingState, 
    route_tokens, # don't remove forwarded import
    OutOfBalance, # don't remove forwarded import
    get_base_quote,
    get_base_quote_intermediary
)
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
 
logger = logging.getLogger(__name__)


class UniswapV2RoutingState(EthereumRoutingState):

    def __init__(self,
                 pair_universe: PandasPairUniverse,
                 tx_builder: Optional[HotWalletTransactionBuilder] = None,
                 web3: Optional[Web3] = None,
                 swap_gas_limit=2_000_000):
        super().__init__(pair_universe=pair_universe,
                         tx_builder=tx_builder,
                         swap_gas_limit=swap_gas_limit,
                         web3=web3)
    
    def __repr__(self):
        return f"<UniswapV2RoutingState Tx builder: {self.tx_builder} web3: {self.web3}>"
    
    def get_uniswap_for_pair(self, factory_router_map: dict, target_pair: TradingPairIdentifier) -> UniswapV2Deployment:
        """Get a router for a trading pair."""
        return get_uniswap_for_pair(self.web3, factory_router_map, target_pair)
    
    def trade_on_router_two_way(self,
            uniswap: UniswapV2Deployment,
            target_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: int,
            max_slippage: float,
            check_balances: False,
            asset_deltas: Optional[List[AssetDelta]] = None,
        ):
        """Prepare the actual swap. Same for Uniswap V2 and V3.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """

        base_token, quote_token = get_base_quote(self.web3, target_pair, reserve_asset)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)
        
        if target_pair.fee:
            bps_fee = target_pair.fee * 10_000
            bound_swap_func = swap_with_slippage_protection(
                uniswap,
                recipient_address=self.tx_builder.get_token_delivery_address(),
                base_token=base_token,
                quote_token=quote_token,
                amount_in=reserve_amount,
                max_slippage=max_slippage * 100,  # In BPS
                fee=bps_fee
            )
        else:
            logger.warning("Pair supplied without fee, using default fee")
            
            bound_swap_func = swap_with_slippage_protection(
                uniswap,
                recipient_address=self.tx_builder.get_token_delivery_address(),
                base_token=base_token,
                quote_token=quote_token,
                amount_in=reserve_amount,
                max_slippage=max_slippage * 100,  # In BPS
            )
        
        return self.create_signed_transaction(
            uniswap.router,
            bound_swap_func,
            self.swap_gas_limit,
            asset_deltas,
        )

    def trade_on_router_three_way(self,
            uniswap: UniswapV2Deployment,
            target_pair: TradingPairIdentifier,
            intermediary_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: int,
            max_slippage: float,
            check_balances: False,
            asset_deltas: Optional[List[AssetDelta]] = None):
        """Prepare the actual swap for three way trade.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """

        self.validate_pairs(target_pair, intermediary_pair)

        self.validate_exchange(target_pair, intermediary_pair)
        self.validate_exchange(target_pair, intermediary_pair)

        base_token, quote_token, intermediary_token = get_base_quote_intermediary(self.web3,target_pair, intermediary_pair, reserve_asset)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        assert target_pair.fee == intermediary_pair.fee, "Uniswap V2 pairs should all have the same fee"
        
        if target_pair.fee:
            bps_fee = target_pair.fee * 10_000
            bound_swap_func = swap_with_slippage_protection(
                uniswap,
                recipient_address=self.tx_builder.get_token_delivery_address(),
                base_token=base_token,
                quote_token=quote_token,
                amount_in=reserve_amount,
                max_slippage=max_slippage * 100,  # In BPS,
                intermediate_token=intermediary_token,
                fee = bps_fee
            )
        else:
            logger.warning("Pair supplied without fee, using default fee")
            
            bound_swap_func = swap_with_slippage_protection(
                uniswap,
                recipient_address=self.tx_builder.get_token_delivery_address(),
                base_token=base_token,
                quote_token=quote_token,
                amount_in=reserve_amount,
                max_slippage=max_slippage * 100,  # In BPS,
                intermediate_token=intermediary_token,
            )

        tx = self.tx_builder.sign_transaction(
            uniswap.router,
            bound_swap_func,
            self.swap_gas_limit,
            asset_deltas
        )
        return [tx]


class UniswapV2SimpleRoutingModel(EthereumRoutingModel):
    """A simple router that does not optimise the trade execution cost.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(self,
                 factory_router_map: Dict[str, Tuple[str, Optional[str]]],
                 allowed_intermediary_pairs: Dict[str, str],
                 reserve_token_address: str,
                 chain_id: Optional[ChainId] = None,
                 trading_fee: Optional[BPS] = None # TODO remove
                 ):
        """
        :param factory_router_map:
            Defines router smart contracts to be used with each DEX.
            Each Uniswap v2 is uniquely identified by its factory contract.
            Addresses always lowercase.

            Map of factory address -> (router address, init code hash tuple)

        :param allowed_intermediary_pairs:

            Quote token address -> pair smart contract address mapping.

            Because we hold our reserves only in one currecy e.g. BUSD
            and we want to trade e.g. Cake/BNB pairs, we need to whitelist
            BNB as an allowed intermediary token.
            This makes it possible to do BUSD -> BNB -> Cake trade.
            This set is the list of pair smart contract addresses that
            are allowed to be used as a hop.

        :param trading_fee:
            Trading fee express as float bps.

            This is the LP fee applied to all swaps.

        :param chain_id:
            Store the chain id for which these routes were generated for.

        :param reserve_token_address:
            Token address of our reserve currency.
            Relevent for buy/sell routing.
            Lowercase.
        """

        super().__init__(allowed_intermediary_pairs, reserve_token_address, chain_id)

        assert type(factory_router_map) == dict
        self.factory_router_map = self.convert_address_dict_to_lower(factory_router_map)
        
        # TODO remove trading_fee
        if trading_fee is not None:
            assert trading_fee >= 0, f"Got fee: {trading_fee}"
            assert trading_fee <= 1, f"Got fee: {trading_fee}"

        self.trading_fee = trading_fee

    def get_default_trading_fee(self) -> Optional[float]:
        return self.trading_fee

    def create_routing_state(self,
                             universe: StrategyExecutionUniverse,
                             execution_details: dict) -> UniswapV2RoutingState:
        """Create a new routing state for this cycle.

        - Connect routing to web3 and hot wallet

        - Read on-chain data on what gas fee we are going to use

        - Setup transaction builder based on this information
        """

        return super().create_routing_state(universe, execution_details, UniswapV2RoutingState)

    def perform_preflight_checks_and_logging(self,
        pair_universe: PandasPairUniverse):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """

        logger.info("Routing details")
        for factory, router in self.factory_router_map.items():
            logger.info("  Factory %s uses router %s", factory, router[0])

        self.reserve_asset_logging(pair_universe)
        
    def make_direct_trade(
        self, 
        routing_state: EthereumRoutingState,
        target_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        max_slippage: float,
        check_balances=False,
        asset_deltas: Optional[List[AssetDelta]] = None,
    ) -> List[BlockchainTransaction]:
        
        return super().make_direct_trade(
            routing_state,
            target_pair,
            reserve_asset,
            reserve_amount,
            max_slippage,
            self.factory_router_map,
            check_balances,
            asset_deltas=asset_deltas,
        )
    
    def make_multihop_trade(
        self,
        routing_state: EthereumRoutingState,
        target_pair: TradingPairIdentifier,
        intermediary_pair: TradingPairIdentifier,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        max_slippage: float,
        check_balances=False,
        asset_deltas: Optional[List[AssetDelta]] = None,
    ) -> List[BlockchainTransaction]:
        
        return super().make_multihop_trade(
            routing_state,
            target_pair,
            intermediary_pair,
            reserve_asset,
            reserve_amount,
            max_slippage,
            self.factory_router_map,
            check_balances,
            asset_deltas=asset_deltas,
        )
    
    
        
def get_uniswap_for_pair(web3: Web3, factory_router_map: dict, target_pair: TradingPairIdentifier) -> UniswapV2Deployment:
    """Get a router for a trading pair."""
    assert target_pair.exchange_address, f"Exchange address missing for {target_pair}"
    factory_address = Web3.to_checksum_address(target_pair.exchange_address)
    router_address, init_code_hash = factory_router_map[factory_address.lower()]

    try:
        return fetch_deployment(
            web3,
            factory_address,
            Web3.to_checksum_address(router_address),
            init_code_hash=init_code_hash,
        )
    except ContractLogicError as e:
        raise RuntimeError(f"Could not fetch deployment data for router address {router_address} (factory {factory_address}) - data is likely wrong") from e