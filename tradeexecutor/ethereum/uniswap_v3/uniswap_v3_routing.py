"""Route trades to different Uniswap v2 like exchanges."""

import logging
from _decimal import Decimal
from typing import Dict, Optional, List

from eth_typing import HexAddress
from hexbytes import HexBytes

from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeSuccess
from eth_defi.tx import AssetDelta
from eth_defi.uniswap_v3.analysis import analyse_trade_by_receipt
from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import Percent
from tradeexecutor.utils.blockchain import get_block_timestamp
from tradingstrategy.chain import ChainId
from web3 import Web3

from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, fetch_deployment, mock_partial_deployment_for_analysis
from eth_defi.uniswap_v3.swap import swap_with_slippage_protection
from web3.exceptions import ContractLogicError

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.ethereum.routing_state import (
    EthereumRoutingState, 
    route_tokens, # don't remove, forwarded import
    OutOfBalance, # don't remove, forwarded import
    get_base_quote,
    get_base_quote_intermediary
)
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
from tradeexecutor.utils.slippage import get_slippage_in_bps

logger = logging.getLogger(__name__)


class UniswapV3RoutingState(EthereumRoutingState):

    def __init__(
        self,
        pair_universe: PandasPairUniverse,
        tx_builder: Optional[HotWalletTransactionBuilder]=None,
        swap_gas_limit=None,
        approve_gas_limit=None,
    ):
        super().__init__(pair_universe, tx_builder=tx_builder, swap_gas_limit=swap_gas_limit, approve_gas_limit=approve_gas_limit)
    
    def __repr__(self):
        return f"<UniswapV3RoutingState Tx builder: {self.tx_builder} web3: {self.web3}>"
    
    def get_uniswap_for_pair(self, address_map: dict, target_pair: TradingPairIdentifier) -> UniswapV3Deployment:
        """Get a router for a trading pair."""
        return get_uniswap_for_pair(self.web3, address_map, target_pair)
    
    def trade_on_router_two_way(self,
            uniswap: UniswapV3Deployment,
            target_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: int,
            max_slippage: Percent,
            check_balances: False,
            asset_deltas: Optional[List[AssetDelta]] = None,
            notes="",
        ):
        """Prepare the actual swap. Same for Uniswap V2 and V3.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """

        base_token, quote_token = get_base_quote(self.web3, target_pair, reserve_asset)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        raw_fee = int(target_pair.fee * 1_000_000)

        logger.info(
            "Creating a trade for %s, slippage tolerance %f, trade reserve %s, amount in %d",
            target_pair,
            max_slippage,
            reserve_asset,
            reserve_amount,
        )

        bound_swap_func = swap_with_slippage_protection(
            uniswap,
            recipient_address=self.tx_builder.get_token_delivery_address(),
            base_token=base_token,
            quote_token=quote_token,
            amount_in=reserve_amount,
            max_slippage=get_slippage_in_bps(max_slippage),
            pool_fees=[raw_fee]
        )

        # If the receiver (vault) has its own security policy for slippage tolerance check it here
        receiver_slippage_tolerance = self.tx_builder.get_internal_slippage_tolerance()
        if receiver_slippage_tolerance is not None:
            # TODO: TxBuilder configures slippage tolerance as opposite of the trades
            receiver_slippage_tolerance = 1 - receiver_slippage_tolerance
            trade_slippage_tolerance = max_slippage
            assert receiver_slippage_tolerance > trade_slippage_tolerance, f"Receiver (vault) slippage tolerance tighter than the trade slippage tolerance.\nReceiver: {receiver_slippage_tolerance}, trade: {trade_slippage_tolerance}"

        return self.create_signed_transaction(
            uniswap.swap_router,
            bound_swap_func,
            self.swap_gas_limit,
            asset_deltas,
            notes=notes,
        )

    def trade_on_router_three_way(self,
            uniswap: UniswapV3Deployment,
            target_pair: TradingPairIdentifier,
            intermediary_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: int,
            max_slippage: float,
            check_balances: False,
            asset_deltas: Optional[List[AssetDelta]] = None,
            notes="",
        ):
        """Prepare the actual swap for three way trade.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """

        self.validate_pairs(target_pair, intermediary_pair)
        
        self.validate_exchange(target_pair, intermediary_pair)

        base_token, quote_token, intermediary_token = get_base_quote_intermediary(self.web3,target_pair, intermediary_pair, reserve_asset)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        # eth_defi uses raw_fees
        raw_pool_fees = [int(intermediary_pair.fee * 1_000_000), int(target_pair.fee * 1_000_000)]
        
        bound_swap_func = swap_with_slippage_protection(
            uniswap,
            recipient_address=self.tx_builder.get_token_delivery_address(),
            base_token=base_token,
            quote_token=quote_token,
            pool_fees=raw_pool_fees,
            amount_in=reserve_amount,
            max_slippage=get_slippage_in_bps(max_slippage),
            intermediate_token=intermediary_token,
        )
        
        return self.create_signed_transaction(
            uniswap.swap_router,
            bound_swap_func,
            self.swap_gas_limit,
            asset_deltas,
            notes=notes,
        )


class UniswapV3Routing(EthereumRoutingModel):
    """A simple router that does not optimise the trade execution cost. Designed for uniswap-v2 forks.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(self,
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
        logger.info(
            "Initialised %s\nfactory_router_map: %s\nallowed_intermediary_pairs: %s\nReserve token: %s",
            self,
            self.address_map,
            self.allowed_intermediary_pairs,
            reserve_token_address,
        )

    def create_routing_state(self,
                             universe: StrategyExecutionUniverse,
                             execution_details: dict) -> UniswapV3RoutingState:
        """Create a new routing state for this cycle.

        - Connect routing to web3 and hot wallet

        - Read on-chain data on what gas fee we are going to use

        - Setup transaction builder based on this information
        """

        return super().create_routing_state(universe, execution_details, UniswapV3RoutingState)

    def perform_preflight_checks_and_logging(self,
        pair_universe: PandasPairUniverse):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """

        logger.info("Routing details")
        logger.info("  Factory: %s", self.address_map["factory"])
        logger.info("  Router: %s", self.address_map["router"])
        logger.info("  Position Manager: %s", self.address_map["position_manager"])
        logger.info("  Quoter: %s", self.address_map["quoter"])

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
        notes="",
    ) -> list[BlockchainTransaction]:
        
        return super().make_direct_trade(
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
        notes="",
    ) -> list[BlockchainTransaction]:
        
        return super().make_multihop_trade(
            routing_state,
            target_pair,
            intermediary_pair,
            reserve_asset,
            reserve_amount,
            max_slippage,
            self.address_map,
            check_balances,
            asset_deltas=asset_deltas,
            notes="",
        )

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):

        logger.info(f"Settling Uniswap v3 trade: #{trade.trade_id}")

        base_token_details = fetch_erc20_details(web3, trade.pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, trade.pair.quote.checksum_address)
        reserve = trade.reserve_currency

        swap_tx = get_swap_transactions(trade)
        uniswap = self.mock_partial_deployment_for_analysis(web3, swap_tx.contract_address)

        tx_dict = swap_tx.get_transaction()

        try:
            receipt = receipts[HexBytes(swap_tx.tx_hash)]
        except KeyError as e:
            raise KeyError(f"Could not find hash: {swap_tx.tx_hash} in {receipts}") from e

        input_args = swap_tx.get_actual_function_input_args()

        result = analyse_trade_by_receipt(
            web3,
            uniswap=uniswap,
            tx=tx_dict,
            tx_hash=swap_tx.tx_hash,
            tx_receipt=receipt,
            input_args=input_args,
        )

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if isinstance(result, TradeSuccess):

            # v3 path includes fee (int) as well
            path = [a.lower() for a in result.path if type(a) == str]

            if trade.is_buy():
                assert path[0] == reserve.address, f"Was expecting the route path to start with reserve token {reserve}, got path {result.path}"

                price = result.get_human_price(quote_token_details.address == result.token0.address)
                executed_reserve = result.amount_in / Decimal(10 ** reserve.decimals)
                executed_amount = result.amount_out / Decimal(10 ** base_token_details.decimals)

                # lp fee is already in terms of quote token
                lp_fee_paid = result.lp_fee_paid
            else:
                # Ordered other way around
                assert path[0] == base_token_details.address.lower(), f"Path is {path}, base token is {base_token_details}"
                assert path[-1] == reserve.address
                price = result.get_human_price(quote_token_details.address == result.token0.address)
                executed_amount = -result.amount_in / Decimal(10 ** base_token_details.decimals)
                executed_reserve = result.amount_out / Decimal(10 ** reserve.decimals)

                # convert lp fee to be in terms of quote token
                lp_fee_paid = result.lp_fee_paid * float(price) if result.lp_fee_paid else None

            assert (executed_reserve > 0) and (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}, price: {price}, tx info {trade.tx_info}"

            logger.info(f"Executed: {executed_amount} {trade.pair.base.token_symbol}, {executed_reserve} {trade.pair.quote.token_symbol}")

            # Mark as success
            state.mark_trade_success(
                ts,
                trade,
                executed_price=float(price),
                executed_amount=executed_amount,
                executed_reserve=executed_reserve,
                lp_fees=lp_fee_paid,
                native_token_price=0,  # won't fix
                cost_of_gas=result.get_cost_of_gas(),
            )
        else:
            # Trade failed
            report_failure(ts, state, trade, stop_on_execution_failure)

    def mock_partial_deployment_for_analysis(
        self,
        web3: Web3,
        router_address: str
    ) -> UniswapV3Deployment:
        return mock_partial_deployment_for_analysis(web3, router_address)


def get_uniswap_for_pair(web3: Web3, address_map: dict, target_pair: TradingPairIdentifier) -> UniswapV3Deployment:
    """Get a router for a trading pair."""
    assert target_pair.exchange_address, f"Exchange address missing for {target_pair}"
    
    factory_address = Web3.to_checksum_address(target_pair.exchange_address)
    assert factory_address == Web3.to_checksum_address(address_map["factory"]), \
        "address_map[\"factory\"] and target_pair.exchange_address should be equal\n" \
        f"Got {factory_address} and {address_map['factory']} on pair {target_pair}"
    
    router_address = Web3.to_checksum_address(address_map["router"])
    position_manager_address = Web3.to_checksum_address(address_map["position_manager"])
    quoter_address = Web3.to_checksum_address(address_map["quoter"])
    quoter_v2 = address_map.get("quoter_v2")
    router_v2 = address_map.get("router_v2")

    try:
        return fetch_deployment(
            web3,
            factory_address,
            router_address,
            position_manager_address,
            quoter_address,
            quoter_v2=quoter_v2,
            router_v2=router_v2,
        )
    except ContractLogicError as e:
        raise RuntimeError(f"Could not fetch deployment data for router address {router_address} (factory {factory_address}) - data is likely wrong") from e
