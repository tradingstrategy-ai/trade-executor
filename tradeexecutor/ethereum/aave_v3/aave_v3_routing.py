"""Route trades for 1delta."""

import logging
from _decimal import Decimal
from typing import Dict, Optional, List

from eth_defi.one_delta.constants import TradeOperation
from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.exceptions import ContractLogicError

from eth_defi.token import fetch_erc20_details
from eth_defi.trade import TradeSuccess
from tradingstrategy.chain import ChainId
from tradingstrategy.pair import PandasPairUniverse
from eth_defi.abi import get_deployed_contract
from eth_defi.tx import AssetDelta
from eth_defi.gas import estimate_gas_fees
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment, mock_partial_deployment_for_analysis
from eth_defi.aave_v3.deployment import AaveV3Deployment, fetch_deployment as fetch_aave_v3_deployment
from eth_defi.one_delta.deployment import OneDeltaDeployment, fetch_deployment as fetch_one_delta_deployment
from eth_defi.one_delta.position import (
    approve,
    close_short_position,
    open_short_position,
    reduce_short_position,
)
from eth_defi.aave_v3.loan import supply, withdraw
from eth_defi.utils import ZERO_ADDRESS_STR
from eth_defi.aave_v3.constants import MAX_AMOUNT

from tradeexecutor.state.types import Percent
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.trade import TradeFlag
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.interest import set_interest_checkpoint
from tradeexecutor.ethereum.routing_state import (
    EthereumRoutingState, 
    route_tokens, # don't remove, forwarded import
    OutOfBalance, # don't remove, forwarded import
    get_base_quote,
)
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import get_uniswap_for_pair
from tradeexecutor.utils.blockchain import get_block_timestamp
from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.ethereum.aave_v3.analysis import analyse_credit_trade_by_receipt


logger = logging.getLogger(__name__)


class AaveV3RoutingState(EthereumRoutingState):
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

    def get_aave_v3_deployment(self, address_map: dict) -> AaveV3Deployment:
        try:
            return fetch_aave_v3_deployment(
                self.web3,
                pool_address=Web3.to_checksum_address(address_map["aave_v3_pool"]),
                data_provider_address=Web3.to_checksum_address(address_map["aave_v3_data_provider"]),
                oracle_address=Web3.to_checksum_address(address_map["aave_v3_oracle"]),
            )
        except ContractLogicError as e:
            raise RuntimeError(f"Could not fetch deployment data for pool address {address_map['aave_v3_pool']}") from e

    def lend_on_aave_v3(
        self,
        *,
        aave_v3_deployment: AaveV3Deployment,
        target_pair: TradingPairIdentifier,
        reserve_amount: int,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ):
        base_token, quote_token = get_base_quote(self.web3, target_pair.get_pricing_pair(), target_pair.get_pricing_pair().quote)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        logger.info(
            "Creating a trade for %s, reserve amount %d. Trade flags are %s",
            target_pair,
            reserve_amount,
            trade_flags,
        )

        if TradeFlag.open in trade_flags:
            assert reserve_amount > 0

            _, bound_func = supply(
                aave_v3_deployment=aave_v3_deployment,
                token=quote_token,
                amount=reserve_amount,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )
        elif TradeFlag.close in trade_flags:
            bound_func = withdraw(
                aave_v3_deployment=aave_v3_deployment,
                token=quote_token,
                amount=MAX_AMOUNT,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )
        else:
            raise ValueError(f"Wrong trade flags used: {trade_flags}")

        return self.create_signed_transaction(
            aave_v3_deployment.pool,
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
        """Not used for now"""
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
        """Not used for now"""
        pass

    def ensure_multiple_tokens_approved(
        self,
        *,
        aave_v3_deployment: AaveV3Deployment,
        collateral_token_address: str,
        atoken_address: str,
    ) -> list[BlockchainTransaction]:
        """Make sure we have ERC-20 approve() for the 1delta

        - Infinite approval on-chain

        - ...or previous approval in this state,

        :param token_address:

        :return: Create 0 or 1 transactions if needs to be approved
        """
        txs = []

        for token_address in [
            collateral_token_address,
            atoken_address,
        ]:
            txs += self.ensure_token_approved(token_address, aave_v3_deployment.pool.address)

        return txs


class AaveV3Routing(EthereumRoutingModel):
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
    ) -> AaveV3RoutingState:
        """Create a new routing state for this cycle.

        - Connect routing to web3 and hot wallet

        - Read on-chain data on what gas fee we are going to use

        - Setup transaction builder based on this information
        """

        return super().create_routing_state(universe, execution_details, AaveV3RoutingState)

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

    def make_credit_supply_trade(
        self, 
        routing_state: EthereumRoutingState,
        target_pair: TradingPairIdentifier,
        *,
        reserve_asset: AssetIdentifier,
        reserve_amount: int,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ) -> list[BlockchainTransaction]:
        aave_v3_deployment = routing_state.get_aave_v3_deployment(self.address_map)
        
        spot_pair = target_pair.get_pricing_pair()
        
        txs = routing_state.ensure_multiple_tokens_approved(
            aave_v3_deployment=aave_v3_deployment,
            collateral_token_address=spot_pair.quote.address,
            atoken_address=target_pair.quote.address,
        )

        logger.info(
            "Doing credit supply trade. Pair:%s\n Reserve asset:%s Reserve amount: %s",
            target_pair,
            reserve_asset,
            reserve_amount,
        )

        trade_txs = routing_state.lend_on_aave_v3(
            aave_v3_deployment=aave_v3_deployment,
            target_pair=target_pair,
            reserve_amount=reserve_amount,
            trade_flags=trade_flags,
            check_balances=check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
        )

        txs += trade_txs
        return txs

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):
        pricing_pair = trade.pair.get_pricing_pair()
        base_token_details = fetch_erc20_details(web3, pricing_pair.base.checksum_address)
        quote_token_details = fetch_erc20_details(web3, pricing_pair.quote.checksum_address)
        reserve = trade.reserve_currency
        tx = get_swap_transactions(trade)
        aave_v3_deployment = fetch_aave_v3_deployment(web3, tx.contract_address, ZERO_ADDRESS_STR, ZERO_ADDRESS_STR)

        tx_dict = tx.get_transaction()
        receipt = receipts[HexBytes(tx.tx_hash)]
        input_args = tx.get_actual_function_input_args()

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if trade.is_credit_supply():
            result = analyse_credit_trade_by_receipt(
                web3,
                aave_v3_deployment=aave_v3_deployment,
                tx=tx_dict,
                tx_hash=tx.tx_hash,
                tx_receipt=receipt,
                input_args=input_args,
                trade_operation=TradeOperation.OPEN if trade.is_buy() else TradeOperation.CLOSE,
            )

            if isinstance(result, TradeSuccess):                
                if trade.is_buy():
                    executed_amount = result.amount_in / Decimal(10 ** base_token_details.decimals)
                    executed_reserve = executed_amount
                else:
                    executed_amount = -result.amount_out / Decimal(10 ** base_token_details.decimals)
                    executed_reserve = -executed_amount

                assert executed_amount != 0, f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}"

                # Mark as success
                state.mark_trade_success(
                    ts,
                    trade,
                    executed_price=float(1),
                    executed_amount=executed_amount,
                    executed_reserve=executed_reserve,
                    lp_fees=0,
                    native_token_price=0,  # won't fix
                    cost_of_gas=result.get_cost_of_gas(),
                )

                # TODO: This need to be properly accounted and currently there is no mechanism here
                # Set the check point interest balances for new positions
                last_block_number = trade.blockchain_transactions[-1].block_number
                set_interest_checkpoint(state, ts, last_block_number)
            else:
                report_failure(ts, state, trade, stop_on_execution_failure)
        else:
            raise ValueError(f"Unknown trade type {trade}")
