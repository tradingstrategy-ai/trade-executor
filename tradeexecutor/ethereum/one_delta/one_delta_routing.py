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
from tradeexecutor.ethereum.one_delta.analysis import (
    analyse_leverage_trade_by_receipt,
    analyse_credit_trade_by_receipt,
)
from tradeexecutor.ethereum.swap import get_swap_transactions, report_failure
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
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
from eth_defi.one_delta.lending import supply, withdraw
from eth_defi.one_delta.price import (
    OneDeltaPriceHelper,
    estimate_buy_received_amount, 
    estimate_sell_received_amount,
)
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
    get_base_quote_intermediary,
    get_token_for_asset,
    DEFAULT_APPROVE_GAS_LIMIT,
    APPROVE_GAS_LIMITS,
)
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import get_uniswap_for_pair
from tradeexecutor.utils.blockchain import get_block_timestamp

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
        return get_one_delta(self.web3, address_map)

    def trade_on_one_delta(
        self,
        *,
        one_delta: OneDeltaDeployment,
        uniswap: UniswapV3Deployment,
        target_pair: TradingPairIdentifier,
        collateral_amount: int,
        borrow_amount: int,
        reserve_amount: int,
        max_slippage: Percent,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
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
            "Creating a trade for %s, slippage tolerance %f, borrow amount %d, collateral amount %d. Trade flags are %s",
            target_pair,
            max_slippage,
            borrow_amount,
            collateral_amount,
            trade_flags,
        )

        pool_fee_raw = int(target_pair.get_pricing_pair().fee * 1_000_000)
        slippage_bps = max_slippage * 10_000
        price_helper = OneDeltaPriceHelper(one_delta)

        if TradeFlag.open in trade_flags:
            assert  borrow_amount < 0
            # TODO: planned_reserve-planned_collateral_allocation refactor later
            assert collateral_amount == 0
            assert reserve_amount > 0

            min_collateral_amount_out = price_helper.get_amount_out(
                -borrow_amount,
                [base_token.address, quote_token.address],
                [pool_fee_raw],
                slippage=slippage_bps,
            )

            bound_func = open_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                pool_fee=pool_fee_raw,
                collateral_amount=reserve_amount,
                borrow_amount=-borrow_amount,
                wallet_address=self.tx_builder.get_token_delivery_address(),
                min_collateral_amount_out=min_collateral_amount_out,
            )
        elif TradeFlag.close in trade_flags or TradeFlag.close_protocol_last in trade_flags:
            # TODO: planned_reserve-planned_collateral_allocation refactor later
            assert collateral_amount < 0
            assert reserve_amount == 0

            if TradeFlag.close_protocol_last in trade_flags:
                withdraw_collateral_amount = MAX_AMOUNT
            else:
                withdraw_collateral_amount = -collateral_amount

            bound_func = close_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                atoken=atoken,
                pool_fee=pool_fee_raw,
                wallet_address=self.tx_builder.get_token_delivery_address(),
                withdraw_collateral_amount=withdraw_collateral_amount,
            )
        elif TradeFlag.increase in trade_flags:
            assert borrow_amount < 0

            min_collateral_amount_out = price_helper.get_amount_out(
                -borrow_amount,
                [base_token.address, quote_token.address],
                [pool_fee_raw],
                slippage=slippage_bps,
            )

            bound_func = open_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                pool_fee=pool_fee_raw,
                collateral_amount=reserve_amount,
                borrow_amount=-borrow_amount,
                wallet_address=self.tx_builder.get_token_delivery_address(),
                min_collateral_amount_out=min_collateral_amount_out,
            )
        elif TradeFlag.reduce in trade_flags:
            assert borrow_amount > 0

            max_collateral_amount_in = price_helper.get_amount_in(
                borrow_amount,
                [base_token.address, quote_token.address],
                [pool_fee_raw],
                slippage=slippage_bps,
            )

            bound_func = reduce_short_position(
                one_delta_deployment=one_delta,
                collateral_token=quote_token,
                borrow_token=base_token,
                atoken=atoken,
                pool_fee=pool_fee_raw,
                wallet_address=self.tx_builder.get_token_delivery_address(),
                reduce_borrow_amount=borrow_amount,
                withdraw_collateral_amount=-collateral_amount,
                max_collateral_amount_in=max_collateral_amount_in,
            )
        else:
            raise ValueError(f"Wrong trade flags used: {trade_flags}")

        return self.create_signed_transaction(
            one_delta.broker_proxy,
            bound_func,
            self.swap_gas_limit,
            asset_deltas,
            notes=notes,
        )

    def lend_via_one_delta(
        self,
        *,
        one_delta: OneDeltaDeployment,
        target_pair: TradingPairIdentifier,
        reserve_amount: int,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ):
        base_token, quote_token = get_base_quote(self.web3, target_pair.get_pricing_pair(), target_pair.get_pricing_pair().quote)
        atoken = get_token_for_asset(self.web3, target_pair.base)

        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        logger.info(
            "Creating a trade for %s, reserve amount %d. Trade flags are %s",
            target_pair,
            reserve_amount,
            trade_flags,
        )

        price_helper = OneDeltaPriceHelper(one_delta)

        if TradeFlag.open in trade_flags:
            assert reserve_amount > 0

            bound_func = supply(
                one_delta_deployment=one_delta,
                token=quote_token,
                amount=reserve_amount,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )
        elif TradeFlag.close in trade_flags:
            bound_func = withdraw(
                one_delta_deployment=one_delta,
                token=quote_token,
                atoken=atoken,
                amount=MAX_AMOUNT,
                wallet_address=self.tx_builder.get_token_delivery_address(),
            )
        else:
            raise ValueError(f"Wrong trade flags used: {trade_flags}")

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
        one_delta: OneDeltaDeployment,
        aave_v3: AaveV3Deployment,
        collateral_token_address: str,
        borrow_token_address: str,
        atoken_address: str,
        vtoken_address: str,
        uniswap_v3: UniswapV3Deployment | None = None,
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
        if uniswap_v3:
            uniswap_router_address = uniswap_v3.swap_router.address

        for token_address in [
            collateral_token_address,
            borrow_token_address,
            atoken_address,
        ]:
            txs += self.ensure_token_approved(token_address, broker_proxy_address)
            txs += self.ensure_token_approved(token_address, aave_v3_pool_address)
            if uniswap_v3:
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


class OneDeltaRouting(EthereumRoutingModel):
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
        reserve_amount: int,
        max_slippage: float,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ) -> list[BlockchainTransaction]:
        
        return super().make_leverage_trade(
            routing_state,
            target_pair,
            borrow_amount=borrow_amount,
            collateral_amount=collateral_amount,
            reserve_amount=reserve_amount,
            max_slippage=max_slippage,
            address_map=self.address_map,
            trade_flags=trade_flags,
            check_balances=check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
        )

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
        
        return super().make_credit_supply_trade(
            routing_state,
            target_pair,
            reserve_asset=reserve_asset,
            reserve_amount=reserve_amount,
            address_map=self.address_map,
            trade_flags=trade_flags,
            check_balances=check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
        )

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
        one_delta = fetch_one_delta_deployment(web3, tx.contract_address, tx.contract_address, ZERO_ADDRESS_STR)
        uniswap = mock_partial_deployment_for_analysis(web3, ZERO_ADDRESS_STR)
        aave = fetch_aave_v3_deployment(web3, ZERO_ADDRESS_STR, ZERO_ADDRESS_STR, ZERO_ADDRESS_STR)

        tx_dict = tx.get_transaction()
        receipt = receipts[HexBytes(tx.tx_hash)]
        input_args = tx.get_actual_function_input_args()

        ts = get_block_timestamp(web3, receipt["blockNumber"])

        if trade.is_leverage():
            result, collateral_amount = analyse_leverage_trade_by_receipt(
                web3,
                one_delta=one_delta,
                uniswap=uniswap,
                aave=aave,
                tx=tx_dict,
                tx_hash=tx.tx_hash,
                tx_receipt=receipt,
                input_args=input_args,
                trade_operation=TradeOperation.OPEN if trade.is_sell() else TradeOperation.CLOSE,
            )

            if isinstance(result, TradeSuccess):
                price = result.get_human_price(quote_token_details.address == result.token0.address)
                
                if trade.is_buy():
                    executed_amount = -result.amount_out / Decimal(10 ** base_token_details.decimals)
                    executed_collateral_consumption = -result.amount_in / Decimal(10 ** reserve.decimals)
                    # TODO: planned_reserve-planned_collateral_allocation refactor later
                    executed_collateral_allocation = Decimal(collateral_amount) / Decimal(10 ** reserve.decimals)
                    executed_reserve = 0
                else:
                    executed_amount = result.amount_in / Decimal(10 ** base_token_details.decimals)
                    executed_collateral_consumption = result.amount_out / Decimal(10 ** reserve.decimals)
                    executed_collateral_allocation = Decimal(0)
                    executed_reserve = Decimal(collateral_amount) / Decimal(10 ** reserve.decimals)

                if trade.is_short():
                    executed_amount = -executed_amount

                lp_fee_paid = result.lp_fee_paid

                assert (executed_amount != 0) and (price > 0), f"Executed amount {executed_amount}, executed collateral consumption: {executed_collateral_consumption},  executed_reserve: {executed_reserve}, price: {price}"

                logger.info("1delta routing\nPlanned: %f %f %f\nExecuted: %f %f %f", trade.planned_collateral_consumption, trade.planned_collateral_allocation, trade.planned_reserve, executed_collateral_consumption, executed_collateral_allocation, executed_reserve)

                # Mark as success
                state.mark_trade_success(
                    ts,
                    trade,
                    executed_price=float(price),
                    executed_amount=executed_amount,
                    executed_reserve=executed_reserve,
                    executed_collateral_consumption=executed_collateral_consumption,
                    executed_collateral_allocation=executed_collateral_allocation,
                    lp_fees=lp_fee_paid,
                    native_token_price=0,  # won't fix
                    cost_of_gas=result.get_cost_of_gas(),
                )

                # TODO: This need to be properly accounted and currently there is no mechanism here
                # Set the check point interest balances for new positions
                last_block_number = trade.blockchain_transactions[-1].block_number
                set_interest_checkpoint(state, ts, last_block_number)
            else:
                report_failure(ts, state, trade, stop_on_execution_failure)

        elif trade.is_credit_supply():
            result = analyse_credit_trade_by_receipt(
                web3,
                one_delta=one_delta,
                uniswap=uniswap,
                aave=aave,
                tx=tx_dict,
                tx_hash=tx.tx_hash,
                tx_receipt=receipt,
                input_args=input_args,
                trade_operation=TradeOperation.OPEN if trade.is_buy() else TradeOperation.CLOSE,
            )

            if isinstance(result, TradeSuccess):
                price = 1
                
                if trade.is_buy():
                    executed_amount = result.amount_in / Decimal(10 ** base_token_details.decimals)
                    executed_reserve = executed_amount
                else:
                    executed_amount = -result.amount_out / Decimal(10 ** base_token_details.decimals)
                    executed_reserve = -executed_amount

                assert executed_amount != 0, f"Executed amount {executed_amount}, executed_reserve: {executed_reserve}"

                logger.info("1delta routing\nPlanned: %f\nExecuted: %f", trade.planned_reserve, executed_reserve)

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


def get_one_delta(web3: Web3, address_map: dict) -> OneDeltaDeployment:
    broker_proxy_address = Web3.to_checksum_address(address_map["one_delta_broker_proxy"])
    quoter_address = Web3.to_checksum_address(address_map["one_delta_quoter"])

    try:
        return fetch_one_delta_deployment(
            web3,
            broker_proxy_address,
            broker_proxy_address,
            quoter_address,
        )
    except ContractLogicError as e:
        raise RuntimeError(f"Could not fetch deployment data with address map {address_map} - data is likely wrong") from e