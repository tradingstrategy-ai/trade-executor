import logging
from typing import Type

from eth_defi.tx import AssetDelta
from tradeexecutor.state.state import State
from tradeexecutor.state.types import Percent
from tradeexecutor.strategy.routing import RoutingModel
from typing import Dict, List, Optional, Tuple

from tradingstrategy.chain import ChainId

from eth_defi.gas import estimate_gas_fees

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.strategy.routing import RoutingModel, CannotRouteTrade
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair, \
    translate_token
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.ethereum.routing_state import EthereumRoutingState
from tradeexecutor.utils.slippage import get_slippage_in_bps

logger = logging.getLogger(__name__)


#: Use 1% slippage tolerance if we somehow miss a vaue
DEFAULT_SLIPPAGE_TOLERANCE: Percent = 0.01


class EthereumRoutingModel(RoutingModel):
    """A simple router that does not optimise the trade execution cost. Designed for uniswap-v2 forks.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """
    
    def __init__(self,
                 allowed_intermediary_pairs: Dict[str, str],
                 reserve_token_address: str,
                 chain_id: Optional[ChainId] = None,
                 ):
        """
        
        
        :param addresses:
            Defines router smart contracts to be used with each DEX.
            
            Each Uniswap v2 is uniquely identified by its factory contract. Addresses always lowercase. Factory Router map
            
            For Uniswap V3, addresses is a dict of factory, router, position_manager,
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

        super().__init__(allowed_intermediary_pairs, reserve_token_address)
        self.chain_id = chain_id

    def make_direct_trade(self,
                          routing_state: EthereumRoutingState,
                          target_pair: TradingPairIdentifier,
                          reserve_asset: AssetIdentifier,
                          reserve_amount: int,
                          max_slippage: Percent,
                          address_map: Dict,
                          check_balances=False,
                          asset_deltas: Optional[List[AssetDelta]] = None,
                          notes="",
                          ) -> List[BlockchainTransaction]:
        """Prepare a trade where target pair has out reserve asset as a quote token.

        :param max_slippage:
            Max slippage tolerance as percent.

            E.g. 0.01 for 100 BPS slippage tolerance.

        :return:
            List of approval transactions (if any needed)
        """
        uniswap = routing_state.get_uniswap_for_pair(address_map, target_pair)
        token_address = reserve_asset.address
        
        # TODO find better way of doing this. Use inheritance?
        if hasattr(uniswap, "router"):
            txs = routing_state.ensure_token_approved(token_address, uniswap.router.address)
        elif hasattr(uniswap, "swap_router"):
            txs = routing_state.ensure_token_approved(token_address, uniswap.swap_router.address)
        else:
            raise TypeError("Incorrect Uniswap Instance provided. Can't get router.")

        adjusted_reserve_amount = routing_state.adjust_spend(
            reserve_asset,
            reserve_amount,
            check_balances=check_balances,
        )

        logger.info(
            "Doing two way trade. Pair:%s\n Reserve:%s Adjusted reserve amount: %s Max slippage: %s",
            target_pair,
            reserve_asset,
            adjusted_reserve_amount,
            max_slippage if max_slippage else "-",
        )

        # Validate slippage tolerance a bit
        get_slippage_in_bps(max_slippage)

        trade_txs = routing_state.trade_on_router_two_way(
            uniswap,
            target_pair,
            reserve_asset,
            adjusted_reserve_amount,
            max_slippage,
            check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
            )

        # Leave note of adjustment.
        # Use str() because JSON cannot handle big int
        trade_txs[0].other = {
            "reserve_amount": str(reserve_amount),
            "adjusted_reserve_amount": str(adjusted_reserve_amount),
        }

        txs += trade_txs
        return txs

    def make_multihop_trade(self,
          routing_state: EthereumRoutingState,
          target_pair: TradingPairIdentifier,
          intermediary_pair: TradingPairIdentifier,
          reserve_asset: AssetIdentifier,
          reserve_amount: int,
          max_slippage: float,
          address_map: Dict,
          check_balances=False,
          asset_deltas: Optional[List[AssetDelta]] = None,
          notes="",
          ) -> List[BlockchainTransaction]:
        """Prepare a trade where target pair has out reserve asset as a quote token.

        :return:
            List of approval transactions (if any needed)
        """
        uniswap = routing_state.get_uniswap_for_pair(address_map, target_pair)
        token_address = reserve_asset.address
        
        # TODO find better way of doing this. Use inheritance?
        if hasattr(uniswap, "router"):
            txs = routing_state.ensure_token_approved(token_address, uniswap.router.address)
        elif hasattr(uniswap, "swap_router"):
            txs = routing_state.ensure_token_approved(token_address, uniswap.swap_router.address)
        else:
            raise TypeError("Incorrect Uniswap Instance provided. Can't get router.")

        adjusted_reserve_amount = routing_state.adjust_spend(
            reserve_asset,
            reserve_amount,
            check_balances=check_balances,
        )

        trade_txs = routing_state.trade_on_router_three_way(
            uniswap,
            target_pair,
            intermediary_pair,
            reserve_asset,
            adjusted_reserve_amount,
            max_slippage,
            check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
            )

        txs += trade_txs

        # Leave note of adjustment.
        # Use str() because JSON cannot handle big int
        trade_txs[0].other = {
            "reserve_amount": str(reserve_amount),
            "adjusted_reserve_amount": str(adjusted_reserve_amount),
        }
        return txs

    def make_leverage_trade(
        self,
        routing_state: EthereumRoutingState,
        target_pair: TradingPairIdentifier,
        *,
        borrow_amount: int,
        collateral_amount: int,
        reserve_amount: int,
        max_slippage: Percent,
        address_map: Dict,
        trade_flags: set[TradeFlag],
        check_balances: bool = False,
        asset_deltas: Optional[List[AssetDelta]] = None,
        notes="",
    ) -> List[BlockchainTransaction]:
        """Prepare a short trade where target pair has out reserve asset as a quote token.

        :param max_slippage:
            Max slippage tolerance as percent.

            E.g. 0.01 for 100 BPS slippage tolerance.

        :return:
            List of approval transactions (if any needed)
        """
        uniswap = routing_state.get_uniswap_for_pair(address_map, target_pair)
        one_delta = routing_state.get_one_delta_for_pair(address_map, target_pair)
        aave_v3 = routing_state.get_aave_v3_for_pair(address_map, target_pair)
        
        spot_pair = target_pair.get_pricing_pair()
        
        txs = routing_state.ensure_multiple_tokens_approved(
            one_delta=one_delta,
            aave_v3=aave_v3,
            uniswap_v3=uniswap,
            collateral_token_address=spot_pair.quote.address,
            borrow_token_address=spot_pair.base.address,
            atoken_address=target_pair.quote.address,
            vtoken_address=target_pair.base.address,
        )

        # adjusted_reserve_amount = routing_state.adjust_spend(
        #     reserve_asset,
        #     reserve_amount,
        # )

        logger.info(
            "Doing leverage short trade. Pair:%s\n Borrow asset:%s Adjusted reserve amount: %s Max slippage: %s",
            target_pair,
            target_pair.base,
            borrow_amount,
            max_slippage,
        )

        # Validate slippage tolerance a bit
        get_slippage_in_bps(max_slippage)
        
        trade_txs = routing_state.trade_on_one_delta(
            one_delta=one_delta,
            uniswap=uniswap,
            target_pair=target_pair,
            collateral_amount=collateral_amount,
            borrow_amount=borrow_amount,
            reserve_amount=reserve_amount,
            max_slippage=max_slippage,
            trade_flags=trade_flags,
            check_balances=check_balances,
            asset_deltas=asset_deltas,
            notes=notes,
        )

        # Leave note of adjustment.
        # Use str() because JSON cannot handle big int
        # trade_txs[0].other = {
        #     "reserve_amount": str(reserve_amount),
        #     "adjusted_reserve_amount": str(adjusted_reserve_amount),
        # }

        txs += trade_txs
        return txs

    def trade(self,
              routing_state: EthereumRoutingState,
              target_pair: TradingPairIdentifier,
              reserve_asset: AssetIdentifier,
              reserve_asset_amount: int,  # Raw amount of the reserve asset
              max_slippage: Percent=0.01,
              check_balances=False,
              intermediary_pair: Optional[TradingPairIdentifier] = None,
              asset_deltas: Optional[List[AssetDelta]] = None,
              notes="",
              ) -> List[BlockchainTransaction]:
        """

        :param routing_state:
        :param target_pair:
        :param reserve_asset:
        :param reserve_asset_amount:

        :param max_slippage:
            Max slippage per trade.

            Set the slippage tolerance for this trade.

            0.01 is 100 BPS is 1%.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        :param intermediary_pair:
            If the trade needs to be routed through a intermediary pool, e.g.
            BUSD -> BNB -> Cake.
        :return:
            List of prepared transactions to make this trade.
            These transactions, like approve() may relate to the earlier
            transactions in the `routing_state`.
        """

        logger.info("trade() pair: %s reserve: %s reserve allocated: %s max slippage: %s %%",
                    target_pair,
                    reserve_asset,
                    reserve_asset_amount,
                    max_slippage * 100,
        )

        self.pre_trade_assertions(reserve_asset_amount, max_slippage, target_pair, reserve_asset)

        # Our reserves match directly the asset on trading pair
        # -> we can do one leg trade
        if not intermediary_pair:
            if target_pair.quote == reserve_asset or target_pair.base == reserve_asset:
                return self.make_direct_trade(
                    routing_state,
                    target_pair,
                    reserve_asset,
                    reserve_asset_amount,
                    max_slippage=max_slippage,
                    check_balances=check_balances,
                    asset_deltas=asset_deltas,
                    notes=notes,
                )
            raise RuntimeError(f"Do not how to trade reserve {reserve_asset} with {target_pair}")
        else:

            self.intermediary_pair_assertion(intermediary_pair)

            return self.make_multihop_trade(
                routing_state,
                target_pair,
                intermediary_pair,
                reserve_asset,
                reserve_asset_amount,
                max_slippage=max_slippage,
                check_balances=check_balances,
                asset_deltas=asset_deltas,
                notes=notes,
            )
    
    def execute_trades_internal(
        self,
        pair_universe: PandasPairUniverse,
        routing_state: EthereumRoutingState,
        trades: List[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Split for testability.

        :param check_balances:
            Check that the wallet has enough reserves to perform the trades
            before executing them. Because we are selling before buying.
            sometimes we do no know this until the sell tx has been completed.

        :param max_slippage:
            The max slipppage tolerated before the trade fails.
            0.01 is 1%.
        """

        # Watch out for executing trade twice

        txs: List[BlockchainTransaction] = []

        reserve_asset = self.get_reserve_asset(pair_universe)

        for t in trades:

            if not rebroadcast:
                assert len(t.blockchain_transactions) == 0, f"Trade {t} had already blockchain transactions associated with it"
            else:
                t.blockchain_transactions = []

            # TODO: Add support for accurate multihop asset deltas
            if t.slippage_tolerance is not None:
                asset_deltas = t.calculate_asset_deltas()
            else:
                # Old path that does not slippage tolerances for trades
                asset_deltas = None

            max_slippage = t.slippage_tolerance if t.slippage_tolerance is not None else DEFAULT_SLIPPAGE_TOLERANCE

            logger.info("Slippage tolerance is: %f %%, expected asset deltas: %s", max_slippage * 100, asset_deltas)

            # TODO: hack to bypass route_trade(), fix later
            if t.is_leverage() or t.is_credit_supply():
                target_pair = t.pair
                intermediary_pair = None
            else:
                target_pair, intermediary_pair = self.route_trade(pair_universe, t)

            notes = f"Trade: {t}\n" \
                    f"Position {t.position_id}\n" \
                    f"Asset deltas: {asset_deltas}"

            # TODO: refactor this part as too many nested if else here
            if t.is_leverage():
                trade_txs = self.make_leverage_trade(
                    routing_state,
                    target_pair,
                    borrow_amount=t.get_raw_planned_quantity(),
                    collateral_amount=t.get_raw_planned_collateral_allocation(),
                    reserve_amount=t.get_raw_planned_reserve(),
                    max_slippage=max_slippage,
                    trade_flags=t.flags,
                    check_balances=check_balances,
                    asset_deltas=asset_deltas,
                    notes=notes,
                )
            elif t.is_credit_supply():
                trade_txs = self.make_credit_supply_trade(
                    routing_state,
                    target_pair,
                    reserve_asset=reserve_asset,
                    reserve_amount=t.get_raw_planned_reserve(),
                    trade_flags=t.flags,
                    check_balances=check_balances,
                    asset_deltas=asset_deltas,
                    notes=notes,
                )
            elif intermediary_pair is None:
                # Two way trade
                # Decide between buying and selling
                trade_txs = (
                    self.trade(
                        routing_state,
                        target_pair=target_pair,
                        reserve_asset=reserve_asset,
                        reserve_asset_amount=t.get_raw_planned_reserve(),
                        check_balances=check_balances,
                        asset_deltas=asset_deltas,
                        max_slippage=max_slippage,
                        notes=notes,
                    )
                    if t.is_buy()
                    else self.trade(
                        routing_state,
                        target_pair=target_pair,
                        reserve_asset=target_pair.base,
                        reserve_asset_amount=-t.get_raw_planned_quantity(),
                        check_balances=check_balances,
                        asset_deltas=asset_deltas,
                        max_slippage=max_slippage,
                        notes=notes,
                    )
                )
            elif t.is_buy():
                trade_txs = self.trade(
                    routing_state,
                    target_pair=target_pair,
                    reserve_asset=reserve_asset,
                    reserve_asset_amount=t.get_raw_planned_reserve(),
                    check_balances=check_balances,
                    intermediary_pair=intermediary_pair,
                    asset_deltas=asset_deltas,
                    max_slippage=max_slippage,
                    notes=notes,
                )
            else:
                trade_txs = self.trade(
                    routing_state,
                    target_pair=target_pair,
                    reserve_asset=target_pair.base,
                    reserve_asset_amount=-t.get_raw_planned_quantity(),
                    check_balances=check_balances,
                    intermediary_pair=intermediary_pair,
                    asset_deltas=asset_deltas,
                    max_slippage=max_slippage,
                    notes=notes,
                )

            t.set_blockchain_transactions(trade_txs)
            txs += trade_txs

        # Now all trades have transactions associated with them.
        # We can start to execute transactions.

    def setup_trades(
        self,
        state: State,
        routing_state: EthereumRoutingState,
        trades: List[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Strategy and live execution connection.

        Turns abstract strategy trades to real blockchain transactions.

        - Modifies TradeExecution objects in place and associates a blockchain transaction for each

        - Signs tranactions from the hot wallet and broadcasts them to the network

        :param check_balances:
            Check that the wallet has enough reserves to perform the trades
            before executing them. Because we are selling before buying.
            sometimes we do no know this until the sell tx has been completed.


        :param max_slippage:
            Max slippaeg tolerated per trade. 0.01 is 1%.

        """
        return self.execute_trades_internal(
            routing_state.pair_universe,
            routing_state,
            trades,
            check_balances,
            rebroadcast=rebroadcast,
        )
    
    def create_routing_state(self,
                             universe: StrategyExecutionUniverse,
                             execution_details: dict,
                             Routing_State: Type[EthereumRoutingState] 
                             # Doesn't get full typing
                             # Type[UniswapV2RoutingState] | Type[UniswapV3RoutingState]
                             # throws error due to circular import
                             ) -> EthereumRoutingState:
        """Create a new routing state for this cycle.

        - Connect routing to web3 and hot wallet

        - Read on-chain data on what gas fee we are going to use

        - Setup transaction builder based on this information
        """

        assert isinstance(universe, TradingStrategyUniverse)
        assert universe is not None, "Universe is required"
        assert universe.data_universe.pairs is not None, "Pairs are required"

        tx_builder = execution_details.get("tx_builder")
        if tx_builder is not None:
            # Modern code path
            routing_state = Routing_State(universe.data_universe.pairs, tx_builder=tx_builder)
        else:
            # Legacy code path - do not use

            web3 = execution_details["web3"]

            # Hot wallet is not present in dummy execution model
            hot_wallet = execution_details.get("hot_wallet")

            fees = estimate_gas_fees(web3)

            logger.info("Gas fee estimations for chain %d: %s", web3.eth.chain_id, fees)

            logger.info("Estimated gas fees for chain %d: %s", web3.eth.chain_id, fees)
            if hot_wallet is not None:
                tx_builder = HotWalletTransactionBuilder(web3, hot_wallet)
                routing_state = Routing_State(universe.data_universe.pairs, tx_builder)
            else:
                routing_state = Routing_State(universe.data_universe.pairs,
                                              tx_builder=None,
                                              web3=web3)

        return routing_state
    
    def route_trade(self, pair_universe: PandasPairUniverse, trade: TradeExecution) -> Tuple[TradingPairIdentifier, Optional[TradingPairIdentifier]]:
        """Figure out how to map an abstract trade to smart contracts.

        Decide if we can do a direct trade in the pair pool.
        or if we need to hop through another pool to buy the token we want to buy.

        :return:
            target pair, intermediary pair tuple
        """
        return self.route_pair(pair_universe, trade.pair)
    
    def intermediary_pair_assertion(self, intermediary_pair: TradingPairIdentifier):
        assert intermediary_pair.pool_address.lower() in self.allowed_intermediary_pairs.values(), f"Does not how to trade a pair. Got intermediary pair {intermediary_pair} that is not allowed, allowed intermediary pairs are {self.allowed_intermediary_pairs}"
    
    def reserve_asset_logging(self, pair_universe: PandasPairUniverse) -> None:
        reserve = self.get_reserve_asset(pair_universe)
        logger.info("  Routed reserve asset is %s", reserve)
    
    @staticmethod
    def route_pair_assertions(trading_pair, pair_universe):
        assert isinstance(trading_pair, TradingPairIdentifier), f"Not a trading pair: {trading_pair}: {trading_pair.__class__}"

        # Only issue for legacy code
        assert pair_universe, "PairUniverse must be given so that we know how to route three way trades"