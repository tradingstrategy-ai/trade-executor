"""Map trades to different routing backends based on their type.

See :py:mod:`tradeexecutor.ethereum.routing_data` for per-chain and per-protocol configuration data.
"""

from typing import Dict, TypeAlias, List, Tuple
import logging

from tradingstrategy.chain import ChainId
from web3 import Web3

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.generic.pair_configurator import PairConfigurator
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


logger = logging.getLogger(__name__)

#: Router name to router implementation mapping
#:
#: Router name is like "aave-3", "uniswap-v2" and so on
#:
#:
RoutingMap: TypeAlias = Dict[str, RoutingModel]


class GenericRoutingState(RoutingState):
    """Store one state per router.

    Strategy universe is part of the state, as it will change between
    trading strategy cycles (pairs are added and removed).
    """

    def __init__(
        self,
        strategy_universe: TradingStrategyUniverse,
        state_map: Dict[str, RoutingState]
    ):

        assert isinstance(strategy_universe, TradingStrategyUniverse)
        assert strategy_universe.data_universe.exchange_universe is not None,  "pair_universe is not initialised with exchange data"

        self.strategy_universe = strategy_universe
        self.state_map = state_map


class GenericRouting(RoutingModel):
    """Routes trade to a different routing backend depending on the trade type.

    Based on the pair and exchange data (contained within pairs),
    make choices which router should take the trade.

    Usually there is very direct map exchange -> router,
    as long as the router has the correct smart contract address
    information available for various contracts needed
    to perform the trade.
    """

    def __init__(
        self,
        pair_configurator: PairConfigurator | None,
        three_leg_resolution=True,
    ):
        self.pair_configurator = pair_configurator
        logger.info("Initialised %s", self)

        # Legacy support for old unit tests
        self.three_leg_resolution = three_leg_resolution

    def is_initialised(self) -> bool:
        return self.pair_configurator is not None

    def initialise(self, pair_configurator: PairConfigurator):
        """On a live trading loop GenericRouting gets instiated early, but does not yet have trading universe available.

        This methods allows us to initialise after we have loaded the universe data for the first time.
        """
        self.pair_configurator = pair_configurator

    def create_routing_state(
        self,
        universe: TradingStrategyUniverse,
        execution_details: object
    ) -> RoutingState:
        """Create a new routing state for this cycle.

        :param execution_details:
            A dict of whatever connects live execution to routing.
        """
        assert isinstance(universe, TradingStrategyUniverse)
        substates = {}
        for routing_id in self.pair_configurator.get_supported_routers():
            router = self.pair_configurator.get_config(routing_id, three_leg_resolution=self.three_leg_resolution).routing_model
            substates[routing_id.router_name] = router.create_routing_state(universe, execution_details)

        return GenericRoutingState(universe, substates)

    def get_router(self, pair: TradingPairIdentifier) -> RoutingModel:
        routing_id = self.pair_configurator.match_router(pair)
        protocol_config = self.pair_configurator.get_config(routing_id)
        router = protocol_config.routing_model
        return router, protocol_config

    def setup_trades(
        self,
        state: State,
        routing_state: GenericRoutingState,
        trades: List[TradeExecution],
        check_balances=False,
        rebroadcast=False,
    ):
        """Route trades.

        - The trade order must be preserved (sells must come before buys,
          so that we have cash)

        - Fills in Blockchain transaction details for each trade
          by using the corresponding router
        """

        assert isinstance(state, State)
        assert isinstance(routing_state, GenericRoutingState)

        for t in trades:
            assert not t.pair.is_exchange_account(), \
                f"Unsupported: exchange account trades must not reach routing. Trade: {t}"
            router, protocol_config = self.get_router(t.pair)
            # Set the router, so we know
            # in the post-trade analysis which route this trade took
            t.route = protocol_config.routing_id.router_name

            # Set the router state
            router_state = routing_state.state_map.get(protocol_config.routing_id.router_name)
            assert router_state, f"No router state for: {protocol_config.routing_id.router_name}, we have {list(routing_state.state_map.keys())}"

            # Multichain: if the trade targets a satellite chain, temporarily
            # swap the tx_builder and web3 so transactions are signed for and
            # contract calls hit the correct chain.
            original_tx_builder = None
            original_web3 = None
            web3config = getattr(self.pair_configurator, 'web3config', None)
            if (web3config
                and hasattr(router_state, 'tx_builder')
                and router_state.tx_builder is not None):
                trade_chain_id = t.pair.chain_id
                if trade_chain_id != router_state.tx_builder.chain_id:
                    from eth_defi.hotwallet import HotWallet as EthHotWallet
                    from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
                    satellite_web3 = web3config.get_connection(ChainId(trade_chain_id))
                    original_tx_builder = router_state.tx_builder
                    # Create a separate HotWallet with its own nonce counter
                    # so the original wallet's nonce isn't corrupted
                    satellite_wallet = EthHotWallet(original_tx_builder.hot_wallet.account)
                    satellite_wallet.sync_nonce(satellite_web3)
                    router_state.tx_builder = HotWalletTransactionBuilder(
                        satellite_web3, satellite_wallet,
                    )
                    # Also swap the routing state's web3 for contract calls
                    if hasattr(router_state, 'web3'):
                        original_web3 = router_state.web3
                        router_state.web3 = satellite_web3

            router.setup_trades(
                state=state,
                routing_state=router_state,
                trades=[t],
                check_balances=check_balances,
                rebroadcast=rebroadcast,
            )

            # Restore originals after signing
            if original_tx_builder is not None:
                router_state.tx_builder = original_tx_builder
            if original_web3 is not None:
                router_state.web3 = original_web3

    def settle_trade(
        self,
        web3: Web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[str, dict],
        stop_on_execution_failure=False,
    ):
        assert isinstance(state, State)
        assert isinstance(trade, TradeExecution)
        assert type(receipts) == dict
        assert trade.route is not None, f"TradeExecution lacks TradeExecution.route, it was not executed with GenericRouter?\n{trade}"

        router = self.pair_configurator.get_routing(trade.pair)

        # Multichain: use the satellite chain's web3 if the trade
        # was executed on a different chain than the default.
        web3config = getattr(self.pair_configurator, 'web3config', None)
        trade_chain_id = trade.pair.chain_id
        if web3config and trade_chain_id != web3.eth.chain_id:
            web3 = web3config.get_connection(ChainId(trade_chain_id))

        return router.settle_trade(
            web3,
            state,
            trade,
            receipts,
            stop_on_execution_failure,
        )
