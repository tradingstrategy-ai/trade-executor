"""Find routes between historical pairs."""

import logging
from decimal import Decimal
from typing import Dict, List, Optional, Tuple


from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState
from tradeexecutor.ethereum.routing_model import EthereumRoutingModel
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_token
from tradingstrategy.pair import PandasPairUniverse



logger = logging.getLogger(__name__)


class OutOfBalance(Exception):
    """Did not have enough tokens"""


class BacktestRoutingState(RoutingState):

    def __init__(self,
                 pair_universe: PandasPairUniverse,
                 wallet: SimulatedWallet,
                 ):
        super().__init__(pair_universe)
        
        self.pair_universe = pair_universe
        self.wallet = wallet

    def is_route_approved(self, router_address: str):
        return router_address in self.approved_routes

    def mark_router_approved(self, token_address, router_address):
        self.approved_routes[router_address].add(token_address)

    def check_has_enough_tokens(
            self,
            token: AssetIdentifier,
            amount: Decimal,
    ):
        """Check we have enough buy side tokens to do a trade."""
        balance = self.wallet.get_balance(token.address)
        if balance < amount:
            raise OutOfBalance(f"SimulatedWallet does not have enough {token} tokens to trade. Need {amount}, has {balance}")

    def create_trade(self,
            target_pair: TradingPairIdentifier,
            reserve_asset: AssetIdentifier,
            reserve_amount: Decimal,
            max_slippage: float,
            check_balances: False):
        """Prepare the actual swap.

        :param check_balances:
            Check on-chain balances that the account has enough tokens
            and raise exception if not.
        """

        base_token, quote_token = self.get_base_quote(self.web3, target_pair, reserve_asset)
        
        if check_balances:
            self.check_has_enough_tokens(quote_token, reserve_amount)

        tx = self.create_simulated_trade(target_pair, max)
        return [tx]


class BacktestRoutingModel(RoutingModel):
    """A simple router that does not optimise the trade execution cost.

    - Able to trade on multiple exchanges

    - Able to three-way trades through predefined intermediary hops,
      either on the exchange itself or some outside exchange
    """

    def __init__(self,
                 factory_router_map: Dict[str, Tuple[str, Optional[str]]],
                 allowed_intermediary_pairs: Dict[str, str],
                 reserve_token_address: str,
                 trading_fee: Optional[float] = None,
                 ):
        """
        :param factory_router_map:
            Defines router smart contracts to be used with each DEX.
            Each Uniswap v2 is uniquely identified by its factory contract.
            Addresses always lowercase.

        :param allowed_intermediary_pairs:

            Quote token address -> pair smart contract address mapping.

            Because we hold our reserves only in one currecy e.g. BUSD
            and we want to trade e.g. Cake/BNB pairs, we need to whitelist
            BNB as an allowed intermediary token.
            This makes it possible to do BUSD -> BNB -> Cake trade.
            This set is the list of pair smart contract addresses that
            are allowed to be used as a hop.

        :param reserve_token_address:
            Token address of our reserve currency.
            Relevent for buy/sell routing.
            Lowercase.

        :param trading_fee:
            The trading fee applied to all trades by default,
            unless a pair overrides.
        """

        assert type(factory_router_map) == dict


        # Convert all key addresses to lowercase to
        # avoid mix up with Ethereum address checksums
        self.factory_router_map = self.convert_address_dict_to_lower(factory_router_map)

        self.trading_fee = trading_fee

    def get_default_trading_fee(self) -> Optional[float]:
        return self.trading_fee

    def get_reserve_asset(self, pair_universe: PandasPairUniverse) -> AssetIdentifier:
        """Translate our reserve token address tok an asset description."""
        assert pair_universe is not None, "Pair universe missing"
        reserve_token = pair_universe.get_token(self.reserve_token_address)
        assert reserve_token, f"Pair universe does not contain our reserve asset {self.reserve_token_address}"
        return translate_token(reserve_token)

    def trade(self,
              routing_state: BacktestRoutingState, # TODO remove
              target_pair: TradingPairIdentifier,
              reserve_asset: AssetIdentifier,
              reserve_asset_amount: Decimal,  # Raw amount of the reserve asset
              max_slippage: float=0.01,
              check_balances=False,
              intermediary_pair: Optional[TradingPairIdentifier] = None,
              ) -> List[BlockchainTransaction]:
        """

        :param routing_state:
        :param target_pair:
        :param reserve_asset:
        :param reserve_asset_amount:
        :param max_slippage:
            Max slippage per trade. 0.01 is 1%.
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
        self.pre_trade_assertions(reserve_asset_amount, max_slippage, target_pair, reserve_asset)

        # Our reserves match directly the asset on trading pair
        # -> we can do one leg trade
        if not intermediary_pair:
            if target_pair.quote == reserve_asset or target_pair.base == reserve_asset:
                return self.routing_state.create_and_complete_trade(
                    target_pair,
                    reserve_asset,
                    reserve_asset_amount,
                    max_slippage=max_slippage,
                    check_balances=check_balances,
                )
            raise RuntimeError(f"Do not how to trade reserve {reserve_asset} with {target_pair}")
        else:

            assert intermediary_pair.pool_address.lower() in self.allowed_intermediary_pairs.values(), f"Does not how to trade a pair. Got intermediary pair {intermediary_pair} that is not allowed, allowed intermediary pairs are {self.allowed_intermediary_pairs}"

            return self.routing_state.create_and_complete_trade(
                target_pair,
                reserve_asset,
                reserve_asset_amount,
                max_slippage=max_slippage,
                check_balances=check_balances,
                intermediary_pair=intermediary_pair,
            )

    def route_pair(self, pair_universe: PandasPairUniverse, trading_pair: TradingPairIdentifier) \
            -> Tuple[TradingPairIdentifier, Optional[TradingPairIdentifier]]:
        """Return Uniswap routing information (path components) for a trading pair.

        For three-way pairs, figure out the intermedia step.

        :return:
            (router address, target pair, intermediate pair) tuple
        """
        return self.route_pair(pair_universe, trading_pair)

    def setup_internal(self, routing_state: RoutingState, trade: TradeExecution):
        """Simulate trade braodcast and mark it as success."""

        # 2. Capital allocation
        nonce, tx_hash = routing_state.wallet.fetch_nonce_and_tx_hash()

        trade.blockchain_transactions = [
            BlockchainTransaction(
                nonce=nonce,
                tx_hash=tx_hash,
            )
        ]

    def setup_trades(self,
                     routing_state: BacktestRoutingState,
                     trades: List[TradeExecution],
                     check_balances=False):
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
        for t in trades:
            self.setup_internal(routing_state, t)

    def create_routing_state(self,
                     universe: TradingStrategyUniverse,
                     execution_details: dict) -> BacktestRoutingState:
        """Create a new routing state for this cycle."""
        assert isinstance(universe, TradingStrategyUniverse)
        wallet = execution_details["wallet"]
        return BacktestRoutingState(universe.universe.pairs, wallet)
