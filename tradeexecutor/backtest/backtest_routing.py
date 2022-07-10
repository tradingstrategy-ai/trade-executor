"""Find routes between historical pairs."""

import logging
from collections import defaultdict
from decimal import Decimal
from typing import Dict, Set, List, Optional, Tuple

from eth_typing import HexAddress, ChecksumAddress
from web3 import Web3
from web3.contract import Contract

from eth_defi.abi import get_deployed_contract
from eth_defi.gas import estimate_gas_fees
from eth_defi.token import fetch_erc20_details
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment, fetch_deployment
from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.ethereum.execution import get_token_for_asset
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.state.blockhain_transaction import BlockchainTransaction
from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.routing import RoutingModel, RoutingState, CannotRouteTrade
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, translate_trading_pair, \
    translate_token
from tradingstrategy.pair import PandasPairUniverse

from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse


logger = logging.getLogger(__name__)


class OutOfBalance(Exception):
    """Did not have enough tokens"""


class BacktestRoutingState(RoutingState):

    def __init__(self,
                 pair_universe: PandasPairUniverse,
                 wallet: SimulatedWallet,
                 ):
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

        if reserve_asset == target_pair.quote:
            # Buy with e.g. BUSD
            base_token = get_token_for_asset(web3, target_pair.base)
            quote_token = get_token_for_asset(web3, target_pair.quote)
        elif reserve_asset == target_pair.base:
            # Sell, flip the direction
            base_token = get_token_for_asset(web3, target_pair.quote)
            quote_token = get_token_for_asset(web3, target_pair.base)
        else:
            raise RuntimeError(f"Cannot trade {target_pair}")

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
        """

        assert type(factory_router_map) == dict
        assert type(allowed_intermediary_pairs) == dict
        assert type(reserve_token_address) == str

        assert reserve_token_address.lower() == reserve_token_address

        # Convert all key addresses to lowercase to
        # avoid mix up with Ethereum address checksums
        self.factory_router_map = {k.lower(): v for k, v in factory_router_map.items()}
        self.allowed_intermediary_pairs = {k.lower(): v.lower() for k, v in allowed_intermediary_pairs.items()}
        self.reserve_token_address = reserve_token_address

    def get_reserve_asset(self, pair_universe: PandasPairUniverse) -> AssetIdentifier:
        """Translate our reserve token address tok an asset description."""
        assert pair_universe is not None, "Pair universe missing"
        reserve_token = pair_universe.get_token(self.reserve_token_address)
        assert reserve_token, f"Pair universe does not contain our reserve asset {self.reserve_token_address}"
        return translate_token(reserve_token)

    def trade(self,
              routing_state: BacktestRoutingState,
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

        assert type(reserve_asset_amount) == int
        assert max_slippage is not None, "Max slippage must be given"
        assert type(max_slippage) == float
        assert reserve_asset_amount > 0, f"For sells, switch reserve_asset to different token. Got target_pair: {target_pair}, reserve_asset: {reserve_asset}, amount: {reserve_asset_amount}"

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

        assert isinstance(trading_pair, TradingPairIdentifier)

        reserve_asset = self.get_reserve_asset(pair_universe)

        # We can directly do a two-way trade
        if trading_pair.quote == reserve_asset:
            return trading_pair, None

        # Only issue for legacy code
        assert pair_universe, "PairUniverse must be given so that we know how to route three way trades"

        # Try to find a mid-hop pool for the trade
        intermediate_pair_contract_address = self.allowed_intermediary_pairs.get(trading_pair.quote.address.lower())

        if not intermediate_pair_contract_address:
            raise CannotRouteTrade(f"Does not know how to trade pair {trading_pair} - supported intermediate tokens are {list(self.allowed_intermediary_pairs.keys())}")

        dex_pair = pair_universe.get_pair_by_smart_contract(intermediate_pair_contract_address)
        assert dex_pair is not None, f"Pair universe did not contain pair for a pair contract address {intermediate_pair_contract_address}, quote token is {trading_pair.quote}"

        intermediate_pair = translate_trading_pair(dex_pair)
        if not intermediate_pair:
            raise CannotRouteTrade(f"Universe does not have a trading pair with smart contract address {intermediate_pair_contract_address}")

        return trading_pair, intermediate_pair

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
