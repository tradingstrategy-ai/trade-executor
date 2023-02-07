"""Trade routing instructions.

Each trading universe and strategy can have different trade routing set,
based on the exchanges the universe covers.

Here we define the abstract overview of routing.
"""
import abc
from typing import List, Optional

from web3 import Web3
from web3.contract import Contract
from eth_defi.abi import get_deployed_contract

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse

from tradingstrategy.pair import PandasPairUniverse


class CannotRouteTrade(Exception):
    """The router does not know who to execute a trade decided by a strategy."""


class RoutingState(abc.ABC):
    """Keep the track record of already done transactions.

    When performing multiple blockchain transactions for multiple trades
    in one cycle, we need to know what approvals and such we have already done.

    Life cycle

    - Created early at the cycle

    - Used for price revaluation

    - Used for execution

    - May cache information about the past price lookups

    - Must cache information about the already on approve() etc
      blockchain transactions relevant to trades

    - Discarded at the end of the cycle
    """

    @abc.abstractmethod
    def __init__(self, universe: "tradeexecutor.strategy.universe_model.StrategyExecutionUniverse"):
        #: Each routing state is specific to the current trading universe.
        #: The trade routing will change when new pairs are added and old goes away.
        self.universe = universe
    
    def get_base_quote(self, web3: Web3, target_pair: TradingPairIdentifier, reserve_asset: AssetIdentifier, error_msg: str = None):
        """Get base and quote token from the pair and reserve asset. Called in parent class (RoutingState) with error_msg.
        
        See: https://tradingstrategy.ai/docs/programming/market-data/trading-pairs.html
        
        :param target_pair: Pair to be traded
        :param reserver_asset: Asset to be kept as reserves
        :returns: (base_token: Contract, quote_token: Contract)
        :param error_msg:
            Only provide this argument if error message includes external info such as an intermediary pair
        """
        if error_msg is None:
            error_msg = f"Cannot route trade through {target_pair}"
        
        if reserve_asset == target_pair.quote:
            # Buy with e.g. BUSD
            base_token = self.get_token_for_asset(web3, target_pair.base)
            quote_token = self.get_token_for_asset(web3, target_pair.quote)
            
        elif reserve_asset == target_pair.base:
            # Sell, flip the direction
            base_token = self.get_token_for_asset(web3, target_pair.quote)
            quote_token = self.get_token_for_asset(web3, target_pair.base)
            
        else:
            raise RuntimeError(error_msg)
        
        return base_token, quote_token
    
    @staticmethod
    def get_token_for_asset(web3: Web3, asset: AssetIdentifier) -> Contract:
        """Get ERC-20 contract proxy."""
        erc_20 = get_deployed_contract(web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(asset.address))
        return erc_20


class RoutingModel(abc.ABC):
    """Trade roouting model base class.

    Nothing done here - check the subclasses.
    """

    def perform_preflight_checks_and_logging(self, pair_universe: PandasPairUniverse):
        """"Checks the integrity of the routing.

        - Called from check-wallet to see our routing and balances are good
        """

    def get_default_trading_fee(self) -> Optional[float]:
        """Get the trading/LP fee applied to all trading pairs.

        This is Uni v2 style fee.

        :return:
            Trading fee, BPS * 10000 as a float.

            If information not available return None.
        """
        return None

    @abc.abstractmethod
    def create_routing_state(self,
                             universe: StrategyExecutionUniverse,
                             execution_details: object) -> RoutingState:
        """Create a new routing state for this cycle.

        :param execution_details:
            A dict of whatever connects live execution to routing.
        """

    @abc.abstractmethod
    def setup_trades(self,
                     state: RoutingState,
                     trades: List[TradeExecution],
                     check_balances=False):
        """Setup the trades decided by a strategy.

        - Decides the best way, or a way, to execute a trade

        - Sets up blockchain transactions needed for trades,
          like approves

        - Trade instances are mutated in-place

        :param check_balances:
            Check that the wallet has enough reserves to perform the trades
            before executing them. Because we are selling before buying.
            sometimes we do no know this until the sell tx has been completed.

        :raise CannotExecuteTrade:
            If a trade cannot be executed, e.g. due to an unsupported pair or an exchange,
        """


