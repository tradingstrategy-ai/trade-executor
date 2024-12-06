"""Trade routing instructions.

Each trading universe and strategy can have different trade routing set,
based on the exchanges the universe covers.

Here we define the abstract overview of routing.
"""
import abc
from typing import List, Optional, Dict
import logging

from hexbytes import HexBytes

from tradeexecutor.state.identifier import TradingPairIdentifier, AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.types import JSONHexAddress
from tradeexecutor.strategy.universe_model import StrategyExecutionUniverse
from tradeexecutor.strategy.trading_strategy_universe import translate_token, translate_trading_pair

from tradingstrategy.pair import PandasPairUniverse, PairNotFoundError


logger = logging.getLogger(__name__)


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


class RoutingModel(abc.ABC):
    """Trade roouting model base class.

    Nothing done here - check the subclasses.
    
    Used directly by BacktestRoutingModel and indirectly (through EthereumRoutingModel) by UniswapV2SimpleRoutingModel and UniswapV3SimpleRoutingModel
    """
    
    def __init__(self,
                 allowed_intermediary_pairs: dict[JSONHexAddress, JSONHexAddress],
                 reserve_token_address: str,
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

        assert type(allowed_intermediary_pairs) == dict
        assert type(reserve_token_address) == str, f"Got {reserve_token_address}"

        assert reserve_token_address.lower() == reserve_token_address, f"reserve token address must be specified as lower case, got {reserve_token_address}"

        self.allowed_intermediary_pairs = self.convert_address_dict_to_lower(allowed_intermediary_pairs)

        self.reserve_token_address = reserve_token_address

        logger.info("Initialised %s", self)
        if self.allowed_intermediary_pairs:
            for token_address, pair_address in self.allowed_intermediary_pairs.items():
                logger.info(
                    "Intermediate pair whitelisted. Token: %s, pair: %s",
                    token_address,
                    pair_address,
                )
        else:
            logger.info("No intermediate pairs whitelisted")

    @staticmethod
    def convert_address_dict_to_lower(address_dict) -> dict:
        """Convert all key addresses to lowercase to avoid mix up with Ethereum address checksums"""
        return {k.lower(): v for k, v in address_dict.items()}
    
    @staticmethod
    def pre_trade_assertions(reserve_asset_amount: int, max_slippage: float, target_pair: TradingPairIdentifier, reserve_asset: AssetIdentifier) -> None:
        """Some basic assertions made at the beginning of the trade() method on child class.
        
        returns: None. 
            An error will be raised during method call if assertions aren't met."""
        assert type(reserve_asset_amount) == int
        assert max_slippage is not None, "Max slippage must be given"
        assert type(max_slippage) == float
        assert reserve_asset_amount > 0, f"For sells, switch reserve_asset to different token. Got target_pair: {target_pair}, reserve_asset: {reserve_asset}, amount: {reserve_asset_amount}"
    
    def get_reserve_asset(self, pair_universe: PandasPairUniverse) -> AssetIdentifier:
        """Translate our reserve token address tok an asset description."""
        assert pair_universe is not None, "Pair universe missing"
        reserve_token = pair_universe.get_token(self.reserve_token_address)
        # Dump all tokens here into the error message is the number of tokens is suffiently small
        if reserve_token is None:
            token_data_msg = f"Universe has {pair_universe.get_count():,} pairs\n"
            if pair_universe.get_count() < 20:
                token_data_msg = f"Tokens are:\n"
                for token in pair_universe.get_all_tokens():
                    token_data_msg += f"  {token}\n"
            assert reserve_token, f"Pair universe does not contain our reserve asset {self.reserve_token_address}\n{token_data_msg}\nWe are {self}"
        return translate_token(reserve_token)
     
    def route_pair(
        self,
        pair_universe: PandasPairUniverse,
        trading_pair: TradingPairIdentifier,
        require_same_dex=True,
    ) -> tuple[TradingPairIdentifier, Optional[TradingPairIdentifier]]:
        """Return Uniswap routing information (path components) for a trading pair.

        For three-way pairs, figure out the intermedia step.

        :return:
            (router address, target pair, intermediate pair) tuple
        """

        self.route_pair_assertions(trading_pair, pair_universe)
        
        reserve_asset = self.get_reserve_asset(pair_universe)

        # We can directly do a two-way trade
        if trading_pair.quote == reserve_asset:
            return trading_pair, None

        # Try to find a mid-hop pool for the trade
        intermediate_pair_contract_address = self.allowed_intermediary_pairs.get(trading_pair.quote.address.lower())

        if not intermediate_pair_contract_address:
            raise CannotRouteTrade(f"Does not know how to trade pair {trading_pair} - supported intermediate tokens are {list(self.allowed_intermediary_pairs.keys())}")

        try:
            dex_pair = pair_universe.get_pair_by_smart_contract(intermediate_pair_contract_address)
            pne = None
        except PairNotFoundError as e:
            # We have not trading pair data loaded for the intermediate pair
            dex_pair = None
            pne = e

        assert dex_pair is not None, f"Intermediate pair not found: Pair universe did not contain pair for a pair contract address {intermediate_pair_contract_address}, quote token is {trading_pair.quote}:\n{pne}"

        if intermediate_pair := translate_trading_pair(dex_pair):
            return trading_pair, intermediate_pair
        else:
            raise CannotRouteTrade(f"Universe does not have a trading pair with smart contract address {intermediate_pair_contract_address}")

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
    def setup_trades(
            self,
            state: RoutingState,
            trades: List[TradeExecution],
            check_balances=False,
            rebroadcast=False,
    ):
        """Setup the trades decided by a strategy.

        - Decides the best way, or a way, to execute a trade

        - Sets up blockchain transactions needed for trades,
          like approves

        - Trade instances are mutated in-place

        :param check_balances:
            Check that the wallet has enough reserves to perform the trades
            before executing them. Because we are selling before buying.
            sometimes we do no know this until the sell tx has been completed.

        :param rebroadcast:
            Allow rebroadcast of already broadcasted trades, but for which we did not get a receipt yet.

        :raise CannotExecuteTrade:
            If a trade cannot be executed, e.g. due to an unsupported pair or an exchange,
        """

    def settle_trade(
        self,
        web3,
        state: State,
        trade: TradeExecution,
        receipts: Dict[HexBytes, dict],
        stop_on_execution_failure=False,
    ):
        """Post-trade

        - Read on-chain data about the execution success and performance

        - Mark trade succeed or failed

        :param state:
            Strategy state

        :param web3:
            Web3 connection.

            TODO: Breaks abstraction. Figure better way to pass
            this around later. Maybe create an Ethereum-specific
            routing parent class?

        :param trade:
            Trade executed in this execution batch

        :param receipts:
            Blockchain receipts we received in this execution batch.

            Hash -> receipt mapping.

        :param stop_on_execution_failure:
            Raise an error if the trade failed.

            Used in unit testing.
        """
        raise NotImplementedError()