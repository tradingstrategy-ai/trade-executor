import datetime
from abc import ABC, abstractmethod
from decimal import Decimal
from web3 import Web3
from web3.contract import Contract

from eth_defi.hotwallet import HotWallet
from eth_defi.abi import get_deployed_contract

from tradeexecutor.ethereum.uniswap_v2_execution import UniswapV2Deployment
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition

from tradingstrategy.pair import PandasPairUniverse

class EthereumTrader(ABC):

    @abstractmethod
    def __init__(self, web3: Web3, uniswap: UniswapV2Deployment, hot_wallet: HotWallet, state: State, pair_universe: PandasPairUniverse):
        self.web3 = web3
        self.uniswap = uniswap
        self.state = state
        self.hot_wallet = hot_wallet
        self.pair_universe = pair_universe

        self.ts = datetime.datetime(2022, 1, 1, tzinfo=None)
        self.lp_fees = 2.50  # $2.5
        self.gas_units_consumed = 150_000  # 150k gas units per swap
        self.gas_price = 15 * 10**9  # 15 Gwei/gas unit

        self.native_token_price = 1
        self.confirmation_block_count = 0

    @abstractmethod
    def buy(self, pair: TradingPairIdentifier, amount_in_usd: Decimal, execute=True) -> tuple[TradingPosition, TradeExecution]:
        """Buy token (trading pair) for a certain value."""
        
    @abstractmethod
    def sell(self, pair: TradingPairIdentifier, quantity: Decimal, execute=True) -> tuple[TradingPosition, TradeExecution]:
        """Sell token token (trading pair) for a certain quantity."""
        
    @abstractmethod
    def execute_trades_simple(
        self,
        trades: list[TradeExecution],
        max_slippage=0.01, 
        stop_on_execution_failure=True
    ):
        """Execute trades on web3 instance.

        A testing shortcut

        - Create `BlockchainTransaction` instances

        - Execute them on Web3 test connection (EthereumTester / Ganache)

        - Works with single Uniswap test deployment
        """
        
# TODO check if duplicated, maybe in routing_model or routing_state
def get_base_quote_contracts(web3: Web3, pair: TradingPairIdentifier) -> tuple[Contract]:
    return (
        get_mock_erc20_contract(web3, pair.base.address),
        get_mock_erc20_contract(web3, pair.quote.address),
    )

def get_mock_erc20_contract(web3: Web3, address: str):
    return get_deployed_contract(web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(address))