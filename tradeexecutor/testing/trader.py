import datetime
from decimal import Decimal
from typing import Tuple, List

from web3 import Web3

from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.ethereum.execution import broadcast_and_resolve
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2RoutingState
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier


class DummyTestTrader:
    """Helper class to generate trades for tests.

    This trade helper is not connected to any blockchain - it just simulates txid and nonce values.
    """

    def __init__(self, state: State, lp_fees=2.50, price_impact=0.99):
        self.state = state
        self.nonce = 1
        self.ts = datetime.datetime(2022, 1, 1, tzinfo=None)

        self.lp_fees = lp_fees
        self.price_impact = price_impact
        self.native_token_price = 1

    def create(self, pair: TradingPairIdentifier, quantity: Decimal, price: float) -> Tuple[TradingPosition, TradeExecution]:
        """Open a new trade."""
        # 1. Plan
        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=None,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        self.ts += datetime.timedelta(seconds=1)
        return position, trade

    def create_and_execute(self, pair: TradingPairIdentifier, quantity: Decimal, price: float) -> Tuple[TradingPosition, TradeExecution]:

        price_impact = self.price_impact

        # 1. Plan
        position, trade = self.create(
            pair=pair,
            quantity=quantity,
            price=price)

        # 2. Capital allocation
        txid = hex(self.nonce)
        nonce = self.nonce
        self.state.start_execution(self.ts, trade, txid, nonce)

        # 3. broadcast
        self.nonce += 1
        self.ts += datetime.timedelta(seconds=1)

        self.state.mark_broadcasted(self.ts, trade)
        self.ts += datetime.timedelta(seconds=1)

        # 4. executed
        executed_price = price * price_impact
        if trade.is_buy():
            executed_quantity = quantity * Decimal(price_impact)
            executed_reserve = Decimal(0)
        else:
            executed_quantity = quantity
            executed_reserve = abs(quantity * Decimal(executed_price))

        self.state.mark_trade_success(self.ts, trade, executed_price, executed_quantity, executed_reserve, self.lp_fees, self.native_token_price)
        return position, trade

    def buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, quantity, price)

    def prepare_buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create(pair, quantity, price)

    def sell(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, -quantity, price)


def execute_trades_simple(
        state: State,
        trades: List[TradeExecution],
        web3: Web3,
        hot_wallet: HotWallet,
        uniswap: UniswapV2Deployment,
        max_slippage=0.01, stop_on_execution_failure=True) -> Tuple[List[TradeExecution], List[TradeExecution]]:
    """Execute trades on web3 instance.

    A testing shortcut

    - Create `BlockchainTransaction` instances

    - Execute them on Web3 test connection (EthereumTester / Ganache)

    - Works with single Uniswap test deployment
    """

    fees = estimate_gas_fees(web3)

    tx_builder = TransactionBuilder(
        web3,
        hot_wallet,
        fees,
    )

    reserve_asset, rate = state.portfolio.get_default_reserve_currency()

    # We know only about one exchange
    routing_model = UniswapV2SimpleRoutingModel(
        factory_router_map={
            uniswap.factory.address: (uniswap.router.address, uniswap.init_code_hash),
        },
        allowed_intermediary_pairs={},
        reserve_asset=reserve_asset,
        max_slippage=max_slippage,
    )

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_state = UniswapV2RoutingState(tx_builder)
    routing_model.execute_trades(None, routing_state, trades)
    broadcast_and_resolve(web3, state, trades, stop_on_execution_failure=stop_on_execution_failure)

    # Clean up failed trades
    freeze_position_on_failed_trade(datetime.datetime.utcnow(), state, trades)

    success = [t for t in trades if t.is_success()]
    failed = [t for t in trades if t.is_failed()]

    return success, failed