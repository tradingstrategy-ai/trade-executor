"""Ethereum test trading."""

import datetime
from decimal import Decimal
from typing import Tuple, List

from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_buy_quantity, estimate_sell_price
from tradeexecutor.ethereum.execution import broadcast_and_resolve
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2RoutingState
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class EthereumTestTrader:
    """Helper class to trade against EthereumTester unit testing network."""

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

    def execute(self, trades: List[TradeExecution]):
        execute_trades_simple(
            self.state,
            self.pair_universe,
            trades,
            self.web3,
            self.hot_wallet,
            self.uniswap,
        )

    def buy(self, pair: TradingPairIdentifier, amount_in_usd: Decimal, execute=True) -> Tuple[TradingPosition, TradeExecution]:
        """Buy token (trading pair) for a certain value."""
        # Estimate buy price
        base_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(pair.base.address))
        quote_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(pair.quote.address))
        raw_assumed_quantity = estimate_buy_quantity(self.uniswap, base_token, quote_token, amount_in_usd * (10 ** pair.quote.decimals))
        assumed_quantity = Decimal(raw_assumed_quantity) / Decimal(10**pair.base.decimals)
        assumed_price = amount_in_usd / assumed_quantity

        position, trade, created= self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=assumed_quantity,
            reserve=None,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        if execute:
            self.execute([trade])
        return position, trade

    def sell(self, pair: TradingPairIdentifier, quantity: Decimal, execute=True) -> Tuple[TradingPosition, TradeExecution]:
        """Sell token token (trading pair) for a certain quantity."""

        assert isinstance(quantity, Decimal)

        base_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(pair.base.address))
        quote_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.toChecksumAddress(pair.quote.address))

        raw_quantity = int(quantity * 10**pair.base.decimals)
        raw_assumed_quote_token = estimate_sell_price(self.uniswap, base_token, quote_token, raw_quantity)
        assumed_quota_token = Decimal(raw_assumed_quote_token) / Decimal(10**pair.quote.decimals)

        # assumed_price = quantity / assumed_quota_token
        assumed_price = assumed_quota_token / quantity

        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=-quantity,
            reserve=None,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        if execute:
            self.execute([trade])
        return position, trade


def execute_trades_simple(
        state: State,
        pair_universe: PandasPairUniverse,
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

    assert isinstance(pair_universe, PandasPairUniverse)

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
        reserve_token_address=reserve_asset.address,
    )

    state.start_trades(datetime.datetime.utcnow(), trades)
    routing_state = UniswapV2RoutingState(pair_universe, tx_builder)
    routing_model.execute_trades_internal(pair_universe, routing_state, trades)
    broadcast_and_resolve(web3, state, trades, stop_on_execution_failure=stop_on_execution_failure)

    # Clean up failed trades
    freeze_position_on_failed_trade(datetime.datetime.utcnow(), state, trades)

    success = [t for t in trades if t.is_success()]
    failed = [t for t in trades if t.is_failed()]

    return success, failed