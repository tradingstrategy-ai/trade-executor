"""Uniswap v2 test trade builder."""

import datetime
from decimal import Decimal
from typing import Tuple, List, Optional

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import UniswapV2LivePricing
from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3

from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_buy_quantity, estimate_sell_price

from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder, TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel, UniswapV2RoutingState
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.ethereum.ethereumtrader import EthereumTrader, get_base_quote_contracts

class UniswapV2TestTrader(EthereumTrader):
    """Helper class to trade against a locally deployed Uniswap v2 contract.

    Allows to execute individual trades without need to go through `decide_trades()`

    May be used with or without :py:attr:`pricing_model`.
    """

    def __init__(self,
                 uniswap: UniswapV2Deployment,
                 state: State,
                 pair_universe: PandasPairUniverse,
                 tx_builder: TransactionBuilder,
                 pricing_model: Optional[UniswapV2LivePricing] = None,
                 ):
        """

        :param web3:
        :param uniswap:
        :param hot_wallet:
        :param state:
        :param pair_universe:
        :param tx_builder:

        :param pricing_model:
            Give if you want to get the lp fees estimated
        """

        assert isinstance(uniswap, UniswapV2Deployment)
        assert isinstance(tx_builder, TransactionBuilder)

        super().__init__(tx_builder, state, pair_universe)

        self.uniswap = uniswap
        self.execution_model = UniswapV2ExecutionModel(tx_builder)
        self.pricing_model = pricing_model

    def buy(self,
            pair: TradingPairIdentifier,
            amount_in_usd: Decimal,
            execute=True,
            slippage_tolerance: Optional[float] = None,
            ) -> Tuple[TradingPosition, TradeExecution]:
        """Buy token (trading pair) for a certain value."""

        base_token, quote_token = get_base_quote_contracts(self.web3, pair)

        if self.pricing_model:
            price_structure = self.pricing_model.get_buy_price(datetime.datetime.utcnow(), pair, amount_in_usd)
            assumed_price = price_structure.price
            estimated_lp_fees = price_structure.get_total_lp_fees()
            assumed_quantity = None
            reserve = amount_in_usd
        else:
            # Shortcut for testing
            raw_assumed_quantity = estimate_buy_quantity(self.uniswap, base_token, quote_token, amount_in_usd * (10 ** pair.quote.decimals))
            assumed_quantity = Decimal(raw_assumed_quantity) / Decimal(10**pair.base.decimals)
            assumed_price = amount_in_usd / assumed_quantity
            price_structure = estimated_lp_fees = None
            reserve = None

        position, trade, created = self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=assumed_quantity,
            reserve=reserve,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0,
            pair_fee=pair.fee,
            slippage_tolerance=slippage_tolerance,
            price_structure=price_structure,
            lp_fees_estimated=estimated_lp_fees,
        )

        if execute:
            self.execute_trades_simple([trade])

        return position, trade

    def sell(
            self,
            pair: TradingPairIdentifier,
            quantity: Decimal,
            execute=True,
            slippage_tolerance: Optional[float] = None,
    ) -> Tuple[TradingPosition, TradeExecution]:
        """Sell tokens on an open position for a certain quantity."""

        assert isinstance(quantity, Decimal)

        base_token, quote_token = get_base_quote_contracts(self.web3, pair)

        if self.pricing_model:
            price_structure = self.pricing_model.get_sell_price(datetime.datetime.utcnow(), pair, quantity)
            assumed_price = price_structure.price
            estimated_lp_fees = price_structure.get_total_lp_fees()
        else:
            # Shortcut in test
            raw_quantity = int(quantity * 10**pair.base.decimals)
            raw_assumed_quote_token = estimate_sell_price(self.uniswap, base_token, quote_token, raw_quantity)
            assumed_quota_token = Decimal(raw_assumed_quote_token) / Decimal(10**pair.quote.decimals)

            # assumed_price = quantity / assumed_quota_token
            assumed_price = assumed_quota_token / quantity
            price_structure = estimated_lp_fees = None

        position, trade, created = self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=-quantity,
            reserve=None,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0,
            pair_fee=pair.fee,
            slippage_tolerance=slippage_tolerance,
            price_structure=price_structure,
            lp_fees_estimated=estimated_lp_fees,
        )

        if execute:
            self.execute_trades_simple([trade])

        return position, trade
    
    def execute_trades_simple(
        self,
        trades: List[TradeExecution],
        stop_on_execution_failure=True,
        broadcast=True,
    ) -> Tuple[List[TradeExecution], List[TradeExecution]]:
        """Execute trades on web3 instance.

        A testing shortcut

        - Create `BlockchainTransaction` instances of each gives `TradeExecution`

        - Execute them on Web3 test connection (EthereumTester / Ganache)

        - Works with single Uniswap test deployment

        :param trades:
            Trades to be converted to blockchain transactions

        :param stop_on_execution_failure:
            Raise exception on an error

        :param broadcast:
            Broadcast trades over web3 connection
        """

        pair_universe = self.pair_universe   
        uniswap = self.uniswap
        state = self.state
        
        assert isinstance(pair_universe, PandasPairUniverse)

        reserve_asset, rate = state.portfolio.get_default_reserve()

        # We know only about one exchange
        routing_model = UniswapV2SimpleRoutingModel(
            factory_router_map={
                uniswap.factory.address: (uniswap.router.address, uniswap.init_code_hash),
            },
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_asset.address,
            trading_fee=0.030,
        )

        state.start_trades(datetime.datetime.utcnow(), trades)
        routing_state = UniswapV2RoutingState(pair_universe, self.tx_builder)
        routing_model.execute_trades_internal(pair_universe, routing_state, trades)

        if broadcast:
            self.broadcast_trades(trades, stop_on_execution_failure)

    def broadcast_trades(self, trades: List[TradeExecution], stop_on_execution_failure=True):
        """Broadcast prepared trades."""

        state = self.state

        execution_model = UniswapV2ExecutionModel(self.tx_builder)
        execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=stop_on_execution_failure)

        # Clean up failed trades
        freeze_position_on_failed_trade(datetime.datetime.utcnow(), state, trades)

        success = [t for t in trades if t.is_success()]
        failed = [t for t in trades if t.is_failed()]

        return success, failed
