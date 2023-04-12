"""Ethereum test trading for uniswap v3."""

import datetime
from decimal import Decimal
from typing import Tuple, List, Optional

from tradingstrategy.pair import PandasPairUniverse
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.gas import estimate_gas_fees
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v3.deployment import UniswapV3Deployment
from eth_defi.uniswap_v3.price import UniswapV3PriceHelper

from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder, TransactionBuilder
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_routing import UniswapV3SimpleRoutingModel, UniswapV3RoutingState
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.state.freeze import freeze_position_on_failed_trade
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.ethereum.ethereumtrader import EthereumTrader


class UniswapV3TestTrader(EthereumTrader):
    """Helper class to trade against EthereumTester unit testing network."""

    def __init__(self,
                 uniswap: UniswapV3Deployment,
                 state: State,
                 pair_universe: PandasPairUniverse,
                 tx_builder: Optional[TransactionBuilder] = None,
                 ):

        super().__init__(tx_builder, state, pair_universe)
        self.uniswap = uniswap

        self.execution_model = UniswapV3ExecutionModel(tx_builder)
        self.price_helper = UniswapV3PriceHelper(uniswap)
        self.tx_builder = tx_builder

    def buy(self, pair: TradingPairIdentifier, amount_in_usd: Decimal, execute=True) -> Tuple[TradingPosition, TradeExecution]:
        """Buy token (trading pair) for a certain value."""
        # Estimate buy price
        base_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.to_checksum_address(pair.base.address))
        quote_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.to_checksum_address(pair.quote.address))
        
        amount_in = int(amount_in_usd * (10 ** pair.quote.decimals))
        
        raw_fee = int(pair.fee * 1_000_000)
        
        # TODO see estimate_buy_quantity in eth_defi/uniswap_v2/fees
        raw_assumed_quantity = self.price_helper.get_amount_out(
            amount_in,
            [quote_token.address, base_token.address],
            [raw_fee]
        )
        
        assumed_quantity = Decimal(raw_assumed_quantity) / Decimal(10**pair.base.decimals)
        assumed_price = amount_in_usd / assumed_quantity

        position, trade, created= self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=assumed_quantity,
            reserve=None,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0,
            pair_fee=pair.fee
        )

        if execute:
            self.execute_trades_simple([trade])
        return position, trade

    def sell(self, pair: TradingPairIdentifier, quantity: Decimal, execute=True) -> Tuple[TradingPosition, TradeExecution]:
        """Sell token token (trading pair) for a certain quantity."""

        assert isinstance(quantity, Decimal)
        
        base_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.to_checksum_address(pair.base.address))
        quote_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", Web3.to_checksum_address(pair.quote.address))

        raw_quantity = int(quantity * 10**pair.base.decimals)
        
        raw_fee = int(pair.fee * 1_000_000)
        
        # TODO see estimate_sell_price() in eth_defi/uniswap_v2/fees.py
        raw_assumed_quote_token = self.price_helper.get_amount_out(
            raw_quantity,
            [base_token.address, quote_token.address],
            [raw_fee]
        )
        
        assumed_quota_token = Decimal(raw_assumed_quote_token) / Decimal(10**pair.quote.decimals)

        # assumed_price = quantity / assumed_quota_token
        assumed_price = assumed_quota_token / quantity

        position, trade, created = self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=-quantity,
            reserve=None,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0,
            pair_fee=pair.fee
        )

        if execute:
            self.execute_trades_simple([trade])
        return position, trade

    def execute_trades_simple(
            self,
            trades: List[TradeExecution],
            max_slippage=0.01, 
            stop_on_execution_failure=True
    ) -> Tuple[List[TradeExecution], List[TradeExecution]]:
        """Execute trades on web3 instance.

        A testing shortcut

        - Create `BlockchainTransaction` instances

        - Execute them on Web3 test connection (EthereumTester / Ganache)

        - Works with single Uniswap test deployment
        """

        pair_universe = self.pair_universe   
        web3 = self.web3
        uniswap = self.uniswap
        state = self.state   
        
        assert isinstance(pair_universe, PandasPairUniverse)

        fees = estimate_gas_fees(web3)

        tx_builder = self.tx_builder

        reserve_asset, rate = state.portfolio.get_default_reserve()

        # We know only about one exchange
        routing_model = UniswapV3SimpleRoutingModel(
            address_map={
                "factory": uniswap.factory.address,
                "router": uniswap.swap_router.address,
                "position_manager": uniswap.position_manager.address,
                "quoter": uniswap.quoter.address
            },
            allowed_intermediary_pairs={},
            reserve_token_address=reserve_asset.address,
        )

        state.start_trades(datetime.datetime.utcnow(), trades)
        routing_state = UniswapV3RoutingState(pair_universe, tx_builder)
        routing_model.execute_trades_internal(pair_universe, routing_state, trades)
        
        execution_model = UniswapV3ExecutionModel(self.tx_builder)
        execution_model.broadcast_and_resolve(state, trades, stop_on_execution_failure=stop_on_execution_failure)

        # Clean up failed trades
        freeze_position_on_failed_trade(datetime.datetime.utcnow(), state, trades)

        success = [t for t in trades if t.is_success()]
        failed = [t for t in trades if t.is_failed()]

        return success, failed