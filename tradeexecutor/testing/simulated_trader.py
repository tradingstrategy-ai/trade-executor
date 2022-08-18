"""Helper to simulate trades without going through the whole backtesting integration."""

import datetime
from decimal import Decimal

from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel


class SimulatedTestTrader:
    """Create trades.
    """

    def __init__(self,
                 state: State,
                 pricing_model: PricingModel):
        self.state = state
        self.nonce = 1
        self.pricing_model = pricing_model

    def buy(self,
            ts: datetime.datetime,
            pair: TradingPairIdentifier,
            amount_in_usd: Decimal) -> TradeExecution:
        """Buy token (trading pair) for a certain value."""

        price = self.pricing_model.get_buy_price(ts, pair, amount_in_usd)

        reserve_currency, exchange_rate = self.state.portfolio.get_default_reserve_currency()

        position, trade, created = self.state.create_trade(
            ts=datetime.datetime.utcnow(),
            pair=pair,
            assumed_price=price,
            quantity=None,
            reserve=amount_in_usd,
            trade_type=TradeType.rebalance,
            reserve_currency=reserve_currency,
            reserve_currency_price=1.0)

        return trade

    def sell(self, pair: TradingPairIdentifier, quantity: Decimal) -> TradeExecution:
        """Sell token token (trading pair) for a certain quantity."""

        reserve_currency, exchange_rate = self.state.portfolio.get_default_reserve_currency()

        position, trade, created = self.state.create_trade(
            ts=datetime.datetime.utcnow(),
            pair=pair,
            assumed_price=1.0,
            quantity=-quantity,
            reserve=None,
            trade_type=TradeType.rebalance,
            reserve_currency=reserve_currency,
            reserve_currency_price=1.0)

        return trade

    def simulate_execution(self, state: State, trade: TradeExecution, price_impact=1):
        ts = trade.opened_at
        price = trade.planned_price
        quantity = trade.planned_quantity
        assert price
        assert quantity

        # 2. Capital allocation
        txid = "0x0"
        nonce = self.nonce
        self.state.start_execution(ts, trade, txid, nonce)
        self.nonce += 1

        # 3. Broadcast
        self.state.mark_broadcasted(ts, trade)

        # 3. Executed
        executed_price = price * price_impact
        if trade.is_buy():
            executed_quantity = quantity * Decimal(price_impact)
            executed_reserve = Decimal(0)
        else:
            executed_quantity = quantity
            executed_reserve = abs(quantity * Decimal(executed_price))

        state.mark_trade_success(
            ts,
            trade,
            executed_price,
            executed_quantity,
            executed_reserve,
            lp_fees=0,
            native_token_price=1)
