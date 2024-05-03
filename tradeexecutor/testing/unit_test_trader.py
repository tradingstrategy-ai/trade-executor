"""Simple trade generator without execution."""
import datetime
from decimal import Decimal
from typing import Tuple

import pandas as pd

from tradeexecutor.state.balance_update import BalanceUpdate, BalanceUpdatePositionType, BalanceUpdateCause
from tradeexecutor.state.portfolio import Portfolio
from tradeexecutor.state.reserve import ReservePosition
from tradingstrategy.candle import GroupedCandleUniverse

from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeFlag 
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.utils.leverage_calculations import LeverageEstimate
from tradingstrategy.types import Percent


class UnitTestTrader:
    """Helper class to generate and settle trades in unit tests.

    This trade helper is not connected to any blockchain - it just simulates txid and nonce values.
    """

    def __init__(self, state: State, lp_fees=2.50, price_impact=0.99):
        self.state = state
        self.nonce = 1
        self.ts = datetime.datetime(2022, 1, 1, tzinfo=None)

        self.lp_fees = lp_fees
        self.price_impact = price_impact
        self.native_token_price = 1

    def time_travel(self, timestamp: datetime.datetime):
        self.ts = timestamp

    def create(self, pair: TradingPairIdentifier, quantity: Decimal, price: float) -> Tuple[TradingPosition, TradeExecution]:
        """Open a new trade."""
        # 1. Plan
        
        position, trade, created = self.state.create_trade(
            strategy_cycle_at=self.ts,
            pair=pair,
            quantity=quantity,
            reserve=None,
            assumed_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0,
            planned_mid_price=price,
            pair_fee=pair.fee
        )

        self.ts += datetime.timedelta(seconds=1)
        return position, trade

    def create_and_execute(self, pair: TradingPairIdentifier, quantity: Decimal, price: float, underflow_check=True) -> Tuple[TradingPosition, TradeExecution]:

        assert price > 0
        assert quantity != 0

        price_impact = self.price_impact

        # 1. Plan
        position, trade = self.create(
            pair=pair,
            quantity=quantity,
            price=price)

        # 2. Capital allocation
        txid = hex(self.nonce)
        nonce = self.nonce
        self.state.start_execution(self.ts, trade, txid, nonce, underflow_check=underflow_check)

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

        self.state.mark_trade_success(
            self.ts,
            trade,
            executed_price,
            executed_quantity,
            executed_reserve,
            self.lp_fees,
            self.native_token_price)
        return position, trade

    def set_perfectly_executed(self, trade: TradeExecution):
        """Sets trade to a executed state.

        - There are no checks whether the wallet contains relevant balances or not
        """

        # 2. Capital allocation
        txid = hex(self.nonce)
        nonce = self.nonce
        self.state.start_execution(self.ts, trade, txid, nonce, underflow_check=False)

        # 3. broadcast
        self.nonce += 1
        self.ts += datetime.timedelta(seconds=1)

        self.state.mark_broadcasted(self.ts, trade)
        self.ts += datetime.timedelta(seconds=1)

        # 4. executed
        executed_price = trade.planned_price
        executed_collateral_consumption = trade.planned_collateral_consumption
        executed_collateral_allocation = trade.planned_collateral_allocation

        if trade.is_spot():
            if trade.is_buy():
                executed_quantity = trade.planned_quantity
                executed_reserve = Decimal(0)  # TODO: Check if we can reorg code here more cleanly
            else:
                # short reduction also changes the reserve (releases collateral)
                executed_quantity = trade.planned_quantity
                executed_reserve = trade.planned_reserve
        else:
            executed_quantity = trade.planned_quantity
            executed_reserve = trade.planned_reserve

        if trade.planned_loan_update:
            trade.executed_loan_update = trade.planned_loan_update

        lp_fees = trade.lp_fees_estimated or self.lp_fees

        self.state.mark_trade_success(
            self.ts,
            trade,
            executed_price,
            executed_quantity,
            executed_reserve,
            lp_fees,
            self.native_token_price,
            executed_collateral_consumption=executed_collateral_consumption,
            executed_collateral_allocation=executed_collateral_allocation,
        )

    def buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, quantity, price)

    def prepare_buy(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create(pair, quantity, price)

    def sell(self, pair, quantity, price) -> Tuple[TradingPosition, TradeExecution]:
        return self.create_and_execute(pair, -quantity, price)

    def buy_with_price_data(self, pair, quantity, candle_universe: GroupedCandleUniverse) -> Tuple[TradingPosition, TradeExecution]:
        price = candle_universe.get_closest_price(pair.internal_id, pd.Timestamp(self.ts))
        return self.create_and_execute(pair, quantity, float(price))

    def sell_with_price_data(self, pair, quantity, candle_universe: GroupedCandleUniverse) -> Tuple[TradingPosition, TradeExecution]:
        price = candle_universe.get_closest_price(pair.internal_id, pd.Timestamp(self.ts))
        return self.create_and_execute(pair, -quantity, float(price))

    def open_short(self, pair, quantity, price, leverage=1) -> tuple[TradingPosition, TradeExecution]:
        assert pair.kind.is_leverage()

        estimation = LeverageEstimate.open_short(
            starting_reserve=quantity,
            leverage=leverage,
            borrowed_asset_price=price,
            shorting_pair=pair,
            fee=pair.get_pricing_pair().fee,
        )

        position, trade, _ = self.state.trade_short(
            strategy_cycle_at=self.ts,
            pair=pair,
            borrowed_quantity=-estimation.borrowed_quantity,
            collateral_quantity=quantity,
            borrowed_asset_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.get_pricing_pair().quote,
            collateral_asset_price=1.0,
            planned_collateral_consumption=estimation.additional_collateral_quantity,
            lp_fees_estimated=estimation.lp_fees,
            flags={TradeFlag.open},
        )

        if trade.planned_loan_update:
            trade.executed_loan_update = trade.planned_loan_update

        self.ts += datetime.timedelta(seconds=1)
        return position, trade

    def close_short(self, pair, quantity, price, leverage=1) -> tuple[TradingPosition, TradeExecution]:
        assert pair.kind.is_leverage()

        position, trade, _ = self.state.trade_short(
            strategy_cycle_at=self.ts,
            closing=True,
            pair=pair,
            borrowed_asset_price=price,
            planned_mid_price=price,
            trade_type=TradeType.rebalance,
            reserve_currency=pair.get_pricing_pair().quote,
            collateral_asset_price=1.0,
            flags={TradeFlag.close},
        )

        if trade.planned_loan_update:
            trade.executed_loan_update = trade.planned_loan_update

        self.ts += datetime.timedelta(seconds=1)
        return position, trade

    def redeem_in_kind(
        self,
        timestamp: datetime.datetime,
        portfolio: Portfolio,
        position: TradingPosition,
        quantity: Decimal,
        redeemer="0x0000000000000000000000000000000000000000",
    ) -> BalanceUpdate:
        """Simulate in-kind redemption.

        :param quantity:
            How much to redeem, in the spot units hold.
        """

        assert position.is_long()
        assert position.is_spot()
        assert isinstance(quantity, Decimal)

        asset = position.pair.base

        event_id = portfolio.next_balance_update_id
        portfolio.next_balance_update_id += 1

        if isinstance(position, ReservePosition):
            position_id = None
            old_balance = position.quantity
            position.quantity -= quantity
            position_type = BalanceUpdatePositionType.reserve
            # TODO: USD stablecoin hardcoded to 1:1 with USD
            usd_value = float(quantity)
        elif isinstance(position, TradingPosition):
            position_id = position.position_id
            old_balance = position.get_quantity()
            position_type = BalanceUpdatePositionType.open_position
            usd_value = position.calculate_quantity_usd_value(quantity)
        else:
            raise NotImplementedError()

        assert old_balance - quantity >= 0, f"Position went to negative: {position}\n" \
                                            f"Quantity: {quantity}, old balance: {old_balance}"

        assert timestamp is not None, f"Timestamp cannot be none: {timestamp}"

        evt = BalanceUpdate(
            balance_update_id=event_id,
            cause=BalanceUpdateCause.redemption,
            position_type=position_type,
            asset=asset,
            block_mined_at=timestamp,
            strategy_cycle_included_at=timestamp,
            chain_id=asset.chain_id,
            quantity=-quantity,
            old_balance=old_balance,
            owner_address=redeemer,
            tx_hash=None,
            log_index=None,
            position_id=position_id,
            usd_value=usd_value,
            block_number=None
        )

        position.add_balance_update_event(evt)

        return evt

