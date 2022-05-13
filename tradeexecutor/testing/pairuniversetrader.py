"""Ethereum test trading."""

import datetime
from decimal import Decimal
from typing import Tuple, List

from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.hotwallet import HotWallet
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from eth_defi.uniswap_v2.fees import estimate_buy_quantity, estimate_sell_price
from tradeexecutor.ethereum.execution import approve_tokens, prepare_swaps, confirm_approvals, broadcast, \
    wait_trades_to_complete, resolve_trades
from tradeexecutor.state.state import State, TradeType
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.state.identifier import TradingPairIdentifier


class PairUniverseTestTrader:
    """Helper class to create trades for testing..

    Trades are executed by the routing model.
    """

    def __init__(self, state: State):
        self.state = state

    def buy(self, pair: TradingPairIdentifier, amount_in_usd: Decimal) -> TradeExecution:
        """Buy token (trading pair) for a certain value."""

        reserve_currency, exchange_rate = self.state.portfolio.get_default_reserve_currency()

        position, trade, created = self.state.create_trade(
            ts=datetime.datetime.utcnow(),
            pair=pair,
            assumed_price=1.0,
            quantity=None,
            reserve=amount_in_usd,
            trade_type=TradeType.rebalance,
            reserve_currency=reserve_currency,
            reserve_currency_price=1.0)

        return trade

    def sell(self, pair: TradingPairIdentifier, quantity: Decimal, execute=True) -> Tuple[TradingPosition, TradeExecution]:
        """Sell token token (trading pair) for a certain quantity."""

        assert isinstance(quantity, Decimal)

        base_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", pair.base.address)
        quote_token = get_deployed_contract(self.web3, "ERC20MockDecimals.json", pair.quote.address)

        raw_quantity = int(quantity * 10**pair.base.decimals)
        raw_assumed_quote_token = estimate_sell_price(self.uniswap, base_token, quote_token, raw_quantity)
        assumed_quota_token = Decimal(raw_assumed_quote_token) / Decimal(10**pair.quote.decimals)

        # assumed_price = quantity / assumed_quota_token
        assumed_price = assumed_quota_token / quantity

        position, trade, created = self.state.create_trade(
            ts=self.ts,
            pair=pair,
            quantity=-quantity,
            assumed_price=float(assumed_price),
            trade_type=TradeType.rebalance,
            reserve_currency=pair.quote,
            reserve_currency_price=1.0)

        if execute:
            self.execute([trade])
        return position, trade
