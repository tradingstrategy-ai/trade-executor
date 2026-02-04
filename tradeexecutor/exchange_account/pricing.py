"""Exchange account position pricing model.

Pricing model for exchange account positions (Derive, Hyperliquid, etc.)
where the account value is already denominated in USD.
"""

import datetime
import logging
from decimal import Decimal
from typing import Callable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing

logger = logging.getLogger(__name__)


class ExchangeAccountPricingModel(PricingModel):
    """Price exchange account positions - always 1:1 USD.

    The account_value_func is protocol-specific and returns
    the total account value in USD from the exchange API.

    Example:

    .. code-block:: python

        from eth_defi.derive.account import fetch_account_summary

        def get_derive_value(pair):
            summary = fetch_account_summary(client, pair.get_exchange_account_id())
            return summary.total_value_usd

        pricing = ExchangeAccountPricingModel(get_derive_value)
    """

    def __init__(
        self,
        account_value_func: Callable[[TradingPairIdentifier], Decimal],
    ):
        """Initialise pricing model.

        :param account_value_func:
            Function that takes a pair and returns account value in USD.
            Signature: (pair: TradingPairIdentifier) -> Decimal
        """
        self.account_value_func = account_value_func

    def get_account_value(self, pair: TradingPairIdentifier) -> Decimal:
        """Get account value from the configured function.

        :param pair:
            Exchange account trading pair
        :return:
            Account value in USD
        """
        return self.account_value_func(pair)

    def get_buy_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        reserve: Decimal | None,
    ) -> TradePricing:
        """Get price for depositing to exchange.

        When depositing, 1 USD = 1 USD (1:1 ratio).

        :param ts:
            Timestamp
        :param pair:
            Exchange account trading pair
        :param reserve:
            Amount of reserve currency being deposited
        :return:
            TradePricing with price=1.0
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        assert isinstance(reserve, Decimal), f"Reserve must be Decimal, got {type(reserve)}"

        return TradePricing(
            price=1.0,
            mid_price=1.0,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=True,  # Buy
            path=[pair],
            read_at=datetime.datetime.utcnow(),
            block_number=None,
            token_in=reserve,
            token_out=reserve,
        )

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Decimal | None,
    ) -> TradePricing:
        """Get price for withdrawing from exchange.

        When withdrawing, 1 USD tracked = 1 USD withdrawn.

        :param ts:
            Timestamp
        :param pair:
            Exchange account trading pair
        :param quantity:
            Amount being withdrawn
        :return:
            TradePricing with price=1.0
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"

        if quantity is None:
            quantity = Decimal("1.0")

        return TradePricing(
            price=1.0,
            mid_price=1.0,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,  # Sell
            path=[pair],
            read_at=datetime.datetime.utcnow(),
            block_number=None,
            token_in=quantity,
            token_out=quantity,
        )

    def get_mid_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> float:
        """Get mid price (always 1.0 for USD denominated accounts).

        :param ts:
            Timestamp
        :param pair:
            Exchange account trading pair
        :return:
            Mid price (always 1.0)
        """
        assert pair.is_exchange_account(), f"Not an exchange account pair: {pair}"
        return 1.0

    def get_pair_fee(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> float | None:
        """Get trading fee for the pair.

        No fees for position tracking (fees are internal to exchange).

        :return:
            0.0 (no fees)
        """
        return 0.0
