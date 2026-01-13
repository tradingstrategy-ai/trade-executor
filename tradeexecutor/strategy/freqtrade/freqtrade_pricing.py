"""Freqtrade position pricing model."""

import datetime
import logging
from decimal import Decimal
from typing import Optional

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.freqtrade.freqtrade_client import FreqtradeClient

logger = logging.getLogger(__name__)


class FreqtradePricingModel(PricingModel):
    """Price Freqtrade positions by querying Freqtrade REST API.

    Freqtrade positions track capital allocation to a Freqtrade bot instance.
    Pricing is always 1.0 (1:1 ratio with reserve currency like USDT).
    """

    def __init__(self, freqtrade_clients: dict[str, FreqtradeClient]):
        """Initialize pricing model.

        Args:
            freqtrade_clients: Dict mapping freqtrade_id -> FreqtradeClient
        """
        self.clients = freqtrade_clients

    def _get_freqtrade_balance(self, pair: TradingPairIdentifier) -> Decimal:
        """Query Freqtrade API for current total balance.

        Args:
            pair: Trading pair with Freqtrade metadata in other_data

        Returns:
            Current balance in reserve currency (e.g., USDT)

        Raises:
            KeyError: If freqtrade_id not in pair.other_data
            Exception: If API call fails
        """
        freqtrade_id = pair.other_data["freqtrade_id"]
        client = self.clients[freqtrade_id]

        try:
            balance_data = client.get_balance()
            # API returns: {"total": X, "free": Y, "used": Z}
            # Total includes unrealized PnL from open positions
            total = balance_data.get("total", 0)
            return Decimal(str(total))
        except Exception as e:
            logger.error(f"Failed to get Freqtrade balance for {freqtrade_id}: {e}")
            raise

    def get_buy_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        reserve: Optional[Decimal],
    ) -> TradePricing:
        """Get price for depositing to Freqtrade.

        When depositing to Freqtrade, 1 USDT = 1 USDT (1:1 ratio).

        Args:
            ts: Timestamp
            pair: Freqtrade trading pair
            reserve: Amount of reserve currency being deposited

        Returns:
            TradePricing with price=1.0
        """
        assert pair.is_freqtrade(), f"Not a Freqtrade pair: {pair}"
        assert isinstance(reserve, Decimal), f"Reserve must be Decimal, got {type(reserve)}"

        # Price is always 1.0 for deposits (no conversion)
        price = 1.0
        mid_price = 1.0

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=True,  # Buy
            path=[pair],
            read_at=datetime.datetime.utcnow(),
            block_number=None,  # No blockchain
            token_in=reserve,  # USDT deposit
            token_out=reserve,  # Get same amount tracked
        )

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Optional[Decimal],
    ) -> TradePricing:
        """Get price for withdrawing from Freqtrade.

        When withdrawing from Freqtrade, 1 tracked USDT = 1 USDT withdrawn.

        Args:
            ts: Timestamp
            pair: Freqtrade trading pair
            quantity: Amount of tracked units being withdrawn

        Returns:
            TradePricing with price=1.0
        """
        assert pair.is_freqtrade(), f"Not a Freqtrade pair: {pair}"

        if quantity is None:
            quantity = Decimal("1.0")

        # Price is 1.0 (1 tracked unit = 1 reserve currency)
        price = 1.0
        mid_price = 1.0

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,  # Sell
            path=[pair],
            read_at=datetime.datetime.utcnow(),
            block_number=None,  # No blockchain
            token_in=quantity,
            token_out=quantity,  # 1:1 withdrawal
        )

    def get_mid_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> float:
        """Get mid price (always 1.0 for Freqtrade).

        Args:
            ts: Timestamp
            pair: Freqtrade trading pair

        Returns:
            Mid price (always 1.0)
        """
        assert pair.is_freqtrade(), f"Not a Freqtrade pair: {pair}"
        return 1.0

    def get_pair_fee(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> Optional[float]:
        """Get trading fee for the pair.

        No fees for Freqtrade position tracking (fees are internal to Freqtrade).

        Args:
            ts: Timestamp
            pair: Freqtrade trading pair

        Returns:
            0.0 (no fees)
        """
        return 0.0
