"""Orderly vault share price estimator."""
import logging
import datetime
from decimal import Decimal
from typing import Optional

from eth_defi.orderly.vault import OrderlyVault
from web3 import Web3

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing

logger = logging.getLogger(__name__)


class OrderlyPricing(PricingModel):
    """Always pull the latest live share price of an Orderly vault.

    .. note::

        Only supports stablecoin-nominated vaults (USDC for Orderly)
    """

    def __init__(
        self,
        web3: Web3,
        vault: OrderlyVault,
    ):
        self.web3 = web3
        self.vault = vault

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Optional[Decimal],
    ) -> TradePricing:
        """Get live price on Orderly vault for redeeming our shares.

        For Orderly, this represents withdrawing from the vault.
        """

        assert pair.is_vault()

        if quantity is None:
            quantity = Decimal(self.very_small_amount)

        assert isinstance(quantity, Decimal)

        block_number = self.web3.eth.block_number

        # For Orderly vault, the withdrawal is typically 1:1 with USDC
        # but we should check the actual vault state
        # TODO: Implement proper Orderly vault share-to-asset conversion
        # For now, assume 1:1 ratio for USDC vault shares
        estimated_usd = quantity

        price = float(estimated_usd / quantity) if quantity != 0 else 1.0
        mid_price = price

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[],  # Simplified for vault operations
            read_at=datetime.datetime.utcnow(),
            block_number=block_number,
            token_in=quantity,
            token_out=estimated_usd,
        )

    def get_buy_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        reserve: Optional[Decimal],
    ) -> TradePricing:
        """Get live price on Orderly vault for depositing our assets.

        For Orderly, this represents depositing to the vault.
        """

        assert pair.is_vault()
        assert isinstance(reserve, Decimal)

        block_number = self.web3.eth.block_number

        # For Orderly vault, the deposit is typically 1:1 with USDC
        # TODO: Implement proper Orderly vault asset-to-share conversion
        # For now, assume 1:1 ratio for USDC vault deposits
        estimated_shares = reserve

        price = float(reserve / estimated_shares) if estimated_shares != 0 else 1.0
        mid_price = price

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[],  # Simplified for vault operations
            read_at=datetime.datetime.utcnow(),
            block_number=block_number,
            token_in=reserve,
            token_out=estimated_shares,
        )

    def get_mid_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        estimate = self.get_buy_price(ts, pair, Decimal(1))
        return estimate.mid_price

    def get_pair_fee(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
    ) -> Optional[float]:
        # Orderly vault operations may have fees, but they're not trading fees
        # They're more like deposit/withdrawal fees
        return 0.0
