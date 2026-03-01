"""Vault share price estimator."""
import logging
import datetime
from decimal import Decimal
from typing import Optional

from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.estimate import estimate_4626_redeem, estimate_4626_deposit
from eth_defi.erc_4626.vault import ERC4626Vault
from web3 import Web3

from tradeexecutor.ethereum.vault.vault_routing import get_vault_for_pair
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.types import USDollarAmount
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from eth_defi.compat import native_datetime_utc_now

logger = logging.getLogger(__name__)


class VaultPricing(PricingModel):
    """Always pull the latest live share price of a vault.

    .. note::

        Only supports stablecoin-nominated vaults
    """

    def __init__(
        self,
        web3: Web3,
        web3config=None,
    ):
        self.web3 = web3
        self.web3config = web3config

    def get_web3_for_pair(self, pair: TradingPairIdentifier) -> Web3:
        """Resolve the correct web3 connection for a pair.

        For cross-chain vaults (e.g. a vault on Base when the home chain
        is Arbitrum), use the satellite chain's web3 from web3config.
        """
        if self.web3config is not None and pair.base.chain_id != self.web3.eth.chain_id:
            from tradingstrategy.chain import ChainId
            return self.web3config.get_connection(ChainId(pair.base.chain_id))
        return self.web3

    def get_vault(self, target_pair: TradingPairIdentifier) -> ERC4626Vault:
        """Helper function to speed up vault deployment resolution."""
        web3 = self.get_web3_for_pair(target_pair)
        return get_vault_for_pair(web3, target_pair)

    def get_sell_price(
        self,
        ts: datetime.datetime,
        pair: TradingPairIdentifier,
        quantity: Optional[Decimal],
    ) -> TradePricing:
        """Get live price on vault for dumping our shares."""

        assert pair.is_vault()

        if quantity is None:
            quantity = Decimal(self.very_small_amount)

        assert isinstance(quantity, Decimal)

        web3 = self.get_web3_for_pair(pair)
        block_number = web3.eth.block_number
        vault = self.get_vault(pair)

        estimated_usd = estimate_4626_redeem(
            vault=vault,
            owner=None,
            share_amount=quantity,
            block_identifier=block_number,
        )

        price = float(estimated_usd / quantity)
        mid_price = price

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=native_datetime_utc_now(),
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
        """Get live price on vault for dumping our shares."""

        assert pair.is_vault()
        assert isinstance(reserve, Decimal)

        web3 = self.get_web3_for_pair(pair)
        block_number = web3.eth.block_number
        vault = self.get_vault(pair)

        estimated_shares = estimate_4626_deposit(
            vault=vault,
            denomination_token_amount=reserve,
            block_identifier=block_number,
        )

        price = float(reserve / estimated_shares)
        mid_price = price

        return TradePricing(
            price=price,
            mid_price=mid_price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=native_datetime_utc_now(),
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
        return 0.0

    def get_usd_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        """Get the TVL of a vault pair."""
        assert pair.quote.is_stablecoin(), f"Only stablecoin vaults are supported for TVL, got: {pair}"
        web3 = self.get_web3_for_pair(pair)
        block_number = web3.eth.block_number
        vault = self.get_vault(pair)
        tvl_tokens = vault.fetch_total_assets(block_identifier=block_number)
        assert tvl_tokens is not None, f"Failed to fetch TVL for vault {pair} at block {block_number}"
        return float(tvl_tokens)

    def get_quote_token_tvl(
        self,
        timestamp: datetime.datetime | None,
        pair: TradingPairIdentifier
    ) -> USDollarAmount:
        return self.get_usd_tvl(timestamp, pair)

