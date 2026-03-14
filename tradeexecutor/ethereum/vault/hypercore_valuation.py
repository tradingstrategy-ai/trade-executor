"""Valuation and pricing models for Hypercore native vault positions.

Hypercore vault equity is queried via the Hyperliquid info API,
not from on-chain contracts.

Position model:

- **quantity** tracks cumulative USDC deposited minus withdrawn.
- **price** = equity / quantity (the return multiplier per deposited USDC).
- **value** = quantity × price = equity (by construction).

Trade pricing is always 1:1 USDC — equity growth is captured solely
by the valuation model, not the pricing model.

In simulate mode (Anvil forks), the Hyperliquid API has no data for the
forked Safe address, so the API is skipped and a 1:1 USDC price is assumed.
"""

import datetime
import logging
from decimal import Decimal
from typing import Callable

from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.trade_pricing import TradePricing
from tradeexecutor.strategy.valuation import ValuationModel
from eth_defi.compat import native_datetime_utc_now

logger = logging.getLogger(__name__)


class HypercoreVaultPricing(PricingModel):
    """Pricing model for Hypercore vault deposit/withdrawal trades.

    Always returns 1.0 USDC per unit because vault trades exchange USDC
    at face value.  Equity growth is reflected by the valuation model
    (:class:`HypercoreVaultValuator`), not by trade pricing.

    :param simulate:
        When ``True``, skip the Hyperliquid API and use 1.0 USDC per unit.
        Used in Anvil fork mode where the API has no data for the forked Safe.
    """

    def __init__(
        self,
        value_func: Callable[[TradingPairIdentifier], Decimal],
        simulate: bool = False,
    ):
        self.value_func = value_func
        self.simulate = simulate

    def _make_pricing(self, pair: TradingPairIdentifier, token_in: Decimal | None = None, token_out: Decimal | None = None) -> TradePricing:
        """Build a :class:`TradePricing` for vault deposits/withdrawals.

        Vault deposits and withdrawals are 1:1 USDC, so the trade price
        is always 1.0.  The vault equity growth is reflected in the
        valuation model (per-unit price = equity / quantity), not here.
        """
        price = 1.0
        return TradePricing(
            price=price,
            mid_price=price,
            lp_fee=[0.0],
            pair_fee=[0.0],
            side=False,
            path=[pair],
            read_at=native_datetime_utc_now(),
            block_number=None,
            token_in=token_in,
            token_out=token_out,
        )

    def get_sell_price(self, ts, pair, quantity) -> TradePricing:
        return self._make_pricing(pair, token_in=quantity)

    def get_buy_price(self, ts, pair, reserve) -> TradePricing:
        return self._make_pricing(pair, token_in=reserve)

    def get_mid_price(self, ts, pair) -> float:
        # Vault trades are 1:1 USDC.  Equity growth is tracked by
        # the valuation model, not the pricing model.
        return 1.0

    def get_pair_fee(self, ts, pair):
        return 0.0

    def get_pair_for_id(self, internal_id):
        raise NotImplementedError("Hypercore vault pricing does not support pair lookup by ID")


class HypercoreVaultValuator(ValuationModel):
    """Re-value Hypercore vault positions using the Hyperliquid info API.

    Queries the vault equity for the Safe address and computes
    a per-unit price so that ``value = quantity × price = equity``.

    - **quantity** = cumulative USDC deposited minus withdrawn
    - **price** = equity / quantity (return multiplier per deposited USDC)

    :param simulate:
        When ``True``, skip the Hyperliquid API and use 1.0 USDC per unit.
        Used in Anvil fork mode where the API has no data for the forked Safe.
    """

    def __init__(
        self,
        value_func: Callable[[TradingPairIdentifier], Decimal],
        simulate: bool = False,
    ):
        self.value_func = value_func
        self.simulate = simulate

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition,
    ) -> ValuationUpdate:
        assert position.is_vault(), f"Not a vault position: {position}"

        position.last_pricing_at = ts

        if self.simulate:
            # Anvil fork: Hyperliquid API has no data for forked Safe
            equity = position.get_quantity()
            logger.info(
                "Hypercore vault position %s: simulate mode, using quantity %s as equity",
                position, equity,
            )
        else:
            try:
                equity = self.value_func(position.pair)
            except Exception as e:
                # Intentionally crash the tick cycle: if the API is down we
                # cannot value the position and must halt rather than use
                # stale data.
                logger.error(
                    "Failed to get Hypercore vault equity for position %s: %s",
                    position, e,
                )
                raise

        old_price = position.last_token_price
        old_value = position.get_value()

        # Compute per-unit price so that value = quantity * price = equity.
        # Position quantity tracks cumulative USDC deposited/withdrawn;
        # the price reflects the return on each deposited USDC.
        quantity = position.get_quantity()
        if float(quantity) > 0:
            new_price = float(equity) / float(quantity)
        else:
            new_price = 1.0

        new_value = position.revalue_base_asset(ts, new_price)

        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=position.get_quantity(),
        )

        position.last_token_price = new_price

        logger.info(
            "Hypercore vault position %s, valuation updated: equity=$%.2f, old=$%.2f, new=$%.2f",
            position, equity, old_value, new_value,
        )

        return evt
