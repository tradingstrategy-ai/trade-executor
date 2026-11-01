"""Value model based on a vault mid-share price.

Async shares remain in the same position until the claim completes. Atoma burns
shares at settlement, so their ticket entitlement replaces the share quote.
Each redemption ticket belongs to one trade. Aggregate claims are outside the
supported operating model and must halt valuation, never be split across trades.
See .claude/docs/vault-deposit-redeem.md for the accounting and code map.
"""
import datetime
import logging
from decimal import Decimal

from eth_defi.vault.deposit_redeem import VaultDepositManager, VaultFlowUnavailable

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.valuation import ValuationModel


logger = logging.getLogger(__name__)

class VaultValuator(ValuationModel):
    """Re-value assets based on what vault tells us."""

    def __init__(self, pricing_model: VaultPricing):
        assert isinstance(pricing_model, VaultPricing)
        self.pricing_model = pricing_model

    def __call__(
        self,
        ts: datetime.datetime,
        position: TradingPosition
    ) -> ValuationUpdate:
        assert position.is_vault()
        shares_amount = position.get_quantity()

        if shares_amount == 0:
            # Frozen position, deposit has failed?
            logger.warning(f"Creating null valuation update: Shares amount must be greater than 0, got {shares_amount} for position {position}")
            evt = ValuationUpdate(
                created_at=ts,
                position_id=position.position_id,
                valued_at=ts,
                old_value=0,
                new_value=0,
                old_price=0,
                new_price=0,
                quantity=shares_amount,
            )
            position.last_token_price = 0
            position.last_pricing_at = ts
            return evt

        assert position.pair.quote.is_stablecoin(), f"Vault position {position} must be a stablecoin pair, got {position.pair}"

        old_price = position.last_token_price
        old_value = position.get_value()
        manager = self.pricing_model.get_vault(position.pair).get_deposit_manager()
        quote_hook = getattr(type(manager), "fetch_pending_redemption_value", None)
        ticket_valuation = quote_hook is not None and quote_hook is not VaultDepositManager.fetch_pending_redemption_value
        if ticket_valuation and position.has_pending_vault_settlement():
            block = manager.web3.eth.block_number
            remaining_shares = shares_amount
            claim_value = Decimal(0)
            for trade in position.trades.values():
                if not trade.is_vault_settlement_in_flight() or not trade.is_sell():
                    continue
                if trade.get_status() != TradeStatus.vault_settlement_pending:
                    raise VaultFlowUnavailable("Vault request must be parsed before valuation")
                ticket = manager.reconstruct_redemption_ticket(trade.other_data)
                net = manager.fetch_pending_redemption_value(ticket, block_identifier=block)
                if net is not None:
                    remaining_shares -= manager.vault.share_token.convert_to_decimals(ticket.raw_shares)
                    claim_value += net
            assert remaining_shares >= 0, "Vault tickets exceed the position quantity"
            remaining_value = manager.estimate_redeem(
                owner=self.pricing_model.get_owner_address(position.pair),
                shares=remaining_shares,
                block_identifier=block,
            ) if remaining_shares else Decimal(0)
            new_price = float((remaining_value + claim_value) / shares_amount)
        else:
            price_structure = self.pricing_model.get_sell_price(ts, position.pair, shares_amount)
            new_price = price_structure.price
        new_value = position.revalue_base_asset(ts, float(new_price))
        evt = ValuationUpdate(
            created_at=ts,
            position_id=position.position_id,
            valued_at=ts,
            old_value=old_value,
            new_value=new_value,
            old_price=old_price,
            new_price=new_price,
            quantity=shares_amount,
        )
        logger.info("Vault position %s, valuation updated: %s", position, evt)
        position.last_token_price = new_price
        return evt


def vault_valuation_factory(pricing_model):
    return VaultValuator(pricing_model)
