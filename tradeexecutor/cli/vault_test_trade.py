"""Small safety checks for the public test-trade command.

These checks are opt-in for the command, not a change to strategy admission or
the standalone vault diagnostic runner.
"""

from decimal import Decimal

from eth_defi.erc_4626.vault_protocol.atoma.deposit_redeem import AtomaDepositManager

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.pricing_model import PricingModel


def is_synchronous_vault_test(pricing_model: PricingModel, pair: TradingPairIdentifier) -> bool:
    """Keep native and queued vaults on their existing command lifecycle."""
    pricing = pricing_model.route(pair) if isinstance(pricing_model, GenericPricing) else pricing_model
    if not isinstance(pricing, VaultPricing):
        return False
    manager = pricing.get_vault(pair).get_deposit_manager()
    return manager.has_synchronous_deposit() and manager.has_synchronous_redemption()


def is_atoma_vault_test(pricing_model: PricingModel, pair: TradingPairIdentifier) -> bool:
    """Use the existing pending-ticket lifecycle for Atoma public test trades."""
    pricing = pricing_model.route(pair) if isinstance(pricing_model, GenericPricing) else pricing_model
    return isinstance(pricing, VaultPricing) and isinstance(pricing.get_vault(pair).get_deposit_manager(), AtomaDepositManager)


def get_test_position(state: State, pair: TradingPairIdentifier, *, allow_pending_redemption: bool = False) -> TradingPosition | None:
    """Refuse ordinary or externally modified holdings before a test touches them."""
    positions = [p for p in state.portfolio.get_open_and_frozen_positions() if p.pair == pair]
    if len(positions) > 1:
        raise RuntimeError(f"Multiple positions for {pair}; select and reconcile them before testing")
    if not positions:
        return None
    position = positions[0]
    if not position.trades or not all(t.is_test() for t in position.trades.values()) or position.balance_updates:
        raise RuntimeError(f"Refusing to trade non-test or externally modified position #{position.position_id}")
    if position.is_frozen() or any(
        not t.is_success() and not (
            allow_pending_redemption and t.is_sell() and t.get_status() == TradeStatus.vault_settlement_pending
        )
        for t in position.trades.values()
    ):
        raise RuntimeError(f"Test position #{position.position_id} has unresolved trades; reconcile existing transactions before retrying")
    return position


def check_vault_test_capacity(
    pricing_model: PricingModel,
    pair: TradingPairIdentifier,
    amount: Decimal,
    *,
    closing: bool,
) -> None:
    """Use shared manager admission checks for the actual custody account.

    Test commands additionally require the funded quote asset and a full exit
    of the test-owned holding; protocol capacity policy stays in the manager.
    """
    if not pair.is_vault():
        return
    pricing = pricing_model.route(pair) if isinstance(pricing_model, GenericPricing) else pricing_model
    if not isinstance(pricing, VaultPricing):
        return
    vault = pricing.get_vault(pair)
    manager = vault.get_deposit_manager()
    if not isinstance(manager, AtomaDepositManager) and not (manager.has_synchronous_deposit() and manager.has_synchronous_redemption()):
        return
    owner = pricing.get_owner_address(pair)
    if owner is None:
        raise RuntimeError(f"Cannot resolve custody owner for {pair}")
    if vault.denomination_token.address.lower() != pair.quote.address.lower():
        raise RuntimeError(f"Vault asset differs from the funded quote token for {pair}")
    raw_amount = (vault.share_token if closing else vault.denomination_token).convert_to_raw(amount)
    if raw_amount <= 0:
        raise ValueError(f"Test amount rounds to zero for {pair}")
    if closing:
        if raw_amount != vault.share_token.fetch_raw_balance_of(owner):
            raise RuntimeError("Test redemption must match its exact custody share balance")
        manager.create_redemption_request(owner=owner, raw_shares=raw_amount)
    else:
        manager.create_deposit_request(owner=owner, raw_amount=raw_amount)


def check_vault_test_closed(pricing_model: PricingModel, pair: TradingPairIdentifier, position: TradingPosition) -> None:
    """A successful sell alone does not prove the test holding was exhausted."""
    if not position.is_closed() or position.get_quantity() != 0:
        raise RuntimeError(f"Test position #{position.position_id} has residual shares: {position.get_quantity()}")
    pricing = pricing_model.route(pair) if isinstance(pricing_model, GenericPricing) else pricing_model
    if isinstance(pricing, VaultPricing):
        owner = pricing.get_owner_address(pair)
        remaining = pricing.get_vault(pair).share_token.fetch_raw_balance_of(owner)
        if remaining:
            raise RuntimeError(f"Test exit left {remaining} raw shares in custody for {pair}")
