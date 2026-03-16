"""Test ERC-4626 vault price reading and estimation."""
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
import datetime

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.ethereum.vault.vault_valuation import VaultValuator
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.state.valuation import ValuationUpdate
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.state.identifier import TradingPairIdentifier
from eth_defi.compat import native_datetime_utc_now


class _FixedCall:
    def __init__(self, value: int):
        self.value = value

    def call(self, block_identifier=None) -> int:
        return self.value


@pytest.fixture()
def vault_pricing(
    pricing_model: GenericPricing,
    ipor_usdc: TradingPairIdentifier
) -> VaultPricing:
    """Create a vault pricing model fixture."""

    vault_pricing = pricing_model.pair_configurator.get_pricing(ipor_usdc)
    assert isinstance(vault_pricing, VaultPricing)
    return vault_pricing


def test_vault_estimate_buy(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Estimate buy price / deosit estimate for a vault."""

    estimate = vault_pricing.get_buy_price(
        ts=None,
        pair=ipor_usdc,
        reserve=Decimal("100.00")
    )
    assert estimate.block_number > 0
    # Price of one share
    assert estimate.mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_estimate_sell(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Estimate sell price / redeem estimate for a vault."""

    estimate = vault_pricing.get_sell_price(
        ts=None,
        pair=ipor_usdc,
        quantity=Decimal("100.00")
    )
    assert estimate.block_number > 0
    # Price of one share
    assert estimate.mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_mid_price(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Because there is no trading fees, vaults mid price is same as share price."""

    mid_price = vault_pricing.get_mid_price(
        ts=None,
        pair=ipor_usdc,
    )
    assert mid_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet


def test_vault_tvl(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier
):
    """Get the real-time TVL of ERC-4626 vault."""
    tvl = vault_pricing.get_usd_tvl(
        timestamp=None,
        pair=ipor_usdc,
    )
    assert tvl == pytest.approx(1437072.77357)  # We use forked by block mainnet


def test_vault_max_deposit(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier,
):
    """Read the live max deposit from the ERC-4626 vault."""

    max_deposit = vault_pricing.get_max_deposit(
        ts=None,
        pair=ipor_usdc,
    )
    assert max_deposit is not None
    assert max_deposit > 0
    assert vault_pricing.can_deposit(None, ipor_usdc) is True


def test_vault_deposit_closed_when_max_deposit_zero(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier,
    monkeypatch: pytest.MonkeyPatch,
):
    """A zero maxDeposit disables new deposits."""

    fake_vault = MagicMock()
    fake_vault.denomination_token.convert_to_decimals.return_value = Decimal(0)
    fake_vault.vault_contract.functions.maxDeposit.return_value = _FixedCall(0)
    monkeypatch.setattr(vault_pricing, "get_vault", lambda pair: fake_vault)

    max_deposit = vault_pricing.get_max_deposit(None, ipor_usdc)
    assert max_deposit == 0
    assert vault_pricing.can_deposit(None, ipor_usdc) is False


def test_vault_redemption_closed_when_max_redeem_zero(
    vault_pricing,
    ipor_usdc: TradingPairIdentifier,
    monkeypatch: pytest.MonkeyPatch,
):
    """A zero maxRedeem disables redemptions."""

    fake_vault = MagicMock()
    fake_vault.share_token.convert_to_decimals.return_value = Decimal(0)
    fake_vault.vault_contract.functions.maxRedeem.return_value = _FixedCall(0)
    monkeypatch.setattr(vault_pricing, "get_vault", lambda pair: fake_vault)

    max_redemption = vault_pricing.get_max_redemption(None, ipor_usdc)
    assert max_redemption == 0
    assert vault_pricing.can_redeem(None, ipor_usdc) is False


def test_vault_position_valuation(
    vault_pricing,
    ipor_usdc
):
    """Check valuation function works."""
    valuation_model = VaultValuator(vault_pricing)

    position = TradingPosition(
        position_id=1,
        pair=ipor_usdc,
        opened_at=native_datetime_utc_now(),
        last_pricing_at=native_datetime_utc_now(),
        last_token_price=1.0,
        last_reserve_price=1.0,
        last_trade_at=1.0,
        reserve_currency=ipor_usdc.quote,
        trades={
            1: TradeExecution(
                trade_id=1,
                position_id=1,
                trade_type=TradeType.rebalance,
                opened_at=native_datetime_utc_now(),
                pair=ipor_usdc,
                executed_at=native_datetime_utc_now(),
                executed_quantity=Decimal(100),
                planned_quantity=Decimal(100),
                planned_reserve=Decimal(100),
                planned_price=1.0,
                reserve_currency=ipor_usdc.quote,
            )
        }
    )

    timestamp = datetime.datetime(2029, 1, 1)

    valuation = valuation_model(
        ts=timestamp,
        position=position
    )
    assert isinstance(valuation, ValuationUpdate)
    assert valuation.new_price == pytest.approx(1.0335669634763602)  # We use forked by block mainnet
    assert valuation.new_value == pytest.approx(103.35669634763602)  # 100 shares * price
    assert position.get_last_valued_at() == timestamp
