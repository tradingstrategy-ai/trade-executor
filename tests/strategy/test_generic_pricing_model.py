import datetime
from decimal import Decimal

from tradeexecutor.ethereum.vault.vault_live_pricing import VaultPricing
from tradeexecutor.strategy.generic.generic_pricing_model import EthereumGenericPricingFactory, GenericPricing
from tradeexecutor.strategy.pricing_model import PricingModel
from tradeexecutor.strategy.redemption import DepositBlockReason, DepositCheckResult, DepositCheckStage
from tradeexecutor.strategy.trade_pricing import TradePricing


class DummyPricingModel(PricingModel):
    def get_sell_price(self, ts, pair, quantity) -> TradePricing:
        raise NotImplementedError

    def get_buy_price(self, ts, pair, reserve) -> TradePricing:
        raise NotImplementedError

    def get_mid_price(self, ts, pair) -> float:
        return 1.0

    def get_pair_fee(self, ts, pair) -> float | None:
        return 0.0


def test_pricing_model_tradeability_defaults_to_true():
    pricing_model = DummyPricingModel()

    assert pricing_model.get_max_deposit(None, pair=None) is None
    assert pricing_model.get_max_redemption(None, pair=None) is None
    assert pricing_model.can_deposit(None, pair=None) is True
    assert pricing_model.can_redeem(None, pair=None) is True
    assert pricing_model.is_tradeable(None, pair=None) is True


def test_generic_pricing_delegates_can_deposit():
    """Check GenericPricing keeps the route-specific deposit gate intact.

    1. Build a child pricing model that reports deposits as closed.
    2. Route all pairs from GenericPricing to that child pricing model.
    3. Verify GenericPricing returns the child model's deposit availability.
    """

    class ClosedDepositPricingModel(DummyPricingModel):
        def can_deposit(self, ts, pair) -> bool:
            return False

        def check_deposit(self, ts, pair, *, stage):
            return DepositCheckResult(
                timestamp=ts,
                stage=stage,
                can_deposit=False,
                reason_code=DepositBlockReason.vault_deposits_closed,
                message="Child pricing model has closed deposits",
            )

    child_pricing = ClosedDepositPricingModel()

    class FakePairConfigurator:
        def get_pricing(self, pair):
            return child_pricing

    pricing_model = GenericPricing(FakePairConfigurator())

    # 1. Build a child pricing model that reports deposits as closed.
    pair = object()

    # 2. Route all pairs from GenericPricing to that child pricing model.
    can_deposit = pricing_model.can_deposit(None, pair)
    check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)

    # 3. Verify GenericPricing returns the child model's deposit availability.
    assert can_deposit is False
    assert check.reason_code == DepositBlockReason.vault_deposits_closed
    assert check.message == "Child pricing model has closed deposits"


def test_generic_pricing_delegates_historical_vault_settlement_event():
    """The generic backtest route must preserve settlement evidence."""
    event_at = datetime.datetime(2026, 1, 3, 12)

    class SettlementPricingModel(DummyPricingModel):
        def get_vault_settlement_event_at(self, ts, pair):
            return event_at

    class FakePairConfigurator:
        def get_pricing(self, pair):
            return SettlementPricingModel()

    pricing_model = GenericPricing(FakePairConfigurator())
    assert pricing_model.get_vault_settlement_event_at(datetime.datetime(2026, 1, 4), object()) == event_at


def test_vault_pricing_check_deposit_records_request_and_cap_blocks(monkeypatch):
    """Check live vault pricing delegates admission to DepositManager."""

    class FakePair:
        pool_address = "0x0000000000000000000000000000000000000001"

        @staticmethod
        def get_ticker() -> str:
            return "Vault USDC"

    pricing_model = VaultPricing(web3=object())
    pair = FakePair()

    class FakeToken:
        @staticmethod
        def convert_to_decimals(value):
            return Decimal(value)

    class FakeManager:
        def __init__(self, can_create_request, max_deposit):
            self.can_create_request = can_create_request
            self.max_deposit = max_deposit

        @staticmethod
        def has_synchronous_deposit():
            return True

        def can_create_deposit_request(self, owner):
            return self.can_create_request

        def fetch_depositable_raw_assets(self, owner):
            return self.max_deposit

    class FakeVault:
        denomination_token = FakeToken()

        def __init__(self, manager, closed_reason=None):
            self.manager = manager
            self.closed_reason = closed_reason

        def get_deposit_manager(self):
            return self.manager

        def fetch_deposit_closed_reason(self):
            return self.closed_reason

    monkeypatch.setattr(pricing_model, "get_owner_address", lambda pair: "0xowner")

    monkeypatch.setattr(pricing_model, "get_vault", lambda pair: FakeVault(FakeManager(False, None)))
    request_check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert request_check.reason_code == DepositBlockReason.deposit_request_unavailable

    monkeypatch.setattr(pricing_model, "get_vault", lambda pair: FakeVault(FakeManager(True, 0)))
    cap_check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert cap_check.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert cap_check.max_deposit == 0.0

    monkeypatch.setattr(pricing_model, "get_vault", lambda pair: FakeVault(FakeManager(True, None), "Vault deposits are closed"))
    closed_check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert closed_check.reason_code == DepositBlockReason.vault_deposits_closed
    assert closed_check.message == "Vault deposits are closed"


def test_ethereum_generic_pricing_factory_passes_execution_model(monkeypatch):
    captured: dict[str, object] = {}

    class FakePairConfigurator:
        def __init__(self, web3, universe, execution_model=None):
            captured["web3"] = web3
            captured["universe"] = universe
            captured["execution_model"] = execution_model
            self.configs = {}

        def get_pricing(self, pair):
            raise NotImplementedError

    monkeypatch.setattr(
        "tradeexecutor.strategy.generic.generic_pricing_model.EthereumPairConfigurator",
        FakePairConfigurator,
    )

    factory = EthereumGenericPricingFactory(web3="web3")
    execution_model = object()
    universe = object()

    pricing = factory(execution_model, universe, routing_model=None)

    assert isinstance(pricing, GenericPricing)
    assert captured == {
        "web3": "web3",
        "universe": universe,
        "execution_model": execution_model,
    }
