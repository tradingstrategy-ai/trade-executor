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


def test_vault_pricing_check_deposit_records_request_and_cap_blocks(monkeypatch):
    """Check live vault pricing retains the concrete deposit gate."""

    class FakePair:
        pool_address = "0x0000000000000000000000000000000000000001"

        @staticmethod
        def get_ticker() -> str:
            return "Vault USDC"

    pricing_model = VaultPricing(web3=object())
    pair = FakePair()

    monkeypatch.setattr(pricing_model, "_can_create_deposit_request", lambda pair: False)
    request_check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert request_check.reason_code == DepositBlockReason.deposit_request_unavailable

    monkeypatch.setattr(pricing_model, "_can_create_deposit_request", lambda pair: None)
    monkeypatch.setattr(pricing_model, "get_max_deposit", lambda ts, pair: Decimal(0))
    cap_check = pricing_model.check_deposit(None, pair, stage=DepositCheckStage.buy_rebalance)
    assert cap_check.reason_code == DepositBlockReason.vault_max_deposit_zero
    assert cap_check.max_deposit == 0.0


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
