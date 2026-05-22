from tradeexecutor.strategy.generic.generic_pricing_model import EthereumGenericPricingFactory, GenericPricing
from tradeexecutor.strategy.pricing_model import PricingModel
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

    child_pricing = ClosedDepositPricingModel()

    class FakePairConfigurator:
        def get_pricing(self, pair):
            return child_pricing

    pricing_model = GenericPricing(FakePairConfigurator())

    # 1. Build a child pricing model that reports deposits as closed.
    pair = object()

    # 2. Route all pairs from GenericPricing to that child pricing model.
    can_deposit = pricing_model.can_deposit(None, pair)

    # 3. Verify GenericPricing returns the child model's deposit availability.
    assert can_deposit is False


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
