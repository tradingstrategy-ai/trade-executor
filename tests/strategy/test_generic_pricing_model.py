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
