"""Regression tests for GenericRouting temporary state swaps."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_router import GenericRouting, GenericRoutingState
from tradeexecutor.strategy.generic.pair_configurator import ProtocolRoutingConfig, ProtocolRoutingId
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


class DummyPair:
    """Minimal trade pair for exercising GenericRouting failure handling."""

    chain_id = 8453

    def is_exchange_account(self) -> bool:
        return False

    def is_cctp_bridge(self) -> bool:
        return False


class DummyPairConfigurator:
    """Minimal pair configurator for GenericRouting state-restore tests."""

    def __init__(
        self,
        protocol_config: ProtocolRoutingConfig,
        web3config: object,
    ):
        self.protocol_config = protocol_config
        self.web3config = web3config
        self.satellite_vaults = {}

    def match_router(self, pair: DummyPair) -> ProtocolRoutingId:
        del pair
        return self.protocol_config.routing_id

    def get_config(
        self,
        router: ProtocolRoutingId,
        pairs=None,
        three_leg_resolution=True,
    ) -> ProtocolRoutingConfig:
        del pairs
        del three_leg_resolution
        assert router == self.protocol_config.routing_id
        return self.protocol_config


@pytest.mark.timeout(300)
def test_generic_router_restores_temporary_state_after_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test GenericRouting restores temporary chain bindings after setup failure.

    1. Create a generic router whose trade setup temporarily swaps to a satellite-chain tx builder and Web3.
    2. Make the underlying router fail inside ``setup_trades()`` after the temporary swap has happened.
    3. Confirm the original routing state is restored even though the trade setup raised.
    """
    protocol_id = ProtocolRoutingId(router_name="uniswap-v3", exchange_slug="uniswap-v3")
    failing_router = Mock()
    failing_router.setup_trades.side_effect = RuntimeError("boom")
    protocol_config = ProtocolRoutingConfig(
        routing_id=protocol_id,
        routing_model=failing_router,
        pricing_model=Mock(),
        valuation_model=Mock(),
    )

    primary_web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=1))
    satellite_web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=8453))

    class FakeWeb3Config:
        def get_connection(self, chain_id):
            assert chain_id.value == 8453
            return satellite_web3

    class FakeHotWallet:
        def __init__(self, account):
            self.account = account

        def sync_nonce(self, web3) -> None:
            self.web3 = web3

    class FakeHotWalletTransactionBuilder:
        def __init__(self, web3, hot_wallet):
            self.chain_id = web3.eth.chain_id
            self.hot_wallet = hot_wallet

    # 1. Create a generic router whose trade setup temporarily swaps to a satellite-chain tx builder and Web3.
    monkeypatch.setattr("eth_defi.hotwallet.HotWallet", FakeHotWallet)
    monkeypatch.setattr("tradeexecutor.ethereum.tx.HotWalletTransactionBuilder", FakeHotWalletTransactionBuilder)

    pair_configurator = DummyPairConfigurator(protocol_config, FakeWeb3Config())
    routing_model = GenericRouting(pair_configurator)

    strategy_universe = object.__new__(TradingStrategyUniverse)
    strategy_universe.data_universe = SimpleNamespace(exchange_universe=object())

    original_tx_builder = SimpleNamespace(
        chain_id=1,
        hot_wallet=SimpleNamespace(account=object()),
    )
    router_state = SimpleNamespace(
        tx_builder=original_tx_builder,
        web3=primary_web3,
    )
    generic_routing_state = GenericRoutingState(
        strategy_universe,
        {"uniswap-v3": router_state},
    )
    trade = SimpleNamespace(pair=DummyPair(), route=None)

    # 2. Make the underlying router fail inside ``setup_trades()`` after the temporary swap has happened.
    with pytest.raises(RuntimeError, match="boom"):
        routing_model.setup_trades(
            state=State(),
            routing_state=generic_routing_state,
            trades=[trade],
        )

    # 3. Confirm the original routing state is restored even though the trade setup raised.
    assert trade.route == "uniswap-v3"
    assert router_state.tx_builder is original_tx_builder
    assert router_state.web3 is primary_web3
