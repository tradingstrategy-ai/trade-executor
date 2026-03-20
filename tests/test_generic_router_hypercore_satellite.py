"""Test that GenericRouting maps synthetic Hypercore chain 9999 to HyperEVM 999.

Hypercore vault pairs use synthetic chain_id 9999, but satellite vaults
are keyed by the real HyperEVM chain_id 999. Without the mapping,
GenericRouting.setup_trades() crashes with:

    AssertionError: No satellite vault configured for chain 9999

1. Build a GenericRouting with satellite_vaults keyed by 999
2. Create a real GenericRoutingState with a HypercoreVaultRouting state entry
3. Use a real LagoonTransactionBuilder subclass for the isinstance check
4. Call setup_trades — verify it does NOT crash with the satellite lookup error
5. It will fail later (nonce sync on SimpleNamespace web3) but the satellite
   lookup at line 201 must succeed first
"""

from types import SimpleNamespace

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_account import Account
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.lagoon.tx import LagoonTransactionBuilder
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.generic.generic_router import GenericRouting, GenericRoutingState
from tradeexecutor.strategy.generic.pair_configurator import ProtocolRoutingId, ProtocolRoutingConfig
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse


TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
TEST_ACCOUNT = Account.from_key(TEST_PRIVATE_KEY)


class _TestLagoonTransactionBuilder(LagoonTransactionBuilder):
    """Subclass of LagoonTransactionBuilder that skips the real constructor.

    We need isinstance(tx_builder, LagoonTransactionBuilder) to return True
    at generic_router.py line 199, but we cannot construct a real
    LagoonTransactionBuilder without deployed contracts. This subclass
    sets the required attributes directly.
    """

    def __init__(self, chain_id: int, hot_wallet: HotWallet):
        # Skip super().__init__() — it needs vault.web3 from a real contract.
        # Set only the attributes that GenericRouting.setup_trades() reads.
        self.chain_id = chain_id
        self.hot_wallet = hot_wallet
        self.vault = SimpleNamespace(safe_address="0x4444444444444444444444444444444444444444")
        self.extra_gnosis_gas = 0
        self.web3 = SimpleNamespace(eth=SimpleNamespace(chain_id=chain_id))


def test_hypercore_satellite_vault_lookup_maps_chain_9999_to_999():
    """Verify GenericRouting maps synthetic chain 9999 to HyperEVM 999 for satellite lookup.

    1. Build a GenericRouting with satellite_vaults={999: vault} and match_router/get_config
    2. Build a real GenericRoutingState with a state_map entry for hypercore_vault
    3. Use _TestLagoonTransactionBuilder so isinstance check at line 199 passes
    4. Create a Hypercore vault trade (chain_id=9999) via create_hypercore_vault_pair
    5. Call setup_trades — should NOT crash with 'No satellite vault for chain 9999'
    6. It will fail later (nonce sync on fake web3) but that proves the satellite lookup passed
    """
    # 1. Create pair
    usdc = AssetIdentifier(
        chain_id=ChainId.hyperliquid.value,
        address="0xb88339CB7199b77E23DB6E890353E22632Ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    pair = create_hypercore_vault_pair(quote=usdc, vault_address="0x1111111111111111111111111111111111111111")
    assert pair.chain_id == ChainId.hypercore.value  # 9999

    # 2. Build pair_configurator with match_router/get_config and satellite_vaults
    satellite_vault = SimpleNamespace(
        safe_address="0x2222222222222222222222222222222222222222",
        trading_strategy_module_address="0x3333333333333333333333333333333333333333",
        web3=SimpleNamespace(eth=SimpleNamespace(chain_id=999)),
    )

    # A stub routing model — setup_trades will be called on this after
    # the satellite lookup. It doesn't need to work, just not crash before
    # the satellite lookup code.
    stub_routing_model = SimpleNamespace()

    hypercore_routing_id = ProtocolRoutingId(router_name="hypercore_vault", exchange_slug=None)
    hypercore_config = ProtocolRoutingConfig(
        routing_id=hypercore_routing_id,
        routing_model=stub_routing_model,
        pricing_model=None,
        valuation_model=None,
    )

    web3_999 = SimpleNamespace(
        eth=SimpleNamespace(
            chain_id=999,
            get_transaction_count=lambda addr: 0,
        ),
    )

    pair_configurator = SimpleNamespace(
        satellite_vaults={ChainId.hyperliquid.value: satellite_vault},
        web3config=SimpleNamespace(
            get_connection=lambda chain_id: web3_999,
        ),
        match_router=lambda p: hypercore_routing_id,
        get_config=lambda rid: hypercore_config,
    )

    routing = GenericRouting(pair_configurator)

    # 3. Build real GenericRoutingState with a LagoonTransactionBuilder
    hot_wallet = HotWallet(TEST_ACCOUNT)
    tx_builder = _TestLagoonTransactionBuilder(
        chain_id=ChainId.arbitrum.value,  # Primary chain (not HyperEVM)
        hot_wallet=hot_wallet,
    )

    # The router_state needs tx_builder for the satellite chain swap
    router_state = SimpleNamespace(tx_builder=tx_builder)

    # Use a minimal TradingStrategyUniverse — GenericRoutingState asserts on it
    from tradingstrategy.universe import Universe
    from tradingstrategy.exchange import ExchangeUniverse
    data_universe = Universe(
        time_bucket=None,
        chains={ChainId.hypercore},
        exchange_universe=ExchangeUniverse.from_collection([]),
    )
    strategy_universe = TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets=[usdc],
    )

    routing_state = GenericRoutingState(
        strategy_universe=strategy_universe,
        state_map={"hypercore_vault": router_state},
    )

    # 4. Create trade
    trade = TradeExecution(
        trade_id=1,
        position_id=1,
        trade_type=TradeType.rebalance,
        pair=pair,
        planned_quantity=5.0,
        planned_reserve=5.0,
        planned_price=1.0,
        reserve_currency=pair.quote,
        opened_at=native_datetime_utc_now(),
    )
    state = State()

    # 5. Call setup_trades — the satellite lookup at line 201 should find
    # satellite_vaults[999] (not satellite_vaults[9999]) thanks to the mapping.
    # It will fail AFTER the lookup (nonce sync on fake web3, or
    # LagoonTransactionBuilder construction with the namespace vault).
    with pytest.raises(Exception) as exc_info:
        routing.setup_trades(
            state=state,
            routing_state=routing_state,
            trades=[trade],
        )

    # 6. Verify the error is NOT the satellite vault lookup failure.
    # Any other error means we got past the 9999 → 999 mapping.
    error_msg = str(exc_info.value)
    assert "No satellite vault configured for chain 9999" not in error_msg, (
        f"Chain 9999 → 999 mapping failed. The satellite vault lookup used "
        f"the synthetic chain_id 9999 instead of HyperEVM 999: {error_msg}"
    )
