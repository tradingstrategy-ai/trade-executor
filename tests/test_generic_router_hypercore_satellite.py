"""Test that GenericRouting maps synthetic Hypercore chain 9999 to HyperEVM 999.

Hypercore vault pairs use synthetic chain_id 9999, but satellite vaults
are keyed by the real HyperEVM chain_id 999. Without the mapping,
GenericRouting.setup_trades() crashes with:

    AssertionError: No satellite vault configured for chain 9999

1. Build a GenericRouting with a pair configurator that has satellite_vaults keyed by 999
2. Create a Hypercore vault trade with chain_id=9999
3. Call setup_trades — verify it does NOT crash with the satellite lookup error
4. The call will fail later (no real web3) but the satellite lookup must succeed
"""

from types import SimpleNamespace

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.hyperliquid.core_writer import CORE_WRITER_ADDRESS
from eth_account import Account
from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeType
from tradeexecutor.strategy.generic.generic_router import GenericRouting


# Use a deterministic test account
TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
TEST_ACCOUNT = Account.from_key(TEST_PRIVATE_KEY)
TEST_VAULT_ADDRESS = "0x1111111111111111111111111111111111111111"


def _make_hypercore_vault_pair() -> "TradingPairIdentifier":
    """Create a real Hypercore vault pair with chain_id=9999."""
    usdc = AssetIdentifier(
        chain_id=ChainId.hyperliquid.value,
        address="0xb88339CB7199b77E23DB6E890353E22632Ba630f",
        token_symbol="USDC",
        decimals=6,
    )
    return create_hypercore_vault_pair(
        quote=usdc,
        vault_address=TEST_VAULT_ADDRESS,
    )


def _make_pair_configurator(satellite_vaults: dict) -> SimpleNamespace:
    """Build a pair configurator namespace with satellite_vaults and a web3config.

    Uses SimpleNamespace instead of mocks — just plain data containers
    with the attributes that GenericRouting.setup_trades() reads.
    """
    # web3config that returns a namespace for get_connection()
    # GenericRouting needs get_connection(ChainId) to return a web3-like object
    web3_999 = SimpleNamespace(
        eth=SimpleNamespace(chain_id=999),
    )

    def get_connection(chain_id):
        if chain_id == ChainId.hyperliquid or chain_id == ChainId(999):
            return web3_999
        raise KeyError(f"No connection for chain {chain_id}")

    web3config = SimpleNamespace(
        get_connection=get_connection,
    )

    return SimpleNamespace(
        satellite_vaults=satellite_vaults,
        web3config=web3config,
    )


def test_hypercore_satellite_vault_lookup_maps_chain_9999_to_999():
    """Verify GenericRouting maps synthetic chain 9999 to HyperEVM 999 for satellite lookup.

    1. Build a GenericRouting with satellite_vaults keyed by 999
    2. Create a Hypercore vault trade (chain_id=9999) via create_hypercore_vault_pair
    3. Call setup_trades — should NOT crash with 'No satellite vault for chain 9999'
    4. It will fail later (nonce sync needs real web3) but satellite lookup must succeed
    """
    pair = _make_hypercore_vault_pair()
    assert pair.chain_id == ChainId.hypercore.value  # 9999

    # Satellite vault namespace — just needs the attributes that
    # LagoonTransactionBuilder reads during construction
    satellite_vault = SimpleNamespace(
        safe_address="0x2222222222222222222222222222222222222222",
        trading_strategy_module_address="0x3333333333333333333333333333333333333333",
        web3=SimpleNamespace(eth=SimpleNamespace(chain_id=999)),
    )

    # satellite_vaults keyed by 999 (HyperEVM), NOT 9999 (synthetic Hypercore)
    satellite_vaults = {ChainId.hyperliquid.value: satellite_vault}

    pair_configurator = _make_pair_configurator(satellite_vaults)
    routing = GenericRouting(pair_configurator)

    # tx_builder namespace simulating a LagoonTransactionBuilder on Arbitrum (primary chain)
    hot_wallet = HotWallet(TEST_ACCOUNT)
    tx_builder = SimpleNamespace(
        chain_id=ChainId.arbitrum.value,
        hot_wallet=hot_wallet,
        vault=SimpleNamespace(safe_address="0x4444444444444444444444444444444444444444"),
        extra_gnosis_gas=0,
    )

    # Routing state with the tx_builder — GenericRouting reads this
    # to detect satellite chain and swap builders
    router_state = SimpleNamespace(
        tx_builder=tx_builder,
    )

    routing_state = SimpleNamespace(
        state_map={"hypercore_vault": router_state},
    )

    # Create a trade directly — State.create_trade() requires many params
    # that are irrelevant to this test. We only need a TradeExecution with
    # the correct pair to trigger the satellite lookup.
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
    assert trade.pair.chain_id == 9999

    # Without the fix, this crashes:
    #   AssertionError: No satellite vault configured for chain 9999.
    #   Available satellite chains: [999]
    #
    # With the fix, GenericRouting maps 9999 → 999, finds the satellite vault,
    # and attempts to create a LagoonTransactionBuilder. This will fail because
    # the satellite_vault namespace isn't a real LagoonVault, but the important
    # thing is we got PAST the satellite vault lookup.
    with pytest.raises(Exception) as exc_info:
        routing.setup_trades(
            state=state,
            routing_state=routing_state,
            trades=[trade],
        )

    # The error should NOT be about missing satellite vault for chain 9999
    error_msg = str(exc_info.value)
    assert "No satellite vault configured for chain 9999" not in error_msg, (
        f"Chain 9999 → 999 mapping failed: {error_msg}"
    )
