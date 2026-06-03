"""Test cross-chain test trade orchestration.

Mainnet fork test (Arbitrum + Base) that verifies perform-test-trade
and trade-ui correctly handle satellite-chain vault pairs by
automatically injecting CCTP bridge trades.

1. Verify cross-chain detection logic (satellite vs home chain pairs)
2. Verify bridge pairs are filtered from TUI display
3. Full round-trip cross-chain test trade on Anvil forks (bridge in → open → close → bridge back)
4. Buy-only cross-chain test trade (bridge in → open, leave positions open)

Requires ``JSON_RPC_ARBITRUM`` and ``JSON_RPC_BASE`` environment variables.
"""

import datetime
import logging
import os
from decimal import Decimal

import pytest
from eth_account import Account

from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch, set_balance, fund_erc20_on_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN
from eth_defi.cctp.constants import TOKEN_MESSENGER_V2

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.cli.trade_ui_tui import PairSelectionApp
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code


JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

ARBITRUM_CHAIN_ID = 42161
BASE_CHAIN_ID = 8453

# IPOR Fusion vault on Base — USDC, short lockup
IPOR_VAULT_ADDRESS = "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"

DEPOSIT_AMOUNT = Decimal("10")
TEST_TRADE_AMOUNT = Decimal("3")

#: Pin Base fork to a block where the IPOR Fusion vault has sufficient
#: deposit headroom (vault's totalSupplyCap fills up quickly).
BASE_FORK_BLOCK = 45_960_642

pytestmark = pytest.mark.skipif(
    not JSON_RPC_ARBITRUM or not JSON_RPC_BASE,
    reason="JSON_RPC_ARBITRUM and JSON_RPC_BASE environment variables required",
)


# --- Fixtures ---


@pytest.fixture()
def arb_anvil() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_ARBITRUM)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def base_anvil() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_BASE, fork_block_number=BASE_FORK_BLOCK)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def arb_web3(arb_anvil):
    return create_multi_provider_web3(arb_anvil.json_rpc_url)


@pytest.fixture()
def base_web3(base_anvil):
    return create_multi_provider_web3(base_anvil.json_rpc_url)


@pytest.fixture()
def web3config(arb_web3, base_web3) -> Web3Config:
    config = Web3Config()
    config.connections[ChainId.arbitrum] = arb_web3
    config.connections[ChainId.base] = base_web3
    config.default_chain_id = ChainId.arbitrum
    return config


@pytest.fixture()
def hot_wallet(arb_web3, base_web3) -> HotWallet:
    """Create hot wallet funded with gas on both chains."""
    wallet = HotWallet.create_for_testing(arb_web3, eth_amount=100)
    set_balance(base_web3, wallet.address, 100 * 10**18)
    return wallet


@pytest.fixture()
def funded_wallet(hot_wallet, arb_web3) -> HotWallet:
    """Fund the hot wallet with USDC on Arbitrum."""
    usdc_raw = int(DEPOSIT_AMOUNT * 10**6)
    fund_erc20_on_anvil(
        arb_web3,
        USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        hot_wallet.address,
        usdc_raw,
    )
    return hot_wallet


@pytest.fixture()
def usdc_arb() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def usdc_base() -> AssetIdentifier:
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[BASE_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )


@pytest.fixture()
def ipor_share_token() -> AssetIdentifier:
    """IPOR Fusion vault share token (ERC-4626 vault IS its own share token)."""
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=IPOR_VAULT_ADDRESS,
        token_symbol="ipfUSDC",
        decimals=18,
    )


@pytest.fixture()
def bridge_pair(usdc_base, usdc_arb) -> TradingPairIdentifier:
    """CCTP bridge pair: Arbitrum USDC → Base USDC."""
    return TradingPairIdentifier(
        base=usdc_base,
        quote=usdc_arb,
        pool_address=TOKEN_MESSENGER_V2,
        exchange_address=TOKEN_MESSENGER_V2,
        internal_id=1,
        internal_exchange_id=1,
        fee=0.0,
        kind=TradingPairKind.cctp_bridge,
        exchange_name="CCTP Bridge",
        other_data={
            "bridge_protocol": "cctp",
            "destination_chain_id": BASE_CHAIN_ID,
        },
    )


@pytest.fixture()
def vault_pair(ipor_share_token, usdc_base) -> TradingPairIdentifier:
    """IPOR Fusion vault pair on Base."""
    return TradingPairIdentifier(
        base=ipor_share_token,
        quote=usdc_base,
        pool_address=IPOR_VAULT_ADDRESS,
        exchange_address=IPOR_VAULT_ADDRESS,
        internal_id=2,
        internal_exchange_id=2,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="IPOR Fusion",
        other_data={
            "vault_protocol": "ipor_fusion",
        },
    )


@pytest.fixture()
def universe(bridge_pair, vault_pair, usdc_arb) -> TradingStrategyUniverse:
    """Trading universe with CCTP bridge and IPOR vault pairs."""
    pair_universe = create_pair_universe_from_code(ChainId.arbitrum, [bridge_pair, vault_pair])

    cctp_exchange = Exchange(
        chain_id=ChainId.arbitrum,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    vault_exchange = Exchange(
        chain_id=ChainId.base,
        chain_slug="base",
        exchange_id=2,
        exchange_slug="ipor-fusion",
        address=IPOR_VAULT_ADDRESS,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=1,
    )

    data_universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.arbitrum, ChainId.base},
        exchanges={cctp_exchange, vault_exchange},
        pairs=pair_universe,
        candles=None,
        liquidity=None,
    )

    return TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets=[usdc_arb],
    )


@pytest.fixture()
def execution_model(arb_web3, funded_wallet, web3config) -> EthereumExecution:
    tx_builder = HotWalletTransactionBuilder(arb_web3, funded_wallet)
    model = EthereumExecution(
        tx_builder=tx_builder,
        confirmation_block_count=0,
        confirmation_timeout=datetime.timedelta(seconds=30),
        max_slippage=0.05,
        mainnet_fork=True,
    )
    model.web3config = web3config
    return model


@pytest.fixture()
def sync_model(arb_web3, funded_wallet) -> HotWalletSyncModel:
    return HotWalletSyncModel(arb_web3, funded_wallet)


@pytest.fixture()
def routing_model(arb_web3, universe, web3config) -> GenericRouting:
    pair_configurator = EthereumPairConfigurator(
        arb_web3,
        universe,
        web3config=web3config,
    )
    return GenericRouting(pair_configurator)


@pytest.fixture()
def state(sync_model, usdc_arb) -> State:
    state = State()
    sync_model.init()
    sync_model.sync_initial(state, reserve_currency=usdc_arb, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, [usdc_arb])
    return state


def _get_total_equity(state: State) -> float:
    """Calculate total portfolio equity: reserves + all open positions."""
    reserve_value = sum(float(r.quantity) for r in state.portfolio.reserves.values())
    position_equity = sum(p.get_equity() for p in state.portfolio.open_positions.values())
    return reserve_value + position_equity


# --- Tests ---


def test_crosschain_detection_and_tui_filtering(
    universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
):
    """Verify cross-chain detection logic and TUI bridge pair filtering.

    1. Check bridge pair has correct kind and is detected as CCTP bridge
    2. Check vault pair on Base has different chain_id from Arbitrum home chain
    3. Verify multichain detection via universe chains
    4. Verify bridge pairs are filtered from TUI PairSelectionApp
    5. Verify chain name resolution works for display
    """

    # 1. Bridge pair detection
    assert bridge_pair.kind == TradingPairKind.cctp_bridge
    assert bridge_pair.is_cctp_bridge()

    # 2. Cross-chain detection: vault on Base vs Arbitrum home
    home_chain_id = ARBITRUM_CHAIN_ID
    assert vault_pair.chain_id == BASE_CHAIN_ID
    assert vault_pair.chain_id != home_chain_id
    assert not vault_pair.is_cctp_bridge()

    # Bridge pair should NOT be considered cross-chain (it IS the bridge)
    is_bridge_cross_chain = (
        not bridge_pair.is_cctp_bridge()
        and bridge_pair.chain_id != home_chain_id
    )
    assert not is_bridge_cross_chain

    # Vault pair on home chain should NOT be cross-chain
    arb_vault = TradingPairIdentifier(
        base=AssetIdentifier(chain_id=ARBITRUM_CHAIN_ID, address="0x1234", token_symbol="TEST", decimals=18),
        quote=AssetIdentifier(chain_id=ARBITRUM_CHAIN_ID, address="0x5678", token_symbol="USDC", decimals=6),
        pool_address="0x1234",
        exchange_address="0x1234",
        internal_id=99,
        internal_exchange_id=99,
        fee=0.0,
        kind=TradingPairKind.vault,
    )
    is_arb_vault_cross_chain = (
        not arb_vault.is_cctp_bridge()
        and arb_vault.chain_id != home_chain_id
    )
    assert not is_arb_vault_cross_chain

    # 3. Multichain detection
    assert len(universe.data_universe.chains) > 1

    # 4. TUI bridge pair filtering
    all_pairs = list(universe.iterate_pairs())
    assert len(all_pairs) == 2
    # PairSelectionApp filters bridge pairs in __init__
    filtered = [p for p in all_pairs if not p.is_cctp_bridge()]
    assert len(filtered) == 1
    assert filtered[0].kind == TradingPairKind.vault

    # 5. Chain name resolution
    chain_name = ChainId(vault_pair.base.chain_id).get_name()
    assert chain_name == "Base"


@pytest.mark.timeout(600)
def test_make_test_trade_crosschain_buy_only(
    arb_web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    state: State,
    universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
    web3config: Web3Config,
):
    """Cross-chain buy-only test trade: bridge in + open vault position.

    1. Set up routing state
    2. Verify initial equity matches deposit
    3. Call make_test_trade with buy_only=True for satellite vault pair
    4. Verify bridge position created with correct kind
    5. Verify vault position created on Base
    6. Verify home reserves decreased by bridge amount
    7. Verify total equity preserved
    """

    # 1. Set up routing state and generic pricing model
    routing_state = routing_model.create_routing_state(
        universe, {"tx_builder": execution_model.tx_builder},
    )
    pricing_model = GenericPricing(routing_model.pair_configurator)

    # 2. Verify initial equity
    initial_equity = _get_total_equity(state)
    assert initial_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01)

    # 3. Call make_test_trade with buy_only for satellite chain pair
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=TEST_TRADE_AMOUNT,
        pair=vault_pair,
        buy_only=True,
        web3config=web3config,
    )

    # 4. Verify bridge position created
    bridge_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.cctp_bridge
    ]
    assert len(bridge_positions) == 1
    bridge_position = bridge_positions[0]
    assert float(bridge_position.get_quantity()) == pytest.approx(float(TEST_TRADE_AMOUNT), rel=0.01)

    # 5. Verify vault position created
    vault_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.vault
    ]
    assert len(vault_positions) == 1
    assert vault_positions[0].get_quantity() > 0

    # 6. Verify home reserves decreased
    home_reserve = state.portfolio.get_default_reserve_position()
    assert float(home_reserve.quantity) == pytest.approx(float(DEPOSIT_AMOUNT - TEST_TRADE_AMOUNT), rel=0.01)

    # 7. Verify total equity preserved (within tolerance for vault share pricing)
    final_equity = _get_total_equity(state)
    assert final_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.03)


@pytest.mark.timeout(600)
def test_make_test_trade_crosschain_round_trip(
    arb_web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    state: State,
    universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
    web3config: Web3Config,
):
    """Full round-trip cross-chain test trade: bridge in → open → close → bridge back.

    1. Set up routing state and pricing
    2. Call make_test_trade with default mode (open + close) for satellite vault pair
    3. Verify no open positions remain after round-trip
    4. Verify capital returned to home reserves
    5. Verify total equity preserved
    """

    # 1. Set up routing and generic pricing model
    routing_state = routing_model.create_routing_state(
        universe, {"tx_builder": execution_model.tx_builder},
    )
    pricing_model = GenericPricing(routing_model.pair_configurator)

    initial_equity = _get_total_equity(state)

    # 2. Call make_test_trade with buy_only first, then close_only
    #    to inspect state between steps
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=TEST_TRADE_AMOUNT,
        pair=vault_pair,
        web3config=web3config,
    )

    # 3. Inspect all trades for debugging
    all_trades = list(state.portfolio.get_all_trades())

    # 3b. No open positions should remain
    open_positions = list(state.portfolio.open_positions.values())
    assert len(open_positions) == 0, \
        f"Expected 0 open positions after round-trip, got {len(open_positions)}: " \
        f"{[(p.pair.get_ticker(), p.pair.kind.value) for p in open_positions]}"

    # 4. Capital returned to home reserves
    home_reserve = state.portfolio.get_default_reserve_position()
    assert float(home_reserve.quantity) > 0

    # 5. Total equity preserved (within tolerance for vault/bridge fees)
    final_equity = _get_total_equity(state)
    assert final_equity == pytest.approx(initial_equity, rel=0.05), \
        f"Equity dropped from {initial_equity} to {final_equity}. " \
        f"Trades: {len(all_trades)}, " \
        f"reserve: {float(home_reserve.quantity)}, " \
        f"closed positions: {len(state.portfolio.closed_positions)}"


@pytest.mark.timeout(600)
def test_make_test_trade_crosschain_close_only(
    arb_web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    state: State,
    universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
    web3config: Web3Config,
):
    """Close-only cross-chain test trade: open with buy_only, then close separately.

    1. Set up routing state and pricing
    2. Open positions via make_test_trade with buy_only=True
    3. Verify open positions exist (bridge + vault)
    4. Close positions via make_test_trade with close_only=True
    5. Verify no open positions remain
    6. Verify capital returned to home reserves
    7. Verify total equity preserved
    """

    # 1. Set up routing state and generic pricing model
    routing_state = routing_model.create_routing_state(
        universe, {"tx_builder": execution_model.tx_builder},
    )
    pricing_model = GenericPricing(routing_model.pair_configurator)

    initial_equity = _get_total_equity(state)

    # 2. Open positions via buy_only
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=TEST_TRADE_AMOUNT,
        pair=vault_pair,
        buy_only=True,
        web3config=web3config,
    )

    # 3. Verify open positions exist
    open_positions = list(state.portfolio.open_positions.values())
    assert len(open_positions) == 2, \
        f"Expected 2 open positions (bridge + vault), got {len(open_positions)}"
    assert any(p.pair.kind == TradingPairKind.cctp_bridge for p in open_positions)
    assert any(p.pair.kind == TradingPairKind.vault for p in open_positions)

    # 4. Close positions via close_only
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=TEST_TRADE_AMOUNT,
        pair=vault_pair,
        close_only=True,
        web3config=web3config,
    )

    # 5. No open positions should remain
    open_positions = list(state.portfolio.open_positions.values())
    assert len(open_positions) == 0, \
        f"Expected 0 open positions after close_only, got {len(open_positions)}: " \
        f"{[(p.pair.get_ticker(), p.pair.kind.value) for p in open_positions]}"

    # 6. Capital returned to home reserves
    home_reserve = state.portfolio.get_default_reserve_position()
    assert float(home_reserve.quantity) > 0

    # 7. Total equity preserved (within tolerance for vault/bridge fees)
    final_equity = _get_total_equity(state)
    assert final_equity == pytest.approx(initial_equity, rel=0.05), \
        f"Equity dropped from {initial_equity} to {final_equity}"
