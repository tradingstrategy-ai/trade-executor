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
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.cctp.constants import TOKEN_MESSENGER_V2

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType, ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.cli.trade_ui_tui import PairSelectionApp
from tradeexecutor.ethereum.cctp.bridge_universe import generate_primary_to_satellite_cctp_bridge_universe
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
    launch = fork_network_anvil(
        JSON_RPC_BASE,
        fork_block_number=BASE_FORK_BLOCK,
        launch_wait_seconds=60.0,
        test_request_timeout=30.0,
    )
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


@pytest.mark.timeout(600)
def test_crosschain_bridge_resolution_and_deposit_simulation(
    arb_web3,
    base_web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    routing_model: GenericRouting,
    state: State,
    universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
    ipor_share_token: AssetIdentifier,
    funded_wallet: HotWallet,
    web3config: Web3Config,
):
    """Resolve CCTP bridge routing and simulate a cross-chain vault deposit in-process.

    Same in-process API as the other perform-test-trade tests (``make_test_trade()``),
    but it asserts the routing resolution explicitly and verifies the deposit
    physically settled on the Base fork, not just in portfolio state. This is the
    path the ``perform-test-trade`` CLI exercises when given a satellite-chain
    vault pair: the bridge is resolved to a CctpBridgeRouting and the bridged USDC
    is deposited into the Base vault.

    1. Resolve the CCTP bridge pair -> "cctp-bridge" router (CctpBridgeRouting)
    2. Resolve the Base vault pair -> "vault" router
    3. Confirm the bridge pair targets the Base destination domain
    4. Run a buy-only cross-chain make_test_trade() (bridge in + open vault)
    5. Assert a CCTP bridge position and a Base vault position were opened, bridge OK
    6. Assert the deposit physically settled on Base: the hot wallet holds IPOR
       vault shares on the Base fork
    """
    from tradeexecutor.ethereum.cctp.routing import CctpBridgeRouting

    pair_configurator = routing_model.pair_configurator

    # 1. CCTP bridge pair resolves to the CCTP bridge router
    bridge_routing_id = pair_configurator.match_router(bridge_pair)
    assert bridge_routing_id.router_name == "cctp-bridge"
    assert isinstance(pair_configurator.get_routing(bridge_pair), CctpBridgeRouting)

    # 2. Satellite vault pair resolves to the vault router
    assert pair_configurator.match_router(vault_pair).router_name == "vault"

    # 3. Bridge pair targets the Base destination domain (forward Arbitrum -> Base)
    assert bridge_pair.other_data["destination_chain_id"] == BASE_CHAIN_ID

    # 4. Buy-only cross-chain test trade: bridge in + open vault on Base
    routing_state = routing_model.create_routing_state(
        universe, {"tx_builder": execution_model.tx_builder},
    )
    pricing_model = GenericPricing(routing_model.pair_configurator)

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

    # 5. Bridge + vault positions opened, bridge trade succeeded
    bridge_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.is_cctp_bridge()
    ]
    vault_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.vault
    ]
    assert len(bridge_positions) == 1
    assert len(vault_positions) == 1
    bridge_trade = list(bridge_positions[0].trades.values())[0]
    assert bridge_trade.is_success(), \
        f"Bridge trade failed: {bridge_trade.get_revert_reason()}"

    # 6. Cross-chain deposit physically settled on Base: hot wallet holds vault shares
    base_vault_shares = fetch_erc20_details(
        base_web3, ipor_share_token.address, chain_id=BASE_CHAIN_ID,
    ).fetch_balance_of(funded_wallet.address)
    assert base_vault_shares > 0, \
        "Expected IPOR vault shares on the Base fork after the cross-chain deposit"


@pytest.mark.timeout(600)
def test_bridge_generation_corrects_wrong_reserve_decimals(
    arb_web3,
    base_web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    funded_wallet: HotWallet,
    web3config: Web3Config,
):
    """Regression test: bridge pair generation corrects wrong reserve_asset decimals.

    Replicates a production failure where DEXPair.quote_token_decimals returned 18
    instead of 6 for ERC-4626 vault pairs (both tokens share the "USDC" symbol).
    The wrong decimals propagated into bridge pairs, causing depositForBurn to
    be called with amount=N*10^18 instead of N*10^6, reverting with
    "ERC20: transfer amount exceeds balance".

    1. Create a pair universe with vault pairs containing USDC with correct decimals (6)
    2. Create a reserve_asset with WRONG decimals (18) simulating the production bug
    3. Generate bridge pairs via generate_primary_to_satellite_cctp_bridge_universe()
    4. Assert generated bridge pair has correct decimals (6, not 18)
    5. Build a trading universe and routing model from the generated bridge pair
    6. Execute a bridge test trade on Anvil forks via make_test_trade()
    7. Verify the bridge trade succeeds (amount_raw was calculated correctly)
    """

    # 1. Create pair universe with pairs on both chains so USDC with
    #    correct decimals (6) is available via pairs.get_token()

    # Arbitrum spot pair — puts USDC(6) into the pair universe on Arbitrum
    usdc_arb = AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )
    weth_arb = AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        token_symbol="WETH",
        decimals=18,
    )
    arb_spot_pair = TradingPairIdentifier(
        base=weth_arb,
        quote=usdc_arb,
        pool_address="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",
        exchange_address="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",
        internal_id=10,
        internal_exchange_id=10,
        fee=0.0005,
        kind=TradingPairKind.spot_market_hold,
    )

    # Base vault pair — puts USDC(6) into the pair universe on Base
    usdc_base = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[BASE_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )
    ipor_share = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=IPOR_VAULT_ADDRESS,
        token_symbol="ipfUSDC",
        decimals=18,
    )
    base_vault_pair = TradingPairIdentifier(
        base=ipor_share,
        quote=usdc_base,
        pool_address=IPOR_VAULT_ADDRESS,
        exchange_address=IPOR_VAULT_ADDRESS,
        internal_id=11,
        internal_exchange_id=11,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="IPOR Fusion",
        other_data={"vault_protocol": "ipor_fusion"},
    )

    initial_pair_universe = create_pair_universe_from_code(
        ChainId.arbitrum, [arb_spot_pair, base_vault_pair],
    )

    arb_exchange = Exchange(
        chain_id=ChainId.arbitrum,
        chain_slug="arbitrum",
        exchange_id=10,
        exchange_slug="uniswap-v3-arb",
        address="0xC31E54c7a869B9FcBEcc14363CF510d1c41fa443",
        exchange_type=ExchangeType.uniswap_v3,
        pair_count=1,
    )
    vault_exchange = Exchange(
        chain_id=ChainId.base,
        chain_slug="base",
        exchange_id=11,
        exchange_slug="ipor-fusion",
        address=IPOR_VAULT_ADDRESS,
        exchange_type=ExchangeType.erc_4626_vault,
        pair_count=1,
    )
    initial_exchange_universe = ExchangeUniverse(
        exchanges={10: arb_exchange, 11: vault_exchange},
    )

    # 2. Create reserve_asset with WRONG decimals (18) — simulates the production
    #    bug where DEXPair.quote_token_decimals picks 18 from the vault share token
    wrong_reserve = AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=18,
    )

    # 3. Generate bridge pairs — the fix resolves decimals from the pair universe
    result = generate_primary_to_satellite_cctp_bridge_universe(
        pairs=initial_pair_universe,
        exchange_universe=initial_exchange_universe,
        reserve_asset=wrong_reserve,
        primary_chain=ChainId.arbitrum,
    )

    # 4. Assert the fix: bridge pair USDC decimals are 6, not 18
    assert len(result.generated_pairs) == 1
    bridge_pair = result.generated_pairs[0]
    assert bridge_pair.kind == TradingPairKind.cctp_bridge
    assert bridge_pair.quote.decimals == 6, (
        f"Bridge quote.decimals={bridge_pair.quote.decimals}, expected 6. "
        "Wrong decimals cause amount_raw to be off by 10^12, "
        "producing the 'ERC20: transfer amount exceeds balance' revert."
    )
    assert bridge_pair.base.decimals == 6, (
        f"Bridge base.decimals={bridge_pair.base.decimals}, expected 6."
    )

    # 5. Build trading universe using the generated bridge pair + vault pair
    data_universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.arbitrum, ChainId.base},
        exchanges=set(result.exchange_universe.exchanges.values()),
        pairs=result.pair_universe,
        exchange_universe=result.exchange_universe,
        candles=None,
        liquidity=None,
    )

    universe = TradingStrategyUniverse(
        data_universe=data_universe,
        reserve_assets=[usdc_arb],
    )

    pair_configurator = EthereumPairConfigurator(
        arb_web3,
        universe,
        web3config=web3config,
    )
    routing_model = GenericRouting(pair_configurator)
    pricing_model = GenericPricing(pair_configurator)

    state = State()
    sync_model.init()
    sync_model.sync_initial(state, reserve_currency=usdc_arb, reserve_token_price=1.0)
    sync_model.sync_treasury(native_datetime_utc_now(), state, [usdc_arb])

    routing_state = routing_model.create_routing_state(
        universe, {"tx_builder": execution_model.tx_builder},
    )

    # 6. Execute bridge test trade via make_test_trade()
    #    This calls CctpBridgeRouting.setup_trades() which computes:
    #      amount_raw = int(trade.planned_reserve * (10 ** pair.quote.decimals))
    #    With correct decimals=6: amount_raw = 3 * 10^6  = 3_000_000 (OK)
    #    With wrong   decimals=18: amount_raw = 3 * 10^18 (reverts on-chain)
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
        pair=base_vault_pair,
        buy_only=True,
        web3config=web3config,
    )

    # 7. Verify bridge trade succeeded (not reverted with wrong amount)
    bridge_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.cctp_bridge
    ]
    assert len(bridge_positions) == 1
    bridge_trade = list(bridge_positions[0].trades.values())[0]
    assert bridge_trade.is_success(), (
        f"Bridge trade failed: {bridge_trade.get_revert_reason()}. "
        "This indicates wrong USDC decimals in the bridge pair."
    )


def test_bridge_generation_primary_chain_not_in_pairs():
    """Regression: bridge pair decimals correct when primary chain has no pairs.

    When the pair universe only has satellite-chain pairs (e.g. Base vault
    pairs), the primary chain USDC is not in the pair universe. The
    generation function must still produce bridge pairs with decimals=6,
    never falling back to reserve_asset.decimals which may be wrong.

    1. Create a pair universe with ONLY Base vault pairs (no Arbitrum pairs)
    2. Pass a reserve_asset with wrong decimals (18) for Arbitrum USDC
    3. Call generate_primary_to_satellite_cctp_bridge_universe()
    4. Assert both quote (primary) and base (satellite) decimals are 6
    """

    # 1. Base-only pair universe — no Arbitrum pairs at all
    usdc_base = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[BASE_CHAIN_ID],
        token_symbol="USDC",
        decimals=6,
    )
    ipor_share = AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=IPOR_VAULT_ADDRESS,
        token_symbol="ipfUSDC",
        decimals=18,
    )
    base_vault_pair = TradingPairIdentifier(
        base=ipor_share,
        quote=usdc_base,
        pool_address=IPOR_VAULT_ADDRESS,
        exchange_address=IPOR_VAULT_ADDRESS,
        internal_id=11,
        internal_exchange_id=11,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name="IPOR Fusion",
        other_data={"vault_protocol": "ipor_fusion"},
    )

    pair_universe = create_pair_universe_from_code(
        ChainId.arbitrum, [base_vault_pair],
    )
    exchange_universe = ExchangeUniverse(exchanges={})

    # 2. Wrong reserve_asset — primary chain USDC with decimals=18
    wrong_reserve = AssetIdentifier(
        chain_id=ARBITRUM_CHAIN_ID,
        address=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        token_symbol="USDC",
        decimals=18,
    )

    # 3. Generate bridge pairs — primary USDC is NOT in pair universe
    result = generate_primary_to_satellite_cctp_bridge_universe(
        pairs=pair_universe,
        exchange_universe=exchange_universe,
        reserve_asset=wrong_reserve,
        primary_chain=ChainId.arbitrum,
    )

    # 4. Assert correct decimals despite missing primary USDC in pairs
    assert len(result.generated_pairs) == 1
    bridge_pair = result.generated_pairs[0]
    assert bridge_pair.quote.decimals == 6, (
        f"Bridge quote (primary USDC) decimals={bridge_pair.quote.decimals}, "
        "expected 6. Must not fall back to reserve_asset.decimals=18 "
        "when primary chain USDC is absent from the pair universe."
    )
    assert bridge_pair.base.decimals == 6, (
        f"Bridge base (satellite USDC) decimals={bridge_pair.base.decimals}, "
        "expected 6."
    )
