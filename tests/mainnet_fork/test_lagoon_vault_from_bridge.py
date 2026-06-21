"""Test vault deposit/redeem on a satellite chain using bridged USDC.

Mainnet fork test (Arbitrum + Base) that exercises:

1. Bridge USDC from Arbitrum to Base via CCTP
2. Deposit bridged USDC into IPOR Fusion vault on Base
3. Redeem from the vault back to USDC on Base
4. Bridge USDC back from Base to Arbitrum

Verifies that ``start_execution()`` correctly routes vault deposit
capital from the bridge position (not from home reserves) and that
portfolio equity stays consistent across all operations.

Uses hot wallet execution with ``make_test_trade()`` for individual
operations and forged CCTP attestations on Anvil forks.

Requires ``JSON_RPC_ARBITRUM`` and ``JSON_RPC_BASE`` environment variables.
"""

import datetime
import logging
import os
import secrets
from decimal import Decimal

import pytest
from eth_account import Account
from eth_typing import HexAddress
from web3 import Web3

from eth_defi.cctp.constants import TOKEN_MESSENGER_V2
from eth_defi.cctp.testing import replace_attester_on_fork, craft_cctp_message, forge_attestation
from eth_defi.cctp.bridge import prepare_receive_message
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fork_network_anvil, AnvilLaunch, set_balance, fund_erc20_on_anvil, mine
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details

from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import Exchange, ExchangeType
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe

from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.utils.hex import hexbytes_to_hex_str


logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")

ARBITRUM_CHAIN_ID = 42161
BASE_CHAIN_ID = 8453

# IPOR Fusion vault on Base — USDC, no lockup beyond ~1 hour
IPOR_VAULT_ADDRESS = "0x45aa96f0b3188d47a1dafdbefce1db6b37f58216"
MUSCADINE_VAULT_ADDRESS = "0xf7e26fa48a568b8b0038e104dfd8abdf0f99074f"
MO_EARN_MAX_VAULT_ADDRESS = "0x3094b241aade60f91f1c82b0628a10d9501462f9"

DEPOSIT_AMOUNT = Decimal("30")  # Total USDC deposited
BRIDGE_AMOUNT = Decimal("5")    # USDC bridged to Base
VAULT_AMOUNT = Decimal("3")     # USDC deposited into vault
SHARED_BRIDGE_TRADE_AMOUNT = Decimal("4")

#: Pin Base fork to a block where the IPOR Fusion vault has sufficient
#: deposit headroom.  The vault's ``totalSupplyCap`` has been nearly full
#: since mid-May 2026 (``maxDeposit`` < 4 USDC at latest blocks), causing
#: ``ERC4626ExceededMaxDeposit`` reverts on unpinned forks.  At this block
#: the vault has ~14 000 USDC of headroom.
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
def arb_web3(arb_anvil) -> Web3:
    return create_multi_provider_web3(arb_anvil.json_rpc_url)


@pytest.fixture()
def base_web3(base_anvil) -> Web3:
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
    """Create hot wallet funded on both chains."""
    wallet = HotWallet.create_for_testing(arb_web3, eth_amount=100)
    # Fund gas on Base
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
def test_attesters(arb_web3, base_web3) -> dict:
    return {
        ARBITRUM_CHAIN_ID: replace_attester_on_fork(arb_web3),
        BASE_CHAIN_ID: replace_attester_on_fork(base_web3),
    }


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
def muscadine_share_token() -> AssetIdentifier:
    """Muscadine USDC Vault share token on Base."""
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=MUSCADINE_VAULT_ADDRESS,
        token_symbol="mvUSDC",
        decimals=18,
    )


@pytest.fixture()
def mo_earn_max_share_token() -> AssetIdentifier:
    """Mo Earn Max USDC share token on Base."""
    return AssetIdentifier(
        chain_id=BASE_CHAIN_ID,
        address=MO_EARN_MAX_VAULT_ADDRESS,
        token_symbol="maxUSD",
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


def _make_base_vault_pair(
    *,
    share_token: AssetIdentifier,
    address: str,
    internal_id: int,
    exchange_name: str,
    vault_protocol: str = "erc_4626",
    vault_features: list[str] | None = None,
) -> TradingPairIdentifier:
    """Create a Base USDC vault pair for fork tests."""
    return TradingPairIdentifier(
        base=share_token,
        quote=AssetIdentifier(
            chain_id=BASE_CHAIN_ID,
            address=USDC_NATIVE_TOKEN[BASE_CHAIN_ID],
            token_symbol="USDC",
            decimals=6,
        ),
        pool_address=address,
        exchange_address=address,
        internal_id=internal_id,
        internal_exchange_id=internal_id,
        fee=0.0,
        kind=TradingPairKind.vault,
        exchange_name=exchange_name,
        other_data={
            "vault_protocol": vault_protocol,
            "vault_features": vault_features,
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
            "vault_features": ["ipor_like"],
        },
    )


@pytest.fixture()
def muscadine_vault_pair(muscadine_share_token) -> TradingPairIdentifier:
    """Muscadine USDC Vault pair on Base."""
    return _make_base_vault_pair(
        share_token=muscadine_share_token,
        address=MUSCADINE_VAULT_ADDRESS,
        internal_id=3,
        exchange_name="Muscadine USDC Vault",
        vault_features=["morpho_like", "euler_earn_like"],
    )


@pytest.fixture()
def mo_earn_max_vault_pair(mo_earn_max_share_token) -> TradingPairIdentifier:
    """Mo Earn Max USDC pair on Base."""
    return _make_base_vault_pair(
        share_token=mo_earn_max_share_token,
        address=MO_EARN_MAX_VAULT_ADDRESS,
        internal_id=4,
        exchange_name="Mo Earn Max USDC",
        vault_features=["morpho_like", "euler_earn_like"],
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
def shared_bridge_universe(
    bridge_pair,
    vault_pair,
    muscadine_vault_pair,
    mo_earn_max_vault_pair,
    usdc_arb,
) -> TradingStrategyUniverse:
    """Trading universe with one CCTP bridge and three real Base vaults."""

    vault_pairs = [muscadine_vault_pair, mo_earn_max_vault_pair, vault_pair]
    pair_universe = create_pair_universe_from_code(ChainId.arbitrum, [bridge_pair] + vault_pairs)

    cctp_exchange = Exchange(
        chain_id=ChainId.arbitrum,
        chain_slug="arbitrum",
        exchange_id=1,
        exchange_slug="cctp-bridge",
        address=TOKEN_MESSENGER_V2,
        exchange_type=ExchangeType.uniswap_v2,
        pair_count=1,
    )

    vault_exchanges = [
        Exchange(
            chain_id=ChainId.base,
            chain_slug="base",
            exchange_id=pair.internal_exchange_id,
            exchange_slug=pair.exchange_name.lower().replace(" ", "-"),
            address=pair.exchange_address,
            exchange_type=ExchangeType.erc_4626_vault,
            pair_count=1,
        )
        for pair in vault_pairs
    ]

    data_universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={ChainId.arbitrum, ChainId.base},
        exchanges={cctp_exchange, *vault_exchanges},
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


def get_total_equity(state: State) -> float:
    """Calculate total portfolio equity: reserves + all open positions."""
    reserve_value = sum(float(r.quantity) for r in state.portfolio.reserves.values())
    position_equity = sum(p.get_equity() for p in state.portfolio.open_positions.values())
    return reserve_value + position_equity


def _create_routing_model(arb_web3: Web3, universe: TradingStrategyUniverse, web3config: Web3Config) -> GenericRouting:
    """Create generic multichain routing for a test universe."""
    pair_configurator = EthereumPairConfigurator(
        arb_web3,
        universe,
        web3config=web3config,
    )
    return GenericRouting(pair_configurator)


def _get_bridge_position(state: State, bridge_pair: TradingPairIdentifier):
    """Return the single shared bridge position for a bridge pair."""
    bridge_position = state.portfolio.get_position_by_trading_pair(bridge_pair)
    assert bridge_position is not None and bridge_position.is_open()
    assert len([p for p in state.portfolio.open_positions.values() if p.pair.is_cctp_bridge()]) == 1
    return bridge_position


def _open_cross_chain_vault_position(
    *,
    arb_web3: Web3,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    sync_model: HotWalletSyncModel,
    state: State,
    universe: TradingStrategyUniverse,
    web3config: Web3Config,
    pair: TradingPairIdentifier,
    amount: Decimal,
) -> None:
    """Open a Base vault position through the production cross-chain test-trade path."""
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=GenericPricing(routing_model.pair_configurator),
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=amount,
        pair=pair,
        buy_only=True,
        web3config=web3config,
    )


def _open_shared_bridge_vault_position(
    *,
    arb_web3: Web3,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    sync_model: HotWalletSyncModel,
    state: State,
    universe: TradingStrategyUniverse,
    web3config: Web3Config,
    pair: TradingPairIdentifier,
    amount: Decimal,
    expected_name: str,
):
    """Open a Base vault position through the shared CCTP bridge."""
    _open_cross_chain_vault_position(
        arb_web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        universe=universe,
        web3config=web3config,
        pair=pair,
        amount=amount,
    )
    position = state.portfolio.get_position_by_trading_pair(pair)
    assert position is not None and position.is_open()
    assert expected_name in position.pair.exchange_name
    return position


def _warm_vault_pricing(
    *,
    routing_model: GenericRouting,
    pairs: list[TradingPairIdentifier],
) -> None:
    """Warm vault adapters before the transaction-heavy test sequence starts."""
    ts = native_datetime_utc_now()
    pricing_model = GenericPricing(routing_model.pair_configurator)
    for pair in pairs:
        pricing_model.get_buy_price(ts, pair, SHARED_BRIDGE_TRADE_AMOUNT)


def _execute_vault_sell(
    *,
    base_web3: Web3,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    state: State,
    universe: TradingStrategyUniverse,
    position,
    quantity: Decimal | None = None,
) -> None:
    """Sell all or part of a Base vault position, mining past lockups if needed."""
    pricing_model = routing_model.pair_configurator.get_config(
        routing_model.pair_configurator.match_router(position.pair)
    ).pricing_model

    for attempt in range(2):
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts,
            universe,
            state,
            pricing_model,
            default_slippage_tolerance=0.05,
        )
        if quantity is None:
            trades = position_manager.close_position(position)
        else:
            trades = position_manager.close_spot_position(position, quantity=quantity, slippage_tolerance=0.05)

        execution_model.execute_trades(
            ts,
            state,
            trades,
            routing_model,
            routing_state,
        )
        trade = trades[0]
        if trade.is_success():
            return
        if attempt == 0:
            mine(base_web3, increase_timestamp=3600)

    raise AssertionError(f"Vault sell failed for {position.pair}: {trade.get_revert_reason()}")


# --- Tests ---


@pytest.mark.timeout(600)
def test_vault_deposit_from_bridge_capital(
    arb_web3,
    base_web3,
    hot_wallet: HotWallet,
    test_attesters,
    execution_model,
    sync_model,
    routing_model,
    state,
    universe,
    bridge_pair,
    vault_pair,
    usdc_arb,
    web3config,
):
    """Full round-trip: bridge → vault deposit → vault redeem → bridge back.

    Verifies that vault deposits on a satellite chain correctly allocate
    capital from the CCTP bridge position, not from home chain reserves.
    """

    ts = native_datetime_utc_now()
    routing_state = routing_model.create_routing_state(universe, {"tx_builder": execution_model.tx_builder})

    # --- Step 1: Verify initial state ---
    initial_equity = get_total_equity(state)
    assert initial_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01), \
        f"Initial equity {initial_equity} != deposit {DEPOSIT_AMOUNT}"

    # Chain equity: all on Arbitrum
    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(0, abs=0.01)

    # --- Step 2: Bridge USDC to Base ---
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=routing_model.pair_configurator.get_config(
            routing_model.pair_configurator.match_router(bridge_pair)
        ).pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=BRIDGE_AMOUNT,
        pair=bridge_pair,
        buy_only=True,
    )

    # Verify bridge trade succeeded
    bridge_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.cctp_bridge
    ]
    assert len(bridge_positions) == 1, f"Expected 1 bridge position, got {len(bridge_positions)}"
    bridge_position = bridge_positions[0]
    assert float(bridge_position.get_quantity()) == pytest.approx(float(BRIDGE_AMOUNT), rel=0.01)

    # Forge attestation and receive USDC on Base
    message = craft_cctp_message(
        source_domain=3,              # Arbitrum
        destination_domain=6,         # Base
        nonce=999_999_006,
        mint_recipient=hot_wallet.address,
        amount=int(BRIDGE_AMOUNT * 10**6),
        burn_token=USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
    )
    attestation = forge_attestation(message, test_attesters[BASE_CHAIN_ID])
    receive_fn = prepare_receive_message(base_web3, message, attestation)
    tx_hash = receive_fn.transact({"from": base_web3.eth.accounts[0]})
    assert_transaction_success_with_explanation(base_web3, tx_hash)

    # Verify USDC arrived on Base
    base_usdc = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[BASE_CHAIN_ID])
    base_balance = base_usdc.fetch_balance_of(hot_wallet.address)
    assert float(base_balance) == pytest.approx(float(BRIDGE_AMOUNT), rel=0.01), \
        f"Base USDC balance {base_balance} != bridge amount {BRIDGE_AMOUNT}"

    # Equity check: home reserve (DEPOSIT - BRIDGE) + bridge position (BRIDGE)
    equity_after_bridge = get_total_equity(state)
    assert equity_after_bridge == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01), \
        f"Equity after bridge {equity_after_bridge} != {DEPOSIT_AMOUNT}"

    # Chain equity: Arb reserves + Base bridge position
    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT - BRIDGE_AMOUNT), rel=0.01)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(float(BRIDGE_AMOUNT), rel=0.01)

    # --- Step 3: Deposit into IPOR vault on Base ---
    ts = native_datetime_utc_now()
    make_test_trade(
        web3=arb_web3,
        execution_model=execution_model,
        pricing_model=routing_model.pair_configurator.get_config(
            routing_model.pair_configurator.match_router(vault_pair)
        ).pricing_model,
        sync_model=sync_model,
        state=state,
        universe=universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=VAULT_AMOUNT,
        pair=vault_pair,
        buy_only=True,
    )

    # Verify vault position opened
    vault_positions = [
        p for p in state.portfolio.open_positions.values()
        if p.pair.kind == TradingPairKind.vault
    ]
    assert len(vault_positions) == 1, f"Expected 1 vault position, got {len(vault_positions)}"
    vault_position = vault_positions[0]
    assert vault_position.get_quantity() > 0, "Vault position should have shares"

    # Bridge capital should be allocated
    assert float(bridge_position.bridge_capital_allocated) == pytest.approx(float(VAULT_AMOUNT), rel=0.01), \
        f"Bridge capital allocated {bridge_position.bridge_capital_allocated} != {VAULT_AMOUNT}"

    # Home reserves should NOT have changed (capital came from bridge, not reserves)
    home_reserve = state.portfolio.get_default_reserve_position()
    assert float(home_reserve.quantity) == pytest.approx(float(DEPOSIT_AMOUNT - BRIDGE_AMOUNT), rel=0.01), \
        f"Home reserve {home_reserve.quantity} changed unexpectedly"

    # Equity check
    equity_after_vault = get_total_equity(state)
    assert equity_after_vault == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.02), \
        f"Equity after vault deposit {equity_after_vault} != {DEPOSIT_AMOUNT}"

    # Chain equity: Arb reserves + Base (bridge unallocated + vault)
    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT - BRIDGE_AMOUNT), rel=0.01)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(float(BRIDGE_AMOUNT), rel=0.02)

    # --- Step 4: Redeem from IPOR vault ---
    # IPOR Fusion has a short lockup — forward time 1 hour
    mine(base_web3, increase_timestamp=3600)

    ts = native_datetime_utc_now()
    position_manager = PositionManager(
        ts,
        universe,
        state,
        routing_model.pair_configurator.get_config(
            routing_model.pair_configurator.match_router(vault_pair)
        ).pricing_model,
        default_slippage_tolerance=0.05,
    )

    trades = position_manager.close_position(vault_position)
    assert len(trades) == 1, f"Expected 1 close trade, got {len(trades)}"

    execution_model.execute_trades(
        ts,
        state,
        trades,
        routing_model,
        routing_state,
    )

    sell_trade = trades[0]
    assert sell_trade.is_success(), f"Vault redeem failed: {sell_trade.get_revert_reason()}"

    # Capital should be returned to bridge position
    assert float(bridge_position.bridge_capital_allocated) < float(VAULT_AMOUNT), \
        "Bridge capital allocated should decrease after vault redeem"

    equity_after_redeem = get_total_equity(state)
    assert equity_after_redeem == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.03), \
        f"Equity after vault redeem {equity_after_redeem} != {DEPOSIT_AMOUNT}"

    # Chain equity: Arb reserves + Base bridge (capital returned)
    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT - BRIDGE_AMOUNT), rel=0.01)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(float(BRIDGE_AMOUNT), rel=0.03)

    # --- Step 5: Bridge USDC back to Arbitrum ---
    # Check Base USDC balance before bridge-back
    base_usdc_check = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[BASE_CHAIN_ID])
    base_balance_before_bridge_back = base_usdc_check.fetch_balance_of(hot_wallet.address)
    logger.info(f"Base USDC balance before bridge-back: {base_balance_before_bridge_back}")

    # Create bridge sell trade manually (PositionManager doesn't support
    # closing CCTP bridge positions yet).
    # Use the actual on-chain balance — vault rounding may leave slightly
    # less than the position quantity.
    ts = native_datetime_utc_now()
    bridge_sell_quantity = base_balance_before_bridge_back
    reserve_asset = universe.get_reserve_asset()
    _, bridge_back_trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=bridge_pair,
        quantity=-bridge_sell_quantity,
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        position=bridge_position,
        closing=True,
        slippage_tolerance=0.05,
    )
    trades = [bridge_back_trade]

    execution_model.execute_trades(
        ts,
        state,
        trades,
        routing_model,
        routing_state,
    )

    assert bridge_back_trade.is_success(), f"Bridge back failed: {bridge_back_trade.get_revert_reason()}"

    # Forge attestation and receive USDC on Arbitrum
    message_back = craft_cctp_message(
        source_domain=6,              # Base
        destination_domain=3,         # Arbitrum
        nonce=999_999_003,
        mint_recipient=hot_wallet.address,
        amount=int(bridge_back_trade.executed_reserve * 10**6),
        burn_token=USDC_NATIVE_TOKEN[BASE_CHAIN_ID],
    )
    attestation_back = forge_attestation(message_back, test_attesters[ARBITRUM_CHAIN_ID])
    receive_fn_back = prepare_receive_message(arb_web3, message_back, attestation_back)
    tx_hash_back = receive_fn_back.transact({"from": arb_web3.eth.accounts[0]})
    assert_transaction_success_with_explanation(arb_web3, tx_hash_back)

    # --- Final equity check ---
    # Sync home chain reserves to pick up returned USDC
    sync_model.sync_treasury(native_datetime_utc_now(), state, [universe.get_reserve_asset()])

    final_equity = get_total_equity(state)
    assert final_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.03), \
        f"Final equity {final_equity} != {DEPOSIT_AMOUNT}"

    # All positions should be closed
    open_positions = list(state.portfolio.open_positions.values())
    assert len(open_positions) == 0, \
        f"Expected 0 open positions after full round-trip, got {len(open_positions)}: {open_positions}"

    # Chain equity: all back on Arbitrum
    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.03)
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(0, abs=0.01)


@pytest.mark.timeout(900)
def test_multiple_base_vault_positions_share_one_bridge(
    arb_web3: Web3,
    base_web3: Web3,
    execution_model: EthereumExecution,
    sync_model: HotWalletSyncModel,
    state: State,
    shared_bridge_universe: TradingStrategyUniverse,
    bridge_pair: TradingPairIdentifier,
    muscadine_vault_pair: TradingPairIdentifier,
    mo_earn_max_vault_pair: TradingPairIdentifier,
    vault_pair: TradingPairIdentifier,
    web3config: Web3Config,
):
    """Three real Base vault positions share and mutate one CCTP bridge.

    1. Bridge from Arbitrum to Base and open Muscadine USDC Vault (X).
    2. Bridge from Arbitrum to Base and open Mo Earn Max USDC (Y).
    3. Close Muscadine USDC Vault (X).
    4. Bridge from Arbitrum to Base and open IPOR Fusion USDC (Z).
    5. Decrease Mo Earn Max USDC (Y).
    6. Close IPOR Fusion USDC (Z).
    7. Close Mo Earn Max USDC (Y).
    """
    routing_model = _create_routing_model(arb_web3, shared_bridge_universe, web3config)
    routing_state = routing_model.create_routing_state(
        shared_bridge_universe,
        {"tx_builder": execution_model.tx_builder},
    )
    _warm_vault_pricing(
        routing_model=routing_model,
        pairs=[muscadine_vault_pair, mo_earn_max_vault_pair, vault_pair],
    )
    initial_equity = get_total_equity(state)
    assert initial_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01)

    # 1. Bridge from Arbitrum to Base and open Muscadine USDC Vault (X).
    position_x = _open_shared_bridge_vault_position(
        arb_web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        universe=shared_bridge_universe,
        web3config=web3config,
        pair=muscadine_vault_pair,
        amount=SHARED_BRIDGE_TRADE_AMOUNT,
        expected_name="Muscadine",
    )
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert float(bridge_position.bridge_capital_allocated) == pytest.approx(float(SHARED_BRIDGE_TRADE_AMOUNT), rel=0.05)

    # 2. Bridge from Arbitrum to Base and open Mo Earn Max USDC (Y).
    position_y = _open_shared_bridge_vault_position(
        arb_web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        universe=shared_bridge_universe,
        web3config=web3config,
        pair=mo_earn_max_vault_pair,
        amount=SHARED_BRIDGE_TRADE_AMOUNT,
        expected_name="Mo Earn",
    )
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert float(bridge_position.bridge_capital_allocated) == pytest.approx(float(SHARED_BRIDGE_TRADE_AMOUNT * 2), rel=0.05)

    # 3. Close Muscadine USDC Vault (X).
    _execute_vault_sell(
        base_web3=base_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        universe=shared_bridge_universe,
        position=position_x,
    )
    assert position_x.is_closed()
    bridge_position = _get_bridge_position(state, bridge_pair)
    allocation_after_x_close = bridge_position.bridge_capital_allocated
    assert float(allocation_after_x_close) == pytest.approx(float(SHARED_BRIDGE_TRADE_AMOUNT), rel=0.10)

    # 4. Bridge from Arbitrum to Base and open IPOR Fusion USDC (Z).
    position_z = _open_shared_bridge_vault_position(
        arb_web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        universe=shared_bridge_universe,
        web3config=web3config,
        pair=vault_pair,
        amount=SHARED_BRIDGE_TRADE_AMOUNT,
        expected_name="IPOR",
    )
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert float(bridge_position.bridge_capital_allocated) == pytest.approx(float(SHARED_BRIDGE_TRADE_AMOUNT * 2), rel=0.10)

    # 5. Decrease Mo Earn Max USDC (Y).
    y_quantity_before_decrease = position_y.get_quantity()
    y_decrease_quantity = y_quantity_before_decrease / Decimal(2)
    allocation_before_y_decrease = bridge_position.bridge_capital_allocated
    _execute_vault_sell(
        base_web3=base_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        universe=shared_bridge_universe,
        position=position_y,
        quantity=y_decrease_quantity,
    )
    assert position_y.is_open()
    assert position_y.get_quantity() < y_quantity_before_decrease
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert bridge_position.bridge_capital_allocated < allocation_before_y_decrease
    assert bridge_position.bridge_capital_allocated > 0

    # 6. Close IPOR Fusion USDC (Z).
    allocation_before_z_close = bridge_position.bridge_capital_allocated
    _execute_vault_sell(
        base_web3=base_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        universe=shared_bridge_universe,
        position=position_z,
    )
    assert position_z.is_closed()
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert bridge_position.bridge_capital_allocated < allocation_before_z_close
    assert bridge_position.bridge_capital_allocated > 0

    # 7. Close Mo Earn Max USDC (Y).
    _execute_vault_sell(
        base_web3=base_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        universe=shared_bridge_universe,
        position=position_y,
    )
    assert position_y.is_closed()
    bridge_position = _get_bridge_position(state, bridge_pair)
    assert float(bridge_position.bridge_capital_allocated) == pytest.approx(0, abs=0.05)
    assert len([p for p in state.portfolio.open_positions.values() if p.pair.kind == TradingPairKind.vault]) == 0
    assert get_total_equity(state) == pytest.approx(initial_equity, rel=0.05)
