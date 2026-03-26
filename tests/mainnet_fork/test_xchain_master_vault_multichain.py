"""Test the xchain master-vault fork-simulated multichain lifecycle.

1. Fork Arbitrum, Base, and HyperEVM, deploy a multichain Lagoon vault, and
   create the auto-bridge trading universe.
2. Bridge USDC from Arbitrum to Base and HyperEVM, then open vault positions
   on Arbitrum, Base, and HyperEVM.
3. Close all vault positions, sell the forward bridge positions to return
   capital to Arbitrum, and verify equity ends back on the primary chain.
"""

import datetime
import importlib.util
import json
import logging
import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.cctp.bridge import prepare_receive_message
from eth_defi.cctp.testing import craft_cctp_message, forge_attestation, replace_attester_on_fork
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import (
    AnvilLaunch,
    fork_network_anvil,
    fund_erc20_on_anvil,
    mine,
    set_balance,
)
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from typer.main import get_command
from web3 import Web3

from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client

from tradeexecutor.cli.main import app
from tradeexecutor.cli.testtrade import make_test_trade
from tradeexecutor.ethereum.ethereum_protocol_adapters import EthereumPairConfigurator
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import HotWalletTransactionBuilder
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.state.identifier import TradingPairKind
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.trading_universe_input import CreateTradingUniverseInput
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.universe_model import UniverseOptions

logger = logging.getLogger(__name__)

JSON_RPC_ARBITRUM = os.environ.get("JSON_RPC_ARBITRUM")
JSON_RPC_BASE = os.environ.get("JSON_RPC_BASE")
JSON_RPC_HYPERLIQUID = os.environ.get("JSON_RPC_HYPERLIQUID")
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

ARBITRUM_CHAIN_ID = 42161
BASE_CHAIN_ID = 8453
HYPEREVM_CHAIN_ID = 999

CCTP_DOMAIN_BY_CHAIN = {
    ARBITRUM_CHAIN_ID: 3,
    BASE_CHAIN_ID: 6,
    HYPEREVM_CHAIN_ID: 19,
}

DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
HOT_WALLET_PRIVATE_KEY = "0x7b5b648bde0b4ef1ceb56d2c3e1fb938a6b8fb3d399fa6320ef1639705c2df08"

TEST_VAULTS = {
    ARBITRUM_CHAIN_ID: "0x0df2e3a0b5997adc69f8768e495fd98a4d00f134",
    BASE_CHAIN_ID: "0x3094b241aade60f91f1c82b0628a10d9501462f9",
    HYPEREVM_CHAIN_ID: "0x8a862fd6c12f9ad34c9c2ff45ab2b6712e8cea27",
}

DEPOSIT_AMOUNT = Decimal("20")
BRIDGE_BASE_AMOUNT = Decimal("6")
BRIDGE_HYPER_AMOUNT = Decimal("6")
VAULT_ARB_AMOUNT = Decimal("4")
VAULT_BASE_AMOUNT = Decimal("4")
VAULT_HYPER_AMOUNT = Decimal("4")

pytestmark = pytest.mark.skipif(
    not all([JSON_RPC_ARBITRUM, JSON_RPC_BASE, JSON_RPC_HYPERLIQUID, TRADING_STRATEGY_API_KEY]),
    reason="JSON_RPC_ARBITRUM, JSON_RPC_BASE, JSON_RPC_HYPERLIQUID, and TRADING_STRATEGY_API_KEY are required",
)


def _load_strategy_module(strategy_file: Path):
    spec = importlib.util.spec_from_file_location("xchain_master_vault_test_strategy", strategy_file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _get_total_equity(state: State) -> float:
    reserve_value = sum(float(r.quantity) for r in state.portfolio.reserves.values())
    position_equity = sum(position.get_equity() for position in state.portfolio.open_positions.values())
    return reserve_value + position_equity


def _find_bridge_pair(strategy_universe, destination_chain_id: int):
    for pair in strategy_universe.iterate_pairs():
        if pair.kind == TradingPairKind.cctp_bridge and pair.get_destination_chain_id() == destination_chain_id:
            return pair
    raise AssertionError(f"Bridge pair missing for destination chain {destination_chain_id}")


def _find_vault_pair(strategy_universe, chain_id: int, vault_address: str):
    for pair in strategy_universe.iterate_pairs():
        if pair.kind == TradingPairKind.vault and pair.base.chain_id == chain_id and pair.pool_address.lower() == vault_address.lower():
            return pair
    raise AssertionError(f"Vault pair missing for chain {chain_id} and address {vault_address}")


def _receive_cctp_transfer(
    *,
    source_chain_id: int,
    destination_chain_id: int,
    amount: Decimal,
    recipient: str,
    web3_by_chain: dict[int, Web3],
    test_attesters: dict[int, str],
    nonce: int,
) -> None:
    message = craft_cctp_message(
        source_domain=CCTP_DOMAIN_BY_CHAIN[source_chain_id],
        destination_domain=CCTP_DOMAIN_BY_CHAIN[destination_chain_id],
        nonce=nonce,
        mint_recipient=recipient,
        amount=int(amount * 10**6),
        burn_token=USDC_NATIVE_TOKEN[source_chain_id],
    )
    attestation = forge_attestation(message, test_attesters[destination_chain_id])
    receive_fn = prepare_receive_message(web3_by_chain[destination_chain_id], message, attestation)
    tx_hash = receive_fn.transact({"from": web3_by_chain[destination_chain_id].eth.accounts[0]})
    assert_transaction_success_with_explanation(web3_by_chain[destination_chain_id], tx_hash)


def _open_trade(
    *,
    web3: Web3,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    sync_model: HotWalletSyncModel,
    state: State,
    strategy_universe,
    pair,
    amount: Decimal,
) -> None:
    make_test_trade(
        web3=web3,
        execution_model=execution_model,
        pricing_model=routing_model.pair_configurator.get_config(
            routing_model.pair_configurator.match_router(pair)
        ).pricing_model,
        sync_model=sync_model,
        state=state,
        universe=strategy_universe,
        routing_model=routing_model,
        routing_state=routing_state,
        max_slippage=0.05,
        amount=amount,
        pair=pair,
        buy_only=True,
    )


def _close_vault_position(
    *,
    web3: Web3,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    state: State,
    strategy_universe,
    pair,
) -> None:
    position = state.portfolio.get_position_by_trading_pair(pair)
    assert position is not None and position.is_open()
    pricing_model = routing_model.pair_configurator.get_config(
        routing_model.pair_configurator.match_router(pair)
    ).pricing_model

    for attempt in range(2):
        ts = native_datetime_utc_now()
        position_manager = PositionManager(
            ts,
            strategy_universe,
            state,
            pricing_model,
            default_slippage_tolerance=0.05,
        )
        trades = position_manager.close_position(position)
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
            mine(web3, increase_timestamp=3600)

    raise AssertionError(f"Vault close failed for {pair}: {trade.get_revert_reason()}")


def _close_bridge_position(
    *,
    execution_model: EthereumExecution,
    routing_model: GenericRouting,
    routing_state,
    state: State,
    strategy_universe,
    pair,
    quantity: Decimal,
) -> None:
    bridge_position = state.portfolio.get_position_by_trading_pair(pair)
    assert bridge_position is not None and bridge_position.is_open()
    reserve_asset = strategy_universe.get_reserve_asset()
    ts = native_datetime_utc_now()
    _, trade, _ = state.create_trade(
        strategy_cycle_at=ts,
        pair=pair,
        quantity=-quantity,
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        position=bridge_position,
        closing=True,
        slippage_tolerance=0.05,
    )
    execution_model.execute_trades(
        ts,
        state,
        [trade],
        routing_model,
        routing_state,
    )
    assert trade.is_success(), f"Bridge close failed for {pair}: {trade.get_revert_reason()}"


@pytest.fixture()
def strategy_file() -> Path:
    path = Path(__file__).resolve().parent / ".." / ".." / "strategies" / "test_only" / "xchain-master-vault-test.py"
    path = path.resolve()
    assert path.exists(), f"Strategy file not found: {path}"
    return path


@pytest.fixture()
def client() -> Client:
    return Client.create_live_client(TRADING_STRATEGY_API_KEY)


@pytest.fixture()
def anvil_arbitrum() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_ARBITRUM)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_base() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_BASE)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def anvil_hyperevm() -> AnvilLaunch:
    launch = fork_network_anvil(JSON_RPC_HYPERLIQUID, gas_limit=30_000_000)
    try:
        yield launch
    finally:
        launch.close(log_level=logging.ERROR)


@pytest.fixture()
def arb_web3(anvil_arbitrum: AnvilLaunch) -> Web3:
    return create_multi_provider_web3(anvil_arbitrum.json_rpc_url)


@pytest.fixture()
def base_web3(anvil_base: AnvilLaunch) -> Web3:
    return create_multi_provider_web3(anvil_base.json_rpc_url)


@pytest.fixture()
def hyper_web3(anvil_hyperevm: AnvilLaunch) -> Web3:
    return create_multi_provider_web3(anvil_hyperevm.json_rpc_url)


@pytest.fixture()
def web3config(arb_web3: Web3, base_web3: Web3, hyper_web3: Web3) -> Web3Config:
    config = Web3Config()
    config.connections[ChainId.arbitrum] = arb_web3
    config.connections[ChainId.base] = base_web3
    config.connections[ChainId.hyperliquid] = hyper_web3
    config.default_chain_id = ChainId.arbitrum
    return config


@pytest.fixture()
def hot_wallet(arb_web3: Web3, base_web3: Web3, hyper_web3: Web3) -> HotWallet:
    wallet = HotWallet.from_private_key(HOT_WALLET_PRIVATE_KEY)
    set_balance(arb_web3, wallet.address, 100 * 10**18)
    set_balance(base_web3, wallet.address, 100 * 10**18)
    set_balance(hyper_web3, wallet.address, 100 * 10**18)
    wallet.sync_nonce(arb_web3)
    return wallet


@pytest.fixture()
def funded_wallet(hot_wallet: HotWallet, arb_web3: Web3) -> HotWallet:
    fund_erc20_on_anvil(
        arb_web3,
        USDC_NATIVE_TOKEN[ARBITRUM_CHAIN_ID],
        hot_wallet.address,
        int(DEPOSIT_AMOUNT * 10**6),
    )
    return hot_wallet


@pytest.fixture()
def execution_model(arb_web3: Web3, funded_wallet: HotWallet, web3config: Web3Config) -> EthereumExecution:
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
def sync_model(arb_web3: Web3, funded_wallet: HotWallet) -> HotWalletSyncModel:
    return HotWalletSyncModel(arb_web3, funded_wallet)


@pytest.fixture()
def state(sync_model: HotWalletSyncModel, strategy_universe) -> State:
    state = State()
    sync_model.init()
    sync_model.sync_initial(
        state,
        reserve_asset=strategy_universe.get_reserve_asset(),
        reserve_token_price=1.0,
    )
    sync_model.sync_treasury(native_datetime_utc_now(), state, [strategy_universe.get_reserve_asset()])
    return state


@pytest.fixture()
def strategy_universe(strategy_file: Path, client: Client):
    strategy = _load_strategy_module(strategy_file)
    return strategy.create_trading_universe(
        CreateTradingUniverseInput(
            client=client,
            timestamp=native_datetime_utc_now(),
            parameters=None,
            execution_context=ExecutionContext(mode=ExecutionMode.preflight_check),
            execution_model=None,
            universe_options=UniverseOptions(history_period=datetime.timedelta(days=365)),
        )
    )


@pytest.fixture()
def routing_model(arb_web3: Web3, strategy_universe, web3config: Web3Config) -> GenericRouting:
    pair_configurator = EthereumPairConfigurator(
        arb_web3,
        strategy_universe,
        web3config=web3config,
    )
    return GenericRouting(pair_configurator)


@pytest.fixture()
def test_attesters(arb_web3: Web3, base_web3: Web3, hyper_web3: Web3) -> dict[int, str]:
    return {
        ARBITRUM_CHAIN_ID: replace_attester_on_fork(arb_web3),
        BASE_CHAIN_ID: replace_attester_on_fork(base_web3),
        HYPEREVM_CHAIN_ID: replace_attester_on_fork(hyper_web3),
    }


@pytest.mark.timeout(600)
def test_xchain_master_vault_multichain_round_trip(
    anvil_arbitrum: AnvilLaunch,
    anvil_base: AnvilLaunch,
    anvil_hyperevm: AnvilLaunch,
    arb_web3: Web3,
    base_web3: Web3,
    hyper_web3: Web3,
    client: Client,
    execution_model: EthereumExecution,
    funded_wallet: HotWallet,
    mocker,
    routing_model: GenericRouting,
    state: State,
    strategy_file: Path,
    strategy_universe,
    sync_model: HotWalletSyncModel,
    test_attesters: dict[int, str],
    tmp_path: Path,
):
    """Exercise the full forward-bridge vault lifecycle across three chains.

    1. Create the auto-bridge universe and deploy the Lagoon multichain vault.
    2. Open bridge positions and vault positions with ``make_test_trade()``.
    3. Close vaults, sell bridge positions, receive CCTP transfers back on
       Arbitrum, and verify the portfolio is flat.
    """

    # 1. Create the auto-bridge universe and deploy the Lagoon multichain vault.
    vault_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.kind == TradingPairKind.vault]
    bridge_pairs = [pair for pair in strategy_universe.iterate_pairs() if pair.kind == TradingPairKind.cctp_bridge]
    assert len(vault_pairs) == 3, f"Expected 3 vault pairs, got {len(vault_pairs)}"
    assert len(bridge_pairs) == 2, f"Expected 2 bridge pairs, got {len(bridge_pairs)}"
    assert {pair.get_source_chain_id() for pair in bridge_pairs} == {ARBITRUM_CHAIN_ID}
    assert {pair.get_destination_chain_id() for pair in bridge_pairs} == {BASE_CHAIN_ID, HYPEREVM_CHAIN_ID}

    deployer = Account.from_key(DEPLOYER_PRIVATE_KEY)
    for web3 in (arb_web3, base_web3, hyper_web3):
        web3.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])

    vault_record_file = tmp_path / "xchain-master-vault-record.txt"
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "test_xchain_master_vault_multichain",
        "NAME": "test_xchain_master_vault_multichain",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ARBITRUM": anvil_arbitrum.json_rpc_url,
        "JSON_RPC_BASE": anvil_base.json_rpc_url,
        "JSON_RPC_HYPERLIQUID": anvil_hyperevm.json_rpc_url,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "warning",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Xchain master vault test",
        "FUND_SYMBOL": "XMVT",
        "ANY_ASSET": "true",
        "SAFE_SALT_NONCE": "42",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
    }
    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    deploy_record = json.load(vault_record_file.with_suffix(".json").open("rt"))
    assert deploy_record["multichain"] is True
    assert set(deploy_record["deployments"].keys()) == {"arbitrum", "base", "hyperliquid"}

    routing_state = routing_model.create_routing_state(
        strategy_universe,
        {"tx_builder": execution_model.tx_builder},
    )
    web3_by_chain = {
        ARBITRUM_CHAIN_ID: arb_web3,
        BASE_CHAIN_ID: base_web3,
        HYPEREVM_CHAIN_ID: hyper_web3,
    }

    bridge_base_pair = _find_bridge_pair(strategy_universe, BASE_CHAIN_ID)
    bridge_hyper_pair = _find_bridge_pair(strategy_universe, HYPEREVM_CHAIN_ID)
    arb_vault_pair = _find_vault_pair(strategy_universe, ARBITRUM_CHAIN_ID, TEST_VAULTS[ARBITRUM_CHAIN_ID])
    base_vault_pair = _find_vault_pair(strategy_universe, BASE_CHAIN_ID, TEST_VAULTS[BASE_CHAIN_ID])
    hyper_vault_pair = _find_vault_pair(strategy_universe, HYPEREVM_CHAIN_ID, TEST_VAULTS[HYPEREVM_CHAIN_ID])

    initial_equity = _get_total_equity(state)
    assert initial_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.01)

    # 2. Open bridge positions and vault positions with `make_test_trade()`.
    _open_trade(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        strategy_universe=strategy_universe,
        pair=bridge_base_pair,
        amount=BRIDGE_BASE_AMOUNT,
    )
    _receive_cctp_transfer(
        source_chain_id=ARBITRUM_CHAIN_ID,
        destination_chain_id=BASE_CHAIN_ID,
        amount=BRIDGE_BASE_AMOUNT,
        recipient=funded_wallet.address,
        web3_by_chain=web3_by_chain,
        test_attesters=test_attesters,
        nonce=999_001,
    )

    _open_trade(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        strategy_universe=strategy_universe,
        pair=bridge_hyper_pair,
        amount=BRIDGE_HYPER_AMOUNT,
    )
    _receive_cctp_transfer(
        source_chain_id=ARBITRUM_CHAIN_ID,
        destination_chain_id=HYPEREVM_CHAIN_ID,
        amount=BRIDGE_HYPER_AMOUNT,
        recipient=funded_wallet.address,
        web3_by_chain=web3_by_chain,
        test_attesters=test_attesters,
        nonce=999_002,
    )

    _open_trade(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        strategy_universe=strategy_universe,
        pair=arb_vault_pair,
        amount=VAULT_ARB_AMOUNT,
    )
    _open_trade(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        strategy_universe=strategy_universe,
        pair=base_vault_pair,
        amount=VAULT_BASE_AMOUNT,
    )
    _open_trade(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        sync_model=sync_model,
        state=state,
        strategy_universe=strategy_universe,
        pair=hyper_vault_pair,
        amount=VAULT_HYPER_AMOUNT,
    )

    bridge_position_base = state.portfolio.get_position_by_trading_pair(bridge_base_pair)
    bridge_position_hyper = state.portfolio.get_position_by_trading_pair(bridge_hyper_pair)
    assert bridge_position_base is not None and bridge_position_base.bridge_capital_allocated > 0
    assert bridge_position_hyper is not None and bridge_position_hyper.bridge_capital_allocated > 0
    assert state.portfolio.get_position_by_trading_pair(arb_vault_pair) is not None
    assert state.portfolio.get_position_by_trading_pair(base_vault_pair) is not None
    assert state.portfolio.get_position_by_trading_pair(hyper_vault_pair) is not None

    # 3. Close vaults, sell bridge positions, receive CCTP transfers back on Arbitrum, and verify the portfolio is flat.
    _close_vault_position(
        web3=arb_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        strategy_universe=strategy_universe,
        pair=arb_vault_pair,
    )
    _close_vault_position(
        web3=base_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        strategy_universe=strategy_universe,
        pair=base_vault_pair,
    )
    _close_vault_position(
        web3=hyper_web3,
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        strategy_universe=strategy_universe,
        pair=hyper_vault_pair,
    )

    base_usdc = fetch_erc20_details(base_web3, USDC_NATIVE_TOKEN[BASE_CHAIN_ID], chain_id=BASE_CHAIN_ID)
    hyper_usdc = fetch_erc20_details(hyper_web3, USDC_NATIVE_TOKEN[HYPEREVM_CHAIN_ID], chain_id=HYPEREVM_CHAIN_ID)
    base_balance_before_return = base_usdc.fetch_balance_of(funded_wallet.address)
    hyper_balance_before_return = hyper_usdc.fetch_balance_of(funded_wallet.address)
    assert base_balance_before_return > 0
    assert hyper_balance_before_return > 0

    _close_bridge_position(
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        strategy_universe=strategy_universe,
        pair=bridge_base_pair,
        quantity=base_balance_before_return,
    )
    _receive_cctp_transfer(
        source_chain_id=BASE_CHAIN_ID,
        destination_chain_id=ARBITRUM_CHAIN_ID,
        amount=Decimal(base_balance_before_return),
        recipient=funded_wallet.address,
        web3_by_chain=web3_by_chain,
        test_attesters=test_attesters,
        nonce=999_101,
    )

    _close_bridge_position(
        execution_model=execution_model,
        routing_model=routing_model,
        routing_state=routing_state,
        state=state,
        strategy_universe=strategy_universe,
        pair=bridge_hyper_pair,
        quantity=hyper_balance_before_return,
    )
    _receive_cctp_transfer(
        source_chain_id=HYPEREVM_CHAIN_ID,
        destination_chain_id=ARBITRUM_CHAIN_ID,
        amount=Decimal(hyper_balance_before_return),
        recipient=funded_wallet.address,
        web3_by_chain=web3_by_chain,
        test_attesters=test_attesters,
        nonce=999_102,
    )

    sync_model.sync_treasury(native_datetime_utc_now(), state, [strategy_universe.get_reserve_asset()])

    final_equity = _get_total_equity(state)
    assert final_equity == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.05)
    assert len(state.portfolio.open_positions) == 0

    chain_equity = state.portfolio.calculate_total_equity_chain()
    assert chain_equity.get(ChainId.base, 0) == pytest.approx(0, abs=0.05)
    assert chain_equity.get(ChainId.hyperliquid, 0) == pytest.approx(0, abs=0.05)
    assert chain_equity.get(ChainId.arbitrum, 0) == pytest.approx(float(DEPOSIT_AMOUNT), rel=0.05)
