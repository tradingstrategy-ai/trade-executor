"""Command line application initialisation helpers."""
import datetime
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple

from eth_defi.enzyme.vault import Vault
from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from web3 import Web3

from tradeexecutor.backtest.backtest_execution import BacktestExecutionModel
from tradeexecutor.backtest.backtest_pricing import backtest_pricing_factory
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.backtest_valuation import backtest_valuation_factory
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2SimpleRoutingModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3ExecutionModel
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import uniswap_v3_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import uniswap_v3_sell_valuation_factory
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.metadata import Metadata, OnChainData
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore, StateStore
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation
from tradeexecutor.strategy.sync_model import SyncModel, DummySyncModel
from tradeexecutor.strategy.valuation import ValuationModelFactory
from tradeexecutor.testing.dummy_wallet import DummyWalletSyncer
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalType, ApprovalModel
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_model import AssetManagementMode, ExecutionModel
from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient, Client
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient

logger = logging.getLogger(__name__)


def validate_executor_id(id: str):
    """Check that given executor id is good.

    No spaces.

    - Will be used in filenames

    - Will be used in URLs

    :raise AssertionError:
        If the user gives us non-id like id
    """

    assert id, f"EXECUTOR_ID must be given so that executor instances can be identified"
    assert " " not in id, f"Bad EXECUTOR_ID: {id}"


def create_web3_config(
    json_rpc_binance,
    json_rpc_polygon,
    json_rpc_avalanche,
    json_rpc_ethereum,
    json_rpc_arbitrum,
    json_rpc_anvil,
    gas_price_method: Optional[GasPriceMethod]=None,
) -> Web3Config:
    """Create Web3 connection to the live node we are executing against.

    :return web3:
        Connect to any passed JSON RPC URL

    """
    web3config = Web3Config.setup_from_environment(
        gas_price_method,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_anvil=json_rpc_anvil,
    )
    return web3config


def create_execution_model(
    routing_hint: Optional[TradeRouting],
    tx_builder: Optional[TransactionBuilder],
    confirmation_timeout: datetime.timedelta,
    confirmation_block_count: int,
    max_slippage: float,
    min_gas_balance: Optional[Decimal],
    mainnet_fork=False,
):
    """Set up the code transaction building logic.

    Choose between Uniswap v2 and v3 trade routing.
    """

    # TODO: user_supplied_routing_model can be uni v3 as well
    if routing_hint is None or routing_hint.is_uniswap_v2() or routing_hint == TradeRouting.user_supplied_routing_model:
        logger.info("Uniswap v2 like exchange. Routing hint is %s", routing_hint)
        execution_model = UniswapV2ExecutionModel(
            tx_builder,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_gas_balance,
            mainnet_fork=mainnet_fork,
        )
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory
    elif routing_hint.is_uniswap_v3():
        logger.info("Uniswap v3 like exchange. Routing hint is %s", routing_hint)
        execution_model = UniswapV3ExecutionModel(
            tx_builder,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_gas_balance,
            mainnet_fork=mainnet_fork,
        )
        valuation_model_factory = uniswap_v3_sell_valuation_factory
        pricing_model_factory = uniswap_v3_live_pricing_factory
    else:
        raise RuntimeError(f"Does not know how to route: {routing_hint}")

    return execution_model, valuation_model_factory, pricing_model_factory


def create_execution_and_sync_model(
        asset_management_mode: AssetManagementMode,
        private_key: str,
        web3config: Web3Config,
        confirmation_timeout: datetime.timedelta,
        confirmation_block_count: int,
        max_slippage: float,
        min_gas_balance: Optional[Decimal],
        vault_address: Optional[str],
        vault_adapter_address: Optional[str],
        vault_payment_forwarder_address: Optional[str],
        routing_hint: Optional[TradeRouting] = None,
) -> Tuple[ExecutionModel, SyncModel, ValuationModelFactory, PricingModelFactory]:
    """Set up the wallet sync and execution mode for the command line client."""

    assert isinstance(confirmation_timeout, datetime.timedelta), f"Got {confirmation_timeout}"

    if asset_management_mode == AssetManagementMode.dummy:
        # Used in test_strategy_cycle_trigger.py
        web3 = web3config.get_default()
        execution_model = DummyExecutionModel(web3)
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory
        sync_model = DummySyncModel()
        return execution_model, sync_model, valuation_model_factory, pricing_model_factory
    elif asset_management_mode in (AssetManagementMode.hot_wallet, AssetManagementMode.enzyme):
        assert private_key, "Private key is needed for live trading"
        web3 = web3config.get_default()
        hot_wallet = HotWallet.from_private_key(private_key)
        sync_model = create_sync_model(
            asset_management_mode,
            web3,
            hot_wallet,
            vault_address,
            vault_adapter_address,
            vault_payment_forwarder_address,
        )

        logger.info("Creating execution model. Asset management mode is %s, routing hint is %s", asset_management_mode.value, routing_hint.value)

        execution_model, valuation_model_factory, pricing_model_factory = create_execution_model(
            routing_hint=routing_hint,
            tx_builder=sync_model.create_transaction_builder(),
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_gas_balance=min_gas_balance,
            mainnet_fork=web3config.is_mainnet_fork(),
        )
        return execution_model, sync_model, valuation_model_factory, pricing_model_factory

    elif asset_management_mode == AssetManagementMode.backtest:
        logger.info("TODO: Command line backtests are always executed with initial deposit of $10,000")
        wallet = SimulatedWallet()
        execution_model = BacktestExecutionModel(wallet, max_slippage=0.01, stop_loss_data_available=True)
        sync_model = BacktestSyncModel(wallet, Decimal(10_000))
        pricing_model_factory = backtest_pricing_factory
        valuation_model_factory = backtest_valuation_factory
        return execution_model, sync_model, valuation_model_factory, pricing_model_factory
    else:
        raise NotImplementedError(f"Unsupported asset management mode: {asset_management_mode} - did you pass ASSET_MANAGEMENT_MODE environment variable?")


def create_approval_model(approval_type: ApprovalType) -> ApprovalModel:
    if approval_type == ApprovalType.unchecked:
        return UncheckedApprovalModel()
    elif approval_type == ApprovalType.cli:
        return CLIApprovalModel()
    else:
        raise NotImplementedError()


def create_state_store(state_file: Path) -> StateStore:
    store = JSONFileStore(state_file)
    return store


def prepare_cache(executor_id: str, cache_path: Optional[Path]) -> Path:
    """Fail early if the cache path is not writable.

    Otherwise Docker might spit misleading "Device or resource busy" message.
    """

    assert executor_id

    if not cache_path:
        cache_path = Path("cache").joinpath(executor_id)

    logger.info("Dataset cache is %s", os.path.realpath(cache_path))

    os.makedirs(cache_path, exist_ok=True)

    with open(cache_path.joinpath("cache.pid"), "wt") as out:
        print(os.getpid(), file=out)

    return cache_path


def create_metadata(
        name,
        short_description,
        long_description,
        icon_url,
        asset_management_mode: AssetManagementMode,
        chain_id: ChainId,
        vault: Optional[Vault],
        backtest_file: Optional[Path]=None,
) -> Metadata:
    """Create metadata object from the configuration variables."""

    on_chain_data = OnChainData(asset_management_mode=asset_management_mode, chain_id=chain_id)

    if vault:
        on_chain_data.smart_contracts.update(vault.deployment.contracts.get_all_addresses())

        on_chain_data.smart_contracts.update({
            "vault": vault.vault.address,
            "comptroller": vault.comptroller.address,
            "generic_adapter": vault.generic_adapter.address,
            "payment_forwarder": vault.payment_forwarder.address if vault.payment_forwarder else None,
        })

        if vault.deployment.contracts.fund_value_calculator is None:
            # Hot fix for Polygon
            # TODO: Fix properly
           on_chain_data.smart_contracts.update({
                "fund_value_calculator": "0xcdf038Dd3b66506d2e5378aee185b2f0084B7A33",
            })

    if backtest_file is not None:
        backtest_state = State.read_json_file(backtest_file)
    else:
        backtest_state = None

    metadata = Metadata(
        name,
        short_description,
        long_description,
        icon_url,
        datetime.datetime.utcnow(),
        executor_running=True,
        on_chain_data=on_chain_data,
        backtest_state=backtest_state,
    )

    return metadata


def prepare_executor_id(id: Optional[str], strategy_file: Path) -> str:
    """Autodetect exeuctor id."""

    if id:
        # Explicitly passed
        pass
    else:
        # Guess id from the strategy file
        if strategy_file:
            id = Path(strategy_file).stem
            pass
        else:
            raise RuntimeError("EXECUTOR_ID or STRATEGY_FILE must be given")

    validate_executor_id(id)

    return id


def monkey_patch():
    """Apply all monkey patches."""
    patch_dataclasses_json()


def create_sync_model(
        asset_management_mode: AssetManagementMode,
        web3: Web3,
        hot_wallet: Optional[HotWallet],
        vault_address: Optional[str],
        vault_adapter_address: Optional[str] = None,
        vault_payment_forwarder_address: Optional[str] = None,
) -> SyncModel:
    match asset_management_mode:
        case AssetManagementMode.hot_wallet:
            return HotWalletSyncModel(web3, hot_wallet)
        case AssetManagementMode.enzyme:
            reorg_mon = create_reorganisation_monitor(web3)
            return EnzymeVaultSyncModel(
                web3,
                vault_address,
                reorg_mon,
                only_chain_listener=True,
                hot_wallet=hot_wallet,
                generic_adapter_address=vault_adapter_address,
                vault_payment_forwarder_address=vault_payment_forwarder_address,
                scan_chunk_size=50_000,
            )
        case _:
            raise NotImplementedError()


def create_client(
    mod: StrategyModuleInformation,
    web3config: Web3Config,
    trading_strategy_api_key: Optional[str],
    cache_path: Optional[Path],
    test_evm_uniswap_v2_factory: Optional[str],
    test_evm_uniswap_v2_router: Optional[str],
    test_evm_uniswap_v2_init_code_hash: Optional[str],
    clear_caches: bool,
) -> Tuple[BaseClient | None, RoutingModel | None]:
    """Create a Trading Strategy client instance.

    - Read env inputs to determine which kind of enviroment/client we need to have

    - May create mock client if we run e2e tests

    - Otherwise create a real client

    :return:
        Client, routing model tuple.

        Routing  model is only returned if a test EVM routing set up is used (smart contracts
        deployed in the test).
    """

    client = None
    routing_model = None

    # Create our data client
    if test_evm_uniswap_v2_factory:
        # Running against a local dev chain
        client = UniswapV2MockClient(
            web3config.get_default(),
            test_evm_uniswap_v2_factory,
            test_evm_uniswap_v2_router,
            test_evm_uniswap_v2_init_code_hash,
        )

        if mod.trade_routing == TradeRouting.user_supplied_routing_model:
            routing_model = UniswapV2SimpleRoutingModel(
                factory_router_map={
                    test_evm_uniswap_v2_factory: (test_evm_uniswap_v2_router, test_evm_uniswap_v2_init_code_hash)},
                allowed_intermediary_pairs={},
                reserve_token_address=client.get_default_quote_token_address(),
            )

    elif trading_strategy_api_key:
        # Backtest / real trading
        client = Client.create_live_client(trading_strategy_api_key, cache_path=cache_path)
        if clear_caches:
            client.clear_caches()
    else:
        # This run does not need to download any data
        pass

    return client, routing_model

