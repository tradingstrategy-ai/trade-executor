"""Command line application initialisation helpers."""
import datetime
import logging
import os
import shutil
from decimal import Decimal
from pathlib import Path
from typing import Optional, Tuple, Set, cast

import pandas as pd
from eth_typing import HexAddress

from eth_defi.compat import native_datetime_utc_now
from eth_defi.enzyme.vault import Vault
from eth_defi.event_reader.reorganisation_monitor import create_reorganisation_monitor
from eth_defi.gas import GasPriceMethod
from eth_defi.hotwallet import HotWallet
from web3 import Web3

from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.vault.base import VaultSpec
from eth_defi.velvet import VelvetVault
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.velvet.velvet_enso_routing import VelvetEnsoRouting

from tradingstrategy.chain import ChainId
from tradingstrategy.client import BaseClient, Client
from tradingstrategy.testing.uniswap_v2_mock_client import UniswapV2MockClient


from tradeexecutor.backtest.backtest_execution import BacktestExecution
from tradeexecutor.backtest.backtest_pricing import backtest_pricing_factory
from tradeexecutor.backtest.backtest_sync import BacktestSyncModel
from tradeexecutor.backtest.backtest_valuation import backtest_valuation_factory
from tradeexecutor.backtest.simulated_wallet import SimulatedWallet
from tradeexecutor.cli.approval import CLIApprovalModel
from tradeexecutor.ethereum.address_sync_model import AddressSyncModel
from tradeexecutor.ethereum.enzyme.vault import EnzymeVaultSyncModel
from tradeexecutor.ethereum.hot_wallet_sync_model import HotWalletSyncModel
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.ethereum.one_delta.one_delta_execution import OneDeltaExecution
from tradeexecutor.ethereum.one_delta.one_delta_live_pricing import one_delta_live_pricing_factory
from tradeexecutor.ethereum.one_delta.one_delta_valuation import one_delta_valuation_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2Execution
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_routing import UniswapV2Routing
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_execution import UniswapV3Execution
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_live_pricing import uniswap_v3_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v3.uniswap_v3_valuation import uniswap_v3_sell_valuation_factory
from tradeexecutor.ethereum.velvet.execution import VelvetExecution
from tradeexecutor.ethereum.velvet.vault import VelvetVaultSyncModel
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.metadata import Metadata, OnChainData
from tradeexecutor.state.state import State
from tradeexecutor.state.store import JSONFileStore, SimulateStore
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.generic.generic_pricing_model import  EthereumGenericPricingFactory
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import  GenericValuationModelFactory
from tradeexecutor.strategy.pricing_model import PricingModelFactory
from tradeexecutor.strategy.routing import RoutingModel
from tradeexecutor.strategy.strategy_module import StrategyModuleInformation
from tradeexecutor.strategy.tag import StrategyTag
from tradeexecutor.strategy.sync_model import SyncModel, DummySyncModel
from tradeexecutor.strategy.valuation import ValuationModelFactory
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalType, ApprovalModel
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_model import AssetManagementMode, ExecutionModel
from eth_defi.compat import native_datetime_utc_now

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
    json_rpc_base,
    json_rpc_anvil,
    json_rpc_derive=None,
    json_rpc_arbitrum_sepolia=None,
    json_rpc_base_sepolia=None,
    json_rpc_hyperliquid=None,
    json_rpc_hyperliquid_testnet=None,
    json_rpc_monad=None,
    gas_price_method: Optional[GasPriceMethod] = None,
    unit_testing: bool=False,
    simulate: bool=False,
    mev_endpoint_disabled: bool=False,
    simulate_http_timeout: tuple[float, float] | None = None,
) -> Web3Config:
    """Create Web3 connection to the live node we are executing against.

    :param simulate:
        Set up a mainnet fork with Anvil for transaction simulation.

    :param mev_endpoint_disabled:
        MEV endpoints do not work when deploying contracts with Forge.

    :return web3:
        Connect to any passed JSON RPC URL

    Environment variables
    ---------------------

    ``MAINNET_FORK``
        Set to ``true`` to signal mainnet fork mode when Anvil forks
        are created externally (outside the CLI).  This sets
        ``web3config.mainnet_fork_simulation = True`` without launching
        a new Anvil instance, enabling simulate-mode behaviour in
        routing models (e.g. batched multicall in
        :py:class:`~tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting`).
    """
    web3config = Web3Config.setup_from_environment(
        gas_price_method,
        json_rpc_ethereum=json_rpc_ethereum,
        json_rpc_base=json_rpc_base,
        json_rpc_binance=json_rpc_binance,
        json_rpc_polygon=json_rpc_polygon,
        json_rpc_avalanche=json_rpc_avalanche,
        json_rpc_arbitrum=json_rpc_arbitrum,
        json_rpc_anvil=json_rpc_anvil,
        json_rpc_derive=json_rpc_derive,
        json_rpc_arbitrum_sepolia=json_rpc_arbitrum_sepolia,
        json_rpc_base_sepolia=json_rpc_base_sepolia,
        json_rpc_hyperliquid=json_rpc_hyperliquid,
        json_rpc_hyperliquid_testnet=json_rpc_hyperliquid_testnet,
        json_rpc_monad=json_rpc_monad,
        unit_testing=unit_testing,
        simulate=simulate,
        mev_endpoint_disabled=mev_endpoint_disabled,
        simulate_http_timeout=simulate_http_timeout,
    )

    # Allow externally-created Anvil forks to signal mainnet fork mode
    # without launching a new Anvil instance.  Used when the test harness
    # forks mainnets itself and passes the Anvil RPC URLs as JSON_RPC_xxx.
    if os.environ.get("MAINNET_FORK", "").lower() in ("true", "1", "yes"):
        web3config.mainnet_fork_simulation = True

    return web3config


def create_execution_model(
    asset_management_mode: AssetManagementMode,
    routing_hint: Optional[TradeRouting],
    tx_builder: Optional[TransactionBuilder],
    confirmation_timeout: datetime.timedelta,
    confirmation_block_count: int,
    max_slippage: float,
    sync_model: SyncModel,
    min_gas_balance: Optional[Decimal],
    mainnet_fork=False,
):
    """Set up the code transaction building logic.

    Choose between Uniswap v2 and v3 trade routing.
    """

    logger.info("create_execution_model(): confirmation_timeout: %s", confirmation_timeout)
    # TODO: user_supplied_routing_model can be uni v3 as well
    if asset_management_mode == AssetManagementMode.velvet:
        # Velvet vaults are hardcoded to use Enzo
        assert isinstance(sync_model, VelvetVaultSyncModel)
        logger.info("Velvet + Enso execution model")
        execution_model = VelvetExecution(
            vault=sync_model.vault,
            tx_builder=tx_builder,
            confirmation_timeout = confirmation_timeout,
            confirmation_block_count = confirmation_block_count,
            max_slippage = max_slippage,
            min_balance_threshold = min_gas_balance,
            mainnet_fork = mainnet_fork,
        )
        valuation_model_factory = GenericValuationModelFactory()
        pricing_model_factory = EthereumGenericPricingFactory(sync_model.web3)
    elif asset_management_mode == AssetManagementMode.lagoon:
        # Velvet vaults are hardcoded to use Enzo
        assert isinstance(sync_model, LagoonVaultSyncModel)
        logger.info("Lagoon execution model for vault: %s", sync_model.vault)
        execution_model = LagoonExecution(
            vault=sync_model.vault,
            tx_builder=tx_builder,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_gas_balance,
            mainnet_fork=mainnet_fork,
        )
        valuation_model_factory = GenericValuationModelFactory()
        pricing_model_factory = EthereumGenericPricingFactory(sync_model.web3)
    elif routing_hint is None or routing_hint.is_uniswap_v2() or routing_hint == TradeRouting.user_supplied_routing_model:
        logger.info("Uniswap v2 like exchange. Routing hint is %s", routing_hint)
        execution_model = UniswapV2Execution(
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
        execution_model = UniswapV3Execution(
            tx_builder,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_gas_balance,
            mainnet_fork=mainnet_fork,
        )
        valuation_model_factory = uniswap_v3_sell_valuation_factory
        pricing_model_factory = uniswap_v3_live_pricing_factory
    elif routing_hint.is_one_delta():
        logger.info("1delta routing. Routing hint is %s", routing_hint)
        execution_model = OneDeltaExecution(
            tx_builder,
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_gas_balance,
            mainnet_fork=mainnet_fork,
        )
        valuation_model_factory = one_delta_valuation_factory
        pricing_model_factory = one_delta_live_pricing_factory
    elif routing_hint == TradeRouting.ignore:
        # For strategies that don't do any on-chain trading (e.g., exchange account monitoring)
        # Use a dummy execution model with basic valuation
        logger.info("Ignoring routing - no on-chain trading expected")
        execution_model = DummyExecutionModel(sync_model.web3)
        valuation_model_factory = GenericValuationModelFactory()
        pricing_model_factory = EthereumGenericPricingFactory(sync_model.web3)
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
    unit_testing: bool = False,
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
    elif asset_management_mode.is_live_trading():
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
            unit_testing=unit_testing,
        )

        # Pass web3config for multichain balance queries (e.g. CCTP bridge)
        if isinstance(sync_model, AddressSyncModel):
            sync_model.web3config = web3config

        logger.info(
            "Creating execution model. Asset management mode is %s, routing hint is %s, confirmation timeout %s",
            asset_management_mode.value,
            routing_hint,
            confirmation_timeout,
        )

        execution_model, valuation_model_factory, pricing_model_factory = create_execution_model(
            asset_management_mode=asset_management_mode,
            routing_hint=routing_hint,
            tx_builder=sync_model.create_transaction_builder(),
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_gas_balance=min_gas_balance,
            mainnet_fork=web3config.is_mainnet_fork(),
            sync_model=sync_model,
        )

        # Pass web3config for multichain execution (e.g. CCTP bridge)
        execution_model.web3config = web3config

        # Populate satellite vaults for multichain Lagoon deployments
        if isinstance(execution_model, LagoonExecution):
            satellite_modules_json = os.environ.get("SATELLITE_MODULES")
            if satellite_modules_json:
                import json as _json
                from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonSatelliteVault
                satellite_modules = _json.loads(satellite_modules_json)
                # The Safe address is shared across all chains (CREATE2)
                safe_address = sync_model.vault.safe_address
                satellite_vaults = {}
                for chain_slug, module_address in satellite_modules.items():
                    sat_chain_id = ChainId[chain_slug]
                    sat_web3 = web3config.get_connection(sat_chain_id)
                    assert sat_web3, f"No Web3 connection for satellite chain {chain_slug}"
                    sat_vault = LagoonSatelliteVault(
                        sat_web3,
                        safe_address=sat_web3.to_checksum_address(safe_address),
                        trading_strategy_module_address=sat_web3.to_checksum_address(module_address),
                    )
                    satellite_vaults[sat_chain_id.value] = sat_vault
                    logger.info("Satellite vault on %s: safe=%s, module=%s", chain_slug, safe_address, module_address)
                execution_model.satellite_vaults = satellite_vaults

        return execution_model, sync_model, valuation_model_factory, pricing_model_factory

    elif asset_management_mode == AssetManagementMode.backtest:
        logger.info("TODO: Command line backtests are always executed with initial deposit of $10,000")
        wallet = SimulatedWallet()
        execution_model = BacktestExecution(wallet, max_slippage=0.01, stop_loss_data_available=True)
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


def create_state_store(state_file: Path, simulate: bool = False) -> JSONFileStore | SimulateStore:
    if simulate:
        return SimulateStore(state_file)
    return JSONFileStore(state_file)


def prepare_cache(
    executor_id: str,
    cache_path: Optional[Path],
    unit_testing=False,
) -> Path:
    """Prepare a cache location for this trade-executor.
    
    - Fail early if the cache path is not writable.
      Otherwise Docker might spit misleading "Device or resource busy" message.
    
    - Default is `./cache/$EXECUTOR_ID`
    
    - Unit tests share a fixed cache path across tests to speed
      up them
    """

    assert executor_id

    if not cache_path:
        # Avoid polluting user caches during test runs,
        # so we use different default
        generated = True
        if unit_testing:
            cache_path = Path("/tmp/trading-strategy-tests")
        else:
            # Default is ./cache/$EXECUTOR_ID
            cache_path = Path("cache").joinpath(executor_id)
    else:
        generated = False

    logger.info("Dataset cache is %s, was generated from the executor id %s",
                os.path.realpath(cache_path),
                generated)

    os.makedirs(cache_path, exist_ok=True)

    # Check that the cache is writable
    with open(cache_path.joinpath("cache.pid"), "wt") as out:
        print(os.getpid(), file=out)

    return cache_path


def prepare_token_cache(
    cache_path: Path,
    unit_testing: bool = False,
) -> "TokenDiskCache":
    """Prepare token cache for this trade-executor.

    - Token cache stores ERC-20 token metadata (symbol, decimals, name) to avoid
      repeated RPC calls when working with vaults and token contracts

    - Uses SQLite for persistent storage across executor runs

    - Stored as `{cache_path}/eth-defi-tokens.sqlite`

    :param cache_path:
        Cache directory path, already prepared by prepare_cache()

    :param unit_testing:
        Pass through to get_default_token_cache() to ensure consistent
        cache location with tradingstrategy.Client in test fixtures

    :return:
        Initialised token disk cache instance
    """

    from tradeexecutor.ethereum.token_cache import get_default_token_cache

    assert cache_path, "cache_path must be provided"
    assert cache_path.exists(), f"Cache path {cache_path} does not exist - call prepare_cache() first"

    token_cache = get_default_token_cache(cache_path, unit_testing=unit_testing)

    logger.info(
        "Token cache initialised at %s with %d entries",
        token_cache.filename,
        len(token_cache),
    )

    return token_cache


def create_metadata(
    name,
    short_description,
    long_description,
    icon_url,
    asset_management_mode: AssetManagementMode,
    chain_id: ChainId,
    vault: Optional[Vault | VelvetVault | LagoonVault],
    backtest_result: Optional[Path] = None,
    backtest_notebook: Optional[Path] = None,
    backtest_html: Optional[Path] = None,
    key_metrics_backtest_cut_off_days: float = 90,
    badges: Optional[str] = None,
    tags: Optional[Set[StrategyTag]] = None,
    hot_wallet: Optional[HotWallet] = None,
    sort_priority=0,
    fees=None,
) -> Metadata:
    """Create metadata object from the configuration variables.

    - This metadata is exposed to frontend to be displayed on the website

    """

    on_chain_data = OnChainData(
        asset_management_mode=asset_management_mode,
        chain_id=chain_id,
        trade_executor_hot_wallet=hot_wallet.address if hot_wallet else None,
    )

    if vault:

        assert asset_management_mode.is_vault()

        match asset_management_mode:
            case AssetManagementMode.enzyme:
                assert isinstance(vault, Vault)
                on_chain_data.owner = vault.get_owner()
                on_chain_data.smart_contracts.update(vault.deployment.contracts.get_all_addresses())
                on_chain_data.smart_contracts.update({
                    "vault": vault.vault.address,
                    "comptroller": vault.comptroller.address,
                    "generic_adapter": vault.generic_adapter.address,
                    "payment_forwarder": vault.payment_forwarder.address if vault.payment_forwarder else None,
                    "terms_of_service": vault.terms_of_service_contract.address if vault.terms_of_service_contract else None,
                    "guard": vault.guard_contract.address if vault.guard_contract else None,
                })

                if vault.deployment.contracts.fund_value_calculator is None:
                    # Hot fix for Polygon
                    # TODO: Fix properly, legacy
                    match vault.web3.eth.chain_id:
                        case 137 | 31337:
                            on_chain_data.smart_contracts.update({
                                "fund_value_calculator": "0xcdf038Dd3b66506d2e5378aee185b2f0084B7A33",
                            })
                        case 1:
                            on_chain_data.smart_contracts.update({
                                "fund_value_calculator": "0x490e64E0690b4aa481Fb02255aED3d052Bad7BF1",
                            })
                        case 42161:
                            on_chain_data.smart_contracts.update({
                                "fund_value_calculator": "0xea609eeb38d1ee8e8719597d47cc9276df9f8707",
                            })
                        case _:
                            raise NotImplementedError(f"Chain {vault.web3.eth.chain_id}")
            case AssetManagementMode.velvet:
                assert isinstance(vault, VelvetVault)
                on_chain_data.smart_contracts.update(vault.info)
                on_chain_data.owner = vault.owner_address
            case AssetManagementMode.lagoon:
                assert isinstance(vault, LagoonVault)
                on_chain_data.smart_contracts.update(vault.info)
                on_chain_data.owner = vault.safe_address
            case _:
                raise NotImplementedError(f"Unsupported asset management mode: {asset_management_mode}")

    if backtest_result is not None:
        backtested_state = State.read_json_file(backtest_result)
    else:
        backtested_state = None

    metadata = Metadata(
        name,
        short_description,
        long_description,
        icon_url,
        native_datetime_utc_now(),
        executor_running=True,
        on_chain_data=on_chain_data,
        backtested_state=backtested_state,
        backtest_notebook=backtest_notebook,
        backtest_html=backtest_html,
        key_metrics_backtest_cut_off=datetime.timedelta(days=key_metrics_backtest_cut_off_days),
        badges=Metadata.parse_badges_configuration(badges),
        tags=tags or set(),  # Always fill empty set
        sort_priority=sort_priority,
        fees=fees,
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
    hot_wallet: HotWallet,
    vault_address: Optional[str] = None,
    vault_adapter_address: Optional[str] = None,
    vault_payment_forwarder_address: Optional[str] = None,
    init=False,
    unit_testing=False,
) -> SyncModel:
    """Create sync model for wallet type

    :param init:
        This is for state init only - no chain functions called
    """
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
        case AssetManagementMode.velvet:
            vault = VelvetVault(
                web3,
                VaultSpec(web3.eth.chain_id, cast(HexAddress, vault_address)),
            )
            vault.check_valid_contract()
            return VelvetVaultSyncModel(
                vault,
                hot_wallet
            )
        case AssetManagementMode.lagoon:
            assert vault_adapter_address, "TradingStrategyModuleV0 address must be given with Lagoon vault using VAULT_ADAPTER_ACCESS env option"
            vault = LagoonVault(
                web3,
                VaultSpec(web3.eth.chain_id, cast(HexAddress, vault_address)),
                trading_strategy_module_address=vault_adapter_address,
                default_block_identifier="latest",
            )
            return LagoonVaultSyncModel(
                vault,
                hot_wallet,
                unit_testing=unit_testing,
                valuation_data_freshness=datetime.timedelta(days=2) if unit_testing else datetime.timedelta(hours=4),
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
    asset_management_mode: AssetManagementMode | None = None,
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

        client.initialise_mock_data()

        # For test strategies, running against a locally deployed Uniswap v2
        if mod.trade_routing == TradeRouting.user_supplied_routing_model:
            routing_model = UniswapV2Routing(
                factory_router_map={
                    test_evm_uniswap_v2_factory: (test_evm_uniswap_v2_router, test_evm_uniswap_v2_init_code_hash)},
                allowed_intermediary_pairs={},
                reserve_token_address=client.get_default_quote_token_address(),
            )

    elif trading_strategy_api_key:
        # Backtest / real trading
        client = Client.create_live_client(
            trading_strategy_api_key,
            cache_path=cache_path,
            settings_path=None,  # No interactive settings file with live execution
        )
        if clear_caches:
            client.clear_caches()
    else:
        # This run does not need to download any data
        pass

    # TODO: Move this to its own function
    if mod.trade_routing == TradeRouting.default:

        if asset_management_mode == AssetManagementMode.velvet:
            # Will be created later?
            routing_model = None
        else:
            # We eill call GenericRouting.initialise() later with pair universe data loaded
            logger.info("Using GenericRouting for Ethereum chains")
            routing_model = GenericRouting(pair_configurator=None)

    return client, routing_model


def backup_state(state_file: Path | str, backup_suffix="backup", unit_testing=False) -> Tuple[JSONFileStore, State]:
    """Take a copy of the state file, then read the original file."""

    logger.info("Backing up %s", state_file)

    state_file = Path(state_file)
    store = create_state_store(state_file)
    assert not store.is_pristine(), f"State does not exists yet: {state_file}"

    # Make a backup
    # https://stackoverflow.com/a/47528275/315168
    backup_file = None
    backup_attempts = 1 if unit_testing else 99  # Don't pollute folders when unit testing
    for i in range(1, backup_attempts):  # Try 99 different iterateive backup filenames
        backup_file = state_file.with_suffix(f".{backup_suffix}-{i}.json")
        if os.path.exists(backup_file):
            continue

        shutil.copy(state_file, backup_file)
        break
    else:
        if not unit_testing:
            raise RuntimeError(f"Could not create backup {backup_file}")

    logger.info("Old state backed up as %s", backup_file)

    state = store.load()
    return store, state


def configure_default_chain(
    web3config: Web3Config,
    mod: StrategyModuleInformation,
):
    if web3config is not None:

        if isinstance(mod, StrategyModuleInformation):
            # This path is not enabled for legacy strategy modules
            if mod.get_default_chain_id():
                # Strategy tells what chain to use
                web3config.set_default_chain(mod.get_default_chain_id())
                web3config.check_default_chain_id()
            else:
                # User has configured only one chain, use it
                web3config.choose_single_chain()

        else:
            # Legacy unit testing path.
            # All chain_ids are 56 (BNB Chain)
            logger.warning("Legacy strategy module: makes assumption of BNB Chain")
            web3config.set_default_chain(ChainId.bsc)


def check_universe_chains_have_rpc(
    web3config: "Web3Config",
    universe: "TradingStrategyUniverse",
):
    """Validate that all chains in the trading universe have RPC connections configured.

    Call after create_trading_universe() to catch missing JSON-RPC
    configurations early, rather than failing during trade execution.

    :param web3config:
        Web3 configuration with RPC connections.

    :param universe:
        Trading universe with chains to check.
    """
    if web3config is None:
        return

    from tradeexecutor.ethereum.web3config import TEST_CHAIN_IDS

    configured_chains = set(web3config.connections.keys())

    # Skip check when running on test chains (e.g. Anvil forks of mainnet)
    # because the fork simulates a different chain than its chain ID suggests
    if configured_chains & set(TEST_CHAIN_IDS):
        return

    universe_chains = universe.data_universe.chains

    missing = set(universe_chains) - configured_chains
    if missing:
        missing_names = ", ".join(c.get_name() for c in sorted(missing, key=lambda c: c.value))
        configured_names = ", ".join(c.get_name() for c in sorted(configured_chains, key=lambda c: c.value))
        raise RuntimeError(
            f"Strategy universe uses chains that do not have JSON-RPC connections configured.\n"
            f"Missing RPCs for: {missing_names}\n"
            f"Configured RPCs for: {configured_names}\n"
            f"Set the corresponding JSON_RPC_* environment variables."
        )


def check_universe_chains_have_gas(
    web3config: "Web3Config",
    universe: "TradingStrategyUniverse",
    wallet_address: str,
    min_gas_balance: float = 0,
):
    """Check that the hot wallet has native token for gas on all universe chains.

    :param web3config:
        Web3 configuration with RPC connections.

    :param universe:
        Trading universe with chains to check.

    :param wallet_address:
        The hot wallet address (same across all chains).

    :param min_gas_balance:
        Minimum native token balance required per chain.
    """
    if web3config is None:
        return

    if min_gas_balance <= 0:
        return

    universe_chains = universe.data_universe.chains
    low_balance_chains = []

    for chain_id in universe_chains:
        try:
            web3 = web3config.get_connection(chain_id)
        except KeyError:
            continue  # Already handled by check_universe_chains_have_rpc

        if web3 is None:
            continue

        raw_balance = web3.eth.get_balance(wallet_address)
        balance = Decimal(raw_balance) / Decimal(10**18)

        if balance < min_gas_balance:
            low_balance_chains.append((chain_id, balance))

    if low_balance_chains:
        details = "\n".join(
            f"  {c.get_name()}: {balance:.8f} native tokens"
            for c, balance in low_balance_chains
        )
        raise RuntimeError(
            f"Hot wallet {wallet_address} does not have enough gas on all universe chains.\n"
            f"Minimum required: {min_gas_balance} native tokens per chain.\n"
            f"Low balance chains:\n{details}\n"
            f"Top up the wallet or adjust MIN_GAS_BALANCE environment variable."
        )
