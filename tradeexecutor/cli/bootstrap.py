"""Command line application initialisation helpers."""
import datetime
import logging
import os
from decimal import Decimal
from pathlib import Path
from typing import Optional

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
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_execution import UniswapV2ExecutionModel
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_live_pricing import uniswap_v2_live_pricing_factory
from tradeexecutor.ethereum.uniswap_v2.uniswap_v2_valuation import uniswap_v2_sell_valuation_factory
from tradeexecutor.ethereum.web3config import Web3Config
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.metadata import Metadata
from tradeexecutor.state.store import JSONFileStore, StateStore
from tradeexecutor.strategy.sync_model import SyncModel, DummySyncModel
from tradeexecutor.testing.dummy_wallet import DummyWalletSyncer
from tradeexecutor.strategy.approval import UncheckedApprovalModel, ApprovalType, ApprovalModel
from tradeexecutor.strategy.dummy import DummyExecutionModel
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradingstrategy.chain import ChainId

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
) -> Optional[Web3Config]:
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


def create_trade_execution_model(
        asset_management_mode: AssetManagementMode,
        private_key: str,
        web3config: Web3Config,
        confirmation_timeout: datetime.timedelta,
        confirmation_block_count: int,
        max_slippage: float,
        min_balance_threshold: Optional[Decimal],
        vault_address: Optional[str],
        vault_adapter_address: Optional[str],
):
    """Set up the execution mode for the command line client.

    :param max_slippage:
        Legacy max slippage parameter. Do not used.
    """

    assert isinstance(confirmation_timeout, datetime.timedelta), f"Got {confirmation_timeout}"

    if asset_management_mode == AssetManagementMode.dummy:
        # Used in test_strategy_cycle_trigger.py
        web3 = web3config.get_default()
        execution_model = DummyExecutionModel(web3)
        sync_method = DummyWalletSyncer()
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory
        sync_model = DummySyncModel()
        return execution_model, sync_model, valuation_model_factory, pricing_model_factory
    elif asset_management_mode in (AssetManagementMode.hot_wallet, AssetManagementMode.enzyme):
        assert private_key, "Private key is needed for live trading"
        web3 = web3config.get_default()
        hot_wallet = HotWallet.from_private_key(private_key)
        sync_model = create_sync_model(asset_management_mode, web3, hot_wallet, vault_address, vault_adapter_address)
        execution_model = UniswapV2ExecutionModel(
            sync_model.create_transaction_builder(),
            confirmation_timeout=confirmation_timeout,
            confirmation_block_count=confirmation_block_count,
            max_slippage=max_slippage,
            min_balance_threshold=min_balance_threshold,
        )
        valuation_model_factory = uniswap_v2_sell_valuation_factory
        pricing_model_factory = uniswap_v2_live_pricing_factory

        # TODO: Temporary fix to prevent connections elsewhere
        # Make sure this never happens even though it should not happen
        if ChainId.bsc in web3config.connections or ChainId.polygon in web3config.connections or ChainId.avalanche in web3config.connections:
            if web3config.gas_price_method == GasPriceMethod.london:
                raise RuntimeError(f"Should not happen: {web3config.gas_price_method}")

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

    logger.info("Dataset cache is %s", cache_path)

    os.makedirs(cache_path, exist_ok=True)

    with open(cache_path.joinpath("cache.pid"), "wt") as out:
        print(os.getpid(), file=out)

    return cache_path


def create_metadata(name, short_description, long_description, icon_url) -> Metadata:
    """Create metadata object from the configuration variables."""
    return Metadata(
        name,
        short_description,
        long_description,
        icon_url,
        datetime.datetime.utcnow(),
        executor_running=True,
    )


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
            )
        case _:
            raise NotImplementedError()

