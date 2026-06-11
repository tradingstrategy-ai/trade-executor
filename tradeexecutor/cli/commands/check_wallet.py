"""check-wallet command"""

import datetime
import logging
from pathlib import Path
from typing import Optional

import typer
from web3 import Web3

from eth_defi.balances import fetch_erc20_balances_by_token_list
from eth_defi.hotwallet import HotWallet
from eth_defi.token import fetch_erc20_details

from tradeexecutor.strategy.pandas_trader.create_universe_wrapper import call_create_trading_universe
from tradingstrategy.chain import ChainId
from tradingstrategy.client import Client
from . import shared_options
from .app import app
from ..bootstrap import prepare_executor_id, prepare_cache_and_token_cache, create_web3_config, create_execution_and_sync_model, resolve_deployment_file
from ..log import setup_logging
from ...ethereum.enzyme.vault import EnzymeVaultSyncModel
from ...ethereum.lagoon.vault import LagoonVaultSyncModel
from ...ethereum.velvet.vault import VelvetVaultSyncModel
from ...ethereum.web3config import TEST_CHAIN_IDS
from ...state.identifier import AssetIdentifier
from ...strategy.approval import UncheckedApprovalModel
from ...strategy.bootstrap import make_factory_from_strategy_mod
from ...strategy.description import StrategyExecutionDescription
from ...strategy.execution_context import ExecutionContext, ExecutionMode
from ...strategy.execution_model import AssetManagementMode
from ...strategy.run_state import RunState
from ...strategy.strategy_module import read_strategy_module
from ...utils.fullname import get_object_full_name
from ...utils.timer import timed_task


logger = logging.getLogger(__name__)
WALLET_CHECK_SEPARATOR = "-" * 80


def _normalise_wallet_check_chain_id(chain_id: ChainId | int) -> ChainId:
    """Normalise synthetic trading chain ids to the EVM chain used for wallet checks."""
    if isinstance(chain_id, int):
        chain_id = ChainId(chain_id)

    if chain_id == ChainId.hypercore:
        return ChainId.hyperliquid

    return chain_id


def _collect_wallet_check_pairs(universe) -> list:
    """Collect trading pairs once so one-shot iterators cannot be exhausted."""
    iterate_pairs = getattr(universe, "iterate_pairs", None)
    if iterate_pairs is None:
        return []

    if callable(iterate_pairs):
        return list(iterate_pairs())

    return list(iterate_pairs)


def _collect_wallet_check_chains(universe, default_chain_id: ChainId, pairs: list) -> list[ChainId]:
    """Collect chain ids that need a wallet balance section."""
    chain_ids: set[ChainId] = set()

    data_universe = getattr(universe, "data_universe", None)
    if data_universe is not None:
        for chain_id in getattr(data_universe, "chains", set()) or set():
            chain_ids.add(_normalise_wallet_check_chain_id(chain_id))

    for asset in getattr(universe, "reserve_assets", []) or []:
        chain_ids.add(_normalise_wallet_check_chain_id(asset.chain_id))

    for pair in pairs:
        chain_ids.add(_normalise_wallet_check_chain_id(pair.base.chain_id))
        chain_ids.add(_normalise_wallet_check_chain_id(pair.quote.chain_id))

    if not chain_ids:
        chain_ids.add(default_chain_id)

    return sorted(chain_ids, key=lambda chain_id: (chain_id != default_chain_id, chain_id.value))


def _collect_wallet_check_assets_by_chain(universe, pairs: list) -> dict[ChainId, list[AssetIdentifier]]:
    """Collect reserve-like assets to check, keyed by their EVM chain."""
    assets_by_key: dict[tuple[ChainId, str], AssetIdentifier] = {}

    def add_asset(asset: AssetIdentifier) -> None:
        chain_id = _normalise_wallet_check_chain_id(asset.chain_id)
        assets_by_key[(chain_id, asset.address.lower())] = asset

    for asset in universe.reserve_assets:
        add_asset(asset)

    for pair in pairs:
        if pair.is_cctp_bridge():
            add_asset(pair.base)
            add_asset(pair.quote)

    assets_by_chain: dict[ChainId, list[AssetIdentifier]] = {}
    for (chain_id, _), asset in assets_by_key.items():
        assets_by_chain.setdefault(chain_id, []).append(asset)

    for chain_assets in assets_by_chain.values():
        chain_assets.sort(key=lambda asset: (asset.token_symbol, asset.address))

    return assets_by_chain


def _get_chain_web3(web3config, chain_id: ChainId):
    """Get the Web3 connection for a wallet check chain."""
    get_connection = getattr(web3config, "get_connection", None)
    if get_connection is None:
        return web3config.get_default()

    try:
        return get_connection(chain_id)
    except KeyError as e:
        raise RuntimeError(
            f"No JSON-RPC connection configured for {chain_id.get_name()} "
            f"(chain id {chain_id.value})."
        ) from e


def _get_lagoon_satellite_vault(execution_model, chain_id: ChainId):
    """Get the Lagoon satellite vault object for this chain, if any."""
    satellite_vaults = getattr(execution_model, "satellite_vaults", {}) or {}
    return satellite_vaults.get(chain_id.value)


def _get_chain_reserve_address(sync_model, execution_model, chain_id: ChainId) -> str:
    """Get the address that stores reserve tokens on this chain."""
    satellite_vault = _get_lagoon_satellite_vault(execution_model, chain_id)
    if satellite_vault is not None:
        return satellite_vault.safe_address

    return sync_model.get_token_storage_address() or sync_model.get_hot_wallet().address


def _log_vault_share_details(sync_model, hot_wallet_address: str) -> None:
    """Log vault share token balance and total supply for the asset manager."""
    vault = getattr(sync_model, "vault", None)
    if vault is None:
        return

    # Enzyme uses shares_token (plural), Lagoon/Velvet use share_token (singular).
    # Velvet exposes fetch_share_token() but not a cached share_token property,
    # so fall back to calling it directly.
    share_token = getattr(vault, "share_token", None) or getattr(vault, "shares_token", None)
    if share_token is None:
        fetch_fn = getattr(vault, "fetch_share_token", None)
        if callable(fetch_fn):
            share_token = fetch_fn()
    if share_token is None:
        return

    share_balance = share_token.fetch_balance_of(hot_wallet_address)

    # Total supply: Enzyme exposes get_total_supply() returning raw int,
    # Lagoon/Velvet share_token is a standard ERC-20 so use contract.functions.totalSupply()
    raw_total_supply_fn = getattr(vault, "get_total_supply", None)
    if callable(raw_total_supply_fn):
        total_supply = share_token.convert_to_decimals(raw_total_supply_fn())
    else:
        raw = share_token.contract.functions.totalSupply().call()
        total_supply = share_token.convert_to_decimals(raw)

    logger.info("  Share token: %s (%s)", share_token.symbol, share_token.address)
    logger.info("  Asset manager share balance: %s %s", share_balance, share_token.symbol)
    logger.info("  Total share supply: %s %s", total_supply, share_token.symbol)

    _log_lagoon_unclaimed_shares(vault, share_token, hot_wallet_address)


def _log_lagoon_unclaimed_shares(vault, share_token, hot_wallet_address: str) -> None:
    """Check and log unclaimed Lagoon vault deposits and redemptions for the asset manager.

    Lagoon vaults use ERC-7540 async deposit/redeem with intermediate states:

    Deposits:
    - Settled but unclaimed (maxDeposit > 0): shares ready to claim via finalise_deposit

    Redemptions:
    - Settled but unclaimed (maxRedeem > 0): denomination tokens ready to claim via finalise_redeem
    - Pending unsettled (pendingRedeemRequest > 0): redemption requested but not yet settled
    """
    vault_contract = getattr(vault, "vault_contract", None)
    if vault_contract is None:
        return

    denomination_token = getattr(vault, "denomination_token", None)

    # Check unclaimed deposits (settled deposit, shares not yet claimed)
    try:
        max_deposit_raw = vault_contract.functions.maxDeposit(hot_wallet_address).call()
    except Exception:
        max_deposit_raw = 0

    if max_deposit_raw > 0:
        if denomination_token is not None:
            deposit_human = denomination_token.convert_to_decimals(max_deposit_raw)
            deposit_symbol = denomination_token.symbol
        else:
            deposit_human = max_deposit_raw
            deposit_symbol = "raw"
        logger.warning(
            "  UNCLAIMED DEPOSIT: %s %s settled but unclaimed (run lagoon-redeem to claim shares)",
            deposit_human, deposit_symbol,
        )

    # Check settled but unclaimed redemptions
    try:
        max_redeemable_raw = vault_contract.functions.maxRedeem(hot_wallet_address).call()
    except Exception:
        max_redeemable_raw = 0

    if max_redeemable_raw > 0:
        redeemable_human = share_token.convert_to_decimals(max_redeemable_raw)
        logger.warning(
            "  UNCLAIMED REDEMPTION: %s %s settled but unclaimed shares (run lagoon-redeem to claim)",
            redeemable_human, share_token.symbol,
        )

    # Check pending unsettled redemptions
    try:
        pending_redeem_raw = vault_contract.functions.pendingRedeemRequest(
            0, hot_wallet_address
        ).call()
    except Exception:
        pending_redeem_raw = 0

    if pending_redeem_raw > 0:
        pending_human = share_token.convert_to_decimals(pending_redeem_raw)
        logger.warning(
            "  PENDING REDEMPTION: %s %s pending unsettled shares (run lagoon-redeem to settle and claim)",
            pending_human, share_token.symbol,
        )


def _log_chain_custody_details(sync_model, execution_model, chain_id: ChainId) -> None:
    """Log vault and custody addresses for a chain."""
    satellite_vault = _get_lagoon_satellite_vault(execution_model, chain_id)
    if satellite_vault is not None:
        logger.info("  Vault address is N/A (satellite chain)")
        logger.info("  Safe address is %s", satellite_vault.safe_address)
    elif isinstance(sync_model, EnzymeVaultSyncModel):
        logger.info("  Vault address is %s", sync_model.get_key_address())
    elif isinstance(sync_model, VelvetVaultSyncModel):
        logger.info("  Vault address is %s", sync_model.vault_address)
    elif isinstance(sync_model, LagoonVaultSyncModel):
        logger.info("  Vault address is %s", sync_model.vault_address)
        logger.info("  Safe address is %s", sync_model.get_token_storage_address())
    else:
        logger.info("  Vault address lookup not implemented")


@app.command()
@shared_options.with_json_rpc_options()
def check_wallet(
    id: str = shared_options.id,
    state_file: Optional[Path] = shared_options.state_file,

    strategy_file: Path = shared_options.strategy_file,
    private_key: str = shared_options.private_key,
    trading_strategy_api_key: str = shared_options.trading_strategy_api_key,
    cache_path: Optional[Path] = shared_options.cache_path,

    # Get minimum gas balance from the env
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,

    # Live trading or backtest
    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,

    # Web3 connection options
    rpc_kwargs: dict | None = None,

    log_level: Optional[str] = shared_options.log_level,

    # Debugging and unit testing
    unit_testing: bool = shared_options.unit_testing,
    unit_test_force_anvil: bool = typer.Option(bool, envvar="UNIT_TEST_FORCE_ANVIL", help="Use Anvil backend regardless of what chain strategy module suggests"),
):
    """Print out the token balances of the hot wallet.

    Check that our hot wallet has cash deposits and gas token deposits.
    """

    # To run this from command line with .env file you can do
    # set -o allexport ; source ~/pancake-eth-usd-sma-final.env ; set +o allexport ;  trade-executor check-wallet

    global logger

    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level)

    mod = read_strategy_module(strategy_file)

    cache_path, token_cache = prepare_cache_and_token_cache(
        id,
        cache_path,
        unit_testing=unit_testing,
    )

    client = Client.create_live_client(
        trading_strategy_api_key,
        cache_path=cache_path,
        settings_path=None,
    )

    web3config = create_web3_config(
        **rpc_kwargs,
        unit_testing=unit_testing,
    )
    assert web3config.has_chain_configured(), "No RPC endpoints given. A working JSON-RPC connection is needed for running this command. Check your JSON-RPC configuration."

    execution_context = ExecutionContext(
        mode=ExecutionMode.preflight_check,
        timed_task_context_manager=timed_task,
        engine_version=mod.trading_strategy_engine_version,
    )

    universe = call_create_trading_universe(
        mod.create_trading_universe,
        client=client,
        universe_options=mod.get_universe_options(),
        execution_context=execution_context,
    )

    # Check that we are connected to the chain strategy assumes
    if unit_test_force_anvil:
        default_chain_id = ChainId.anvil
    else:
        default_chain_id = mod.get_default_chain_id()

    web3config.set_default_chain(default_chain_id)
    web3config.check_default_chain_id()

    execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=6,
        max_slippage=0.01,
        min_gas_balance=min_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
        token_cache=token_cache,
        deployment_file=resolve_deployment_file(id, state_file),
    )

    assert asset_management_mode.is_live_trading(), f"Cannot perform check wallet for non-real modes"
    assert sync_model, f"sync_model not set up"
    assert sync_model.get_hot_wallet(), f"sync_model {sync_model} lacks hot wallet"

    hot_wallet = HotWallet.from_private_key(private_key)

    # Set up the strategy engine
    factory = make_factory_from_strategy_mod(mod)
    run_description: StrategyExecutionDescription = factory(
        execution_model=execution_model,
        execution_context=execution_context,
        timed_task_context_manager=execution_context.timed_task_context_manager,
        sync_model=sync_model,
        valuation_model_factory=valuation_model_factory,
        pricing_model_factory=pricing_model_factory,
        approval_model=UncheckedApprovalModel(),
        client=client,
        run_state=RunState(),
    )

    # Get all tokens from the universe
    if unit_test_force_anvil:
        wallet_check_chains = [ChainId.anvil]
        assets_by_chain = {ChainId.anvil: list(universe.reserve_assets)}
    else:
        wallet_check_pairs = _collect_wallet_check_pairs(universe)
        wallet_check_chains = _collect_wallet_check_chains(universe, default_chain_id, wallet_check_pairs)
        assets_by_chain = _collect_wallet_check_assets_by_chain(universe, wallet_check_pairs)

    default_wallet_check_chain_id = _normalise_wallet_check_chain_id(default_chain_id)

    for chain_id in wallet_check_chains:
        web3 = _get_chain_web3(web3config, chain_id)
        reserve_assets = assets_by_chain.get(chain_id, [])
        tokens = [Web3.to_checksum_address(a.address) for a in reserve_assets]

        logger.info(WALLET_CHECK_SEPARATOR)
        logger.info("%s (chain id %d)", chain_id.get_name(), chain_id.value)
        logger.info("RPC details")

        # Check the chain is online
        logger.info(f"  Chain id is {web3.eth.chain_id:,}")
        logger.info(f"  Latest block is {web3.eth.block_number:,}")
        if chain_id not in TEST_CHAIN_IDS:
            assert web3.eth.chain_id == chain_id.value, (
                f"Wallet check expected chain id {chain_id} ({chain_id.value}), "
                f"RPC says we got {web3.eth.chain_id}"
            )

        # Check balances
        reserve_address = _get_chain_reserve_address(sync_model, execution_model, chain_id)
        logger.info("Balance details")
        logger.info("  Hot wallet is %s", sync_model.get_hot_wallet().address)
        gas_balance = web3.eth.get_balance(hot_wallet.address) / 10**18
        _log_chain_custody_details(sync_model, execution_model, chain_id)
        if chain_id == default_wallet_check_chain_id:
            _log_vault_share_details(sync_model, hot_wallet.address)

        logger.info("  Hot wallet %s has %f native gas tokens", hot_wallet.address, gas_balance)
        logger.info("  The gas error limit is %f tokens", min_gas_balance)

        token_details_by_address = {}
        for asset in reserve_assets:
            logger.info("  Reserve asset: %s (%s)", asset.token_symbol, asset.address)
            details = fetch_erc20_details(web3, asset.address, cache=token_cache)
            token_details_by_address[Web3.to_checksum_address(asset.address)] = details
            hot_wallet_reserve_balance = details.fetch_balance_of(hot_wallet.address)
            logger.info(
                "  Hot wallet reserve balance of %s (%s): %s %s",
                details.name,
                details.address,
                hot_wallet_reserve_balance,
                details.symbol,
            )

        if not tokens:
            logger.info("  No reserve assets detected for this chain")
            continue

        balances = fetch_erc20_balances_by_token_list(web3, reserve_address, tokens)

        for address, balance in balances.items():
            details = token_details_by_address.get(Web3.to_checksum_address(address))
            if details is None:
                details = fetch_erc20_details(web3, address, cache=token_cache)

            if reserve_address != hot_wallet.address:
                satellite_vault = _get_lagoon_satellite_vault(execution_model, chain_id)
                if satellite_vault is not None:
                    balance_owner = "Safe"
                elif isinstance(sync_model, (EnzymeVaultSyncModel, VelvetVaultSyncModel, LagoonVaultSyncModel)):
                    balance_owner = "Vault"
                else:
                    balance_owner = "Token storage"
                logger.info(
                    "  %s reserve balance of %s (%s) at address %s: %s %s",
                    balance_owner,
                    details.name,
                    details.address,
                    reserve_address,
                    details.convert_to_decimals(balance),
                    details.symbol,
                )

    # Check that the routing looks sane
    # E.g. there is no mismatch between strategy reserve token, wallet and pair universe
    runner = run_description.runner
    routing_model = None
    pricing_model = None
    valuation_method = None
    routing_preflight_skipped_reason = None

    try:
        _, pricing_model, valuation_method = runner.setup_routing(universe)
        routing_model = runner.routing_model
    except NotImplementedError as e:
        if universe.cross_chain:
            routing_preflight_skipped_reason = str(e)
            logger.info("Skipping routing preflight for cross-chain universe: %s", e)
        else:
            raise

    logger.info("Execution details")
    logger.info("  Execution model is %s", get_object_full_name(execution_model))
    logger.info("  Routing model is %s", get_object_full_name(routing_model) if routing_model else "<skipped>")
    logger.info("  Token pricing model is %s", get_object_full_name(pricing_model) if pricing_model else "<skipped>")
    logger.info("  Position valuation model is %s", get_object_full_name(valuation_method) if valuation_method else "<skipped>")
    logger.info("  Sync model is %s", get_object_full_name(sync_model))

    if routing_preflight_skipped_reason:
        logger.info("  Routing preflight skipped reason: %s", routing_preflight_skipped_reason)

    # Check we have enough gas
    execution_model.preflight_check()

    # Check our routes
    if routing_model is not None:
        routing_model.perform_preflight_checks_and_logging(universe.data_universe.pairs)

    web3config.close()

    logger.info("All ok")
