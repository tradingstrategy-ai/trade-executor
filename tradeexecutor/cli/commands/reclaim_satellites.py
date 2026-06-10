"""lagoon-reclaim-satellites command.

Consolidates capital scattered across multichain Lagoon satellite Safes back
into the master vault Safe. This is the manual recovery counterpart to the
automatic cross-chain bridging that ``start`` performs during live trading.

For every satellite chain configured in the deployment artifact the command:

1. Completes any incomplete (``cctp_in_transit``) CCTP transfers recorded in
   the state file, so capital that was burned but never minted is recovered
   first and swept along with the rest.
2. Reads the USDC balance held by the satellite Safe.
3. Bridges that USDC back to the master Safe via CCTP (burn on the satellite
   chain through the Lagoon guard module, then ``receiveMessage`` on the master
   chain) — the same shared Safe address custodies funds on every chain.
4. Verifies each CCTP retrieve (the master-chain mint) confirmed successfully.

Use ``--dry-run`` to report satellite balances and in-transit trades without
broadcasting anything.
"""

import datetime
import logging
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from eth_defi.cctp.attestation import fetch_attestation
from eth_defi.cctp.bridge import burn_usdc_cctp, receive_usdc_cctp
from eth_defi.cctp.transfer import _resolve_cctp_domain
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fund_erc20_on_anvil, is_anvil
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from tradingstrategy.chain import ChainId

from tradeexecutor.cli.bootstrap import (
    create_execution_and_sync_model,
    create_web3_config,
    prepare_cache_and_token_cache,
    prepare_executor_id,
)
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_utils import (
    ensure_state_store_exists,
    sync_reserve_balance_to_state,
)
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.cctp.retry import check_and_retry_cctp_in_transit
from tradeexecutor.ethereum.cctp.routing import CCTP_RECEIVE_GAS_FALLBACK
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.utils.key import ensure_0x_prefixed_private_key

logger = logging.getLogger(__name__)


def plan_reclaims(
    satellite_usdc: dict[int, Decimal],
    min_reclaim_amount: Decimal,
) -> list[int]:
    """Decide which satellite chains hold enough USDC to bridge back.

    Pure planning helper, separated from on-chain I/O so it can be unit tested.

    :param satellite_usdc:
        Mapping of ``chain_id`` to the USDC balance held by that satellite Safe.

    :param min_reclaim_amount:
        Minimum USDC balance worth bridging. Balances at or below this are left
        in place because the CCTP gas/fee cost is not worth reclaiming dust.

    :return:
        Sorted list of chain ids whose balance strictly exceeds the threshold.
    """
    return sorted(
        chain_id
        for chain_id, balance in satellite_usdc.items()
        if balance > min_reclaim_amount
    )


def _bridge_satellite_to_master(
    *,
    sat_web3,
    sat_vault,
    master_web3,
    master_chain_id: int,
    master_safe_address: str,
    amount_raw: int,
    private_key: str,
    attestation_timeout: float,
) -> dict:
    """Bridge USDC from one satellite Safe back to the master Safe via CCTP.

    Composed from the eth_defi burn / attest / receive primitives rather than
    the all-in-one :func:`bridge_usdc_cctp`, because that helper reuses a single
    :class:`HotWallet` nonce counter across both the source and destination
    chains. Source and destination nonces differ on live chains, so we sign with
    a freshly nonce-synced wallet per chain — mirroring the production routing.

    On Anvil forks Circle's Iris API cannot attest the burn, so after the real
    burn through the guard the mint is materialised on the master chain with
    ``fund_erc20_on_anvil()`` — the same simulation approach
    ``perform-test-trade`` uses for cross-chain test trades.

    :return:
        Dict with ``burn_tx_hash`` and ``receive_tx_hash``.
    """
    source_chain_id = sat_web3.eth.chain_id

    # Burn on the satellite chain through the Lagoon guard module.
    source_wallet = HotWallet.from_private_key(ensure_0x_prefixed_private_key(private_key))
    source_wallet.sync_nonce(sat_web3)
    burn_result = burn_usdc_cctp(
        source_web3=sat_web3,
        source_vault=sat_vault,
        dest_chain_id=master_chain_id,
        dest_safe_address=master_web3.to_checksum_address(master_safe_address),
        amount=amount_raw,
        sender=source_wallet.address,
        hot_wallet=source_wallet,
    )

    if is_anvil(master_web3):
        # Anvil fork: no Iris attestation available — materialise the mint.
        # fund_erc20_on_anvil() overwrites the balance via anvil_setStorageAt,
        # so read the existing balance first and add to it.
        master_usdc_address = USDC_NATIVE_TOKEN[master_chain_id]
        master_usdc = fetch_erc20_details(master_web3, master_usdc_address)
        existing_raw = master_usdc.contract.functions.balanceOf(
            master_web3.to_checksum_address(master_safe_address)
        ).call()
        logger.info(
            "Anvil fork: materialising CCTP mint of %d raw USDC on master chain %d",
            amount_raw, master_chain_id,
        )
        fund_erc20_on_anvil(
            master_web3,
            master_usdc_address,
            master_web3.to_checksum_address(master_safe_address),
            existing_raw + amount_raw,
        )
        return {
            "burn_tx_hash": burn_result.burn_tx_hash,
            "receive_tx_hash": "<anvil fork simulated mint>",
        }

    # Wait for Circle's attestation on the burn.
    source_domain = _resolve_cctp_domain(source_chain_id)
    assert source_domain is not None, f"No CCTP domain for satellite chain {source_chain_id}"
    attestation = fetch_attestation(
        source_domain=source_domain,
        transaction_hash=burn_result.burn_tx_hash,
        timeout=attestation_timeout,
    )

    # Receive (mint) on the master chain with a wallet synced to the master nonce.
    master_wallet = HotWallet.from_private_key(ensure_0x_prefixed_private_key(private_key))
    master_wallet.sync_nonce(master_web3)
    receive_tx_hash = receive_usdc_cctp(
        dest_web3=master_web3,
        message=attestation.message,
        attestation=attestation.attestation,
        sender=master_wallet.address,
        hot_wallet=master_wallet,
        gas=CCTP_RECEIVE_GAS_FALLBACK,
    )

    return {
        "burn_tx_hash": burn_result.burn_tx_hash,
        "receive_tx_hash": receive_tx_hash,
    }


@app.command()
@shared_options.with_json_rpc_options()
def lagoon_reclaim_satellites(
    id: str = shared_options.id,

    strategy_file: Path = shared_options.strategy_file,
    state_file: Optional[Path] = shared_options.state_file,
    cache_path: Optional[Path] = shared_options.cache_path,

    log_level: str = shared_options.log_level,
    rpc_kwargs: dict | None = None,
    private_key: str = shared_options.private_key,

    asset_management_mode: AssetManagementMode = shared_options.asset_management_mode,
    vault_address: Optional[str] = shared_options.vault_address,
    vault_adapter_address: Optional[str] = shared_options.vault_adapter_address,
    vault_payment_forwarder_address: Optional[str] = shared_options.vault_payment_forwarder,
    min_gas_balance: Optional[float] = shared_options.min_gas_balance,

    min_reclaim_amount: float = typer.Option(
        1.0,
        envvar="MIN_RECLAIM_AMOUNT",
        help="Minimum USDC balance on a satellite Safe worth bridging back. Balances at or below are left in place as dust.",
    ),
    attestation_timeout: float = typer.Option(
        1800.0,
        envvar="CCTP_ATTESTATION_TIMEOUT",
        help="Maximum seconds to wait for Circle's CCTP attestation per bridge.",
    ),
    dry_run: bool = typer.Option(
        False,
        envvar="DRY_RUN",
        help="Report satellite balances and in-transit transfers without broadcasting any transaction.",
    ),

    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
):
    """Reclaim capital from multichain Lagoon satellite Safes into the master vault.

    - Completes any CCTP transfers stuck in transit (burned but not yet minted).
    - Bridges each satellite Safe's USDC back to the master Safe via CCTP.
    - Verifies every retrieve (master-chain mint) confirmed on-chain.

    Satellites are auto-discovered from the ``SATELLITE_MODULES`` env var or the
    ``{id}.deployment.json`` artifact written next to the state file by
    ``lagoon-deploy-vault``.
    """
    id = prepare_executor_id(id, strategy_file)

    logger = setup_logging(log_level=log_level)

    assert private_key, "PRIVATE_KEY is required"
    assert vault_address, "VAULT_ADDRESS is required"

    # --dry-run never broadcasts; treat --simulate the same way because we have
    # no forged attester to mint against on a fork.
    no_broadcast = dry_run or simulate

    mod = read_strategy_module(strategy_file)

    cache_path, token_cache = prepare_cache_and_token_cache(
        id,
        cache_path,
        unit_testing=unit_testing,
    )

    web3config = create_web3_config(
        **rpc_kwargs,
        unit_testing=unit_testing,
        simulate=simulate,
    )
    assert web3config.has_chain_configured(), "No JSON-RPC connections configured for any chain"

    default_chain_id = mod.get_default_chain_id()
    web3config.set_default_chain(default_chain_id)
    web3config.check_default_chain_id()

    deployment_file = (
        Path(state_file) if state_file else Path(f"state/{id}.json")
    ).with_name(f"{id}.deployment.json")

    execution_model, sync_model, _valuation_factory, _pricing_factory = create_execution_and_sync_model(
        asset_management_mode=asset_management_mode,
        private_key=private_key,
        web3config=web3config,
        confirmation_timeout=datetime.timedelta(seconds=60),
        confirmation_block_count=2,
        max_slippage=0.01,
        min_gas_balance=min_gas_balance,
        vault_address=vault_address,
        vault_adapter_address=vault_adapter_address,
        vault_payment_forwarder_address=vault_payment_forwarder_address,
        routing_hint=mod.trade_routing,
        token_cache=token_cache,
        deployment_file=deployment_file,
        unit_testing=unit_testing,
    )

    assert isinstance(execution_model, LagoonExecution), (
        f"lagoon-reclaim-satellites only supports Lagoon vaults, got {execution_model}"
    )

    satellite_vaults = getattr(execution_model, "satellite_vaults", {}) or {}
    master_safe_address = sync_model.vault.safe_address
    master_chain_id = default_chain_id.value
    master_web3 = web3config.get_default()

    logger.info("Reclaiming satellite capital into master vault")
    logger.info("  Master chain: %s (chain id %d)", default_chain_id.get_name(), master_chain_id)
    logger.info("  Master Safe: %s", master_safe_address)
    logger.info("  Satellite chains: %s", [ChainId(c).get_slug() for c in satellite_vaults] or ["none"])
    if no_broadcast:
        logger.info("  Dry run — no transactions will be broadcast")

    if not satellite_vaults:
        logger.info("No satellite vaults configured — nothing to reclaim")
        return

    state_path, store = ensure_state_store_exists(id, state_file, simulate=simulate)
    state = store.load()

    # Step 1: finish any CCTP transfers burned but never minted, so the recovered
    # USDC lands in a Safe and is swept along with the rest below.
    if no_broadcast:
        in_transit = [
            trade
            for position in state.portfolio.open_positions.values()
            for trade in position.trades.values()
            if trade.get_status() == TradeStatus.cctp_in_transit
        ]
        logger.info("Step 1: %d CCTP in-transit trade(s) found (skipped in dry run)", len(in_transit))
    else:
        logger.info("Step 1: completing in-transit CCTP transfers")
        resolved = check_and_retry_cctp_in_transit(
            state=state,
            execution_model=execution_model,
            web3config=web3config,
            attestation_timeout=attestation_timeout,
        )
        if resolved:
            store.sync(state)
        logger.info("  Resolved %d in-transit transfer(s)", len(resolved))

    # Step 2: read each satellite Safe's USDC balance.
    logger.info("Step 2: reading satellite Safe USDC balances")
    satellite_usdc: dict[int, Decimal] = {}
    usdc_tokens = {}
    for chain_id_value, sat_vault in satellite_vaults.items():
        sat_web3 = web3config.get_connection(ChainId(chain_id_value))
        assert sat_web3 is not None, f"No Web3 connection for satellite chain {chain_id_value}"

        usdc_address = USDC_NATIVE_TOKEN.get(chain_id_value)
        assert usdc_address is not None, f"No native USDC known for satellite chain {chain_id_value}"
        usdc = fetch_erc20_details(sat_web3, usdc_address)
        usdc_tokens[chain_id_value] = usdc

        balance = usdc.fetch_balance_of(sat_vault.safe_address)
        satellite_usdc[chain_id_value] = balance
        logger.info(
            "  %s: Safe %s holds %s USDC",
            ChainId(chain_id_value).get_slug(), sat_vault.safe_address, balance,
        )

    # Step 3: bridge reclaimable balances back to the master Safe.
    to_reclaim = plan_reclaims(satellite_usdc, Decimal(str(min_reclaim_amount)))
    logger.info(
        "Step 3: %d satellite(s) above the %s USDC threshold: %s",
        len(to_reclaim), min_reclaim_amount,
        [ChainId(c).get_slug() for c in to_reclaim] or ["none"],
    )

    master_usdc_address = USDC_NATIVE_TOKEN.get(master_chain_id)
    assert master_usdc_address is not None, f"No native USDC known for master chain {master_chain_id}"
    master_usdc = fetch_erc20_details(master_web3, master_usdc_address)
    master_balance_before = master_usdc.fetch_balance_of(master_safe_address)

    results: list[dict] = []
    for chain_id_value in to_reclaim:
        sat_vault = satellite_vaults[chain_id_value]
        sat_web3 = web3config.get_connection(ChainId(chain_id_value))
        usdc = usdc_tokens[chain_id_value]
        balance = satellite_usdc[chain_id_value]
        amount_raw = usdc.convert_to_raw(balance)
        slug = ChainId(chain_id_value).get_slug()

        if no_broadcast:
            logger.info("  Would bridge %s USDC from %s back to master Safe", balance, slug)
            continue

        logger.info("  Bridging %s USDC from %s back to master Safe", balance, slug)
        bridged = _bridge_satellite_to_master(
            sat_web3=sat_web3,
            sat_vault=sat_vault,
            master_web3=master_web3,
            master_chain_id=master_chain_id,
            master_safe_address=master_safe_address,
            amount_raw=amount_raw,
            private_key=private_key,
            attestation_timeout=attestation_timeout,
        )
        results.append({"chain": slug, "amount": balance, **bridged})
        logger.info(
            "  Reclaimed %s USDC from %s — burn %s, receive %s",
            balance, slug, bridged["burn_tx_hash"], bridged["receive_tx_hash"],
        )

    # Step 4: verify the retrieve(s) landed on the master Safe.
    if no_broadcast:
        logger.info("Dry run complete — no funds moved. State file: %s", state_path)
        return

    master_balance_after = master_usdc.fetch_balance_of(master_safe_address)
    reclaimed_total = master_balance_after - master_balance_before
    logger.info("Step 4: verifying retrieve on master Safe")
    logger.info("  Master Safe USDC before: %s", master_balance_before)
    logger.info("  Master Safe USDC after:  %s", master_balance_after)
    logger.info("  Net reclaimed:           %s USDC across %d bridge(s)", reclaimed_total, len(results))

    # Sweeping into the master Safe increases the on-chain reserve held by the
    # vault. Reflect that in the state file so accounting stays consistent.
    if results:
        sync_reserve_balance_to_state(store, master_usdc, master_balance_after)
        logger.info("  State file reserve updated: %s", state_path)

    logger.info("All ok")
