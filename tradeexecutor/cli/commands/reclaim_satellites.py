"""lagoon-reclaim-satellites command.

Consolidates capital scattered across multichain Lagoon satellite Safes back
into the master vault Safe. This is the manual recovery counterpart to the
automatic cross-chain bridging that ``start`` performs during live trading.

The command first gathers a full picture without broadcasting anything:

1. Reads the USDC balance held by every satellite Safe and the master Safe,
   and displays them as a table.
2. Finds incomplete (``cctp_in_transit``) CCTP transfers recorded in the state
   file, auto-detects unreceived burns by scanning ``DepositForBurn`` events
   from our Safes on every chain (checking Circle's Iris API and the
   destination chain's ``usedNonces()``), and resolves any explicit
   ``--complete-burn-tx`` burns — covering transfers that were never written
   to the state file, e.g. when the process crashed before the state was
   saved.
3. Lists every action it is about to perform and asks for y/n confirmation.

Then it executes:

4. Completes the in-transit and unrecorded transfers (``receiveMessage`` on
   their destination chains), so capital that was burned but never minted
   lands in a Safe first.
5. Re-reads satellite balances and bridges everything above the dust threshold
   back to the master Safe via CCTP (burn on the satellite chain through the
   Lagoon guard module, then ``receiveMessage`` on the master chain) — the
   same shared Safe address custodies funds on every chain.
6. Verifies each retrieve (the master-chain mint) confirmed and syncs the
   state file reserve to the new on-chain balance.

Use ``--dry-run`` to stop after the table and action list without
broadcasting anything.
"""

import datetime
import logging
import sys
from decimal import Decimal
from pathlib import Path
from typing import Optional

import typer
from eth_defi.cctp.attestation import fetch_attestation
from eth_defi.cctp.bridge import burn_usdc_cctp, receive_usdc_cctp
from eth_defi.cctp.constants import CCTP_DOMAIN_TO_CHAIN_ID
from eth_defi.cctp.events import fetch_deposit_for_burn_events
from eth_defi.cctp.monitor import CCTPTransferStatus, fetch_transfer_status
from eth_defi.cctp.transfer import _resolve_cctp_domain, get_message_transmitter_v2
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import fund_erc20_on_anvil, is_anvil
from eth_defi.token import USDC_NATIVE_TOKEN, fetch_erc20_details
from tabulate import tabulate
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


def parse_burn_tx_entries(complete_burn_tx: str | None) -> list[tuple[str, str]]:
    """Parse ``--complete-burn-tx`` entries into ``(chain_slug, tx_hash)`` pairs.

    Pure parsing helper, separated from Iris API I/O so it can be unit tested.

    :param complete_burn_tx:
        Comma-separated ``<chain_slug>:<burn_tx_hash>`` entries, e.g.
        ``arbitrum:0x7a3c...`` — or ``None``/empty for no entries.

    :return:
        List of ``(chain_slug, burn_tx_hash)`` tuples.
    """
    entries: list[tuple[str, str]] = []
    if not complete_burn_tx:
        return entries

    for raw in complete_burn_tx.split(","):
        raw = raw.strip()
        if not raw:
            continue
        slug, sep, burn_tx_hash = raw.partition(":")
        slug = slug.strip().lower()
        burn_tx_hash = burn_tx_hash.strip()
        assert sep and slug and burn_tx_hash.startswith("0x"), (
            f"--complete-burn-tx entries must be <chain_slug>:<burn_tx_hash>, got {raw!r}. "
            f"Example: arbitrum:0x7a3c7ba8b770bb7db401e47cf916ceb7592a82335744ee0b4f6f838f7c1b2834"
        )
        entries.append((slug, burn_tx_hash))
    return entries


def _resolve_unrecorded_burns(complete_burn_tx: str | None) -> list[dict]:
    """Resolve ``--complete-burn-tx`` entries against Circle's Iris API.

    Fails fast (before any confirmation prompt) when a burn is not indexed,
    still pending block finality, or its attestation is not signed yet, so the
    operator never confirms an action list that cannot execute.

    :return:
        List of dicts with ``slug``, ``burn_tx_hash``, ``source_chain_id``,
        ``dest_chain_id``, ``dest_slug`` and the Iris ``status``
        (:class:`CCTPTransferStatus`).
    """
    burns: list[dict] = []
    for slug, burn_tx_hash in parse_burn_tx_entries(complete_burn_tx):
        try:
            source_chain_id = ChainId[slug].value
        except KeyError:
            raise RuntimeError(f"Unknown chain slug in --complete-burn-tx: {slug!r}")

        source_domain = _resolve_cctp_domain(source_chain_id)
        assert source_domain is not None, f"Chain {slug} is not CCTP-enabled"

        status = fetch_transfer_status(source_domain, burn_tx_hash)
        assert status is not None, (
            f"Circle Iris API has not indexed burn {burn_tx_hash} on {slug}. "
            f"Check the tx hash, or wait if the burn is very recent."
        )
        assert status.is_complete, (
            f"Attestation for burn {burn_tx_hash} on {slug} is not ready: "
            f"status={status.status}, delay_reason={status.delay_reason}. Retry later."
        )

        dest_chain_id = CCTP_DOMAIN_TO_CHAIN_ID.get(status.dest_domain)
        assert dest_chain_id is not None, f"Unknown CCTP destination domain {status.dest_domain} for burn {burn_tx_hash}"

        burns.append({
            "slug": slug,
            "burn_tx_hash": burn_tx_hash,
            "source_chain_id": source_chain_id,
            "dest_chain_id": dest_chain_id,
            "dest_slug": ChainId(dest_chain_id).get_slug(),
            "status": status,
            "auto": False,
        })
    return burns


def _estimate_lookback_blocks(web3, days: float, sample_span: int = 5_000) -> int:
    """Convert a lookback in days to a block count using the chain's recent block time."""
    latest = web3.eth.get_block("latest")
    past = web3.eth.get_block(max(1, latest["number"] - sample_span))
    span = latest["number"] - past["number"]
    block_time = (latest["timestamp"] - past["timestamp"]) / span if span else 12.0
    return int(days * 86_400 / max(block_time, 0.05))


def _discover_unreceived_burns(
    web3config,
    scan_safes: dict[int, str],
    lookback_days: float,
    known_burn_tx_hashes: set[str],
    hypersync_api_key: str | None = None,
) -> list[dict]:
    """Auto-detect CCTP burns from our Safes whose mint never happened.

    For every chain, scans ``DepositForBurn`` events where the depositor is
    our Safe (via :func:`eth_defi.cctp.events.fetch_deposit_for_burn_events` —
    HyperSync when available, chunked ``eth_getLogs`` fallback otherwise),
    then checks each burn against Circle's Iris API and the destination
    chain's ``MessageTransmitterV2.usedNonces()``. A burn whose attestation
    is ready but whose nonce is unused on the destination chain is stuck
    mid-bridge — typically because ``receiveMessage`` reverted or the process
    died before broadcasting it — and is returned as an actionable completion.

    Unlike explicit ``--complete-burn-tx`` entries, non-actionable burns
    (not indexed by Iris, attestation pending, already received) are logged
    and skipped rather than failing the command.

    :param scan_safes:
        Mapping of ``chain_id`` to the Safe address to scan on that chain.

    :param lookback_days:
        How far back to scan on every chain.

    :param known_burn_tx_hashes:
        Lowercased burn tx hashes already handled elsewhere (explicit
        ``--complete-burn-tx`` entries, in-transit state trades) — skipped.

    :return:
        List of burn dicts in the same shape as :func:`_resolve_unrecorded_burns`,
        with ``auto=True``.
    """
    burns: list[dict] = []
    for chain_id_value, safe_address in scan_safes.items():
        slug = ChainId(chain_id_value).get_slug()
        web3 = web3config.get_connection(ChainId(chain_id_value))
        if web3 is None:
            continue

        source_domain = _resolve_cctp_domain(chain_id_value)
        if source_domain is None:
            continue

        lookback_blocks = _estimate_lookback_blocks(web3, lookback_days)
        logger.info("Scanning %s for unreceived CCTP burns from Safe %s (%d blocks)", slug, safe_address, lookback_blocks)
        burn_events = fetch_deposit_for_burn_events(
            web3,
            depositor=web3.to_checksum_address(safe_address),
            start_block=max(1, web3.eth.block_number - lookback_blocks),
            hypersync_api_key=hypersync_api_key,
        )
        # One Iris lookup per transaction — dedupe, oldest first
        burn_tx_hashes = list(dict.fromkeys(event.transaction_hash for event in burn_events))
        for burn_tx_hash in burn_tx_hashes:
            if burn_tx_hash.lower() in known_burn_tx_hashes:
                continue

            status = fetch_transfer_status(source_domain, burn_tx_hash)
            if status is None:
                logger.warning("Burn %s on %s not indexed by Circle Iris API — skipping", burn_tx_hash, slug)
                continue
            if not status.is_complete:
                logger.warning(
                    "Burn %s on %s attestation not ready (status=%s, delay_reason=%s) — skipping, retry later",
                    burn_tx_hash, slug, status.status, status.delay_reason,
                )
                continue
            if status.nonce is None or (status.cctp_version or 2) != 2:
                logger.warning("Burn %s on %s is not a CCTP V2 message — skipping", burn_tx_hash, slug)
                continue

            dest_chain_id = CCTP_DOMAIN_TO_CHAIN_ID.get(status.dest_domain)
            if dest_chain_id is None:
                logger.warning("Burn %s on %s has unknown destination domain %d — skipping", burn_tx_hash, slug, status.dest_domain)
                continue
            dest_web3 = web3config.get_connection(ChainId(dest_chain_id))
            if dest_web3 is None:
                logger.warning("Burn %s on %s targets chain %d with no JSON-RPC connection — skipping", burn_tx_hash, slug, dest_chain_id)
                continue

            message_transmitter = get_message_transmitter_v2(dest_web3)
            nonce_used = message_transmitter.functions.usedNonces(
                bytes.fromhex(status.nonce.removeprefix("0x"))
            ).call()
            if nonce_used:
                logger.info("Burn %s on %s already received on destination — ok", burn_tx_hash, slug)
                continue

            logger.warning(
                "Detected unreceived CCTP burn %s: %s -> %s, attestation ready but receiveMessage never landed",
                burn_tx_hash, slug, ChainId(dest_chain_id).get_slug(),
            )
            burns.append({
                "slug": slug,
                "burn_tx_hash": burn_tx_hash,
                "source_chain_id": chain_id_value,
                "dest_chain_id": dest_chain_id,
                "dest_slug": ChainId(dest_chain_id).get_slug(),
                "status": status,
                "auto": True,
            })
    return burns


def _complete_unrecorded_burn(*, web3config, burn: dict, private_key: str) -> str:
    """Broadcast ``receiveMessage`` for a burn that is missing from the state file.

    The mint recipient and amount are encoded in the attested message, so this
    simply relays Circle's attestation to the destination chain. If the message
    was already received, the transaction reverts with a nonce-already-used
    error and the command stops with that explanation.

    :return:
        Receive transaction hash as hex string.
    """
    status: CCTPTransferStatus = burn["status"]
    dest_web3 = web3config.get_connection(ChainId(burn["dest_chain_id"]))
    assert dest_web3 is not None, f"No JSON-RPC connection configured for destination chain {burn['dest_slug']}"

    wallet = HotWallet.from_private_key(ensure_0x_prefixed_private_key(private_key))
    wallet.sync_nonce(dest_web3)
    return receive_usdc_cctp(
        dest_web3=dest_web3,
        message=status.message,
        attestation=status.attestation,
        sender=wallet.address,
        hot_wallet=wallet,
        gas=CCTP_RECEIVE_GAS_FALLBACK,
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


def _read_satellite_balances(web3config, satellite_vaults: dict) -> tuple[dict[int, Decimal], dict]:
    """Read the on-chain USDC balance of every satellite Safe.

    :return:
        Tuple of ``({chain_id: balance}, {chain_id: TokenDetails})``.
    """
    satellite_usdc: dict[int, Decimal] = {}
    usdc_tokens: dict = {}
    for chain_id_value, sat_vault in satellite_vaults.items():
        sat_web3 = web3config.get_connection(ChainId(chain_id_value))
        assert sat_web3 is not None, f"No Web3 connection for satellite chain {chain_id_value}"

        usdc_address = USDC_NATIVE_TOKEN.get(chain_id_value)
        assert usdc_address is not None, f"No native USDC known for satellite chain {chain_id_value}"
        usdc = fetch_erc20_details(sat_web3, usdc_address)
        usdc_tokens[chain_id_value] = usdc
        satellite_usdc[chain_id_value] = usdc.fetch_balance_of(sat_vault.safe_address)
    return satellite_usdc, usdc_tokens


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
    complete_burn_tx: Optional[str] = typer.Option(
        None,
        envvar="COMPLETE_BURN_TX",
        help="Complete CCTP burns that are missing from the state file (e.g. the process crashed before the state was saved). "
             "Comma-separated <chain_slug>:<burn_tx_hash> entries, e.g. arbitrum:0x7a3c... "
             "The attestation is fetched from Circle's Iris API and receiveMessage is broadcast on the destination chain.",
    ),
    burn_scan_lookback_days: float = typer.Option(
        7.0,
        envvar="BURN_SCAN_LOOKBACK_DAYS",
        help="Auto-detect unreceived CCTP burns by scanning DepositForBurn events from our Safes this many days back on every chain, "
             "checking each against Circle's Iris API and the destination chain's usedNonces(). Set 0 to disable the scan.",
    ),
    hypersync_api_key: Optional[str] = shared_options.hypersync_api_key,
    dry_run: bool = typer.Option(
        False,
        envvar="DRY_RUN",
        help="Show the balance table and planned actions without broadcasting any transaction.",
    ),

    unit_testing: bool = shared_options.unit_testing,
    simulate: bool = shared_options.simulate,
):
    """Reclaim capital from multichain Lagoon satellite Safes into the master vault.

    - Displays a table of all on-chain Safe USDC balances.
    - Lists the planned reclaim actions and asks for y/n confirmation.
    - Completes CCTP transfers stuck in transit (burned but not yet minted),
      including burns missing from the state file via --complete-burn-tx.
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
    master_slug = default_chain_id.get_slug()
    master_web3 = web3config.get_default()

    logger.info("Reclaiming satellite capital into master vault")
    logger.info("  Master chain: %s (chain id %d)", default_chain_id.get_name(), master_chain_id)
    logger.info("  Master Safe: %s", master_safe_address)
    logger.info("  Satellite chains: %s", [ChainId(c).get_slug() for c in satellite_vaults] or ["none"])
    if no_broadcast:
        logger.info("  Dry run — no transactions will be broadcast")

    if not satellite_vaults and not complete_burn_tx:
        logger.info("No satellite vaults configured — nothing to reclaim")
        return

    state_path, store = ensure_state_store_exists(id, state_file, simulate=simulate)
    state = store.load()

    # Gather phase (read-only): in-transit transfers recorded in the state
    # file, unrecorded burns from --complete-burn-tx resolved against Iris,
    # unreceived burns auto-detected from on-chain events, and all on-chain
    # Safe USDC balances.
    in_transit = [
        trade
        for position in state.portfolio.open_positions.values()
        for trade in position.trades.values()
        if trade.get_status() == TradeStatus.cctp_in_transit
    ]
    unrecorded_burns = _resolve_unrecorded_burns(complete_burn_tx)

    if burn_scan_lookback_days > 0:
        # Burns already handled by explicit entries or the state-based retry
        # must not be double-completed by the auto-scan.
        known_burn_tx_hashes = {burn["burn_tx_hash"].lower() for burn in unrecorded_burns}
        known_burn_tx_hashes |= {
            str(trade.other_data.get("cctp_burn_tx_hash", "")).lower()
            for trade in in_transit
        }
        scan_safes = {master_chain_id: master_safe_address}
        for chain_id_value, sat_vault in satellite_vaults.items():
            scan_safes[chain_id_value] = sat_vault.safe_address
        unrecorded_burns += _discover_unreceived_burns(
            web3config,
            scan_safes,
            burn_scan_lookback_days,
            known_burn_tx_hashes,
            hypersync_api_key=hypersync_api_key,
        )

    satellite_usdc, usdc_tokens = _read_satellite_balances(web3config, satellite_vaults)

    master_usdc_address = USDC_NATIVE_TOKEN.get(master_chain_id)
    assert master_usdc_address is not None, f"No native USDC known for master chain {master_chain_id}"
    master_usdc = fetch_erc20_details(master_web3, master_usdc_address)
    master_balance_before = master_usdc.fetch_balance_of(master_safe_address)

    # Display the on-chain Safe balance table.
    min_reclaim = Decimal(str(min_reclaim_amount))
    to_reclaim = plan_reclaims(satellite_usdc, min_reclaim)
    rows = [(master_slug, "master", master_safe_address, f"{master_balance_before:,.6f}", "receives reclaimed USDC")]
    for chain_id_value in sorted(satellite_usdc):
        balance = satellite_usdc[chain_id_value]
        if chain_id_value in to_reclaim:
            action = "bridge back to master"
        elif balance > 0:
            action = f"leave (dust <= {min_reclaim_amount} USDC)"
        else:
            action = "nothing to do"
        rows.append((
            ChainId(chain_id_value).get_slug(),
            "satellite",
            satellite_vaults[chain_id_value].safe_address,
            f"{balance:,.6f}",
            action,
        ))
    table = tabulate(rows, headers=["Chain", "Role", "Safe address", "USDC", "Planned action"], tablefmt="rounded_outline")
    logger.info("On-chain Safe USDC balances:\n%s", table)

    # Build and display the action list.
    actions: list[str] = []
    for burn in unrecorded_burns:
        origin = "auto-detected" if burn["auto"] else "from --complete-burn-tx"
        actions.append(
            f"Complete unreceived CCTP burn {burn['burn_tx_hash']} "
            f"({burn['slug']} -> {burn['dest_slug']}, attestation ready, {origin}) via receiveMessage"
        )
    for trade in in_transit:
        actions.append(
            f"Complete in-transit CCTP transfer for trade #{trade.trade_id} "
            f"(burn tx {trade.other_data.get('cctp_burn_tx_hash')})"
        )
    for chain_id_value in to_reclaim:
        actions.append(
            f"Bridge {satellite_usdc[chain_id_value]} USDC from {ChainId(chain_id_value).get_slug()} "
            f"Safe back to the master Safe on {master_slug}"
        )
    if unrecorded_burns or in_transit:
        actions.append(
            f"Re-read satellite balances after the completions above and sweep "
            f"any newly minted USDC above the {min_reclaim_amount} USDC threshold"
        )

    if not actions:
        logger.info("Nothing to reclaim — no in-transit transfers and all satellite Safes are at or below the dust threshold")
        return

    logger.info("Planned reclaim actions:")
    for index, action in enumerate(actions, start=1):
        logger.info("  %d. %s", index, action)

    if no_broadcast:
        logger.info("Dry run complete — no funds moved. State file: %s", state_path)
        return

    # Confirmation — skipped in unit testing, where stdin is not interactive.
    if not unit_testing:
        confirm = input(f"Proceed with these {len(actions)} action(s)? [y/n] ")
        if not confirm.lower().startswith("y"):
            print("Aborted")
            sys.exit(1)

    # Execute: complete unrecorded burns first (mints land in a Safe).
    for burn in unrecorded_burns:
        receive_tx_hash = _complete_unrecorded_burn(
            web3config=web3config,
            burn=burn,
            private_key=private_key,
        )
        logger.info(
            "Completed unrecorded burn %s on %s — receive tx %s on %s",
            burn["burn_tx_hash"], burn["slug"], receive_tx_hash, burn["dest_slug"],
        )

    # Complete in-transit transfers recorded in the state file.
    if in_transit:
        logger.info("Completing %d in-transit CCTP transfer(s)", len(in_transit))
        resolved = check_and_retry_cctp_in_transit(
            state=state,
            execution_model=execution_model,
            web3config=web3config,
            attestation_timeout=attestation_timeout,
        )
        if resolved:
            store.sync(state)
        logger.info("  Resolved %d in-transit transfer(s)", len(resolved))

    # Re-read balances: completions above may have minted USDC to satellite
    # Safes, which must be swept home along with the previously seen balances.
    if unrecorded_burns or in_transit:
        satellite_usdc, usdc_tokens = _read_satellite_balances(web3config, satellite_vaults)
        to_reclaim = plan_reclaims(satellite_usdc, min_reclaim)

    # Bridge reclaimable balances back to the master Safe.
    results: list[dict] = []
    for chain_id_value in to_reclaim:
        sat_vault = satellite_vaults[chain_id_value]
        sat_web3 = web3config.get_connection(ChainId(chain_id_value))
        usdc = usdc_tokens[chain_id_value]
        balance = satellite_usdc[chain_id_value]
        amount_raw = usdc.convert_to_raw(balance)
        slug = ChainId(chain_id_value).get_slug()

        logger.info("Bridging %s USDC from %s back to master Safe", balance, slug)
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

    # Verify the retrieve(s) landed on the master Safe.
    master_balance_after = master_usdc.fetch_balance_of(master_safe_address)
    reclaimed_total = master_balance_after - master_balance_before
    logger.info("Verifying retrieve on master Safe")
    logger.info("  Master Safe USDC before: %s", master_balance_before)
    logger.info("  Master Safe USDC after:  %s", master_balance_after)
    logger.info("  Net reclaimed:           %s USDC across %d bridge(s)", reclaimed_total, len(results))

    # Sweeping into the master Safe increases the on-chain reserve held by the
    # vault. Reflect that in the state file so accounting stays consistent.
    if results:
        sync_reserve_balance_to_state(store, master_usdc, master_balance_after)
        logger.info("  State file reserve updated: %s", state_path)

    logger.info("All ok")
