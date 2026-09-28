"""Show HyperCore vaults absent from a Lagoon guard deployment record.

Use this read-only command when a strategy discovers more vaults than its
deployed guard permits. The current vault metadata supplies names, TVL and
three-month performance; the deployment JSON supplies the permission set.
The report does not query the guard on-chain, so refresh the record after a
governance change before using it to decide which vaults to whitelist.

For Hyper-AI, run from the strategies deployment directory with::

    trade-executor lagoon-hypercore-vault-whitelist-status \
        --vault-record-file deploy/hyper-ai-v2-new-guard-vault-info.json
"""

import datetime
import json
from pathlib import Path

from tabulate import tabulate
from tradingstrategy.vault_data_client import VaultDataClient, VaultDataset
from typer import Option

from tradeexecutor.cli.bootstrap import prepare_cache
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.log import setup_logging
from tradeexecutor.ethereum.lagoon.hypercore_whitelist import HypercoreVaultWhitelist, load_hypercore_vault_whitelist


def get_unwhitelisted_hypercore_vaults(
    vaults: list[dict],
    whitelist: HypercoreVaultWhitelist,
) -> list[dict]:
    """Select current HyperCore metadata entries that the recorded guard rejects.

    Called by the CLI after downloading the metadata dataset. No curator risk,
    age or strategy-TVL screen is applied: this is a guard-coverage report,
    not an investable-universe recommendation.

    :param vaults: Raw vault metadata entries from the vault dataset.
    :param whitelist: Permissions read from the Lagoon deployment record.
    :return: Missing vaults ordered by descending current TVL.
    """
    missing = [
        vault
        for vault in vaults
        if vault.get("chain_id") == 9999
        and vault.get("address")
        and not whitelist.allows(vault["address"])
    ]
    return sorted(missing, key=lambda vault: float(vault.get("current_nav") or 0), reverse=True)


def _three_month_metrics(vault: dict) -> tuple[float | None, float | None]:
    """Read the current dataset's 3M gross CAGR and Sharpe for one report row.

    :param vault: Raw vault metadata entry.
    :return: Annualised CAGR as a ratio and Sharpe, or ``None`` if unavailable.
    """
    period = next((item for item in vault.get("period_results") or [] if item.get("period") == "3M"), None)
    if period is None:
        return None, None
    cagr = period.get("cagr_gross")
    sharpe = period.get("sharpe")
    return (float(cagr) if cagr is not None else None, float(sharpe) if sharpe is not None else None)


@app.command()
def lagoon_hypercore_vault_whitelist_status(
    vault_record_file: Path = Option(..., "--vault-record-file", envvar="VAULT_RECORD_FILE", help="Lagoon deployment JSON containing the recorded Hyperliquid guard configuration"),
    vault_pro_api_key: str = shared_options.vault_pro_api_key,
    cache_path: Path | None = shared_options.cache_path,
    log_level: str = shared_options.log_level,
):
    """List HyperCore vaults not permitted by the recorded Lagoon guard.

    Run from a deployment directory with ``--vault-record-file`` pointing to
    its current guard record. This downloads current vault metadata and prints
    every missing vault, sorted by TVL, including its address for governance.
    No transaction or executor state mutation is performed. The fresh metadata
    download updates the local dataset cache.
    """
    logger = setup_logging(log_level)
    whitelist = load_hypercore_vault_whitelist(vault_record_file)
    cache_root = prepare_cache("lagoon-hypercore-vault-whitelist-status", cache_path)
    # An operator status report should use the latest published metadata,
    # even when a cached copy exists from an earlier run.
    client = VaultDataClient(
        api_key=vault_pro_api_key,
        download_root=cache_root / "vaults",
        cache_expiry=datetime.timedelta(0),
    )
    metadata = json.loads(client.download(VaultDataset.vault_metadata).read_bytes())
    vaults = metadata["vaults"]
    missing = get_unwhitelisted_hypercore_vaults(vaults, whitelist)

    rows = []
    for vault in missing:
        cagr, sharpe = _three_month_metrics(vault)
        rows.append((
            vault.get("name") or "Unknown",
            f"${float(vault.get('current_nav') or 0):,.2f}",
            f"{cagr:.2%}" if cagr is not None else "—",
            f"{sharpe:.2f}" if sharpe is not None else "—",
            vault["address"],
        ))

    logger.info(
        "Vault metadata generated at %s; guard module %s; %d listed in record, %d missing of %d HyperCore vaults",
        metadata.get("generated_at"),
        whitelist.module_address,
        len(whitelist.vault_addresses),
        len(missing),
        sum(vault.get("chain_id") == 9999 for vault in vaults),
    )
    print(tabulate(rows, headers=("Name", "TVL", "Gross CAGR 3M", "Sharpe 3M", "Address"), tablefmt="simple"))
