"""Retry a Lagoon guard Safe proposal from its saved deployment record."""

import json
import re
from pathlib import Path

import typer
from typer import Option
from web3 import Web3

from tradeexecutor.cli.bootstrap import create_web3_config
from tradeexecutor.cli.commands import shared_options
from tradeexecutor.cli.commands.app import app
from tradeexecutor.cli.commands.lagoon_deploy_vault import _resolve_deployment_artifact_paths, _write_private_json_file
from tradeexecutor.ethereum.lagoon.guard_proposal import prepare_guard_proposal, submit_guard_migration


def _update_text_proposal(path: Path, slug: str, proposal: dict, *, multichain: bool) -> None:
    """Keep the human record's current proposal status in step with its JSON."""
    text = path.read_text()
    start = text.find(f"Chain: {slug}\n") if multichain else 0
    if start < 0:
        path.write_text(text.rstrip() + f"\nSafe proposal for {slug}: submitted\nSafe transaction: {proposal['url']}\n")
        return
    next_chain = text.find("\nChain: ", start + 1) if multichain else -1
    report = text.find("\nGuard report\n", start + 1) if multichain else -1
    end = min((position for position in (next_chain, report) if position >= 0), default=len(text))
    section = text[start:end]
    section, count = re.subn(
        r"(?m)^([ \t]*)Safe proposal status: pending$",
        lambda match: f"{match.group(1)}Safe proposal status: submitted\n{match.group(1)}Safe transaction: {proposal['url']}",
        section,
        count=1,
    )
    if not count:
        indent = "    " if multichain else "  "
        section = section.rstrip() + f"\n{indent}Safe proposal status: submitted\n{indent}Safe transaction: {proposal['url']}\n"
    path.write_text(text[:start] + section + text[end:])


def resubmit_guard_migration(
    deployment_file: Path,
    chain_web3: dict[str, Web3],
    private_key: str,
    *,
    chain: str | None = None,
    safe_api_key: str | None = None,
) -> dict[str, dict]:
    """Submit missing proposals from a saved guard deployment without redeploying."""
    text_path, json_path = _resolve_deployment_artifact_paths(deployment_file)
    assert json_path is not None
    record = json.loads(json_path.read_text())
    if record.get("multichain"):
        migrations = {
            slug: dep["guard_migration"]
            for slug, dep in record["deployments"].items()
            if dep.get("guard_migration")
        }
    else:
        if len(chain_web3) != 1:
            raise ValueError("A single-chain deployment record requires exactly one configured JSON-RPC connection")
        migration = record.get("Guard migration")
        migrations = {next(iter(chain_web3)): migration} if migration else {}
    if not migrations:
        raise ValueError(f"No guard migration found in {json_path}")
    if chain:
        if chain not in migrations:
            raise ValueError(f"No guard migration for chain {chain} in {json_path}")
        migrations = {chain: migrations[chain]}
    results = {}
    errors = []
    for slug, migration in migrations.items():
        proposal = migration.get("safe_proposal") or {}
        if proposal.get("status") == "submitted":
            if not proposal.get("safe_tx_hash") or not proposal.get("url"):
                errors.append(f"{slug}: submitted proposal is missing its hash or URL in the deployment record")
                continue
            results[slug] = proposal
            continue
        if slug not in chain_web3:
            errors.append(f"{slug}: no JSON-RPC connection configured")
            continue
        try:
            context = prepare_guard_proposal(
                chain_web3[slug],
                migration["safe_address"],
                private_key,
                expected_old_guard_address=migration["old_guard_address"],
            )
            result = submit_guard_migration(
                context,
                migration["new_guard_address"],
                private_key,
                safe_api_key=safe_api_key,
                expected_proposal=proposal,
            )
        except Exception as exc:
            errors.append(f"{slug}: {exc}")
            continue
        migration["safe_proposal"] = result
        _write_private_json_file(json_path, record, indent=2, exclusive=False)
        if text_path and text_path.exists():
            _update_text_proposal(text_path, slug, result, multichain=bool(record.get("multichain")))
        results[slug] = result
    if errors:
        raise RuntimeError("Guard proposal submission incomplete: " + "; ".join(errors) + f". Deployment record: {json_path}")
    return results


@app.command()
@shared_options.with_json_rpc_options()
def lagoon_propose_guard_migration(
    deployment_file: Path = Option(..., "--deployment-file", envvar="VAULT_RECORD_FILE", help="Guard deployment record (TXT or JSON)."),
    chain: str | None = Option(None, "--chain", help="Only retry this chain in a multichain record."),
    private_key: str = shared_options.private_key,
    safe_transaction_service_api_key: str | None = Option(None, envvar="SAFE_TRANSACTION_SERVICE_API_KEY", help="Optional Safe Transaction Service API key."),
    rpc_kwargs: dict | None = None,
):
    """Submit a pending guard migration proposal without deploying another guard."""
    if not private_key:
        raise ValueError("PRIVATE_KEY is required to sign a Safe proposal")
    web3config = create_web3_config(**rpc_kwargs)
    if not web3config.has_any_connection():
        raise ValueError("Pass a JSON-RPC connection for the guard migration chain")
    try:
        chain_web3 = {chain_id.get_slug(): web3 for chain_id, web3 in web3config.connections.items()}
        results = resubmit_guard_migration(
            deployment_file,
            chain_web3,
            private_key,
            chain=chain,
            safe_api_key=safe_transaction_service_api_key,
        )
        for slug, proposal in results.items():
            typer.echo(f"{slug}: {proposal['safe_tx_hash']}\nSafe transaction: {proposal['url']}\nOwner execution is still required.")
    finally:
        web3config.close()
