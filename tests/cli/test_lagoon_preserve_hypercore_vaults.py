"""Guard redeploys must retain vault permissions lost from the current universe."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tradeexecutor.cli.commands.lagoon_deploy_vault import _preserve_hypercore_vaults_from_record, _write_markdown_report


OLD_MODULE = "0x0000000000000000000000000000000000000001"
SAFE = "0x0000000000000000000000000000000000000002"
VAULT = "0x0000000000000000000000000000000000000003"
OLD_ONLY = "0x0000000000000000000000000000000000000004"
SHARED = "0x0000000000000000000000000000000000000005"
NEW_ONLY = "0x0000000000000000000000000000000000000006"


def test_preserve_previous_hypercore_vaults(tmp_path: Path):
    """A replacement keeps old permissions even when a vault falls below today's TVL screen.

    1. Write the old guard's two permitted vaults.
    2. Merge them with a new universe containing one shared and one new vault.
    3. Confirm that the replacement permits all three without duplicates.
    """
    # 1. Write the old guard's two permitted vaults.
    record = tmp_path / "guard.json"
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "vault_address": VAULT,
        "safe_address": SAFE,
        "module_address": OLD_MODULE,
        "config": {"any_hypercore_vault": False, "hypercore_vaults": [OLD_ONLY, SHARED]},
    }}}))

    # 2. Merge them with a new universe containing one shared and one new vault.
    config = SimpleNamespace(hypercore_vaults=[SHARED, NEW_ONLY])
    added = _preserve_hypercore_vaults_from_record(
        {"hyperliquid": config}, record,
        expected_old_guard_address=OLD_MODULE,
        existing_vault_address=VAULT,
        existing_safe_address=SAFE,
    )

    # 3. Confirm that the replacement permits all three without duplicates.
    assert added == 1
    assert {address.lower() for address in config.hypercore_vaults} == {OLD_ONLY.lower(), SHARED.lower(), NEW_ONLY.lower()}
    assert len(config.hypercore_vaults) == 3


def test_preserve_previous_hypercore_vaults_rejects_wrong_module(tmp_path: Path):
    """A stale record must not donate permissions to another Safe module.

    1. Write a guard record with the wrong module address.
    2. Attempt to merge it into the replacement config.
    3. Confirm that deployment preparation stops without changing the config.
    """
    # 1. Write a guard record with the wrong module address.
    record = tmp_path / "guard.json"
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "vault_address": VAULT,
        "safe_address": SAFE,
        "module_address": NEW_ONLY,
        "config": {"any_hypercore_vault": False, "hypercore_vaults": [OLD_ONLY]},
    }}}))

    # 2. Attempt to merge it into the replacement config.
    config = SimpleNamespace(hypercore_vaults=[SHARED])
    with pytest.raises(ValueError, match="not configured module"):
        _preserve_hypercore_vaults_from_record(
            {"hyperliquid": config}, record,
            expected_old_guard_address=OLD_MODULE,
            existing_vault_address=VAULT,
            existing_safe_address=SAFE,
        )

    # 3. Confirm that deployment preparation stops without changing the config.
    assert config.hypercore_vaults == [SHARED]


def test_simulated_guard_deploy_does_not_overwrite_report(tmp_path: Path):
    """A rehearsal must not replace a real deployment's shared Markdown report.

    1. Save a report from a previous real deployment.
    2. Run the simulated report writer and check the original is unchanged.
    3. Run the real writer and check the replacement is saved.
    """
    # 1. Save a report from a previous real deployment.
    report = tmp_path / "deployment-report.md"
    report.write_text("previous deployment")

    # 2. Run the simulated report writer and check the original is unchanged.
    _write_markdown_report(tmp_path / "guard.txt", "simulation", logging.getLogger(__name__), simulate=True)
    assert report.read_text() == "previous deployment"

    # 3. Run the real writer and check the replacement is saved.
    _write_markdown_report(tmp_path / "guard.txt", "new deployment", logging.getLogger(__name__), simulate=False)
    assert report.read_text() == "new deployment"
