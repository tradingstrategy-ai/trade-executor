"""The Lagoon guard record must determine HyperCore strategy eligibility."""

import json
from pathlib import Path

import pytest

from tradeexecutor.cli.commands.lagoon_hypercore_vault_whitelist_status import get_unwhitelisted_hypercore_vaults
from tradeexecutor.ethereum.lagoon.hypercore_whitelist import load_hypercore_vault_whitelist


def test_hypercore_guard_record_filters_vaults_by_address(tmp_path: Path):
    """The report and strategy helper agree on the recorded guard permissions.

    1. Write a Lagoon deployment record with one permitted HyperCore vault.
    2. Load it and compare mixed-chain metadata against the permission set.
    3. Confirm the missing HyperCore vaults are ordered by current TVL.
    """
    # 1. Write a Lagoon deployment record with one permitted HyperCore vault.
    allowed = "0x0034cd90f5a6195a2e282282a9e551502b7b516c"
    missing_high = "0xb8f43ee0513e53309b9d6d6a42dce8fdb694b04d"
    missing_low = "0xa1b6d8efbcb2fb750a84dbc05649fa4968034f04"
    record = tmp_path / "deployment.json"
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "module_address": "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84",
        "config": {"any_hypercore_vault": False, "hypercore_vaults": [allowed]},
    }}}))

    # 2. Load it and compare mixed-chain metadata against the permission set.
    whitelist = load_hypercore_vault_whitelist(record)
    assert whitelist.allows(allowed.upper())
    assert not whitelist.allows(missing_high)
    vaults = [
        {"chain_id": 9999, "address": allowed, "current_nav": 500},
        {"chain_id": 9999, "address": missing_low, "current_nav": 100},
        {"chain_id": 1, "address": missing_high, "current_nav": 2000},
        {"chain_id": 9999, "address": missing_high, "current_nav": 1000},
    ]

    # 3. Confirm the missing HyperCore vaults are ordered by current TVL.
    missing = get_unwhitelisted_hypercore_vaults(vaults, whitelist)
    assert [vault["address"] for vault in missing] == [missing_high, missing_low]


def test_hypercore_guard_record_fails_closed_on_empty_allowlist(tmp_path: Path):
    """An empty or inconsistent restricted record must not admit every vault.

    1. Write a restricted deployment record without permitted vaults.
    2. Confirm loading raises before a strategy could build its universe.
    3. Confirm contradictory configured and recorded permissions also raise.
    """
    # 1. Write a restricted deployment record without permitted vaults.
    record = tmp_path / "deployment.json"
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "module_address": "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84",
        "config": {"any_hypercore_vault": False, "hypercore_vaults": []},
    }}}))

    # 2. Confirm loading raises before a strategy could build its universe.
    with pytest.raises(ValueError, match="allowlist is empty"):
        load_hypercore_vault_whitelist(record)

    # 3. Confirm contradictory configured and recorded permissions also raise.
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "module_address": "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84",
        "config": {"any_hypercore_vault": False, "hypercore_vaults": ["0x0034cd90f5a6195a2e282282a9e551502b7b516c"]},
        "whitelisted_items": [],
    }}}))
    with pytest.raises(ValueError, match="allowlists disagree"):
        load_hypercore_vault_whitelist(record)
