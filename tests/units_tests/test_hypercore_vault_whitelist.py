"""The Lagoon guard record must determine HyperCore strategy eligibility."""

import datetime
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tradeexecutor.cli.commands.lagoon_hypercore_vault_whitelist_status import get_unwhitelisted_hypercore_vaults
from tradeexecutor.ethereum.lagoon.hypercore_whitelist import load_hypercore_vault_whitelist
from tradeexecutor.ethereum.vault.hypercore_vault import create_hypercore_vault_pair
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradingstrategy.chain import ChainId
from tradingstrategy.exchange import ExchangeUniverse
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


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
    4. Confirm an older record without HyperCore fields gives an actionable error.
    5. Confirm a record for the wrong chain does not leak a bare ``KeyError``.
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

    # 4. Confirm an older record without HyperCore fields gives an actionable error.
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "module_address": "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84",
        "config": {},
    }}}))
    with pytest.raises(ValueError, match="has no HyperCore vault permissions"):
        load_hypercore_vault_whitelist(record)

    # 5. Confirm a record for the wrong chain does not leak a bare KeyError.
    record.write_text(json.dumps({"deployments": {"ethereum": {}}}))
    with pytest.raises(ValueError, match="must contain deployments.hyperliquid"):
        load_hypercore_vault_whitelist(record)


def test_position_manager_records_unwhitelisted_vault_once(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """A decision excludes an unlisted vault and records why only once.

    1. Write a restricted Lagoon record and construct real HyperCore pairs.
    2. Create a position manager with a recorder probe for this decision.
    3. Check permitted and rejected pairs, repeating the rejected check.
    4. Check the warning, observations, unrestricted default and wrong module.

    A recorder probe is used here because the DuckDB recorder lifecycle has
    separate integration coverage; this test targets PositionManager's calls.
    """
    # 1. Write a restricted Lagoon record and construct real HyperCore pairs.
    allowed = "0x0034cd90f5a6195a2e282282a9e551502b7b516c"
    rejected = "0xb8f43ee0513e53309b9d6d6a42dce8fdb694b04d"
    module = "0xf79d5540fA3a6ea738Aa21A562c5AD7224406F84"
    record = tmp_path / "deployment.json"
    record.write_text(json.dumps({"deployments": {"hyperliquid": {
        "module_address": module,
        "config": {"any_hypercore_vault": False, "hypercore_vaults": [allowed]},
    }}}))
    usdc = AssetIdentifier(999, "0x0000000000000000000000000000000000000002", "USDC", 6)
    allowed_pair = create_hypercore_vault_pair(usdc, allowed, internal_id=1)
    rejected_pair = create_hypercore_vault_pair(usdc, rejected, internal_id=2)
    universe = Universe(time_bucket=TimeBucket.d1, chains={ChainId.hypercore}, exchange_universe=ExchangeUniverse({}))

    # 2. Create a position manager with a recorder probe for this decision.
    recorder = MagicMock()
    manager = PositionManager(
        timestamp=datetime.datetime(2026, 9, 28),
        universe=universe,
        state=State(),
        pricing_model=object(),
        vault_record_file=record,
        expected_vault_guard_module_address=module,
        recorder=recorder,
    )

    # 3. Check permitted and rejected pairs, repeating the rejected check.
    assert manager.is_whitelisted_vault(allowed_pair)
    assert not manager.is_whitelisted_vault(rejected_pair)
    assert not manager.is_whitelisted_vault(rejected_pair)

    # 4. Check the warning and decision observations, then reject a wrong module.
    assert caplog.text.count("not in Lagoon guard record") == 1
    assert "strategy cycle 2026-09-28" in caplog.text
    assert any(record.levelno == logging.WARNING and "not in Lagoon guard record" in record.message for record in caplog.records)
    assert recorder.record.call_count == 2
    assert recorder.record.call_args_list[0].args[:2] == ("input", "hypercore_guard_whitelist")
    assert recorder.record.call_args_list[0].args[2]["vault_addresses"] == [allowed]
    assert recorder.record.call_args_list[1].args[:2] == ("admission", "hypercore_guard_whitelist_skip")
    assert recorder.record.call_args_list[1].args[2]["reason"] == "not_in_guard_record"
    assert recorder.record.call_args_list[1].kwargs["pair_key"] == rejected
    unrestricted_manager = PositionManager(
        timestamp=datetime.datetime(2026, 9, 28),
        universe=universe,
        state=State(),
        pricing_model=object(),
    )
    assert unrestricted_manager.is_whitelisted_vault(rejected_pair)
    with pytest.raises(ValueError, match="not configured module"):
        PositionManager(
            timestamp=datetime.datetime(2026, 9, 28),
            universe=universe,
            state=State(),
            pricing_model=object(),
            vault_record_file=record,
            expected_vault_guard_module_address="0x0000000000000000000000000000000000000001",
        )
