from types import SimpleNamespace

from tradeexecutor.cli.commands.lagoon_deploy_vault import _augment_guard_only_artifacts


def _addr(value: int) -> str:
    return f"0x{value:040x}"


def test_guard_only_artifacts_include_safe_migration_instructions():
    """Keep manual calls and record the Safe proposal for a guard redeploy.

    1. Arrange a replacement guard and its existing Safe module.
    2. Build the single-chain deployment artefacts with a submitted proposal.
    3. Verify the manual fallback and proposal status are both present.
    """
    # 1. Arrange a replacement guard and its existing Safe module.
    safe_address = _addr(1)
    old_guard_address = _addr(2)
    new_guard_address = _addr(3)
    vault_address = _addr(4)

    deploy_info = SimpleNamespace(
        safe=SimpleNamespace(
            address=safe_address,
            retrieve_modules=lambda: [old_guard_address],
        ),
        trading_strategy_module=SimpleNamespace(address=new_guard_address),
        old_trading_strategy_module=SimpleNamespace(address=old_guard_address),
        vault=SimpleNamespace(address=vault_address),
    )

    # 2. Build the single-chain deployment artefacts with a submitted proposal.
    text_payload, json_payload = _augment_guard_only_artifacts(
        deploy_info,
        text_payload="Deployment summary",
        json_payload={"Trading strategy module": new_guard_address},
        safe_proposal={"status": "submitted", "safe_tx_hash": "0x" + "ab" * 32, "url": "https://safe.invalid/tx"},
    )

    # 3. Verify the manual fallback and proposal status are both present.
    assert "Guard migration instructions" in text_payload
    assert "Safe proposal status: submitted" in text_payload
    assert f"{safe_address}.disableModule(0x0000000000000000000000000000000000000001, {old_guard_address})" in text_payload
    assert f"{safe_address}.enableModule({new_guard_address})" in text_payload

    instructions = json_payload["Guard migration"]
    assert instructions["old_guard_address"] == old_guard_address
    assert instructions["new_guard_address"] == new_guard_address
    assert instructions["safe_address"] == safe_address
    assert instructions["vault_address"] == vault_address
    assert instructions["enabled_modules_at_deployment"] == [old_guard_address]
    assert instructions["proposed_safe_transactions"][0]["function"] == "disableModule"
    assert instructions["proposed_safe_transactions"][1]["function"] == "enableModule"
    assert instructions["safe_proposal"]["safe_tx_hash"] == "0x" + "ab" * 32
