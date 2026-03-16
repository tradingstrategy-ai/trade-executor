from types import SimpleNamespace

from tradeexecutor.cli.commands.lagoon_deploy_vault import _augment_guard_only_artifacts


def _addr(value: int) -> str:
    return f"0x{value:040x}"


def test_guard_only_artifacts_include_safe_migration_instructions():
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

    text_payload, json_payload = _augment_guard_only_artifacts(
        deploy_info,
        vault_adapter_address=old_guard_address,
        text_payload="Deployment summary",
        json_payload={"Trading strategy module": new_guard_address},
    )

    assert "Guard migration instructions" in text_payload
    assert f"{safe_address}.disableModule(0x0000000000000000000000000000000000000001, {old_guard_address})" in text_payload
    assert f"{safe_address}.enableModule({new_guard_address})" in text_payload

    instructions = json_payload["Guard migration"]
    assert instructions["old_guard_address"] == old_guard_address
    assert instructions["new_guard_address"] == new_guard_address
    assert instructions["safe_address"] == safe_address
    assert instructions["vault_address"] == vault_address
    assert instructions["safe_transactions"][0]["function"] == "disableModule"
    assert instructions["safe_transactions"][1]["function"] == "enableModule"
