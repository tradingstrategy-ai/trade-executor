import logging
from types import SimpleNamespace

from tradeexecutor.cli.commands.lagoon_deploy_vault import _resolve_multisig_configuration


def test_resolve_multisig_configuration_reads_existing_safe_when_redeploying(monkeypatch):
    existing_owners = [
        "0x1000000000000000000000000000000000000001",
        "0x2000000000000000000000000000000000000002",
        "0x3000000000000000000000000000000000000003",
    ]

    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.fetch_safe_deployment",
        lambda web3, safe_address: SimpleNamespace(
            retrieve_owners=lambda: existing_owners,
            retrieve_threshold=lambda: 2,
        ),
    )

    owners, threshold = _resolve_multisig_configuration(
        multisig_owners=None,
        hot_wallet=SimpleNamespace(address="0x9000000000000000000000000000000000000009"),
        web3=object(),
        guard_only=True,
        existing_safe_address="0x4000000000000000000000000000000000000004",
        logger=logging.getLogger("test"),
    )

    assert owners == existing_owners
    assert threshold == 2


def test_resolve_multisig_configuration_keeps_explicit_owners(monkeypatch):
    monkeypatch.setattr(
        "tradeexecutor.cli.commands.lagoon_deploy_vault.fetch_safe_deployment",
        lambda web3, safe_address: (_ for _ in ()).throw(AssertionError("should not fetch Safe")),
    )

    owners, threshold = _resolve_multisig_configuration(
        multisig_owners=[
            "0x1000000000000000000000000000000000000001",
            "0x2000000000000000000000000000000000000002",
            "0x3000000000000000000000000000000000000003",
        ],
        hot_wallet=SimpleNamespace(address="0x9000000000000000000000000000000000000009"),
        web3=object(),
        guard_only=True,
        existing_safe_address="0x4000000000000000000000000000000000000004",
        logger=logging.getLogger("test"),
    )

    assert owners == [
        "0x1000000000000000000000000000000000000001",
        "0x2000000000000000000000000000000000000002",
        "0x3000000000000000000000000000000000000003",
    ]
    assert threshold == 2
