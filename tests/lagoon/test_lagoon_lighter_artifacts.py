"""Secret-boundary tests for Lagoon Lighter deployment artifacts."""

import json
import logging
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
)
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX

from tradeexecutor.cli.commands.lagoon_deploy_vault import (
    _build_multichain_artifact_payload,
    _build_single_chain_artifact_payload,
    _remove_private_key_fields,
    _write_deployment_artifacts,
    _write_lighter_private_record,
)

#: Unique fake credential used to detect accidental disclosure in public artefacts.
SECRET = "lighter-private-key-DO-NOT-LOG-7f91"
#: Representative public account metadata used in serialisation fixtures.
LIGHTER_ACCOUNT_INDEX = 123
#: First valid Lighter delegated-key slot used in fixture metadata.
LIGHTER_API_KEY_INDEX = MIN_API_KEY_INDEX
#: Required owner-only permissions for the private operator record.
PRIVATE_RECORD_MODE = 0o600


def _deployment(secret: str = SECRET) -> SimpleNamespace:
    """Build a deployment double so artifact tests need no on-chain deployment."""
    public_setup = {
        "account_index": LIGHTER_ACCOUNT_INDEX,
        "api_key_index": LIGHTER_API_KEY_INDEX,
        "public_key": "0x" + "ab" * 40,
        "activation_amount": "1",
        "deposit_tx_hash": "0xdeposit",
        "change_pubkey_tx_hash": "0xpubkey",
        "observed_collateral": "1",
        "future_sdk_payload": {"credential": secret},
    }
    private_setup = {**public_setup, "private_key": secret}

    return SimpleNamespace(
        get_deployment_data=lambda: {
            "Safe": "0x0000000000000000000000000000000000000001",
            "Trading strategy module": "0x0000000000000000000000000000000000000002",
            "Lighter collateral": "1",
            "Lighter API-key index": LIGHTER_API_KEY_INDEX,
        },
        as_json_friendly_dict=lambda include_secrets=False: {
            "lighter_account_setup": private_setup if include_secrets else public_setup,
        },
    )


def test_single_chain_public_and_private_payloads_are_separate() -> None:
    """Keep the generated key exclusively in the operator payload.

    1. Build public and private payloads through the report boundary.
    2. Verify only the private payload contains the sentinel key.
    3. Verify public metadata remains available for diagnostics.
    """
    # 1. Build public and private payloads through the report boundary.
    deployment = _deployment()
    public = _build_single_chain_artifact_payload(deployment, include_secrets=False)
    private = _build_single_chain_artifact_payload(deployment, include_secrets=True)

    # 2. Verify only the private payload contains the sentinel key.
    assert SECRET not in json.dumps(public)
    assert private["lighter_account_setup"]["private_key"] == SECRET

    # 3. Verify public metadata remains available for diagnostics.
    assert public["lighter_account_setup"]["account_index"] == LIGHTER_ACCOUNT_INDEX
    assert public["lighter_account_setup"]["change_pubkey_tx_hash"] == "0xpubkey"
    assert "future_sdk_payload" not in public["lighter_account_setup"]
    assert public["Lighter API-key index"] == LIGHTER_API_KEY_INDEX


def test_public_lighter_generation_flag_is_not_mistaken_for_a_secret() -> None:
    """Keep the public feature flag while removing actual key material.

    1. Build a configuration payload containing the public flag and a private key.
    2. Pass it through the runtime artefact secret filter.
    3. Verify the feature flag remains and the credential is removed.
    """
    # 1. Build a configuration payload containing the public flag and a private key.
    payload = {
        "generate_lighter_api_key": True,
        "private_key": SECRET,
    }

    # 2. Pass it through the runtime artefact secret filter.
    public = _remove_private_key_fields(payload)

    # 3. The feature flag remains and the credential is removed.
    assert public == {"generate_lighter_api_key": True}


def test_operator_json_is_0600_and_exclusive(tmp_path: Path) -> None:
    """Write a private report safely and refuse a second generated key.

    1. Write a generated-key deployment record.
    2. Verify its JSON mode and secret location.
    3. Verify a second write fails without overwriting the first record.
    """
    # 1. Write a generated-key deployment record.
    record = tmp_path / "vault-record.txt"
    logger = logging.getLogger("test-lighter-artifacts")
    _write_deployment_artifacts(
        record,
        text_payload="public text",
        public_json_payload={"public": True},
        private_json_payload={"lighter_account_setup": {"private_key": SECRET}},
        simulate=False,
        logger=logger,
        include_private_key=True,
    )

    # 2. Verify its JSON mode and secret location.
    operator_json = record.with_suffix(".json")
    assert stat.S_IMODE(operator_json.stat().st_mode) == PRIVATE_RECORD_MODE
    assert json.loads(operator_json.read_text())["lighter_account_setup"]["private_key"] == SECRET

    # 3. Verify a second write fails without overwriting the first record.
    with pytest.raises(FileExistsError):
        _write_deployment_artifacts(
            record,
            text_payload="replacement",
            public_json_payload={},
            private_json_payload={"lighter_account_setup": {"private_key": "other"}},
            simulate=False,
            logger=logger,
            include_private_key=True,
        )
    assert record.read_text() == "public text"


def test_private_writer_redacts_key_from_human_text(tmp_path: Path) -> None:
    """Keep a caller-supplied human payload public at the writer boundary.

    1. Supply a private payload and accidentally include its key in text.
    2. Write the artifacts through the production helper.
    3. Verify only the mode-0600 JSON contains the key.
    """
    # 1. Supply a private payload and accidentally include its key in text.
    record = tmp_path / "vault-record.txt"

    # 2. Write the artifacts through the production helper.
    _write_deployment_artifacts(
        record,
        text_payload=f"accidental key: {SECRET}",
        public_json_payload={},
        private_json_payload={"deployments": {"ethereum": {"lighter_account_setup": {"private_key": SECRET}}}},
        simulate=False,
        logger=logging.getLogger("test-lighter-artifacts-redaction"),
        include_private_key=True,
    )

    # 3. Verify only the mode-0600 JSON contains the key.
    assert SECRET not in record.read_text()
    assert SECRET in json.loads(record.with_suffix(".json").read_text())["deployments"]["ethereum"]["lighter_account_setup"]["private_key"]


def test_public_operator_json_keeps_replacement_semantics(tmp_path: Path) -> None:
    """Replace a non-secret deployment record without retaining old bytes.

    1. Write a deliberately long public JSON payload.
    2. Replace it through the ordinary non-secret path with a shorter payload.
    3. Verify the resulting JSON is exactly the second payload and remains private.
    """
    # 1. Write a deliberately long public JSON payload.
    record = tmp_path / "vault-record.txt"
    logger = logging.getLogger("test-lighter-artifacts-public")
    _write_deployment_artifacts(
        record,
        text_payload="first",
        public_json_payload={"long": "x" * 200},
        simulate=False,
        logger=logger,
    )

    # 2. Replace it through the ordinary non-secret path with a shorter payload.
    _write_deployment_artifacts(
        record,
        text_payload="second",
        public_json_payload={"short": True},
        simulate=False,
        logger=logger,
    )

    # 3. Verify the resulting JSON is exactly the second payload and remains private.
    operator_json = record.with_suffix(".json")
    assert json.loads(operator_json.read_text()) == {"short": True}
    assert stat.S_IMODE(operator_json.stat().st_mode) == PRIVATE_RECORD_MODE


def test_multichain_private_record_survives_report_generation(tmp_path: Path) -> None:
    """Add key material only to the Ethereum source operator payload.

    1. Build a minimal multichain result with a Lighter source deployment.
    2. Persist the private record before building a public report.
    3. Verify report writing does not overwrite the generated key.
    """
    # 1. Build a minimal multichain result with a Lighter source deployment.
    deployment = _deployment()
    config = SimpleNamespace(
        parameters=LagoonDeploymentParameters(
            underlying="0x0000000000000000000000000000000000000001",
            name="Test",
            symbol="TST",
        ),
        safe_owners=[],
        safe_threshold=1,
        asset_manager=None,
        asset_managers=[],
        uniswap_v2=None,
        uniswap_v3=None,
        aave_v3=None,
        cowswap=False,
        velora=False,
        gmx_deployment=None,
        lighter_deployment=None,
        generate_lighter_api_key=True,
        lighter_api_key_index=LIGHTER_API_KEY_INDEX,
        cctp_deployment=None,
        any_asset=False,
        any_hypercore_vault=False,
        max_settlement_amount=None,
        settlement_cooldown=86400,
        etherscan_api_key=None,
        verifier="etherscan",
        verifier_url=None,
        use_forge=False,
        between_contracts_delay_seconds=5,
        erc_4626_vaults=None,
        guard_only=False,
        existing_vault_address=None,
        existing_safe_address=None,
        vault_abi="lagoon/Vault.json",
        factory_contract=True,
        from_the_scratch=False,
        hypercore_vaults=None,
        assets=None,
        safe_salt_nonce=1,
        safe_proxy_factory_address=None,
        forge_cache_dir=None,
        deploy_retries=1,
        satellite_chain=False,
    )
    result = SimpleNamespace(
        safe_address="0x0000000000000000000000000000000000000001",
        deployments={
            "ethereum": SimpleNamespace(
                vault=SimpleNamespace(address="0x0000000000000000000000000000000000000003"),
                safe_address="0x0000000000000000000000000000000000000001",
                trading_strategy_module=SimpleNamespace(address="0x0000000000000000000000000000000000000002"),
                asset_manager="0x0000000000000000000000000000000000000004",
                asset_managers=(),
                valuation_manager="0x0000000000000000000000000000000000000004",
                is_satellite=False,
                whitelisted_items=(),
                get_deployment_data=deployment.get_deployment_data,
                as_json_friendly_dict=deployment.as_json_friendly_dict,
            ),
        },
    )

    # 2. Persist the private record before building a public report.
    record = tmp_path / "vault-record.txt"
    _write_lighter_private_record(
        record,
        {
            "multichain": True,
            "deployments": {
                "ethereum": {
                    "safe_address": result.safe_address,
                    "lighter_account_setup": deployment.as_json_friendly_dict(True)["lighter_account_setup"],
                },
            },
        },
        logging.getLogger("test-lighter-artifacts-multichain"),
    )
    public_text, public = _build_multichain_artifact_payload(
        result,
        1,
        {"ethereum": config},
        "guard",
    )
    _write_deployment_artifacts(
        record,
        text_payload=public_text,
        public_json_payload=public,
        simulate=False,
        logger=logging.getLogger("test-lighter-artifacts-multichain"),
        write_json=False,
    )

    # 3. Verify report writing does not overwrite the generated key.
    assert SECRET not in public_text
    assert SECRET not in json.dumps(public)
    private = json.loads(record.with_suffix(".json").read_text())
    assert private["deployments"]["ethereum"]["lighter_account_setup"]["private_key"] == SECRET
