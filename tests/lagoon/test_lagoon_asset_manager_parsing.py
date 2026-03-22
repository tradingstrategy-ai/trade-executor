import logging
import os
from types import SimpleNamespace

from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.cli.commands.lagoon_deploy_vault import _resolve_asset_managers


def test_resolve_asset_managers_supports_single_and_multiple_addresses():
    deployer_address = Web3.to_checksum_address("0x00000000000000000000000000000000000000dE")
    primary_address = "0x00000000000000000000000000000000000000a1"
    secondary_address = "0x00000000000000000000000000000000000000b2"
    hot_wallet = SimpleNamespace(address=deployer_address)

    assert _resolve_asset_managers(None, hot_wallet) == [deployer_address]
    assert _resolve_asset_managers("   ", hot_wallet) == [deployer_address]
    assert _resolve_asset_managers(primary_address, hot_wallet) == [Web3.to_checksum_address(primary_address)]
    assert _resolve_asset_managers(
        [primary_address, secondary_address],
        hot_wallet,
    ) == [
        Web3.to_checksum_address(primary_address),
        Web3.to_checksum_address(secondary_address),
    ]


def test_cli_strategy_file_parses_multiple_asset_managers_from_env(mocker, tmp_path):
    deployer_address = Web3.to_checksum_address("0x00000000000000000000000000000000000000dE")
    primary_address = "0x00000000000000000000000000000000000000a1"
    secondary_address = "0x00000000000000000000000000000000000000b2"
    captured: dict[str, object] = {}

    class DummyWeb3Config:
        connections = {"base": object()}

        def has_any_connection(self):
            return True

        def close(self):
            pass

    def fake_deploy_multichain(**kwargs):
        captured["asset_managers"] = kwargs["asset_managers"]
        captured["token_cache"] = kwargs["token_cache"]

    environment = {
        "PATH": os.environ["PATH"],
        "PRIVATE_KEY": "0x123",
        "JSON_RPC_BASE": "http://localhost:8545",
        "VAULT_RECORD_FILE": str(tmp_path / "vault-record.txt"),
        "STRATEGY_FILE": __file__,
        "ASSET_MANAGER": f"{primary_address}, {secondary_address}",
        "UNIT_TESTING": "true",
        "SIMULATE": "true",
        "LOG_LEVEL": "disabled",
    }

    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault.setup_logging", return_value=logging.getLogger("test"))
    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault.prepare_cache", return_value=tmp_path)
    token_cache = SimpleNamespace(filename=":memory:")
    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault.prepare_token_cache", return_value=token_cache)
    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault.create_web3_config", return_value=DummyWeb3Config())
    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault.create_hot_wallet", return_value=SimpleNamespace(address=deployer_address))
    mocker.patch("tradeexecutor.cli.commands.lagoon_deploy_vault._deploy_multichain", side_effect=fake_deploy_multichain)

    cli = get_command(app)
    mocker.patch.dict("os.environ", environment, clear=True)
    cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    assert captured["asset_managers"] == [
        Web3.to_checksum_address(primary_address),
        Web3.to_checksum_address(secondary_address),
    ]
    assert captured["token_cache"] is token_cache
