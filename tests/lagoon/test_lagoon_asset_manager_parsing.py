from types import SimpleNamespace

from web3 import Web3

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
    assert _resolve_asset_managers(
        [f" {primary_address} ", f"  {secondary_address}  "],
        hot_wallet,
    ) == [
        Web3.to_checksum_address(primary_address),
        Web3.to_checksum_address(secondary_address),
    ]
