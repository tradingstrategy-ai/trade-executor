"""Lagoon helpers for trade-executor tests."""

from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import Contract


# Lagoon v0.5 uses OpenZeppelin v5 PausableUpgradeable. The pause flag lives in
# the ERC-7201 namespace:
# keccak256(abi.encode(uint256(keccak256("openzeppelin.storage.Pausable")) - 1)) & ~bytes32(uint256(0xff))
LAGOON_VAULT_PAUSABLE_STORAGE_SLOT = "0xcd5ed15c6e187e77e9aee88184c21f4f2182ab5827cb3b7e07fbedcd63f03300"


def set_lagoon_vault_paused_storage_for_testing(
    web3: Web3,
    vault_contract: Contract,
    paused: bool,
) -> bool:
    """Toggle Lagoon's pause gate by mutating Anvil storage directly.

    This helper is intentionally Anvil-only. It lets live CLI blackbox tests
    drive per-cycle Lagoon availability through real EVM reads while avoiding
    extra Safe pause/unpause transactions between every strategy tick.

    :return:
        ``True`` when storage changed, otherwise ``False``.
    """
    currently_paused = bool(vault_contract.functions.paused().call())
    if currently_paused == paused:
        return False

    value = "0x" + int(paused).to_bytes(32, byteorder="big").hex()
    response = web3.provider.make_request(
        "anvil_setStorageAt",
        [
            vault_contract.address,
            LAGOON_VAULT_PAUSABLE_STORAGE_SLOT,
            value,
        ],
    )
    if response.get("error"):
        raise RuntimeError(f"Could not mutate Lagoon paused storage: {response['error']}")

    updated_paused = bool(vault_contract.functions.paused().call())
    assert updated_paused == paused, f"Lagoon paused storage write failed, expected {paused}, got {updated_paused}"
    return True


def set_lagoon_vault_open_for_testing(
    web3: Web3,
    vault_contract: Contract,
    safe_address: HexAddress,
    open_: bool,
    *,
    gas: int = 150_000,
) -> HexBytes | None:
    """Toggle Lagoon's pause gate from an Anvil-impersonated Safe.

    Lagoon exposes its request gate as ``paused()``. Tests can use this helper
    to make the live strategy pricing checks see real EVM state instead of a
    mocked pricing model.

    Use :py:func:`set_lagoon_vault_paused_storage_for_testing` when a test needs
    direct Anvil state mutation instead of controller transactions.

    :return:
        Transaction hash when the state changed, otherwise ``None``.
    """
    currently_paused = bool(vault_contract.functions.paused().call())
    target_paused = not open_
    if currently_paused == target_paused:
        return None

    web3.provider.make_request("anvil_setBalance", [safe_address, hex(5 * 10**18)])
    web3.provider.make_request("anvil_impersonateAccount", [safe_address])
    if target_paused:
        tx_hash = vault_contract.functions.pause().transact({"from": safe_address, "gas": gas})
    else:
        tx_hash = vault_contract.functions.unpause().transact({"from": safe_address, "gas": gas})
    return tx_hash
