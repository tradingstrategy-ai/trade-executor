"""Lagoon helpers for trade-executor tests."""

from eth_typing import HexAddress
from hexbytes import HexBytes
from web3 import Web3
from web3.contract.contract import Contract


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
