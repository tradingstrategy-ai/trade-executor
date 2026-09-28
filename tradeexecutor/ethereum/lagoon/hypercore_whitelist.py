"""Read the HyperCore vault permissions recorded with a Lagoon deployment.

HyperCore vault discovery is independent of the Lagoon guard. A vault can be
open at Hyperliquid yet fail the guard's ``vaultTransfer`` check, after USDC has
already reached the Safe's HyperCore perp account. Strategies use this record
to exclude such vaults before ranking; the whitelist-status CLI uses the same
interpretation when reporting candidates for a governance update.

The deployment record is a configuration snapshot, not an on-chain query.
Operators must update it when they change the deployed guard permissions.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from eth_utils import is_address


@dataclass(frozen=True)
class HypercoreVaultWhitelist:
    """Vault-transfer permissions from one Hyperliquid Lagoon guard record.

    :param module_address: Module this configuration was deployed for.
    :param vault_addresses: Explicitly permitted HyperCore vault addresses.
    :param any_hypercore_vault: Whether the guard permits every vault address.
    """

    module_address: str
    vault_addresses: frozenset[str]
    any_hypercore_vault: bool

    def allows(self, vault_address: str) -> bool:
        """Apply the recorded guard policy during universe filtering and reporting.

        :param vault_address: HyperCore vault address to check.
        :return: Whether a ``vaultTransfer`` to this vault is permitted.
        """
        return self.any_hypercore_vault or vault_address.lower() in self.vault_addresses


def load_hypercore_vault_whitelist(record_file: Path) -> HypercoreVaultWhitelist:
    """Load the guard's HyperCore permissions from a Lagoon deployment JSON file.

    Called at strategy universe construction and by
    ``lagoon-hypercore-vault-whitelist-status``. A missing or malformed record
    raises instead of silently admitting a vault the guard might reject.

    :param record_file: JSON record written by ``lagoon-deploy-vault``.
    :return: Hyperliquid guard module identity and vault-transfer permissions.
    """
    deployment = json.loads(record_file.read_text())["deployments"]["hyperliquid"]
    config = deployment["config"]
    if "hypercore_vaults" not in config or "any_hypercore_vault" not in config:
        raise ValueError(f"Lagoon deployment record {record_file} has no HyperCore vault permissions; use a current guard record")
    module_address = deployment["module_address"]
    if not is_address(module_address):
        raise ValueError(f"Invalid Hyperliquid guard module address in {record_file}: {module_address}")

    addresses = config["hypercore_vaults"]
    if not isinstance(addresses, list) or any(not isinstance(address, str) or not is_address(address) for address in addresses):
        raise ValueError(f"Invalid HyperCore vault allowlist in {record_file}")
    configured_addresses = frozenset(address.lower() for address in addresses)
    if "whitelisted_items" in deployment:
        deployed_addresses = frozenset(
            item["address"].lower()
            for item in deployment["whitelisted_items"]
            if item.get("kind") == "Hypercore vault"
        )
        if deployed_addresses != configured_addresses:
            raise ValueError(f"Configured and recorded HyperCore vault allowlists disagree in {record_file}")
    any_hypercore_vault = config["any_hypercore_vault"]
    if not isinstance(any_hypercore_vault, bool):
        raise ValueError(f"Invalid any_hypercore_vault flag in {record_file}")
    if not any_hypercore_vault and not addresses:
        raise ValueError(f"HyperCore vault allowlist is empty in {record_file}")

    return HypercoreVaultWhitelist(
        module_address=module_address.lower(),
        vault_addresses=configured_addresses,
        any_hypercore_vault=any_hypercore_vault,
    )
