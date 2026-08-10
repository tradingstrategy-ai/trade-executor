"""Whitelist-gated vault analysis tests."""

from types import SimpleNamespace

from tradeexecutor.analysis.vault import (
    BLACKLISTED_VAULT_DIAGNOSTIC_FLAG,
    WHITELISTED_VAULT_DIAGNOSTIC_FLAG,
    build_blacklisted_vault_dataframe,
    build_whitelisted_vault_dataframe,
    mark_blacklisted_vaults_ignored,
    mark_whitelisted_vaults_ignored,
    render_blacklisted_vaults,
    render_whitelisted_vaults,
)
from tradeexecutor.state.identifier import (
    IGNORE_REASON_BLACKLISTED_VAULT,
    IGNORE_REASON_LACKS_FEE_DATA,
    IGNORE_REASON_REQUIRES_WHITELIST,
)
from tradeexecutor.strategy.vault_fee_data import mark_missing_fee_vaults_ignored
from tradingstrategy.chain import ChainId
from tradingstrategy.vault import VaultDepositPermission


class _VaultPair:
    """Minimal vault pair used to exercise whitelist universe diagnostics."""

    def __init__(self, name: str, address: str, permission: VaultDepositPermission, ignore_reason: str | None = None, risk_level: str | None = None) -> None:
        self.base = SimpleNamespace(token_symbol=name)
        self.chain_id = ChainId.ethereum.value
        self.pool_address = address
        self.other_data = {}
        self._metadata = SimpleNamespace(
            deposit_permission=permission,
            protocol_slug="test-protocol",
            risk_level=risk_level,
            fee_internalised=None,
            management_fee=None,
            performance_fee=None,
            deposit_fee=None,
            withdrawal_fee=None,
        )
        self._ignore_reason = ignore_reason

    def is_vault(self) -> bool:
        """Identify the test fixture as a vault."""
        return True

    def get_vault_metadata(self):
        """Return fixture vault metadata."""
        return self._metadata

    def get_ignore_reason(self) -> str | None:
        """Return the current data-only reason."""
        return self._ignore_reason

    def set_ignore_reason(self, reason: str) -> None:
        """Persist an ignore reason like a real trading pair."""
        self._ignore_reason = reason

    def get_vault_name(self) -> str:
        """Return the fixture vault name."""
        return self.base.token_symbol

    def get_vault_protocol(self) -> str:
        """Return the fixture protocol."""
        return self._metadata.protocol_slug

    def get_vault_risk_level(self) -> str | None:
        """Return producer risk metadata."""
        return self._metadata.risk_level


class _Universe:
    """Minimal strategy universe fixture."""

    def __init__(self, pairs: list[_VaultPair]) -> None:
        self._pairs = pairs

    def iterate_pairs(self):
        """Yield fixture pairs in universe order."""
        yield from self._pairs


def test_whitelisted_vaults_are_ignored_and_rendered_for_diagnostics() -> None:
    """Whitelist metadata excludes allocation while preserving diagnostics.

    1. Create a permissioned vault, a fee-invalid permissioned vault and a public vault.
    2. Mark permissioned vaults as data-only and retain an existing fee reason.
    3. Check the diagnostics table and its Trading Strategy links.
    """
    # 1. Create fixture vaults with both new and existing ignore reasons.
    permissioned = _VaultPair(
        "Allow-listed vault",
        "0x1111111111111111111111111111111111111111",
        VaultDepositPermission.whitelisted,
    )
    fee_invalid_permissioned = _VaultPair(
        "Fee-invalid allow-listed vault",
        "0x2222222222222222222222222222222222222222",
        VaultDepositPermission.whitelisted,
        IGNORE_REASON_LACKS_FEE_DATA,
    )
    public = _VaultPair(
        "Public vault",
        "0x3333333333333333333333333333333333333333",
        VaultDepositPermission.permissionless,
    )
    universe = _Universe([permissioned, fee_invalid_permissioned, public])

    # 2. Exclude permissioned vaults, retaining the independent fee diagnostic.
    flagged = mark_whitelisted_vaults_ignored(universe)
    assert flagged == [permissioned, fee_invalid_permissioned]
    assert permissioned.get_ignore_reason() == IGNORE_REASON_REQUIRES_WHITELIST
    assert fee_invalid_permissioned.get_ignore_reason() == IGNORE_REASON_LACKS_FEE_DATA
    assert permissioned.other_data[WHITELISTED_VAULT_DIAGNOSTIC_FLAG] is True
    assert fee_invalid_permissioned.other_data[WHITELISTED_VAULT_DIAGNOSTIC_FLAG] is True
    assert WHITELISTED_VAULT_DIAGNOSTIC_FLAG not in public.other_data

    # 3. Display only allocation-blocked vaults with a page link per address.
    table = build_whitelisted_vault_dataframe(universe)
    assert table.to_dict("records") == [
        {
            "Name": "Allow-listed vault",
            "Protocol": "test-protocol",
            "Chain": "Ethereum",
            "Address": "0x1111111111111111111111111111111111111111",
        },
        {
            "Name": "Fee-invalid allow-listed vault",
            "Protocol": "test-protocol",
            "Chain": "Ethereum",
            "Address": "0x2222222222222222222222222222222222222222",
        },
    ]
    rendered = render_whitelisted_vaults(universe).to_html()
    assert "https://tradingstrategy.ai/trading-view/vaults/address/0x1111111111111111111111111111111111111111" in rendered


def test_blacklisted_vaults_are_ignored_and_rendered_for_diagnostics() -> None:
    """Blacklisted producer metadata blocks allocation but remains inspectable."""
    blacklisted = _VaultPair(
        "Blacklisted USDC vault",
        "0x4444444444444444444444444444444444444444",
        VaultDepositPermission.permissionless,
        risk_level="Blacklisted",
    )
    fee_invalid_blacklisted = _VaultPair(
        "Fee-invalid blacklisted vault",
        "0x5555555555555555555555555555555555555555",
        VaultDepositPermission.permissionless,
        ignore_reason=IGNORE_REASON_LACKS_FEE_DATA,
        risk_level="blacklisted",
    )
    public = _VaultPair(
        "Public vault",
        "0x6666666666666666666666666666666666666666",
        VaultDepositPermission.permissionless,
        risk_level="Minimal",
    )
    universe = _Universe([blacklisted, fee_invalid_blacklisted, public])

    flagged = mark_blacklisted_vaults_ignored(universe)
    mark_missing_fee_vaults_ignored(universe)

    assert flagged == [blacklisted, fee_invalid_blacklisted]
    assert blacklisted.get_ignore_reason() == IGNORE_REASON_BLACKLISTED_VAULT
    assert fee_invalid_blacklisted.get_ignore_reason() == IGNORE_REASON_LACKS_FEE_DATA
    assert blacklisted.other_data[BLACKLISTED_VAULT_DIAGNOSTIC_FLAG] is True
    assert fee_invalid_blacklisted.other_data[BLACKLISTED_VAULT_DIAGNOSTIC_FLAG] is True
    assert BLACKLISTED_VAULT_DIAGNOSTIC_FLAG not in public.other_data

    table = build_blacklisted_vault_dataframe(universe)
    assert table["Name"].tolist() == ["Blacklisted USDC vault", "Fee-invalid blacklisted vault"]
    assert table["Risk"].tolist() == ["Blacklisted", "blacklisted"]
    rendered = render_blacklisted_vaults(universe).to_html()
    assert "https://tradingstrategy.ai/trading-view/vaults/address/0x4444444444444444444444444444444444444444" in rendered
