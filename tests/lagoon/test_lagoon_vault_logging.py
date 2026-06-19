from types import SimpleNamespace

from tradeexecutor.ethereum.lagoon.vault import _get_position_vault_log_suffix


def test_position_vault_log_suffix_uses_pair_vault_name() -> None:
    """Test direct pair vault names are included in Lagoon freshness logs.

    1. Create a position-like object whose pair exposes a vault name.
    2. Build the vault log suffix.
    3. Check the suffix carries the vault name.
    """
    # 1. Create a position-like object whose pair exposes a vault name.
    pair = SimpleNamespace(
        get_vault_name=lambda: "Autopilot USDC Base",
        get_token_metadata=lambda: None,
    )
    position = SimpleNamespace(pair=pair)

    # 2. Build the vault log suffix.
    suffix = _get_position_vault_log_suffix(position)

    # 3. Check the suffix carries the vault name.
    assert suffix == ", vault=Autopilot USDC Base"


def test_position_vault_log_suffix_uses_metadata_vault_name() -> None:
    """Test metadata vault names are included in Lagoon freshness logs.

    1. Create a position-like object whose pair has no direct vault name.
    2. Build the vault log suffix from token metadata.
    3. Check the suffix carries the metadata vault name.
    """
    # 1. Create a position-like object whose pair has no direct vault name.
    metadata = SimpleNamespace(vault_name="IndeFi USDC")
    pair = SimpleNamespace(
        get_vault_name=lambda: None,
        get_token_metadata=lambda: metadata,
    )
    position = SimpleNamespace(pair=pair)

    # 2. Build the vault log suffix from token metadata.
    suffix = _get_position_vault_log_suffix(position)

    # 3. Check the suffix carries the metadata vault name.
    assert suffix == ", vault=IndeFi USDC"


def test_position_vault_log_suffix_is_empty_without_vault_name() -> None:
    """Test non-vault positions keep Lagoon freshness logs unchanged.

    1. Create a position-like object without vault name metadata.
    2. Build the vault log suffix.
    3. Check no suffix is added.
    """
    # 1. Create a position-like object without vault name metadata.
    pair = SimpleNamespace(
        get_vault_name=lambda: None,
        get_token_metadata=lambda: None,
    )
    position = SimpleNamespace(pair=pair)

    # 2. Build the vault log suffix.
    suffix = _get_position_vault_log_suffix(position)

    # 3. Check no suffix is added.
    assert suffix == ""
