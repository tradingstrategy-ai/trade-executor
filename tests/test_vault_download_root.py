"""Vault download root resolution tests."""

from pathlib import Path
from types import SimpleNamespace

from tradeexecutor.strategy.trading_strategy_universe import _resolve_vault_download_root


def test_vault_download_root_uses_client_cache_path(tmp_path: Path) -> None:
    """Verify remote vault data defaults under the Trading Strategy client cache.

    1. Create a client-like object with a transport cache path.
    2. Resolve the vault download root without an explicit override.
    3. Check that vault downloads are placed under the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a transport cache path.
    # 2. Resolve the vault download root without an explicit override.
    resolved = _resolve_vault_download_root(
        client,
        None,
    )

    # 3. Check that vault downloads are placed under the client cache path.
    assert resolved == cache_path / "vaults" / "downloads"


def test_vault_download_root_keeps_explicit_override(tmp_path: Path) -> None:
    """Verify explicit vault download roots stay authoritative.

    1. Create a client-like object with a different transport cache path.
    2. Resolve the vault download root with an explicit override.
    3. Check that the explicit path wins over the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    explicit_download_root = tmp_path / "explicit-vault-downloads"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a different transport cache path.
    # 2. Resolve the vault download root with an explicit override.
    resolved = _resolve_vault_download_root(
        client,
        explicit_download_root,
    )

    # 3. Check that the explicit path wins over the client cache path.
    assert resolved == explicit_download_root


def test_vault_download_root_falls_back_for_mock_clients() -> None:
    """Verify mock clients without usable transport cache paths do not crash.

    1. Resolve the root for a client-like object without transport.
    2. Resolve the root for a client-like object with a ``None`` cache path.
    3. Check that both return ``None`` for the lower-level default fallback.
    """
    client_without_transport = SimpleNamespace()
    client_without_cache_path = SimpleNamespace(transport=SimpleNamespace(cache_path=None))

    # 1. Resolve the root for a client-like object without transport.
    missing_transport_root = _resolve_vault_download_root(
        client_without_transport,
        None,
    )

    # 2. Resolve the root for a client-like object with a ``None`` cache path.
    missing_cache_path_root = _resolve_vault_download_root(
        client_without_cache_path,
        None,
    )

    # 3. Check that both return ``None`` for the lower-level default fallback.
    assert missing_transport_root is None
    assert missing_cache_path_root is None
