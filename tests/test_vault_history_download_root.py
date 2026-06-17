"""Vault history download root resolution tests."""

from pathlib import Path
from types import SimpleNamespace

from tradeexecutor.strategy.trading_strategy_universe import (
    _resolve_vault_data_download_root,
    _resolve_vault_history_download_root,
)


def test_vault_history_download_root_uses_client_cache_path(tmp_path: Path) -> None:
    """Verify remote vault history defaults under the Trading Strategy client cache.

    1. Create a client-like object with a transport cache path.
    2. Resolve the vault history download root for website vault history.
    3. Check that vault downloads are placed under the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a transport cache path.
    # 2. Resolve the vault history download root for website vault history.
    resolved = _resolve_vault_history_download_root(
        client,
        "trading-strategy-website",
        None,
    )

    # 3. Check that vault downloads are placed under the client cache path.
    assert resolved == cache_path / "vaults" / "downloads"


def test_vault_data_download_root_uses_client_cache_path(tmp_path: Path) -> None:
    """Verify remote vault metadata defaults under the Trading Strategy client cache.

    1. Create a client-like object with a transport cache path.
    2. Resolve the generic vault data download root.
    3. Check that vault downloads are placed under the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a transport cache path.
    # 2. Resolve the generic vault data download root.
    resolved = _resolve_vault_data_download_root(
        client,
        None,
    )

    # 3. Check that vault downloads are placed under the client cache path.
    assert resolved == cache_path / "vaults" / "downloads"


def test_vault_history_download_root_keeps_explicit_override(tmp_path: Path) -> None:
    """Verify explicit vault history download roots stay authoritative.

    1. Create a client-like object with a different transport cache path.
    2. Resolve the vault history download root with an explicit override.
    3. Check that the explicit path wins over the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    explicit_download_root = tmp_path / "explicit-vault-downloads"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a different transport cache path.
    # 2. Resolve the vault history download root with an explicit override.
    resolved = _resolve_vault_history_download_root(
        client,
        "trading-strategy-website",
        explicit_download_root,
    )

    # 3. Check that the explicit path wins over the client cache path.
    assert resolved == explicit_download_root


def test_vault_data_download_root_keeps_explicit_override(tmp_path: Path) -> None:
    """Verify explicit vault metadata download roots stay authoritative.

    1. Create a client-like object with a different transport cache path.
    2. Resolve the generic vault data download root with an explicit override.
    3. Check that the explicit path wins over the client cache path.
    """
    cache_path = tmp_path / "client-cache"
    explicit_download_root = tmp_path / "explicit-vault-downloads"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=cache_path))

    # 1. Create a client-like object with a different transport cache path.
    # 2. Resolve the generic vault data download root with an explicit override.
    resolved = _resolve_vault_data_download_root(
        client,
        explicit_download_root,
    )

    # 3. Check that the explicit path wins over the client cache path.
    assert resolved == explicit_download_root


def test_vault_history_download_root_falls_back_for_mock_clients(tmp_path: Path) -> None:
    """Verify mock clients without usable transport cache paths do not crash.

    1. Resolve the root for a client-like object without transport.
    2. Resolve the root for a client-like object with a ``None`` cache path.
    3. Check that both return ``None`` for the lower-level default fallback.
    """
    client_without_transport = SimpleNamespace()
    client_without_cache_path = SimpleNamespace(transport=SimpleNamespace(cache_path=None))

    # 1. Resolve the root for a client-like object without transport.
    missing_transport_root = _resolve_vault_history_download_root(
        client_without_transport,
        "trading-strategy-website",
        None,
    )

    # 2. Resolve the root for a client-like object with a ``None`` cache path.
    missing_cache_path_root = _resolve_vault_history_download_root(
        client_without_cache_path,
        "trading-strategy-website",
        None,
    )

    # 3. Check that both return ``None`` for the lower-level default fallback.
    assert missing_transport_root is None
    assert missing_cache_path_root is None


def test_vault_data_download_root_falls_back_for_mock_clients(tmp_path: Path) -> None:
    """Verify mock clients without usable transport cache paths do not crash for metadata.

    1. Resolve the root for a client-like object without transport.
    2. Resolve the root for a client-like object with a ``None`` cache path.
    3. Check that both return ``None`` for the lower-level default fallback.
    """
    client_without_transport = SimpleNamespace()
    client_without_cache_path = SimpleNamespace(transport=SimpleNamespace(cache_path=None))

    # 1. Resolve the root for a client-like object without transport.
    missing_transport_root = _resolve_vault_data_download_root(
        client_without_transport,
        None,
    )

    # 2. Resolve the root for a client-like object with a ``None`` cache path.
    missing_cache_path_root = _resolve_vault_data_download_root(
        client_without_cache_path,
        None,
    )

    # 3. Check that both return ``None`` for the lower-level default fallback.
    assert missing_transport_root is None
    assert missing_cache_path_root is None


def test_vault_history_download_root_ignores_non_website_sources(tmp_path: Path) -> None:
    """Verify non-website vault history modes do not derive client cache paths.

    1. Create a client-like object with a transport cache path.
    2. Resolve the vault history download root for no remote vault history.
    3. Check that no client-dependent path is derived.
    """
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=tmp_path / "client-cache"))

    # 1. Create a client-like object with a transport cache path.
    # 2. Resolve the vault history download root for no remote vault history.
    resolved = _resolve_vault_history_download_root(
        client,
        "none",
        None,
    )

    # 3. Check that no client-dependent path is derived.
    assert resolved is None


def test_vault_history_download_root_keeps_bundled_override(tmp_path: Path) -> None:
    """Verify bundled vault history mode passes explicit paths through.

    1. Create a client-like object with a transport cache path.
    2. Resolve the vault history download root for bundled vault history.
    3. Check that the explicit path is returned unchanged.
    """
    explicit_download_root = tmp_path / "bundled-vault-downloads"
    client = SimpleNamespace(transport=SimpleNamespace(cache_path=tmp_path / "client-cache"))

    # 1. Create a client-like object with a transport cache path.
    # 2. Resolve the vault history download root for bundled vault history.
    resolved = _resolve_vault_history_download_root(
        client,
        "bundled",
        explicit_download_root,
    )

    # 3. Check that the explicit path is returned unchanged.
    assert resolved == explicit_download_root
