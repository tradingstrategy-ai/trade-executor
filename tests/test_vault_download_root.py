"""Vault download root resolution tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from tradingstrategy.vault_data_client import VaultDataAccessDenied

from tradeexecutor.strategy import trading_strategy_universe
from tradeexecutor.strategy.trading_strategy_universe import (
    _resolve_vault_download_root,
    refresh_vault_universe_metadata_cache,
)


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


def test_refresh_vault_universe_metadata_cache_replaces_only_metadata_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify repair startup can force a fresh vault metadata download.

    1. Create a client-like object with cached vault metadata and another unrelated file.
    2. Refresh the vault universe metadata cache.
    3. Check the vault metadata file was replaced while unrelated cache data remains.
    """
    cache_path = tmp_path / "client-cache"
    vault_download_root = cache_path / "vaults" / "downloads"
    vault_download_root.mkdir(parents=True)
    vault_universe_file = vault_download_root / "vault-metadata.json"
    unrelated_file = vault_download_root / "vault-price-history.parquet"
    vault_universe_file.write_text("old")
    unrelated_file.write_text("keep")

    def _fetch_vault_universe() -> None:
        (vault_download_root / "vault-metadata.json").write_text("fresh")

    client = SimpleNamespace(
        transport=SimpleNamespace(cache_path=cache_path),
        has_vault_data_access=lambda: True,
    )
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: SimpleNamespace(fetch_vault_universe=_fetch_vault_universe),
    )

    # 1. Create a client-like object with cached vault metadata and another unrelated file.
    # 2. Refresh the vault universe metadata cache.
    refreshed_path = refresh_vault_universe_metadata_cache(client)

    # 3. Check the vault metadata file was replaced while unrelated cache data remains.
    assert refreshed_path == vault_universe_file
    assert vault_universe_file.read_text() == "fresh"
    assert unrelated_file.exists()


def test_refresh_vault_universe_metadata_cache_restores_stale_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify repair startup keeps stale vault metadata if refresh fails.

    1. Create a client-like object with cached vault metadata.
    2. Refresh the vault universe metadata cache with a failing downloader.
    3. Check the stale vault metadata file was restored for fallback startup.
    """
    cache_path = tmp_path / "client-cache"
    vault_download_root = cache_path / "vaults" / "downloads"
    vault_download_root.mkdir(parents=True)
    vault_universe_file = vault_download_root / "vault-metadata.json"
    vault_universe_file.write_text("old")

    def _fetch_vault_universe() -> None:
        raise RuntimeError("Download failed")

    client = SimpleNamespace(
        transport=SimpleNamespace(cache_path=cache_path),
        has_vault_data_access=lambda: True,
    )
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: SimpleNamespace(fetch_vault_universe=_fetch_vault_universe),
    )

    # 1. Create a client-like object with cached vault metadata.
    # 2. Refresh the vault universe metadata cache with a failing downloader.
    restored_path = refresh_vault_universe_metadata_cache(client)

    # 3. Check the stale vault metadata file was restored for fallback startup.
    assert restored_path == vault_universe_file
    assert vault_universe_file.read_text() == "old"


def test_refresh_vault_universe_metadata_cache_surfaces_rejected_licence_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a rejected licence key is not mistaken for a data outage.

    Repair restores the stale metadata file when a refresh fails, so that a
    Trading Strategy outage does not make repair worse. An invalid or expired
    licence is not an outage: silently repairing against stale metadata would
    hide the configuration problem.

    1. Create a client-like object with cached vault metadata.
    2. Refresh the cache with a downloader whose licence key is refused.
    3. Check the access error is raised instead of being swallowed.
    """

    # 1. Create a client-like object with cached vault metadata.
    cache_path = tmp_path / "client-cache"
    vault_download_root = cache_path / "vaults" / "downloads"
    vault_download_root.mkdir(parents=True)
    (vault_download_root / "vault-metadata.json").write_text("old")

    def _fetch_vault_universe() -> None:
        raise VaultDataAccessDenied("Licence key rejected")

    client = SimpleNamespace(
        transport=SimpleNamespace(cache_path=cache_path),
        has_vault_data_access=lambda: True,
    )

    # 2. Refresh the cache with a downloader whose licence key is refused.
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: SimpleNamespace(fetch_vault_universe=_fetch_vault_universe),
    )

    # 3. Check the access error is raised instead of being swallowed.
    with pytest.raises(VaultDataAccessDenied):
        refresh_vault_universe_metadata_cache(client)


def test_refresh_vault_universe_metadata_cache_skips_without_a_licence(tmp_path: Path) -> None:
    """Verify repair works on deployments that do not subscribe to vault datasets.

    The refresh is an opportunistic cache pre-warm. A lending strategy
    deployment has no vault licence key and must be able to repair anyway;
    requiring one here broke ``repair`` for every non-vault strategy.

    1. Create a client-like object reporting no vault data access.
    2. Refresh the vault universe metadata cache.
    3. Check the refresh skipped without raising and without touching the cache.
    """

    # 1. Create a client-like object reporting no vault data access.
    cache_path = tmp_path / "client-cache"
    vault_download_root = cache_path / "vaults" / "downloads"
    vault_download_root.mkdir(parents=True)
    stale_file = vault_download_root / "vault-metadata.json"
    stale_file.write_text("stale")
    client = SimpleNamespace(
        transport=SimpleNamespace(cache_path=cache_path),
        has_vault_data_access=lambda: False,
    )

    # 2. Refresh the vault universe metadata cache.
    refreshed_path = refresh_vault_universe_metadata_cache(client)

    # 3. Check the refresh skipped without raising and without touching the cache.
    assert refreshed_path is None
    assert stale_file.read_text() == "stale"
