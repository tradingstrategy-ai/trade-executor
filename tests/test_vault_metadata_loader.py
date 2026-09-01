"""Tests for vault metadata loader options."""

from types import SimpleNamespace

from tradingstrategy.chain import ChainId

import pytest

from tradeexecutor.strategy import trading_strategy_universe
from tradeexecutor.strategy.trading_strategy_universe import load_vault_universe_with_metadata


def _make_client(calls: list[bool]) -> SimpleNamespace:
    """Create a fake client whose vault universe records strictness flags."""
    vault_universe = SimpleNamespace()

    def _limit_to_vaults(
        vaults: list[tuple[ChainId, str]],
        check_all_vaults_found: bool = True,
    ) -> SimpleNamespace:
        calls.append(check_all_vaults_found)
        return vault_universe

    vault_universe.limit_to_vaults = _limit_to_vaults
    return SimpleNamespace(
        transport=SimpleNamespace(cache_path=None),
        vault_universe=vault_universe,
    )


def test_load_vault_universe_with_metadata_forwards_missing_vault_strictness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify vault metadata loading can tolerate fresher source vault lists.

    1. Create a fake vault universe that records the strictness flag it receives.
    2. Load metadata with the default arguments.
    3. Load metadata with missing vault checks disabled.
    4. Confirm both strictness values were forwarded to the vault universe.
    """
    vaults = [(ChainId.hypercore, "0xc8913adaf1174034c1dc5881a2526ee18e03ccf5")]
    calls: list[bool] = []
    client = _make_client(calls)
    # The Creem dataset client downloads a paid dataset over HTTP, so hand the
    # loader a stub returning the fake universe instead.
    monkeypatch.setattr(
        trading_strategy_universe,
        "create_vault_data_client",
        lambda client, download_root=None: SimpleNamespace(
            fetch_vault_universe=lambda: client.vault_universe,
        ),
    )

    # 1. Create a fake vault universe that records the strictness flag it receives.
    # 2. Load metadata with the default arguments.
    load_vault_universe_with_metadata(client, vaults=vaults)

    # 3. Load metadata with missing vault checks disabled.
    load_vault_universe_with_metadata(
        client,
        vaults=vaults,
        check_all_vaults_found=False,
    )

    # 4. Confirm both strictness values were forwarded to the vault universe.
    assert calls == [True, False]
