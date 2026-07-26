"""Regression tests for project-owned Foundry RPC cache seeds."""

from pathlib import Path

from eth_defi.testing.rpc_cache import seed_foundry_rpc_cache


PROJECT_SEED_DIR = Path(__file__).parent / "rpc_cache_seed"


def test_project_rpc_cache_seed_can_populate_empty_cache(tmp_path: Path) -> None:
    """Seed a fresh Foundry cache with the committed fixed-block responses.

    1. Locate every committed Foundry storage cache entry.
    2. Seed an empty cache directory through eth-defi's standard helper.
    3. Verify every generated response is available at the expected cache path.
    """
    # 1. Locate every committed Foundry storage cache entry.
    seed_files = list(PROJECT_SEED_DIR.rglob("storage.json"))
    assert seed_files

    # 2. Seed an empty cache directory through eth-defi's standard helper.
    copied = seed_foundry_rpc_cache(PROJECT_SEED_DIR, tmp_path)

    # 3. Verify every generated response is available at the expected cache path.
    assert copied == len(seed_files)
    for source in seed_files:
        target = tmp_path / source.relative_to(PROJECT_SEED_DIR)
        assert target.read_bytes() == source.read_bytes()
