
from pathlib import Path

from eth_defi.token import TokenDiskCache


def get_default_token_cache(
    cache_path: Path | None = None,
    unit_testing: bool = False,
) -> TokenDiskCache:
    """Get a default token cache with common tokens pre-filled.

    :param cache_path:
        Base cache directory path. If provided, token cache will be stored at
        `{cache_path}/eth-defi-tokens.sqlite`.

        If not provided, falls back based on unit_testing flag.

    :param unit_testing:
        If True and no cache_path provided, uses `~/.cache/trading-strategy-tests`
        to match the path used by tradingstrategy.Client in test fixtures.
        This ensures test cache persistence and consistency.
    """

    if cache_path:
        cache_path = Path(cache_path)
    elif unit_testing:
        # Match the cache path used by tradingstrategy.Client in test fixtures
        # See tests/conftest.py persistent_test_cache_path fixture
        cache_path = Path.home() / ".cache" / "trading-strategy-tests"
    else:
        # Legacy behaviour for backward compatibility
        cache_path = Path("./cache")

    cache_path.mkdir(parents=True, exist_ok=True)
    token_cache_file = cache_path / "eth-defi-tokens.sqlite"
    token_cache = TokenDiskCache(token_cache_file)
    return token_cache
