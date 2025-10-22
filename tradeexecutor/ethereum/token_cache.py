from eth_defi.token import TokenDiskCache


def get_default_token_cache() -> TokenDiskCache:
    """Get a default token cache with common tokens pre-filled.

    Not unit testing friendly.
    """

    # TODO: Move to options, as now just a hack.
    # See lagoon_deploy_vault.py
    cache_path = "./cache"
    token_cache = TokenDiskCache(cache_path / "eth-defi-tokens.sqlite")
    return token_cache
