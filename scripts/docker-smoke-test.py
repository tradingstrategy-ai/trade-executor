"""Smoke test the Docker image runtime dependency set."""

import importlib


RUNTIME_IMPORTS = [
    "brotli",
    "ccxt",
    "click",
    "duckdb",
    "eth_defi.coloured_logging",
    "eth_defi.event_reader.timestamp_cache",
    "eth_defi.hypersync.session",
    "eth_defi.research.vault_metrics",
    "eth_defi.vault.vaultdb",
    "hypersync",
    "pyramid",
    "pyarrow",
    "pyarrow.dataset",
    "rich",
    "sentry_sdk",
    "telegram_bot_logger",
    "textual",
    "tradeexecutor.cli.main",
    "tradeexecutor.strategy.trading_strategy_universe",
    "tradeexecutor.webhook.app",
    "typer",
    "waitress",
    "zstandard",
]


def import_runtime_modules() -> None:
    """Import modules that the release image is expected to provide."""
    missing = []

    for module_name in RUNTIME_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as e:
            missing.append((module_name, e))

    if missing:
        lines = [f"{module_name}: {type(e).__name__}: {e}" for module_name, e in missing]
        raise RuntimeError("Docker runtime import smoke test failed:\n" + "\n".join(lines))


def parse_vault_metadata() -> None:
    """Exercise the vault JSON parsing path that imports eth_defi metrics classes."""
    vault_module = importlib.import_module("tradingstrategy.alternative_data.vault")

    json_data = {
        "generated_at": "2026-07-05T00:00:00",
        "vaults": [
            {
                "address": "0x0000000000000000000000000000000000000001",
                "chain_id": 1,
                "name": "Docker smoke vault",
                "protocol": "Smoke",
                "protocol_slug": "smoke",
                "share_token": "SMOKE",
                "share_token_address": "0x0000000000000000000000000000000000000001",
                "share_token_decimals": 18,
                "denomination": "USDC",
                "denomination_token_address": "0x0000000000000000000000000000000000000002",
                "denomination_decimals": 6,
                "features": [],
                "period_results": [
                    {
                        "period": "1M",
                        "period_start_at": "2026-06-01T00:00:00",
                        "period_end_at": "2026-07-01T00:00:00",
                        "raw_samples": 2,
                        "daily_samples": 2,
                    }
                ],
            }
        ],
    }

    vault_universe = vault_module.load_vault_database_with_metadata(json_data)
    vaults = list(vault_universe.iterate_vaults())
    assert len(vaults) == 1, f"Expected one parsed vault, got {len(vaults)}"
    assert vaults[0].metadata is not None
    assert vaults[0].metadata.period_results is not None


def main() -> None:
    """Run all Docker smoke checks."""
    import_runtime_modules()
    parse_vault_metadata()


if __name__ == "__main__":
    main()
