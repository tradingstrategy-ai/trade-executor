"""Dynamic Hyperliquid (Hypercore) vault universe construction with caching.

Fetches qualifying Hypercore vaults from the Trading Strategy API, filters
them using the same logic as the notebook helpers, and caches the result by
filter parameters.
"""

import datetime
import hashlib
import json
import sys
from pathlib import Path

from eth_defi.compat import native_datetime_utc_fromtimestamp, native_datetime_utc_now
from tradingstrategy.chain import ChainId

from tradeexecutor.curator.curator import EXCLUDED_PROTOCOLS, EXCLUDED_VAULTS, MUST_INCLUDE
from tradeexecutor.curator.vault_universe_creation import (
    fetch_vaults,
    parse_vault,
    select_top_vaults,
)


DATA_URL = "https://top-defi-vaults.tradingstrategy.ai/top_vaults_by_chain.json"

CHAIN_CONFIG = {
    9999: {"name": "Hypercore", "enum": "HYPERCORE_CHAIN_ID", "top_n": 120},
}

CHAIN_ORDER = [9999]

ALLOWED_DENOMINATIONS = {
    "USDC", "USDC.E",
    "USDT", "USD₮0", "USDT0", "USDT.E",
    "CRVUSD",
    "USDS",
}

EXCLUDED_RISKS = {"Blacklisted", "Dangerous"}
EXCLUDED_FLAGS = {"malicious", "broken"}
REQUIRE_KNOWN_PROTOCOL = True

TRACKED_PERIODS = ("1M", "3M", "1Y")

CACHE_DIR = Path.home() / ".cache" / "indicators"
CACHE_TTL = datetime.timedelta(days=1)

SKIP_CAGR_FILTER = True
USE_PEAK_TVL = True


def _curator_fingerprint() -> str:
    """Short hash of all selection policy inputs used by the cached universe."""
    policy = {
        "data_url": DATA_URL,
        "chain_config": {str(chain_id): CHAIN_CONFIG[chain_id] for chain_id in sorted(CHAIN_CONFIG)},
        "chain_order": CHAIN_ORDER,
        "tracked_periods": TRACKED_PERIODS,
        "excluded_vaults": sorted(EXCLUDED_VAULTS),
        "must_include": sorted(MUST_INCLUDE),
        "excluded_protocols": sorted(EXCLUDED_PROTOCOLS.items()),
        "allowed_denominations": sorted(ALLOWED_DENOMINATIONS),
        "excluded_risks": sorted(EXCLUDED_RISKS),
        "excluded_flags": sorted(EXCLUDED_FLAGS),
        "require_known_protocol": REQUIRE_KNOWN_PROTOCOL,
        "skip_cagr_filter": SKIP_CAGR_FILTER,
        "use_peak_tvl": USE_PEAK_TVL,
    }
    digest = hashlib.md5(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:8]
    return digest


def _make_cache_key(min_tvl: float, top_n: int | None, min_age: float, sort_period: str, include_closed_vaults: bool) -> str:
    """Derive a cache key from filter parameters."""
    top_part = "topall" if top_n is None else f"top{top_n}"
    closed_part = "-closed" if include_closed_vaults else ""
    return f"tvl{int(min_tvl)}-{top_part}-age{min_age}-sort{sort_period}{closed_part}-cur{_curator_fingerprint()}"


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"vault-universe-{cache_key}.json"


def _load_cache(cache_key: str) -> list[tuple[ChainId, str]] | None:
    """Load cached vault universe if it exists."""
    path = _cache_path(cache_key)
    if not path.exists():
        return None

    cache_age = native_datetime_utc_now() - native_datetime_utc_fromtimestamp(path.stat().st_mtime)
    if cache_age > CACHE_TTL or path.stat().st_size == 0:
        return None

    with path.open() as inp:
        data = json.load(inp)
    return [(ChainId(entry["chain_id"]), entry["address"]) for entry in data]


def _save_cache(cache_key: str, vaults: list[tuple[ChainId, str]]) -> None:
    """Save vault universe to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    data = [{"chain_id": chain_id.value, "address": address} for chain_id, address in vaults]
    with _cache_path(cache_key).open("w") as out:
        json.dump(data, out, indent=2)


def build_hyperliquid_vault_universe(
    min_tvl: float = 10_000,
    top_n: int | None = None,
    min_age: float = 0.15,
    sort_period: str = "1Y",
    include_closed_vaults: bool = False,
) -> list[tuple[ChainId, str]]:
    """Build a filtered Hypercore vault universe, cached by parameters."""
    cache_key = _make_cache_key(min_tvl, top_n, min_age, sort_period, include_closed_vaults)
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"Loaded {len(cached)} cached Hypercore vaults ({cache_key})", file=sys.stderr)
        return cached

    chain_config = dict(CHAIN_CONFIG)
    if top_n is not None:
        chain_config[9999] = {**chain_config[9999], "top_n": top_n}

    raw_vaults = fetch_vaults(DATA_URL)

    parsed = []
    for rv in raw_vaults:
        v = parse_vault(rv, chain_config, TRACKED_PERIODS)
        if v is not None:
            parsed.append(v)

    print(f"Parsed {len(parsed)} Hypercore vaults from API", file=sys.stderr)

    selected = select_top_vaults(
        parsed,
        min_tvl,
        min_age,
        chain_config,
        CHAIN_ORDER,
        sort_period,
        allowed_denominations=ALLOWED_DENOMINATIONS,
        excluded_risks=EXCLUDED_RISKS,
        excluded_flags=EXCLUDED_FLAGS,
        require_known_protocol=REQUIRE_KNOWN_PROTOCOL,
        hypercore_min_tvl=min_tvl,
        top_n_override=top_n,
        skip_cagr_filter=SKIP_CAGR_FILTER,
        use_peak_tvl=USE_PEAK_TVL,
        include_closed_vaults=include_closed_vaults,
    )

    result = []
    for chain_id in CHAIN_ORDER:
        for v in selected.get(chain_id, []):
            if not v.excluded and v.excluded_protocol_reason is None:
                result.append((ChainId.hypercore, v.address))

    top_label = f"top {top_n}" if top_n is not None else "all"
    print(f"Selected {len(result)} Hypercore vaults (peak TVL >= ${min_tvl:,.0f}, {top_label})", file=sys.stderr)

    _save_cache(cache_key, result)
    return result
