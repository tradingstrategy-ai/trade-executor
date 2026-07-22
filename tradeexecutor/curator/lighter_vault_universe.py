"""Dynamic Lighter (ZK-rollup perps DEX) vault universe construction with caching.

Fetches qualifying Lighter public pools from the Trading Strategy API, filters
them using the same logic as :py:mod:`~tradeexecutor.curator.hyperliquid_vault_universe`,
and caches the result by filter parameters.

Lighter pools are non-EVM: they use synthetic addresses of the form
``lighter-pool-{account_index}`` and a placeholder zero denomination-token
address (see :py:mod:`eth_defi.lighter.constants`). The off-chain metrics
pipeline publishes them under the synthetic chain id ``9998``.
"""

import hashlib
import inspect
import json
from collections.abc import Callable
from pathlib import Path

from eth_defi.lighter.constants import LIGHTER_CHAIN_ID, LIGHTER_DENOMINATION
from tradingstrategy.chain import ChainId

from tradeexecutor.curator import curator as curator_module
from tradeexecutor.curator import vault_universe_creation as vault_universe_creation_module
from tradeexecutor.curator.curator import EXCLUDED_PROTOCOLS, EXCLUDED_VAULTS, MUST_INCLUDE
from tradeexecutor.curator.vault_universe_creation import (
    fetch_vaults,
    parse_vault,
    select_top_vaults,
)


DATA_URL = "https://top-defi-vaults.tradingstrategy.ai/top_vaults_by_chain.json"

#: `LIGHTER_CHAIN_ID` (9998) is the canonical synthetic chain id, imported from
#: :py:mod:`eth_defi.lighter.constants`. `top_n` is a generous non-binding
#: ceiling: the whole Lighter universe is only ~40 pools, far below it.
CHAIN_CONFIG = {
    LIGHTER_CHAIN_ID: {"name": "Lighter", "enum": "LIGHTER_CHAIN_ID", "top_n": 120},
}

CHAIN_ORDER = [LIGHTER_CHAIN_ID]

#: The main (zkSync) Lighter pools are USDC-denominated (`LIGHTER_DENOMINATION`;
#: see the eth_defi Lighter vaults README), matching the USDC strategy reserve,
#: so the allowlist is a single entry — unlike the multi-stablecoin Hypercore
#: universe. Restricting to USDC also excludes Lighter's separate Robinhood Chain
#: deployment, which the off-chain feed publishes under the same synthetic chain
#: 9998 but denominates in USDG (`lighter-pool-robinhood-*` addresses,
#: robinhoodchain.lighter.xyz) — a different venue and stablecoin the USDC-reserve
#: backtest does not model.
ALLOWED_DENOMINATIONS = {LIGHTER_DENOMINATION}

#: Safety filters shared with the Hypercore curator. Note every Lighter pool is
#: currently labelled `Severe` risk in the metrics feed, which is *not* in
#: `EXCLUDED_RISKS`, so pools are admitted; this is an exploratory universe.
EXCLUDED_RISKS = {"Blacklisted", "Dangerous"}
EXCLUDED_FLAGS = {"malicious", "broken"}
REQUIRE_KNOWN_PROTOCOL = True

TRACKED_PERIODS = ("1M", "3M", "1Y")

CACHE_DIR = Path.home() / ".cache" / "indicators"

#: Bump this whenever cache invalidation needs to cover a new policy dimension.
CURATOR_POLICY_VERSION = 1


def _source_digest(*objects) -> str:
    """Create a stable digest from the current selector implementation source."""
    source_parts = []
    for obj in objects:
        source_parts.append(inspect.getsource(obj))
    return hashlib.md5("\n".join(source_parts).encode()).hexdigest()[:8]


def _curator_fingerprint() -> str:
    """Short hash of all selector inputs used by the cached universe.

    Include both explicit policy constants and implementation source so that
    cached universes are invalidated when the selection logic itself changes.
    """
    policy = {
        "policy_version": CURATOR_POLICY_VERSION,
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
        "selection_behaviour": {
            "skip_cagr_filter": True,
            "use_peak_tvl": True,
        },
        "implementation_digest": _source_digest(
            build_lighter_vault_universe,
            fetch_vaults,
            parse_vault,
            select_top_vaults,
            curator_module,
            vault_universe_creation_module,
        ),
    }
    digest = hashlib.md5(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:8]
    return digest


def _make_cache_key(min_tvl: float, top_n: int | None, min_age: float, sort_period: str, include_closed_vaults: bool) -> str:
    """Derive a cache key from filter parameters."""
    top_part = "topall" if top_n is None else f"top{top_n}"
    closed_part = "-closed" if include_closed_vaults else ""
    return f"lighter-tvl{int(min_tvl)}-{top_part}-age{min_age}-sort{sort_period}{closed_part}-cur{_curator_fingerprint()}"


def _cache_path(cache_key: str) -> Path:
    return CACHE_DIR / f"vault-universe-{cache_key}.json"


def _load_cache(cache_key: str) -> list[tuple[ChainId, str]] | None:
    """Load cached vault universe if it exists."""
    path = _cache_path(cache_key)
    if not path.exists():
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


def build_lighter_vault_universe(
    min_tvl: float = 10_000,
    top_n: int | None = None,
    min_age: float = 0.15,
    sort_period: str = "1Y",
    include_closed_vaults: bool = False,
    printer: Callable[[str], None] = print,
) -> list[tuple[ChainId, str]]:
    """Build a filtered Lighter vault universe, cached by parameters.

    Mirrors :py:func:`~tradeexecutor.curator.hyperliquid_vault_universe.build_hyperliquid_vault_universe`
    but targets the synthetic Lighter chain (``9998``).

    :param printer:
        Output callback used for cache and selection diagnostics.
        Defaults to :py:func:`print` so notebook calls surface the
        fingerprint and vault count in cell output.
    """
    cache_key = _make_cache_key(min_tvl, top_n, min_age, sort_period, include_closed_vaults)
    fingerprint = _curator_fingerprint()
    printer(f"Lighter universe fingerprint: {fingerprint}")
    cached = _load_cache(cache_key)
    if cached is not None:
        printer(f"Loaded {len(cached)} cached Lighter vaults ({cache_key})")
        return cached

    chain_config = dict(CHAIN_CONFIG)
    if top_n is not None:
        chain_config[LIGHTER_CHAIN_ID] = {**chain_config[LIGHTER_CHAIN_ID], "top_n": top_n}

    raw_vaults = fetch_vaults(DATA_URL)

    parsed = []
    for rv in raw_vaults:
        v = parse_vault(rv, chain_config, TRACKED_PERIODS)
        if v is not None:
            parsed.append(v)

    printer(f"Parsed {len(parsed)} Lighter vaults from API")

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
        # `hypercore_min_tvl` only affects chain 9999; Lighter uses `min_tvl`.
        hypercore_min_tvl=min_tvl,
        top_n_override=top_n,
        skip_cagr_filter=True,
        use_peak_tvl=True,
        include_closed_vaults=include_closed_vaults,
    )

    result = []
    for chain_id in CHAIN_ORDER:
        for v in selected.get(chain_id, []):
            if v.excluded or v.excluded_protocol_reason is not None:
                continue
            # `select_top_vaults` only auto-drops deposit-closed pools on chain
            # 9999, so filter Lighter's closed pools here to match the
            # survivor-first Hypercore behaviour.
            if not include_closed_vaults and v.deposit_closed_reason is not None:
                continue
            result.append((ChainId.lighter, v.address))

    top_label = f"top {top_n}" if top_n is not None else "all"
    printer(f"Selected {len(result)} Lighter vaults (peak TVL >= ${min_tvl:,.0f}, {top_label})")

    _save_cache(cache_key, result)
    return result
