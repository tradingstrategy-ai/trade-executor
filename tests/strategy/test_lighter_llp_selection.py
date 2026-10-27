"""A USDC reserve strategy must not acquire LIT implicitly to enter LLP."""

from pathlib import Path

import pytest

from tradeexecutor.curator import lighter_vault_universe as lighter
from tradeexecutor.curator.vault_universe_creation import (
    LIGHTER_LLP_ADDRESS,
    lighter_llp_exclusion_reason,
    parse_vault,
)


def test_llp_default_exclusion_and_explicit_opt_in():
    """Exclude the canonical USDC LLP without confusing other deployment identities.

    1. Parse otherwise ordinary USDC LLP metadata with default and explicit opt-in.
    2. Check the LIT reason and confirm XLP/Robinhood addresses remain unaffected.
    """
    # 1. Membership is blocked independently of current access or profitability.
    raw = dict(chain_id=9998, address=LIGHTER_LLP_ADDRESS, protocol_slug="lighter", denomination="USDC")
    blocked = parse_vault(raw, lighter.CHAIN_CONFIG, lighter.TRACKED_PERIODS)
    allowed = parse_vault(raw, lighter.CHAIN_CONFIG, lighter.TRACKED_PERIODS, include_lighter_llp=True)
    # 2. Only the canonical USDC LLP carries the LIT prerequisite exclusion.
    assert blocked.excluded and "staked LIT" in blocked.excluded_reason
    assert not allowed.excluded and allowed.excluded_reason is None
    assert lighter_llp_exclusion_reason(9998, "lighter-pool-281474976680784") is None
    assert lighter_llp_exclusion_reason(9998, "lighter-pool-robinhood-281474976710654") is None
    assert lighter_llp_exclusion_reason(1, LIGHTER_LLP_ADDRESS) is None


def test_builder_cache_keeps_llp_opt_in_separate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Cached defaults must not reintroduce LLP after an opt-in universe build.

    1. Mock only metadata acquisition/cache location with LLP and one manager pool.
    2. Build default, opt-in and cached default universes.
    3. Verify LLP appears only in the explicit opt-in and uses a separate cache.
    """
    # 1. All financial and age filters pass, isolating the LLP membership policy.
    rows = [dict(chain_id=9998, address=address, name=address, protocol_slug="lighter",
                 denomination="USDC", years=1., current_nav=1_000_000., peak_nav=1_000_000.)
            for address in (LIGHTER_LLP_ADDRESS, "lighter-pool-123")]
    monkeypatch.setattr(lighter, "fetch_vaults", lambda client: rows)
    monkeypatch.setattr(lighter, "CACHE_DIR", tmp_path)
    # 2. The builder, parser, filters and cache are real.
    default = lighter.build_lighter_vault_universe(printer=lambda message: None)
    opted_in = lighter.build_lighter_vault_universe(include_lighter_llp=True, printer=lambda message: None)
    cached_default = lighter.build_lighter_vault_universe(printer=lambda message: None)
    # 3. Opt-in cannot contaminate later default calls.
    assert {address for _, address in default} == {"lighter-pool-123"}
    assert {address for _, address in opted_in} == {"lighter-pool-123", LIGHTER_LLP_ADDRESS}
    assert cached_default == default
    assert len(list(tmp_path.glob("vault-universe-*.json"))) == 2
