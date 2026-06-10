"""Unit tests for the lagoon-reclaim-satellites planning logic.

The reclaim command sweeps USDC from multichain Lagoon satellite Safes back to
the master Safe via CCTP. ``plan_reclaims()`` decides which satellite chains
hold enough USDC to be worth bridging, leaving dust in place so the CCTP
gas/fee cost is not wasted.
"""

from decimal import Decimal

from tradeexecutor.cli.commands.reclaim_satellites import plan_reclaims


def test_plan_reclaims_selects_above_threshold_only():
    """Only satellite chains strictly above the threshold are reclaimed.

    1. A chain comfortably above the threshold is selected.
    2. A chain holding dust below the threshold is skipped.
    3. A chain exactly at the threshold is skipped (strict comparison).
    4. The result is sorted by chain id for deterministic output.
    """

    balances = {
        8453: Decimal("12.5"),   # 1. Base — well above threshold, reclaim
        1: Decimal("0.4"),       # 2. Ethereum — dust, skip
        137: Decimal("1.0"),     # 3. Polygon — exactly at threshold, skip
    }

    # 1-4. Plan with a 1.0 USDC threshold
    result = plan_reclaims(balances, Decimal("1.0"))

    assert result == [8453]


def test_plan_reclaims_empty_when_all_dust():
    """No chains are reclaimed when every satellite holds only dust.

    1. All balances are below the threshold.
    2. The plan is empty.
    """

    # 1. Every satellite below threshold
    balances = {8453: Decimal("0.1"), 1: Decimal("0.0")}

    # 2. Nothing to reclaim
    assert plan_reclaims(balances, Decimal("1.0")) == []
