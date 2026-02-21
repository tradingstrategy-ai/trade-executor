"""Master vault correct-history acceptance test.

- No blockchain fork needed -- this command only modifies state statistics
- State file downloaded from https://master-vault.tradingstrategy.ai/
- Cutoff date 2026-01-15: removes the warmup period (Nov 2025) when share
  price was below $1.0
"""
import datetime
import os.path
import shutil
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.main import app
from tradeexecutor.state.store import JSONFileStore

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


CUTOFF_DATE = datetime.datetime(2026, 1, 15)


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Copy the production state file to a temp directory so we can modify it."""
    source = Path(os.path.join(os.path.dirname(__file__), "state.json"))
    assert source.exists(), f"{source} missing"
    dest = tmp_path / "state.json"
    shutil.copy(source, dest)
    return dest


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module for this executor."""
    p = Path(os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "strategies", "master-vault.py",
    ))
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def environment(
    state_file: Path,
    strategy_file: Path,
) -> dict:
    """Passed to correct-history command as environment variables."""
    return {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CUTOFF_DATE": CUTOFF_DATE.strftime("%Y-%m-%d"),
    }


def test_correct_history(environment: dict, state_file: Path):
    """Prune early warmup data from master vault state.

    The master vault share price was below $1.0 during its initial warmup
    period (Nov 2025). Using cutoff 2026-01-15 removes that data while
    keeping the regular operating period.

    Verifies:
    - Portfolio stats count is reduced
    - All remaining portfolio stats have calculated_at >= cutoff
    - Position stats for early-only positions are removed
    - Visualisation messages from the warmup period are removed
    - Uptime cycles from the warmup period are removed
    - Closed positions stats from the warmup period are removed
    - State file size is reduced
    """

    # Load original state to get baseline counts
    original_state = JSONFileStore(state_file).load()
    original_portfolio_count = len(original_state.stats.portfolio)
    original_position_stats_count = sum(
        len(entries) for entries in original_state.stats.positions.values()
    )
    original_vis_messages_count = len(original_state.visualisation.messages)
    original_cycles_count = len(original_state.uptime.cycles_completed_at)
    original_closed_positions_count = len(original_state.stats.closed_positions)
    original_file_size = state_file.stat().st_size

    assert original_portfolio_count == 921
    assert original_vis_messages_count == 5
    assert original_cycles_count == 115

    # Run the correct-history command
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["correct-history"], standalone_mode=False)

    # Load the pruned state
    pruned_state = JSONFileStore(state_file).load()

    # Portfolio stats should be reduced -- entries before 2026-01-15 removed
    assert len(pruned_state.stats.portfolio) < original_portfolio_count
    assert len(pruned_state.stats.portfolio) > 0

    # All remaining portfolio stats must be at or after the cutoff
    for ps in pruned_state.stats.portfolio:
        assert ps.calculated_at >= CUTOFF_DATE, (
            f"Portfolio stat at {ps.calculated_at} is before cutoff {CUTOFF_DATE}"
        )

    # Position stats should be reduced
    pruned_position_stats_count = sum(
        len(entries) for entries in pruned_state.stats.positions.values()
    )
    assert pruned_position_stats_count < original_position_stats_count

    # All remaining position stats must be at or after the cutoff
    for position_id, entries in pruned_state.stats.positions.items():
        for ps in entries:
            assert ps.calculated_at >= CUTOFF_DATE, (
                f"Position {position_id} stat at {ps.calculated_at} is before cutoff"
            )

    # Visualisation messages from Oct/Nov 2025 should be removed
    # All 5 original messages are from Oct-Nov 2025, so all should be gone
    assert len(pruned_state.visualisation.messages) < original_vis_messages_count

    # Uptime cycles from before cutoff should be removed
    # Values are UNIX int timestamps after deserialisation
    import calendar
    cutoff_unix = int(calendar.timegm(CUTOFF_DATE.utctimetuple()))
    assert len(pruned_state.uptime.cycles_completed_at) < original_cycles_count
    for cycle_num, completed_at in pruned_state.uptime.cycles_completed_at.items():
        assert completed_at >= cutoff_unix, (
            f"Cycle {cycle_num} completed at {completed_at} is before cutoff {cutoff_unix}"
        )

    # Closed positions stats from before cutoff should be removed
    # Positions 1, 2, 3 were closed on 2025-11-06, so their stats should be gone
    assert len(pruned_state.stats.closed_positions) < original_closed_positions_count
    for position_id, fps in pruned_state.stats.closed_positions.items():
        assert fps.calculated_at >= CUTOFF_DATE, (
            f"Closed position {position_id} stat at {fps.calculated_at} is before cutoff"
        )

    # state.created_at should be updated to the cutoff date so that
    # key metrics (CAGR, trading period, etc.) use the correct start date
    assert pruned_state.created_at == CUTOFF_DATE

    # File size should be smaller
    pruned_file_size = state_file.stat().st_size
    assert pruned_file_size < original_file_size

    # After pruning the warmup period, the annualised return from the
    # remaining share price data should be positive (vault has been growing
    # since Jan 2026, share price ~0.7735 -> ~0.7755)
    first_ps = pruned_state.stats.portfolio[0]
    last_ps = pruned_state.stats.portfolio[-1]
    assert first_ps.share_price_usd is not None
    assert last_ps.share_price_usd is not None

    share_price_return = (last_ps.share_price_usd / first_ps.share_price_usd) - 1
    duration = last_ps.calculated_at - first_ps.calculated_at
    assert duration.total_seconds() > 0

    annualised_return = share_price_return * (365 * 24 * 3600) / duration.total_seconds()
    assert annualised_return > 0, (
        f"Expected positive annualised return after pruning warmup period, "
        f"got {annualised_return:.4%} "
        f"(share price {first_ps.share_price_usd:.6f} -> {last_ps.share_price_usd:.6f})"
    )
    # Should be a modest positive yield (~5-10% annualised)
    assert annualised_return == pytest.approx(0.08, abs=0.05)

    # initial_share_price should be set to the first remaining share price
    # so that share_price_based_return chart shows positive values
    assert pruned_state.initial_share_price == pytest.approx(first_ps.share_price_usd)

    # Verify all share price based returns are non-negative after pruning
    from tradeexecutor.visual.equity_curve import calculate_share_price
    share_price_df = calculate_share_price(pruned_state)
    assert (share_price_df["returns"] >= 0).all(), (
        f"Expected all share price returns to be non-negative after pruning, "
        f"but found negative values: {share_price_df['returns'].min():.6f}"
    )
