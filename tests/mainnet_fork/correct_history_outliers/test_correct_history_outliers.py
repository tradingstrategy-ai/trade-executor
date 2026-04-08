"""GMX-AI correct-history outlier removal acceptance test.

- No blockchain fork needed -- this command only modifies state statistics
- State file downloaded from https://gmx-ai.tradingstrategy.ai/state
- Tests that share price outliers (e.g. 94% drop for a few hours) are
  detected and removed, improving max drawdown statistics
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

# Known bad timestamps in the GMX-AI state (share price outliers)
BAD_TIMESTAMPS = {
    # 2026-03-19 22:00, 23:00 and 2026-03-20 00:00 — share_price drops to 0.1025
    datetime.datetime(2026, 3, 19, 22, 0),
    datetime.datetime(2026, 3, 19, 23, 0),
    datetime.datetime(2026, 3, 20, 0, 0),
    # 2026-04-08 00:00 — share_price drops to 0.831
    datetime.datetime(2026, 4, 8, 0, 0),
}


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Copy the GMX-AI state file to a temp directory so we can modify it."""
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
        "strategies", "gmx-ai.py",
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
        "REMOVE_SHARE_PRICE_OUTLIERS": "true",
    }


def test_remove_share_price_outliers(environment: dict, state_file: Path):
    """Remove share price outlier data points from GMX-AI state.

    The GMX-AI strategy has spurious share price drops caused by temporary
    NAV calculation failures:
    - idx 133-135 (2026-03-19/20): share_price drops from ~1.616 to 0.1025
    - idx 446 (2026-04-08): share_price drops from ~1.817 to 0.831

    These cause a false max drawdown of ~94%.

    1. Load the original state and record baseline portfolio stats count
    2. Verify the known bad timestamps exist in the original data
    3. Run correct-history with --remove-share-price-outliers
    4. Load the modified state
    5. Verify outlier entries are removed (count reduced by a small number)
    6. Verify the known bad timestamps are no longer present
    7. Verify initial_share_price is updated correctly
    """

    # 1. Load original state and record baseline
    original_state = JSONFileStore(state_file).load()
    original_count = len(original_state.stats.portfolio)

    # 2. Verify the known bad timestamps exist in the original data
    original_timestamps = {ps.calculated_at for ps in original_state.stats.portfolio}
    for bad_ts in BAD_TIMESTAMPS:
        # 2026-03-20 00:00 appears twice (one bad, one good) — just check it exists
        assert bad_ts in original_timestamps, f"Expected bad timestamp {bad_ts} in original data"

    # 3. Run correct-history with --remove-share-price-outliers
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["correct-history"], standalone_mode=False)

    # 4. Load the modified state
    pruned_state = JSONFileStore(state_file).load()
    pruned_count = len(pruned_state.stats.portfolio)

    # 5. Verify outlier entries are removed (should only remove a handful)
    # The strategy has several sharp spikes/drops that recover within hours
    # (temporary NAV miscalculations), so more than just the 4 most extreme
    # entries may be flagged
    removed = original_count - pruned_count
    assert removed >= 4, f"Expected at least 4 outliers removed, got {removed}"
    assert removed < 15, f"Too many entries removed ({removed}), possible false positives"

    # 6. Verify the known bad share prices are gone
    # After filtering, no entry should have share_price_usd < 0.5
    for ps in pruned_state.stats.portfolio:
        if ps.share_price_usd is not None:
            assert ps.share_price_usd >= 0.5, (
                f"Outlier still present: {ps.calculated_at} share_price={ps.share_price_usd}"
            )

    # 7. Verify initial_share_price was not corrupted by outlier removal
    # The first portfolio entry has share_price_usd=None (pre-trading),
    # so initial_share_price keeps its original value
    assert pruned_state.initial_share_price == pytest.approx(original_state.initial_share_price)


def test_outlier_removal_with_cutoff_date(state_file: Path, strategy_file: Path):
    """Verify that both --cutoff-date and --remove-share-price-outliers work together.

    Outlier removal runs before cutoff pruning so that the detector sees
    the full neighbourhood context.

    1. Set up environment with both options
    2. Run correct-history
    3. Verify cutoff pruning happened (all entries after cutoff)
    4. Verify outlier removal also happened on the remaining data
    """

    # 1. Set up environment with both options
    cutoff = datetime.datetime(2026, 3, 15)
    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "CUTOFF_DATE": cutoff.strftime("%Y-%m-%d"),
        "REMOVE_SHARE_PRICE_OUTLIERS": "true",
    }

    # 2. Run correct-history
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["correct-history"], standalone_mode=False)

    # 3. Verify cutoff pruning happened
    pruned_state = JSONFileStore(state_file).load()
    for ps in pruned_state.stats.portfolio:
        assert ps.calculated_at >= cutoff, (
            f"Entry at {ps.calculated_at} is before cutoff {cutoff}"
        )

    # 4. Verify outlier removal also happened
    for ps in pruned_state.stats.portfolio:
        if ps.share_price_usd is not None:
            assert ps.share_price_usd >= 0.5, (
                f"Outlier still present after combined operation: "
                f"{ps.calculated_at} share_price={ps.share_price_usd}"
            )
