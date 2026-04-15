"""GMX-AI state correction acceptance tests.

- No blockchain fork needed -- these tests only read/modify state statistics
- State file downloaded from https://gmx-ai.tradingstrategy.ai/state
- Tests share price outlier removal and exchange account position profit calculation
"""
import datetime
import os.path
import shutil
from decimal import Decimal
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.main import app
from tradeexecutor.state.correct_history import detect_nav_sync_outliers, detect_share_price_outliers
from tradeexecutor.state.state import State
from tradeexecutor.state.statistics import PortfolioStatistics
from tradeexecutor.state.store import JSONFileStore
from tradeexecutor.statistics.core import calculate_position_statistics

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

CURRENT_STATE_BAD_SHARE_PRICE_POINTS = {
    # Public GMX-AI state downloaded on 2026-04-15.
    datetime.datetime(2026, 4, 8, 10, 0): 2.0779597367123466,
    datetime.datetime(2026, 4, 14, 12, 0): 1.2725298415304773,
    datetime.datetime(2026, 4, 15, 10, 0): 1.4161990737674461,
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
def current_state_file(tmp_path: Path) -> Path:
    """Copy the current public GMX-AI state file to a temp directory."""
    source = Path(os.path.join(os.path.dirname(__file__), "gmx-ai-current-state.json"))
    assert source.exists(), f"{source} missing"
    dest = tmp_path / "gmx-ai-current-state.json"
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


def _create_state_with_share_prices(share_prices: list[float]) -> State:
    """Create a minimal state with portfolio statistics."""
    start_at = datetime.datetime(2026, 1, 1)
    state = State(created_at=start_at)
    share_count = Decimal("100")
    state.stats.portfolio = [
        PortfolioStatistics(
            calculated_at=start_at + datetime.timedelta(hours=i),
            total_equity=share_price * float(share_count),
            net_asset_value=share_price * float(share_count),
            free_cash=50,
            open_position_equity=(share_price * float(share_count)) - 50,
            share_count=share_count,
            share_price_usd=share_price,
        )
        for i, share_price in enumerate(share_prices)
    ]
    return state


def _create_state_with_nav_sync_outliers() -> State:
    """Create a state with clustered cash/account sync outliers."""
    start_at = datetime.datetime(2026, 1, 1)
    state = State(created_at=start_at)
    share_count = Decimal("100")
    rows = [
        (1.0, 50, 50),
        (1.0, 50, 50),
        (1.0, 50, 50),
        (1.0, 50, 50),
        # Stale reserve cash briefly double-counts account equity.
        (1.5, 100, 50),
        (1.5, 100, 50),
        (1.5, 100, 50),
        (1.0, 50, 50),
        (1.0, 50, 50),
        (1.0, 50, 50),
        (1.0, 50, 50),
    ]
    state.stats.portfolio = [
        PortfolioStatistics(
            calculated_at=start_at + datetime.timedelta(hours=i),
            total_equity=free_cash + open_position_equity,
            net_asset_value=free_cash + open_position_equity,
            free_cash=free_cash,
            open_position_equity=open_position_equity,
            share_count=share_count,
            share_price_usd=share_price,
        )
        for i, (share_price, free_cash, open_position_equity) in enumerate(rows)
    ]
    return state


def _calculate_largest_share_price_move(state: State) -> float:
    """Calculate the largest absolute adjacent share price move."""
    rows = [
        (ps.calculated_at, ps.share_price_usd)
        for ps in state.stats.portfolio
        if ps.share_price_usd is not None
    ]
    moves = [
        abs(next_price / current_price - 1)
        for (_, current_price), (_, next_price) in zip(rows, rows[1:])
        if current_price
    ]
    return max(moves) if moves else 0


def _calculate_largest_share_price_drop(state: State) -> float:
    """Calculate the largest adjacent share price drop."""
    rows = [
        (ps.calculated_at, ps.share_price_usd)
        for ps in state.stats.portfolio
        if ps.share_price_usd is not None
    ]
    moves = [
        next_price / current_price - 1
        for (_, current_price), (_, next_price) in zip(rows, rows[1:])
        if current_price
    ]
    return min(moves) if moves else 0


def _calculate_max_share_price_drawdown(state: State) -> float:
    """Calculate maximum drawdown from the share price series."""
    peak: float | None = None
    max_drawdown = 0.0

    for ps in state.stats.portfolio:
        price = ps.share_price_usd
        if price is None:
            continue

        if peak is None or price > peak:
            peak = price

        drawdown = price / peak - 1
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    return max_drawdown


def _has_share_price_point(state: State, calculated_at: datetime.datetime, share_price: float) -> bool:
    """Check if a specific share price point is present in the state."""
    return any(
        ps.calculated_at == calculated_at
        and ps.share_price_usd == pytest.approx(share_price)
        for ps in state.stats.portfolio
    )


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
    assert removed < 30, f"Too many entries removed ({removed}), possible false positives"

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


def test_share_price_outlier_default_threshold_is_twenty_percent():
    """Verify the default share price outlier threshold catches 20% moves.

    The default was tightened because GMX-AI still had material sub-30%
    share price breaks after the old 30% cleaner had run.

    1. Build a small state with one 25% share price outlier
    2. Run share price outlier detection with its default threshold
    3. Verify the 25% outlier is detected
    """

    # 1. Build a small state with one 25% share price outlier
    state = _create_state_with_share_prices([1.0, 1.0, 1.25, 1.0, 1.0])

    # 2. Run share price outlier detection with its default threshold
    outliers = detect_share_price_outliers(state, window_size=2)

    # 3. Verify the 25% outlier is detected
    assert outliers == {2}


def test_remove_nav_sync_outliers_cli_flag(tmp_path: Path, strategy_file: Path):
    """Remove clustered NAV sync outliers using the dedicated CLI flag.

    GMX exchange account stats can briefly double-count or under-count reserve
    cash while GMX account equity is being synced. These points form clusters,
    so a wider detector is needed in addition to isolated share price outlier
    detection.

    1. Create a state with a clustered NAV/share price spike around a cash jump
    2. Run correct-history with --remove-nav-sync-outliers enabled
    3. Verify only the clustered sync spike was removed
    """

    # 1. Create a state with a clustered NAV/share price spike around a cash jump
    state = _create_state_with_nav_sync_outliers()
    state_file = tmp_path / "nav-sync-state.json"
    state.write_json_file(state_file)
    original_count = len(state.stats.portfolio)

    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "REMOVE_NAV_SYNC_OUTLIERS": "true",
        "NAV_SYNC_WINDOW_SIZE": "4",
        "NAV_SYNC_THRESHOLD": "0.20",
        "NAV_SYNC_MIN_COMPONENT_CHANGE": "0.10",
    }

    # 2. Run correct-history with --remove-nav-sync-outliers enabled
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["correct-history"], standalone_mode=False)

    # 3. Verify only the clustered sync spike was removed
    corrected_state = JSONFileStore(state_file).load()
    assert len(corrected_state.stats.portfolio) == original_count - 3
    assert [ps.share_price_usd for ps in corrected_state.stats.portfolio] == [1.0] * 8
    assert corrected_state.initial_share_price == pytest.approx(1.0)


def test_current_gmx_ai_state_correct_history_cli_fixes_nav_sync_issues(
    current_state_file: Path,
    strategy_file: Path,
):
    """Correct the copied public GMX-AI state with the CLI flags.

    The copied state contains the newly detected NAV sync issue where share
    price drops and rebounds around reserve cash/exchange account equity sync
    points. This test makes sure one correct-history run fixes the state.

    1. Load the copied public GMX-AI state
    2. Verify the original state has the newly detected broken points
    3. Run correct-history with both outlier cleaner flags
    4. Verify the bad detector candidates are gone
    5. Verify share price drops and drawdown are no longer massive
    """

    # 1. Load the copied public GMX-AI state
    original_state = JSONFileStore(current_state_file).load()
    original_count = len(original_state.stats.portfolio)

    # 2. Verify the original state has the newly detected broken points
    for calculated_at, share_price in CURRENT_STATE_BAD_SHARE_PRICE_POINTS.items():
        assert _has_share_price_point(original_state, calculated_at, share_price)

    original_share_outliers = detect_share_price_outliers(original_state)
    original_nav_sync_outliers = detect_nav_sync_outliers(original_state)
    assert len(original_share_outliers | original_nav_sync_outliers) >= 52
    assert _calculate_largest_share_price_move(original_state) > 0.40
    assert _calculate_largest_share_price_drop(original_state) < -0.30
    assert _calculate_max_share_price_drawdown(original_state) < -0.50

    environment = {
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": current_state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "REMOVE_SHARE_PRICE_OUTLIERS": "true",
        "REMOVE_NAV_SYNC_OUTLIERS": "true",
    }

    # 3. Run correct-history with both outlier cleaner flags
    with mock.patch.dict("os.environ", environment, clear=True):
        app(["correct-history"], standalone_mode=False)

    # 4. Verify the bad detector candidates are gone
    corrected_state = JSONFileStore(current_state_file).load()
    removed = original_count - len(corrected_state.stats.portfolio)
    assert removed >= 52
    assert removed < 60
    assert detect_share_price_outliers(corrected_state) == set()
    assert detect_nav_sync_outliers(corrected_state) == set()

    for calculated_at, share_price in CURRENT_STATE_BAD_SHARE_PRICE_POINTS.items():
        assert not _has_share_price_point(corrected_state, calculated_at, share_price)

    # 5. Verify share price drops and drawdown are no longer massive
    assert _calculate_largest_share_price_move(corrected_state) < 0.20
    assert _calculate_largest_share_price_drop(corrected_state) > -0.20
    assert _calculate_max_share_price_drawdown(corrected_state) > -0.25


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


def test_exchange_account_position_profit():
    """Exchange account position profit uses share price method, not legacy.

    Exchange account positions are created with placeholder trades ($1 reserve,
    price=1.0) that don't represent real capital. The legacy profit calculation
    divides by this tiny buy amount, producing absurd percentages like 9,999%.

    The fix ensures exchange account positions use the internal share price
    method for profit calculation, which tracks actual returns.

    1. Load the GMX-AI state
    2. Get exchange account position #1
    3. Verify it uses share price method (share_price_state is set)
    4. Verify get_total_profit_percent() returns a reasonable value
    5. Verify calculate_position_statistics() also produces reasonable profitability
    """

    # 1. Load the GMX-AI state
    source = Path(os.path.join(os.path.dirname(__file__), "state.json"))
    state = State.read_json_file(source)

    # 2. Get exchange account position #1
    position = state.portfolio.open_positions[1]
    assert position.is_exchange_account()

    # 3. Verify it uses share price method
    assert position.share_price_state is not None

    # 4. Verify profit percentage is reasonable (not 9,999%)
    profit_pct = position.get_total_profit_percent()
    assert -1.0 < profit_pct < 10.0, (
        f"Exchange account position profit {profit_pct:.2%} looks unreasonable, "
        f"expected share price method to produce a value between -100% and +1000%"
    )

    # 5. Verify calculate_position_statistics() also produces reasonable profitability
    stats = calculate_position_statistics(datetime.datetime(2026, 4, 8), position)
    assert -1.0 < stats.profitability < 10.0, (
        f"PositionStatistics.profitability {stats.profitability:.2%} looks unreasonable"
    )
    assert stats.internal_profit_pct is not None, "Exchange account should have internal_profit_pct"
    assert stats.profitability == pytest.approx(stats.internal_profit_pct)
