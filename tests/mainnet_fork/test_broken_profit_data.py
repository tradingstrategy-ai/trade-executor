"""Repair broken profit data on base-ath strategy.
"""
import shutil
import os.path
import datetime
from pathlib import Path

import pandas as pd
import pytest

from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.ethereum.lagoon.share_price_retrofit import retrofit_share_price
from tradeexecutor.state.state import State
from tradeexecutor.statistics.summary import prepare_share_price_summary_statistics
from tradeexecutor.visual.equity_curve import calculate_share_price
from tradeexecutor.visual.web_chart import render_web_chart, WebChartType, WebChartSource, WebChart


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Make a copy of the state file with the broken credit position on a new test cycle"""
    template = Path(__file__).resolve().parent / "profit-data-broken.json"
    assert template.exists(), f"State dump missing: {template}"
    p = tmp_path / Path("credit-position-open-failed.json.json")
    shutil.copy(template, p)
    assert p.exists(), f"{p} missing"
    return p


@pytest.fixture()
def strategy_file() -> Path:
    """The strategy module where the broken accounting happened."""
    p = Path(__file__).resolve().parent / ".." / ".." / "strategies" /  "test_only" / "base-ath.py"
    assert p.exists(), f"{p.resolve()} missing"
    return p


@pytest.fixture()
def web3() -> "Web3":
    return create_multi_provider_web3(os.environ["JSON_RPC_BASE"])


def fix_base_ath_share_price_issue(state):
    """Fix share price issue in base-ath strategy by removing entries that got in broken data.

    - This was caused by IPOR vault transactions not going through due to abnormally high transaction gas requirement
    - This caused net asset value calculation to fail because

    Usage:

    .. code-block:: python

        fix_base_ath_share_price_issue(state)
    """

    cleaned_entries = []
    for s in state.stats.portfolio:

        # Invalid entries caused by bad net asset value calculation discarding broken trades
        if s.calculated_at > datetime.datetime(2025, 6, 1) and s.share_price_usd < 1.10:
            print(f"Removing bad entry: {s.calculated_at} with share price {s.share_price_usd}")
            continue

        cleaned_entries.append(s)

    state.stats.portfolio = cleaned_entries


@pytest.mark.skip(reason="Only for manual scripting testing, takes very long time")
def test_clean_broken_profit_data(
    web3,
    state_file: Path,
):
    """Fix a creidt position that failed to open.

    - Execution crashes in broadcasting phase
    """

    state = State.read_json_file(state_file)

    vault = create_vault_instance(web3, "0x7d8Fab3E65e6C81ea2a940c050A7c70195d1504f", auto_detect=True)

    retrofit_share_price(
        state=state,
        vault=vault,
        max_time=datetime.datetime(2025, 6, 10),
    )

    fix_base_ath_share_price_issue(state)

    profit = [
        {"calculated_at": s.calculated_at, "net_asset_value": s.net_asset_value, "unrealised_profitability": s.unrealised_profitability, "share_price_usd": s.share_price_usd} for s in state.stats.portfolio
    ]
    df = pd.DataFrame(profit)
    df = df.set_index("calculated_at")

    print(df)


def test_share_price_chart(
    web3,
    state_file: Path,
):
    """Calculate the share price chart
    """

    state = State.read_json_file(state_file)
    chart = render_web_chart(state, WebChartType.share_price, source=WebChartSource.live_trading)
    assert isinstance(chart, WebChart)

    assert chart.data[-1][1]  == pytest.approx(1.1592042216816583)

    chart = render_web_chart(state, WebChartType.share_price_based_return, source=WebChartSource.live_trading)
    assert isinstance(chart, WebChart)
    assert chart.data[-1][1] == pytest.approx(0.1592042216816583)

    # Lagoon patched calculations
    share_price_df = calculate_share_price(state)
    returns_annualised, nav_90_days, performance_90_days = prepare_share_price_summary_statistics(
        share_price_df,
        start_at=pd.Timestamp("2025-05-01"),
        age=datetime.timedelta(days=30),
    )
    assert returns_annualised == pytest.approx(-10.229681969539822)
    assert len(performance_90_days) == 26
    assert len(nav_90_days) == 27
