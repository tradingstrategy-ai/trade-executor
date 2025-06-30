"""Repair broken profit data on base-ath strategy.
"""
import shutil
import os.path
import secrets
import datetime
from pathlib import Path

import pandas as pd
import pytest

from eth_defi.erc_4626.classification import create_vault_instance
from eth_defi.provider.multi_provider import create_multi_provider_web3
from tradeexecutor.cli.commands.app import app
from tradeexecutor.ethereum.lagoon.share_price_retrofit import retrofit_share_price
from tradeexecutor.state.state import State

pytestmark = pytest.mark.skipif(not os.environ.get("JSON_RPC_BASE") or not os.environ.get("TRADING_STRATEGY_API_KEY"), reason="Set JSON_RPC_POLYGON and TRADING_STRATEGY_API_KEY environment variables to run this test")


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
    """

    cleaned_entries = []
    for s in state.stats.portfolio:

        # Invalid entries caused by bad net asset value calculation discarding broken trades
        if s.calculated_at > datetime.datetime(2025, 6, 1) and s.share_price_usd < 1:
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

    import ipdb ; ipdb.set_trace()



    # Check accounts now to verify if balance is good
    # with pytest.raises(SystemExit) as sys_exit:

    # assert sys_exit.value.code == 0
