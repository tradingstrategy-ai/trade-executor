"""Rebuild the lighter-ai Lighter account share price from production state.

``lighter-ai.json.gz`` is the public ``https://lighter-ai.tradingstrategy.ai/state``
downloaded on 2026-11-16 13:35 UTC. Position #1, the Lighter master account,
reported a 95.1% loss. On 2026-09-17 ``lagoon-lighter-test-trade`` withdrew
16.995365 USDC of its 18 USDC test capital back to the Safe, and the valuation
booked that as balance update #4, a loss, which took the internal share price
from 1.0 to 0.0556. The later 3,000 and 11,990.89 USDC deposits minted shares
at that price. The real result is about -11.2%.

``lighter-ai.lighter-test-trade.json`` is reconstructed from the state, not
copied from the server: account 748071, a 17 USDC deposit on a 1 USDC
baseline, and the 16.995365 USDC withdrawal requested at 16:30 UTC, before
balance update #4 at 16:37:41 UTC. It has the fields the repair reads.

No RPC or fork is needed: the command only reads and rewrites state.
"""
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest import mock

import pytest

from tradeexecutor.cli.main import app
from tradeexecutor.state.store import JSONFileStore

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")

FIXTURES = Path(__file__).parent / "lighter_ai_share_price"
COMMAND = "rebuild-exchange-account-share-price"


@pytest.fixture()
def state_file(tmp_path) -> Path:
    """Decompress the production state into a writable folder."""
    dest = tmp_path / "lighter-ai.json"
    with gzip.open(FIXTURES / "lighter-ai.json.gz", "rb") as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out)
    return dest


@pytest.fixture()
def journal_file(state_file: Path) -> Path:
    """Put the test-trade journal at its default path next to the state."""
    dest = state_file.parent / "lighter-ai.lighter-test-trade.json"
    shutil.copy(FIXTURES / "lighter-ai.lighter-test-trade.json", dest)
    return dest


def _environment(state_file: Path, **extra) -> dict:
    return {
        "EXECUTOR_ID": "lighter-ai",
        "STATE_FILE": state_file.as_posix(),
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        **extra,
    }


def _run(state_file: Path, *args: str, **env) -> int:
    """Run the command; return its exit code."""
    with mock.patch.dict("os.environ", _environment(state_file, **env), clear=True):
        try:
            app([COMMAND, *args], standalone_mode=False)
        except SystemExit as e:
            return e.code or 0
    return 0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _backups(state_file: Path) -> list[Path]:
    return list(state_file.parent.glob("*share-price-rebuild-backup*"))


def test_lighter_ai_rebuild_with_defaults(state_file: Path, journal_file: Path, capsys):
    """With no options, the journal identifies the withdrawal and the loss becomes -11.2%.

    Also checks the dry run writes nothing and a second run is a no-op.
    """
    before = JSONFileStore(state_file).load()
    position = before.portfolio.open_positions[1]
    assert position.share_price_state.current_share_price == pytest.approx(0.049338, abs=1e-6)
    assert position.get_total_profit_percent() == pytest.approx(-0.9507, abs=1e-4)
    equity_before = before.portfolio.calculate_total_equity()
    samples_before = before.stats.positions[1]
    assert len(samples_before) == 587

    # Dry run: report only
    digest = _sha(state_file)
    assert _run(state_file, "--dry-run") == 0
    output = capsys.readouterr().out
    assert "balance update #4 is the test-trade withdrawal of 16.995365 USDC" in output
    assert "Dry run" in output
    assert _sha(state_file) == digest
    assert _backups(state_file) == []

    # Repair with defaults
    assert _run(state_file, AUTO_APPROVE="true") == 0
    assert len(_backups(state_file)) == 1

    after = JSONFileStore(state_file).load()
    position = after.portfolio.open_positions[1]
    sps = position.share_price_state
    assert sps.current_share_price == pytest.approx(0.887851, abs=1e-6)
    assert sps.total_withdrawn == pytest.approx(16.995365)
    assert sps.total_invested == pytest.approx(15008.885773)
    assert position.get_total_profit_percent() == pytest.approx(-0.11215, abs=1e-5)
    assert position.get_total_profit_usd() == pytest.approx(-1741.54, abs=0.01)
    assert position.balance_updates[4].other_data == {"share_price_capital_flow": "withdrawal"}

    # Accounting is untouched
    assert float(position.get_quantity()) == pytest.approx(13250.354528)
    assert len(position.trades) == 3
    assert len(position.balance_updates) == 505
    assert after.portfolio.calculate_total_equity() == pytest.approx(equity_before)

    # Every sample is corrected; the first comes after balance update #4
    samples = after.stats.positions[1]
    assert len(samples) == 587
    assert samples_before[0].internal_share_price == pytest.approx(1 / 18)
    assert samples[0].internal_share_price == pytest.approx(0.99974, abs=1e-5)
    assert samples[-1].internal_profit_pct == pytest.approx(-0.11215, abs=1e-5)
    assert samples[-1].profitability == pytest.approx(-0.11215, abs=1e-5)
    assert samples[-1].internal_profit_usd == pytest.approx(-1741.54, abs=0.01)
    assert samples[-1].profit_usd == pytest.approx(-1741.54, abs=0.01)
    assert [s.value for s in samples] == [s.value for s in samples_before]

    # Second run: nothing to do
    digest = _sha(state_file)
    assert _run(state_file, AUTO_APPROVE="true") == 0
    assert "Nothing to repair" in capsys.readouterr().out
    assert _sha(state_file) == digest
    assert len(_backups(state_file)) == 1


def test_lighter_ai_rebuild_overrides(state_file: Path, journal_file: Path, tmp_path: Path, capsys):
    """Overrides add or drop withdrawals; without one, balance update #4 is only a suspect."""
    # Ignoring the detected withdrawal applies none and flags it
    assert _run(state_file, "--dry-run", "--ignore-balance-update-id", "4") == 0
    output = capsys.readouterr().out
    assert "Withdrawal: balance update #4" not in output
    assert "Suspect, not applied: balance update #4" in output

    # Without a journal, naming the balance update gives the same repair
    journal_file.unlink()
    assert _run(state_file, "--capital-withdrawal-balance-update-id", "4", AUTO_APPROVE="true") == 0
    assert "Withdrawal: balance update #4 (--capital-withdrawal-balance-update-id)" in capsys.readouterr().out
    position = JSONFileStore(state_file).load().portfolio.open_positions[1]
    assert position.share_price_state.current_share_price == pytest.approx(0.887851, abs=1e-6)


@pytest.mark.parametrize("case", ["unknown-id", "positive", "other-vault", "ambiguous", "altered-sample", "no-terminal"])
def test_lighter_ai_rebuild_refusals(state_file: Path, journal_file: Path, case: str, capsys):
    """Each refusal exits non-zero and leaves the state byte-identical."""
    args = []
    env = {"AUTO_APPROVE": "true"}
    if case == "unknown-id":
        args = ["--capital-withdrawal-balance-update-id", "999999"]
    elif case == "positive":
        journal_file.unlink()
        args = ["--capital-withdrawal-balance-update-id", "5"]
    elif case == "other-vault":
        journal = json.loads(journal_file.read_text())
        journal["vault_address"] = "0x0000000000000000000000000000000000000001"
        journal_file.write_text(json.dumps(journal))
    elif case == "ambiguous":
        data = json.loads(state_file.read_text())
        data["portfolio"]["open_positions"]["1"]["balance_updates"]["5"]["quantity"] = "-16.995365"
        state_file.write_text(json.dumps(data))
    elif case == "altered-sample":
        data = json.loads(state_file.read_text())
        data["stats"]["positions"]["1"][100]["internal_profit_pct"] = 0.5
        state_file.write_text(json.dumps(data))
    elif case == "no-terminal":
        env = {}

    digest = _sha(state_file)
    assert _run(state_file, *args, **env) == 1
    assert "Nothing was written" in capsys.readouterr().out
    assert _sha(state_file) == digest
    assert _backups(state_file) == []
