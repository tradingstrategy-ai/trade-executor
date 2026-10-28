"""Unit checks of the command-line Lighter manual test's building blocks.

The shell script, its ``lighter_manual`` helpers, the ``--lighter-transfer``
parser and the deployment key backup are exercised end to end in
``test_lagoon_lighter_cli_manual_e2e.py``; these checks cover their edge
cases without a chain.
"""

import shutil
import stat
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi.lighter.api_key import LighterApiKey

from tradeexecutor.cli.commands.lagoon_deploy_vault import LighterKeyBackup
from tradeexecutor.cli.testtrade import parse_lighter_transfer
from tradeexecutor.testing.lighter_manual import ManualTestError, check_expectations, format_usdc, parse_expectations

#: The manual test script.
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lagoon" / "manual-lighter-subaccounts-cli.sh"


def _key(number: int) -> LighterApiKey:
    """Return a deterministic API key pair for file-name and content checks."""
    return LighterApiKey(api_key_index=4, private_key=f"0x{number:080x}", public_key="0x" + f"{number:02x}" * 40)


def test_lighter_manual_building_blocks(tmp_path: Path) -> None:
    """The helpers parse, compare and back up as the script expects, and the script parses.

    1. Parse balance expectations and compare them, including the <= form for dust
    2. Format amounts for the shell
    3. Parse --lighter-transfer account pairs
    4. Back up two keys with private modes and remove them
    5. Check the shell script parses, and passes shellcheck when it is installed
    """
    # 1. Parse balance expectations and compare them, including the <= form for dust
    expectations = parse_expectations(["safe=6", "master=1", "trend<=0.99", "arb=3"])
    rows = check_expectations(expectations, {"safe": Decimal("6.005"), "master": Decimal(1), "trend": Decimal("0.5"), "arb": Decimal(3)}, Decimal("0.01"))
    assert [row[4] for row in rows] == [True, True, True, True]
    rows = check_expectations(expectations, {"safe": Decimal(6), "master": Decimal("0.5"), "trend": Decimal("1.5"), "arb": Decimal(3)}, Decimal("0.01"))
    assert [row[0] for row in rows if not row[4]] == ["master", "trend"]

    # 2. Format amounts for the shell
    assert format_usdc(Decimal("3.0000009")) == "3.000000"
    assert format_usdc(Decimal("1E+1")) == "10.000000"

    # 3. Parse --lighter-transfer account pairs
    assert parse_lighter_transfer("trend:arb") == ("trend", "arb")
    assert parse_lighter_transfer(" master : trend ") == ("master", "trend")

    # 4. Back up two keys with private modes and remove them
    backup = LighterKeyBackup(tmp_path / "key-backup")
    backup.save(_key(1), "master")
    backup.save(_key(2), "sub-account trend")
    assert [path.name for path in backup.saved] == ["lighter-api-key-1-0101010101010101.json", "lighter-api-key-2-0202020202020202.json"]
    assert stat.S_IMODE(backup.directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in backup.saved)
    assert '"role": "sub-account trend"' in backup.saved[1].read_text() and _key(2).private_key in backup.saved[1].read_text()
    backup.remove()
    assert list(backup.directory.iterdir()) == []

    # 5. Check the shell script parses, and passes shellcheck when it is installed
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    assert SCRIPT.stat().st_mode & stat.S_IXUSR
    if shutil.which("shellcheck"):
        subprocess.run(["shellcheck", "--severity=warning", str(SCRIPT)], check=True)


def test_lighter_manual_building_blocks_refuse_bad_input(tmp_path: Path) -> None:
    """Malformed input is refused, and a backup never overwrites earlier recovery material.

    A retried deployment that regenerated the same key file name must fail
    rather than replace the only copy of a key registered by the first attempt.

    1. Refuse malformed balance expectations and unknown balance names
    2. Refuse malformed and self-referencing --lighter-transfer pairs
    3. Refuse to overwrite an existing key backup file
    """
    # 1. Refuse malformed balance expectations and unknown balance names
    for bad in ("safe", "safe=-1", "safe>=1", "safe=abc"):
        with pytest.raises(ManualTestError):
            parse_expectations([bad])
    with pytest.raises(ManualTestError, match="Unknown balance 'treasury'"):
        check_expectations(parse_expectations(["treasury=1"]), {"safe": Decimal(1)}, Decimal("0.01"))

    # 2. Refuse malformed and self-referencing --lighter-transfer pairs
    for bad in ("trend", "trend:", ":arb", "trend:arb:master", "trend:trend"):
        with pytest.raises(ValueError):
            parse_lighter_transfer(bad)

    # 3. Refuse to overwrite an existing key backup file
    first = LighterKeyBackup(tmp_path / "key-backup")
    first.save(_key(1), "master")
    retry = LighterKeyBackup(tmp_path / "key-backup")
    with pytest.raises(FileExistsError):
        retry.save(_key(1), "master")
    assert _key(1).private_key in first.saved[0].read_text()
