"""Loading Lighter signer records for the master account and for sub-accounts."""

import json
from pathlib import Path

import pytest

from tradeexecutor.ethereum.lighter.lighter_routing import LighterRoutingConfig
from tradeexecutor.exchange_account.lighter_operator import load_lighter_operator_record, load_lighter_sub_account_operator_records

#: Master account index of the example deployment.
MASTER_INDEX = 127
#: Sub-account indices of the example deployment.
SUB_INDICES = (2**47 + 2, 2**47 + 1)
#: Example addresses; never real deployments.
VAULT = "0x0000000000000000000000000000000000000001"
SAFE = "0x0000000000000000000000000000000000000002"
MODULE = "0x0000000000000000000000000000000000000003"


def _write_private(path: Path, payload: dict) -> Path:
    """Write a mode-0600 JSON record as the deployment does."""
    path.write_text(json.dumps(payload))
    path.chmod(0o600)
    return path


def test_lighter_operator_records_for_master_and_sub_accounts(tmp_path: Path) -> None:
    """Load the master operator record and sub-account hand-over files as signers.

    The executor normally signs only as the master. An operator may also give
    it a sub-account's hand-over file, so transfers sent from that sub-account
    can be executed and reconciled; both record formats must load into the
    same signer type and the routing config must find the right signer.

    1. Load the executor's deployment record holding the master key.
    2. Load two hand-over files and check they are keyed by sub-account index.
    3. Check the routing config returns the master, a sub-account or nothing.
    4. Reject a duplicate sub-account record and an incomplete hand-over file.
    """
    # 1. Load the executor's deployment record holding the master key.
    master_file = _write_private(tmp_path / "vault-record.json", {
        "deployments": {
            "ethereum": {
                "vault_address": VAULT,
                "safe_address": SAFE,
                "module_address": MODULE,
                "lighter_account_setup": {"account_index": MASTER_INDEX, "api_key_index": 4, "private_key": "0x" + "11" * 40},
            },
        },
    })
    master = load_lighter_operator_record(master_file)
    assert (master.account_index, master.api_key_index, master.module_address) == (MASTER_INDEX, 4, MODULE)

    # 2. Load two hand-over files and check they are keyed by sub-account index.
    handover_files = [
        _write_private(tmp_path / f"vault-record.nautilus.{index}.json", {
            "account_index": index,
            "api_key_index": 4,
            "public_key": "0x" + "22" * 40,
            "change_pubkey_tx_hash": "0xdead",
            "safe_address": SAFE,
            "vault_address": VAULT,
            "module_address": MODULE,
            "private_key": "0x" + f"{index:080x}",
        })
        for index in SUB_INDICES
    ]
    subs = load_lighter_sub_account_operator_records(",".join(str(path) for path in handover_files) + ",")
    assert sorted(subs) == sorted(SUB_INDICES)
    assert subs[SUB_INDICES[0]].api_private_key == "0x" + f"{SUB_INDICES[0]:080x}"
    assert subs[SUB_INDICES[0]].safe_address == SAFE
    assert load_lighter_sub_account_operator_records(None) == {}

    # 3. Check the routing config returns the master, a sub-account or nothing.
    config = LighterRoutingConfig(operator_record=master, sub_account_operator_records=subs)
    assert config.get_operator_record(MASTER_INDEX) is master
    assert config.get_operator_record(SUB_INDICES[1]) is subs[SUB_INDICES[1]]
    assert config.get_operator_record(999) is None
    assert LighterRoutingConfig().get_operator_record(MASTER_INDEX) is None

    # 4. Reject a duplicate sub-account record and an incomplete hand-over file.
    with pytest.raises(ValueError, match="more than one operator record"):
        load_lighter_sub_account_operator_records(f"{handover_files[0]},{handover_files[0]}")
    incomplete = _write_private(tmp_path / "incomplete.json", {"account_index": SUB_INDICES[0], "private_key": "0x00"})
    with pytest.raises(ValueError, match="incomplete"):
        load_lighter_operator_record(incomplete)
