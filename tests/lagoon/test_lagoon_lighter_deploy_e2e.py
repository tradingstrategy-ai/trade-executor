"""Black-box Lagoon Lighter deployment test on an Ethereum Anvil fork.

The fork executes the complete Lagoon/Safe/Lighter-L1 ceremony. Only Lighter's
sequencer-owned account indexing and public API observations are mocked.
"""

import json
import logging
import os
import stat
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from eth_account import Account
from eth_defi.lighter.api import LIGHTER_MIN_MAINNET_USDC
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LIGHTER_BOOTSTRAP_SUBSCRIPTION
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.lighter.testing import register_lighter_account_on_anvil
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from pytest import CaptureFixture, LogCaptureFixture
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account.lighter import create_lighter_exchange_account_pair
from tradeexecutor.state.balance_update import BalanceUpdateCause
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State

#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM,
        reason="JSON_RPC_ETHEREUM environment variable required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]

#: Public deterministic Anvil account zero key; never a production secret.
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
#: Synthetic account returned by the mocked Lighter sequencer registration.
LIGHTER_ACCOUNT_INDEX = 123
#: First SDK-supported delegated API-key slot used by deployment.
LIGHTER_API_KEY_INDEX = MIN_API_KEY_INDEX
#: Required owner-only permissions for the generated operator record.
PRIVATE_RECORD_MODE = 0o600


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the deployment test."""
    launch = anvil_fork_pool.get_launch(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
        unlocked_addresses=[USDC_WHALE[1]],
    )
    snapshot = evm_snapshot_revert(launch)
    next(snapshot)
    try:
        yield launch
    finally:
        next(snapshot, None)


@pytest.fixture()
def web3_ethereum(anvil_ethereum: AnvilLaunch) -> Web3:
    web3 = create_multi_provider_web3(
        anvil_ethereum.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 1
    return web3


@pytest.fixture()
def deployer() -> Account:
    return Account.from_key(DEPLOYER_PRIVATE_KEY)


@pytest.fixture()
def strategy_file() -> Path:
    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "minimal_lighter_strategy.py"


@pytest.mark.timeout(600)
def test_cli_lagoon_deploy_lighter_on_external_anvil(
    anvil_ethereum: AnvilLaunch,
    web3_ethereum: Web3,
    deployer: Account,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
    capsys: CaptureFixture[str],
    caplog: LogCaptureFixture,
) -> None:
    """Run the real Typer deployment while mocking only Lighter public state.

    1. Fund the deterministic deployer with fork ETH and the complete initial native-USDC subscription.
    2. Forge only the sequencer-owned Lighter account registration and API polls.
    3. Invoke the real Typer command and all Lagoon/Safe/Lighter-L1 writers.
    4. Initialise the executor and run its normal Lagoon accounting correction.
    5. Verify receipts, public/private artifact boundaries and the fully accounted initial capital.
    """
    # 1. Fund the deterministic deployer with fork ETH and the complete initial native-USDC subscription.
    web3_ethereum.provider.make_request(
        "anvil_setBalance",
        [deployer.address, hex(100 * 10**18)],
    )
    usdc = fetch_erc20_details(
        web3_ethereum,
        USDC_NATIVE_TOKEN[1],
        chain_id=1,
    )
    funding_tx = usdc.contract.functions.transfer(
        deployer.address,
        usdc.convert_to_raw(LIGHTER_BOOTSTRAP_SUBSCRIPTION),
    ).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)

    observed: dict[str, object] = {}

    class FakeLighterSession:
        def close(self) -> None:
            observed["session_closed"] = True

    def fake_wait_for_account(session: Any, owner: str) -> int:
        del session
        observed["safe_address"] = owner
        register_lighter_account_on_anvil(
            web3_ethereum,
            owner,
            LIGHTER_ACCOUNT_INDEX,
        )
        return LIGHTER_ACCOUNT_INDEX

    def fake_wait_for_collateral(
        session: Any,
        account_index: int,
        expected_collateral: Decimal,
        **kwargs: Any,
    ) -> Decimal:
        del session, kwargs
        observed["collateral_account_index"] = account_index
        observed["expected_collateral"] = expected_collateral
        return Decimal("1")

    def fake_wait_for_api_key(
        session: Any,
        account_index: int,
        api_key_index: int,
        public_key: str,
        **kwargs: Any,
    ) -> None:
        del session
        observed["api_key_account_index"] = account_index
        observed["api_key_index"] = api_key_index
        observed["public_key"] = public_key
        observed["api_key_kwargs"] = kwargs
        return None

    # 2. Forge only the sequencer-owned Lighter account registration and API polls.
    deployment_module = "eth_defi.erc_4626.vault_protocol.lagoon.deployment"
    mocker.patch(f"{deployment_module}.create_lighter_session", lambda: FakeLighterSession())
    mocker.patch(f"{deployment_module}.wait_for_lighter_account", fake_wait_for_account)
    mocker.patch(f"{deployment_module}.wait_for_lighter_collateral", fake_wait_for_collateral)
    mocker.patch(f"{deployment_module}.wait_for_lighter_api_key", fake_wait_for_api_key)

    vault_record_file = tmp_path / "lighter-vault-record.txt"
    state_file = tmp_path / "lighter-state.json"
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-blackbox",
        "NAME": "lighter-blackbox",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ETHEREUM": anvil_ethereum.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Lighter Blackbox",
        "FUND_SYMBOL": "LGT",
        "MULTISIG_OWNERS": deployer.address,
        "ANY_ASSET": "true",
        "GENERATE_LIGHTER_API_KEY": "true",
        "LIGHTER_API_KEY_INDEX": str(LIGHTER_API_KEY_INDEX),
        "SAFE_SALT_NONCE": "42",
    }
    mocker.patch.dict("os.environ", environment, clear=True)

    # 3. Invoke the real Typer command and all Lagoon/Safe/Lighter-L1 writers.
    cli = get_command(app)
    with caplog.at_level(logging.INFO):
        cli.main(args=["lagoon-deploy-vault"], standalone_mode=False)

    # 4. Initialise the executor and run its normal Lagoon accounting correction.
    operator_json_path = vault_record_file.with_suffix(".json")
    operator_payload = json.loads(operator_json_path.read_text())
    setup = operator_payload["deployments"]["ethereum"]["lighter_account_setup"]
    private_key = setup["private_key"]
    assert stat.S_IMODE(operator_json_path.stat().st_mode) == PRIVATE_RECORD_MODE

    for tx_key in ("deposit_tx_hash", "change_pubkey_tx_hash"):
        receipt = web3_ethereum.eth.get_transaction_receipt(setup[tx_key])
        assert receipt["status"] == 1

    safe_address = operator_payload["deployments"]["ethereum"]["safe_address"]
    assert usdc.fetch_balance_of(safe_address) == LIGHTER_BOOTSTRAP_SUBSCRIPTION - LIGHTER_MIN_MAINNET_USDC

    assert observed["safe_address"] == operator_payload["deployments"]["ethereum"]["safe_address"]
    assert observed["collateral_account_index"] == LIGHTER_ACCOUNT_INDEX
    assert observed["expected_collateral"] == pytest.approx(LIGHTER_MIN_MAINNET_USDC)
    assert observed["api_key_account_index"] == LIGHTER_ACCOUNT_INDEX
    assert observed["api_key_index"] == LIGHTER_API_KEY_INDEX
    assert observed["public_key"] == setup["public_key"]
    assert observed["session_closed"] is True

    public_paths = [
        vault_record_file,
        vault_record_file.with_name("deployment-report.md"),
        state_file.with_name("lighter-blackbox.deployment.json"),
    ]
    captured_output = capsys.readouterr()
    public_text = "\n".join(path.read_text() for path in public_paths)
    public_text += captured_output.out + captured_output.err
    assert private_key not in public_text
    assert private_key[2:18] not in public_text
    assert private_key not in caplog.text
    assert private_key[2:18] not in caplog.text

    runtime_payload = json.loads(public_paths[-1].read_text())
    runtime_setup = runtime_payload["deployments"]["ethereum"]["lighter_account_setup"]
    deployment = operator_payload["deployments"]["ethereum"]
    assert deployment["vault_address"] == runtime_payload["deployments"]["ethereum"]["vault_address"]
    assert deployment["module_address"] == runtime_payload["deployments"]["ethereum"]["module_address"]
    assert "private_key" not in runtime_setup
    assert runtime_setup["account_index"] == LIGHTER_ACCOUNT_INDEX
    assert runtime_setup["api_key_index"] == LIGHTER_API_KEY_INDEX
    assert runtime_setup["public_key"] == setup["public_key"]

    pair = create_lighter_exchange_account_pair(
        quote=AssetIdentifier(
            chain_id=1,
            address=USDC_NATIVE_TOKEN[1],
            token_symbol="USDC",
            decimals=6,
        ),
        account_index=runtime_setup["account_index"],
    )
    assert pair.get_exchange_account_id() == setup["account_index"]
    for value in (setup["account_index"], setup["api_key_index"], setup["public_key"], setup["deposit_tx_hash"], setup["change_pubkey_tx_hash"]):
        assert str(value) in public_text

    class FakeLighterEquity:
        """Provide the sequencer value Anvil cannot create."""

        collateral = LIGHTER_MIN_MAINNET_USDC
        unrealised_pnl = Decimal(0)

        def get_total(self) -> Decimal:
            """Return the confirmed activation collateral."""
            return LIGHTER_MIN_MAINNET_USDC

    os.environ["VAULT_ADDRESS"] = operator_payload["deployments"]["ethereum"]["vault_address"]
    os.environ["VAULT_ADAPTER_ADDRESS"] = operator_payload["deployments"]["ethereum"]["module_address"]
    os.environ["LIGHTER_ACCOUNT_INDEX"] = str(LIGHTER_ACCOUNT_INDEX)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        return_value=FakeLighterEquity(),
    )
    cli.main(args=["init"], standalone_mode=False)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(args=["correct-accounts", "--process-redemption"], standalone_mode=False)
    assert exit_info.value.code == 0

    # 5. Verify receipts, public/private artifact boundaries and the fully accounted initial capital.
    state = State.read_json_file(state_file)
    reserve = state.portfolio.get_default_reserve_position()
    lighter_position = next(iter(state.portfolio.open_positions.values()))
    settlement = next(iter(reserve.balance_updates.values()))
    activation_receipt = web3_ethereum.eth.get_transaction_receipt(setup["deposit_tx_hash"])
    assert state.sync.deployment.block_number <= activation_receipt["blockNumber"]
    assert settlement.cause == BalanceUpdateCause.deposit_and_redemption
    assert settlement.quantity == LIGHTER_BOOTSTRAP_SUBSCRIPTION
    assert reserve.quantity == LIGHTER_BOOTSTRAP_SUBSCRIPTION - LIGHTER_MIN_MAINNET_USDC
    assert lighter_position.get_quantity() == LIGHTER_MIN_MAINNET_USDC
    assert state.portfolio.get_net_asset_value() == LIGHTER_BOOTSTRAP_SUBSCRIPTION
