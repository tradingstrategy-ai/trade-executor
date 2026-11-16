"""Typer black-box coverage for the resumable Lagoon Lighter test trade.

The fixed Ethereum fork keeps Lagoon, Safe, USDC, state persistence and NAV
posting real. Lighter's sequencer, SDK order transport and L1 withdrawal proof
are mocked because Anvil cannot reproduce those off-chain systems.
"""

import json
import logging
import os
import stat
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.erc_4626.vault_protocol.lagoon.config import get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
    deploy_automated_lagoon_vault,
)
from eth_defi.erc_4626.vault_protocol.lagoon.funding import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.api import LighterAccounts
from eth_defi.lighter.valuation import LighterEquity
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

from tradeexecutor.cli.commands import lagoon_lighter_test_trade
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag


#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: Public deterministic Anvil account zero key; never a production secret.
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
#: Sequencer-only Lighter account represented by the API mocks.
LIGHTER_ACCOUNT_INDEX = 222
#: Public delegated API key slot represented by the operator record.
LIGHTER_API_KEY_INDEX = 4
#: Safe USDC used to fund the real Lagoon vault before the default test deposit.
SAFE_USDC = Decimal("25")
#: Default additional Safe USDC assigned to the simulated Lighter round trip.
LIGHTER_DEPOSIT_USDC = Decimal("20")
#: Public base size returned by the mocked ETH market metadata lookup.
ETH_BASE_SIZE = Decimal("0.01")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM,
        reason="JSON_RPC_ETHEREUM environment variable required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the command test."""
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
    """Connect to the real single-node Ethereum Anvil fork."""
    web3 = create_multi_provider_web3(
        anvil_ethereum.json_rpc_url,
        default_http_timeout=(3, 250.0),
    )
    assert web3.eth.chain_id == 1
    return web3


@pytest.fixture()
def deployer(web3_ethereum: Web3) -> HotWallet:
    """Fund the deterministic deployer used by Lagoon deployment calls."""
    account = Account.from_key(DEPLOYER_PRIVATE_KEY)
    web3_ethereum.provider.make_request(
        "anvil_setBalance",
        [account.address, hex(100 * 10**18)],
    )
    return HotWallet(account)


@pytest.fixture()
def strategy_file() -> Path:
    """Return the static Ethereum Lighter strategy for the Typer command."""
    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "minimal_lighter_strategy.py"


def _lighter_equity(total: Decimal) -> LighterEquity:
    """Build the public Lighter equity response used by off-chain mocks."""
    return LighterEquity(
        account_index=LIGHTER_ACCOUNT_INDEX,
        collateral=total,
        unrealised_pnl=Decimal(0),
        total_asset_value=total,
        available_balance=total,
        initial_margin_requirement=Decimal(0),
        maintenance_margin_requirement=Decimal(0),
        position_count=0,
    )


def _prepare_command(
    *,
    web3_ethereum: Web3,
    deployer: HotWallet,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
    fail_open_once: bool,
) -> dict:
    """Deploy a funded Lagoon vault and mock Lighter for the Typer command.

    The fake deposit moves USDC out of the real Safe and reports its tx hash
    through ``tx_observer``, and the fake claim moves it back, so the booked
    reserve can be checked against the Safe at every treasury sync.
    """
    usdc = fetch_erc20_details(web3_ethereum, USDC_NATIVE_TOKEN[1], chain_id=1)
    funding_tx = usdc.transfer(
        deployer.address,
        SAFE_USDC + Decimal("2"),
    ).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)
    chain_config = get_lagoon_chain_config(1)
    deployment = deploy_automated_lagoon_vault(
        web3=web3_ethereum,
        deployer=deployer,
        asset_manager=deployer.address,
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[1],
            name="Lighter command test",
            symbol="LCT",
        ),
        safe_owners=[deployer.address],
        safe_threshold=1,
        any_asset=True,
        use_forge=True,
        factory_contract=chain_config.factory_contract,
        from_the_scratch=chain_config.from_the_scratch,
    )
    fund_lagoon_vault(
        web3=web3_ethereum,
        vault_address=deployment.vault.address,
        asset_manager=deployer.address,
        test_account_with_balance=deployer.address,
        trading_strategy_module_address=deployment.trading_strategy_module.address,
        amount=SAFE_USDC,
        hot_wallet=deployer,
    )

    operator_record_file = tmp_path / "lighter-operator.json"
    delegated_private_key = "0x" + "11" * 32
    operator_record_file.write_text(json.dumps({
        "deployments": {
            "ethereum": {
                "vault_address": deployment.vault.address,
                "safe_address": deployment.safe_address,
                "module_address": deployment.trading_strategy_module.address,
                "lighter_account_setup": {
                    "account_index": LIGHTER_ACCOUNT_INDEX,
                    "api_key_index": LIGHTER_API_KEY_INDEX,
                    "private_key": delegated_private_key,
                },
            },
        },
    }))
    operator_record_file.chmod(0o600)
    state_file = tmp_path / "lighter-command-state.json"
    journal_file = tmp_path / "lighter-command-journal.json"
    deployment_artifact_file = tmp_path / "lighter-command-test.deployment.json"
    deployment_artifact_file.write_text(json.dumps({
        "multichain": False,
        "deployments": {
            "ethereum": {
                "vault_address": deployment.vault.address,
                "module_address": deployment.trading_strategy_module.address,
                "is_satellite": False,
            },
        },
        "deployment_record": {
            "lighter_account_setup": {
                "account_index": LIGHTER_ACCOUNT_INDEX,
                "api_key_index": LIGHTER_API_KEY_INDEX,
            },
        },
    }))

    session = mocker.Mock()
    lighter_state = {"equity": Decimal(0), "position": Decimal(0), "fail_open_once": fail_open_once}
    observed = {"deposits": 0, "orders": [], "withdrawals": 0, "claims": 0, "deposit_tx_hashes": []}
    safe_address = deployment.safe_address
    web3_ethereum.provider.make_request("anvil_impersonateAccount", [safe_address])
    web3_ethereum.provider.make_request("anvil_setBalance", [safe_address, hex(10**18)])

    def fetch_equity(*_args: object, **_kwargs: object) -> LighterEquity:
        return _lighter_equity(lighter_state["equity"])

    def fetch_account(*_args: object, **_kwargs: object) -> dict[str, object]:
        if lighter_state["position"] == 0:
            return {"positions": []}
        return {
            "positions": [{
                "market_id": 0,
                "position": str(abs(lighter_state["position"])),
                "sign": 1,
            }],
        }

    def fake_deposit(*_args: object, **kwargs: object) -> str:
        deposit = kwargs["deposit_usdc"]
        assert deposit == LIGHTER_DEPOSIT_USDC
        deposit_tx = usdc.transfer(USDC_WHALE[1], deposit).build_transaction({"from": safe_address})
        tx_hash = web3_ethereum.eth.send_transaction(deposit_tx)
        kwargs["tx_observer"](Web3.to_hex(tx_hash), 0, "Deposit USDC from Safe to Lighter")
        assert_transaction_success_with_explanation(web3_ethereum, tx_hash)
        lighter_state["equity"] += deposit
        observed["deposits"] += 1
        observed["deposit_tx_hashes"].append(Web3.to_hex(tx_hash))
        return Web3.to_hex(tx_hash)

    def fake_wait_for_collateral(*args: object, **_kwargs: object) -> Decimal:
        assert args[2] == LIGHTER_DEPOSIT_USDC
        return lighter_state["equity"]

    async def fake_resolve_eth_order(*_args: object) -> tuple[int, Decimal, int, Decimal]:
        return 1_000_000, ETH_BASE_SIZE, 6, Decimal("25")

    async def fake_submit_order(*_args: object, **kwargs: object) -> None:
        is_ask = kwargs["is_ask"]
        observed["orders"].append((is_ask, kwargs["reduce_only"]))
        if not is_ask and lighter_state["fail_open_once"]:
            lighter_state["fail_open_once"] = False
            raise RuntimeError("mock Lighter sequencer is temporarily unavailable")
        lighter_state["position"] = Decimal(0) if is_ask else ETH_BASE_SIZE

    async def fake_wait_for_position(*_args: object, **kwargs: object) -> Decimal:
        if kwargs["expected_long"]:
            assert lighter_state["position"] == ETH_BASE_SIZE
        else:
            assert lighter_state["position"] == 0
        return lighter_state["position"]

    async def fake_withdrawal(_operator: object, amount: Decimal) -> None:
        assert amount == LIGHTER_DEPOSIT_USDC
        lighter_state["equity"] = Decimal(0)
        observed["withdrawals"] += 1

    async def fake_wait_for_claimable(
        _operator: object,
        amount: Decimal,
        _timeout: int,
        requested_at: int,
    ) -> Decimal:
        assert requested_at > 0
        assert amount == LIGHTER_DEPOSIT_USDC
        return amount

    def fake_claim(*args: object, **kwargs: object) -> str:
        claimable = kwargs["claimable_usdc"]
        vault = kwargs["vault"]
        claim_tx = usdc.transfer(
            vault.safe_address,
            claimable,
        ).transact({"from": USDC_WHALE[1]})
        assert_transaction_success_with_explanation(web3_ethereum, claim_tx)
        observed["claims"] += 1
        return Web3.to_hex(claim_tx)

    # The recorded reserve must match the Safe before each treasury sync, so
    # a booking error cannot be hidden by the sync replacing the reserve.
    reserve_checks = []
    original_sync_treasury = LagoonVaultSyncModel.sync_treasury

    def checked_sync_treasury(self, strategy_cycle_ts, state, *args, **kwargs):
        if state.portfolio.reserves:
            reserve_checks.append((
                state.portfolio.get_default_reserve_position().quantity,
                usdc.fetch_balance_of(safe_address),
            ))
        return original_sync_treasury(self, strategy_cycle_ts, state, *args, **kwargs)

    command_module = "tradeexecutor.cli.commands.lagoon_lighter_test_trade"
    mocker.patch.object(LagoonVaultSyncModel, "sync_treasury", checked_sync_treasury)
    mocker.patch(f"{command_module}.create_lighter_session", return_value=session)
    mocker.patch(f"{command_module}.fetch_lighter_total_equity", side_effect=fetch_equity)
    mocker.patch(f"{command_module}.fetch_lighter_account_by_index", side_effect=fetch_account)
    mocker.patch(f"{command_module}.deposit_usdc_from_lagoon_safe_into_lighter", side_effect=fake_deposit)
    mocker.patch(f"{command_module}.wait_for_lighter_collateral", side_effect=fake_wait_for_collateral)
    mocker.patch(f"{command_module}.fetch_lighter_withdrawal_delay", return_value=1_800)
    mocker.patch(f"{command_module}.resolve_eth_order", side_effect=fake_resolve_eth_order)
    mocker.patch(f"{command_module}.submit_lighter_order", side_effect=fake_submit_order)
    mocker.patch(f"{command_module}.wait_for_eth_position", side_effect=fake_wait_for_position)
    mocker.patch(f"{command_module}.request_lighter_withdrawal", side_effect=fake_withdrawal)
    mocker.patch(
        f"{command_module}.wait_for_lighter_withdrawal_claimable",
        side_effect=fake_wait_for_claimable,
    )
    mocker.patch(f"{command_module}.claim_usdc_to_lagoon_safe_from_lighter", side_effect=fake_claim)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        side_effect=fetch_equity,
    )
    # The NAV checks the Safe's Lighter account set, which the fake vault does not have on Lighter.
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_accounts",
        return_value=LighterAccounts(master_index=LIGHTER_ACCOUNT_INDEX, sub_account_indices=()),
    )
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-command-test",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "JSON_RPC_ETHEREUM": web3_ethereum.provider.endpoint_uri,
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": deployment.vault.address,
        "VAULT_ADAPTER_ADDRESS": deployment.trading_strategy_module.address,
        "LIGHTER_ACCOUNT_INDEX": str(LIGHTER_ACCOUNT_INDEX),
        "LIGHTER_OPERATOR_RECORD_FILE": operator_record_file.as_posix(),
        "LIGHTER_TEST_JOURNAL_FILE": journal_file.as_posix(),
        "LOG_LEVEL": "disabled",
        "UNIT_TESTING": "true",
    }
    mocker.patch.dict("os.environ", environment, clear=True)
    return {
        "cli": get_command(app),
        "usdc": usdc,
        "deployment": deployment,
        "state_file": state_file,
        "journal_file": journal_file,
        "observed": observed,
        "session": session,
        "reserve_checks": reserve_checks,
        "delegated_private_key": delegated_private_key,
    }


def _assert_transfers_booked(context: dict) -> TradingPosition:
    """The round trip is two transfer trades, so the share price stays at 1.0."""
    state = State.read_json_file(context["state_file"])
    journal = json.loads(context["journal_file"].read_text())
    lighter_positions = [
        position
        for position in state.portfolio.open_positions.values()
        if position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    ]
    assert len(lighter_positions) == 1
    position = lighter_positions[0]
    transfers = [t for t in position.trades.values() if TradeFlag.external_account_transfer in t.flags]
    assert [(t.is_success(), t.executed_quantity) for t in transfers] == [
        (True, LIGHTER_DEPOSIT_USDC),
        (True, -LIGHTER_DEPOSIT_USDC),
    ]
    assert journal["deposit_trade_id"] == transfers[0].trade_id
    assert journal["withdrawal_trade_id"] == transfers[1].trade_id
    assert transfers[0].other_data["lighter_deposit_tx_hashes"] == context["observed"]["deposit_tx_hashes"]
    assert all(bu.quantity != -LIGHTER_DEPOSIT_USDC for bu in position.balance_updates.values())
    sps = position.share_price_state
    assert sps.total_invested == pytest.approx(float(LIGHTER_DEPOSIT_USDC))
    assert sps.total_withdrawn == pytest.approx(float(LIGHTER_DEPOSIT_USDC))
    assert sps.current_share_price == pytest.approx(1.0)
    assert position.get_value() == pytest.approx(0)
    safe_balance = context["usdc"].fetch_balance_of(context["deployment"].safe_address)
    assert safe_balance == pytest.approx(SAFE_USDC)
    assert state.portfolio.get_default_reserve_position().quantity == pytest.approx(safe_balance)
    # The first sync imports the Safe balance into the fresh reserve
    assert context["reserve_checks"][1:]
    for recorded, onchain in context["reserve_checks"][1:]:
        assert recorded == pytest.approx(onchain)
    return position


@pytest.mark.timeout(600)
def test_cli_lagoon_lighter_test_trade_resumes_after_lighter_failure(
    web3_ethereum: Web3,
    deployer: HotWallet,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
    capsys: CaptureFixture[str],
    caplog: LogCaptureFixture,
) -> None:
    """Resume the complete Typer-managed Lighter test trade after an API failure.

    1. Deploy and fund a real Lagoon vault on the fixed Ethereum Anvil block.
    2. Omit the deposit option, mock Lighter-only steps and interrupt after deposit.
    3. Re-run the real Typer command and resume the open, close, withdrawal and NAV phases.
    4. Verify the transfers are booked as trades, the reserve, journal and secret redaction.
    """
    # 1-2. Deploy, fund, mock Lighter and interrupt after the deposit.
    context = _prepare_command(
        web3_ethereum=web3_ethereum,
        deployer=deployer,
        strategy_file=strategy_file,
        mocker=mocker,
        tmp_path=tmp_path,
        fail_open_once=True,
    )
    caplog.set_level(logging.INFO, logger="tradeexecutor.cli.commands.lagoon_lighter_test_trade")
    cli = context["cli"]
    journal_file = context["journal_file"]
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        cli.main(args=["lagoon-lighter-test-trade"], standalone_mode=False)
    interrupted_journal = json.loads(journal_file.read_text())
    assert interrupted_journal["phase"] == "deposited"
    assert context["observed"]["deposits"] == 1

    # 3. Re-run the real Typer command and resume the open, close, withdrawal and NAV phases.
    cli.main(args=["lagoon-lighter-test-trade"], standalone_mode=False)

    # 4. Verify the transfers are booked as trades, the reserve, journal and secret redaction.
    _assert_transfers_booked(context)
    journal = json.loads(journal_file.read_text())
    captured_output = capsys.readouterr()
    observed = context["observed"]
    delegated_private_key = context["delegated_private_key"]
    assert journal["phase"] == "complete"
    assert stat.S_IMODE(journal_file.stat().st_mode) == 0o600
    assert {key: observed[key] for key in ("deposits", "orders", "withdrawals", "claims")} == {
        "deposits": 1,
        "orders": [(False, False), (False, False), (True, True)],
        "withdrawals": 1,
        "claims": 1,
    }
    assert context["session"].close.call_count == 2
    assert delegated_private_key not in journal_file.read_text()
    assert delegated_private_key not in caplog.text
    assert delegated_private_key not in captured_output.out + captured_output.err
    assert "secure-withdrawal delay of 1800 seconds" in caplog.text


@pytest.mark.timeout(600)
def test_cli_lagoon_lighter_test_trade_books_transfers_once_across_crashes(
    web3_ethereum: Web3,
    deployer: HotWallet,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """A crash after a transfer is booked but before the journal advances books nothing twice.

    1. Crash after the deposit trade completes, before the journal reaches ``deposited``.
    2. Crash after the withdrawal trade completes, before the journal reaches ``withdrawn``.
    3. The third run completes with one deposit, one withdrawal and one claim.
    """
    context = _prepare_command(
        web3_ethereum=web3_ethereum,
        deployer=deployer,
        strategy_file=strategy_file,
        mocker=mocker,
        tmp_path=tmp_path,
        fail_open_once=False,
    )
    original_update_journal = lagoon_lighter_test_trade._update_journal
    crash_phases = {"deposited", "withdrawn"}

    def crashing_update_journal(config, journal, phase, **values):
        if phase in crash_phases:
            crash_phases.remove(phase)
            raise RuntimeError(f"crash before journal phase {phase}")
        return original_update_journal(config, journal, phase, **values)

    mocker.patch.object(lagoon_lighter_test_trade, "_update_journal", side_effect=crashing_update_journal)
    cli = context["cli"]
    journal_file = context["journal_file"]

    # 1. Crash after the deposit trade completes
    with pytest.raises(RuntimeError, match="crash before journal phase deposited"):
        cli.main(args=["lagoon-lighter-test-trade"], standalone_mode=False)
    assert json.loads(journal_file.read_text())["phase"] == "created"

    # 2. Crash after the withdrawal trade completes
    with pytest.raises(RuntimeError, match="crash before journal phase withdrawn"):
        cli.main(args=["lagoon-lighter-test-trade"], standalone_mode=False)
    assert json.loads(journal_file.read_text())["phase"] == "withdrawal_requested"

    # 3. Complete
    cli.main(args=["lagoon-lighter-test-trade"], standalone_mode=False)
    assert json.loads(journal_file.read_text())["phase"] == "complete"
    observed = context["observed"]
    assert (observed["deposits"], observed["withdrawals"], observed["claims"]) == (1, 1, 1)
    _assert_transfers_booked(context)
