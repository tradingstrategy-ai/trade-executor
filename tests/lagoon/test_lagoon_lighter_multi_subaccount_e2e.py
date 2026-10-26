"""Black-box Lagoon vault with a Lighter master account and two sub-accounts.

The fixed Ethereum fork runs every Lagoon, Safe, guard, USDC and ZkLighter
call for real, including the sub-account ``changePubKey`` transactions, the
key-free ``withdraw(subAccountIndex)`` requests and the Safe
``withdrawPendingBalance`` claims. Only Lighter's sequencer is replaced, by
:py:class:`tradeexecutor.testing.lighter_core.FakeLighterCore`, because Anvil
cannot create Lighter accounts, execute L2 transfers or prove withdrawals.
"""

import json
import logging
import os
import stat
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LIGHTER_BOOTSTRAP_SUBSCRIPTION
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.lighter.api import LIGHTER_MIN_MAINNET_USDC
from eth_defi.lighter.pubkey import MIN_API_KEY_INDEX
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.safe.sweep import sweep_safe_to_owner
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.vault.base import VaultSpec
from pytest import CaptureFixture, LogCaptureFixture
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account.lighter import is_lighter_sub_account_pair
from tradeexecutor.state.position import TradingPosition
from tradeexecutor.state.state import State, UncleanState
from tradeexecutor.state.trade import TradeExecution, TradeFlag
from tradeexecutor.testing.lighter_core import FakeLighterCore

#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: API key required by the strategy factory even though this strategy has no market data.
TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

pytestmark = [
    pytest.mark.skipif(
        not JSON_RPC_ETHEREUM or not TRADING_STRATEGY_API_KEY,
        reason="JSON_RPC_ETHEREUM and TRADING_STRATEGY_API_KEY are required",
    ),
    pytest.mark.warm_rpc_test_group,
    pytest.mark.xdist_group("fork:ethereum:midnight"),
]

#: Public deterministic Anvil account zero key; never a production secret.
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
#: Master account index assigned by the fake sequencer.
MASTER_INDEX = 127
#: Lagoon subscription made after deployment.
SUBSCRIPTION_USDC = Decimal(200)
#: USDC kept in the Safe by the strategy.
SAFE_BUFFER_USDC = Decimal(20)
#: Fee charged by the fake sequencer on each funding transfer in the first rebalance.
#:
#: Lighter charges nothing for transfers between accounts of one owner; a
#: non-zero fee exercises the fee accounting path.
FUNDING_FEE_USDC = Decimal("0.5")
#: Owner-only mode required for private records.
PRIVATE_RECORD_MODE = 0o600


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the test."""
    launch = anvil_fork_pool.get_launch(
        JSON_RPC_ETHEREUM,
        ETHEREUM_MIDNIGHT_BLOCK,
        unlocked_addresses=[USDC_WHALE[1]],
        inherit_stdio=True,
    )
    snapshot = evm_snapshot_revert(launch)
    next(snapshot)
    try:
        yield launch
    finally:
        next(snapshot, None)


@pytest.fixture()
def web3_ethereum(anvil_ethereum: AnvilLaunch) -> Web3:
    """Connect to the Ethereum Anvil fork."""
    web3 = create_multi_provider_web3(anvil_ethereum.json_rpc_url, default_http_timeout=(3, 250.0))
    assert web3.eth.chain_id == 1
    return web3


@pytest.fixture()
def usdc(web3_ethereum: Web3) -> TokenDetails:
    """Native Ethereum USDC."""
    return fetch_erc20_details(web3_ethereum, USDC_NATIVE_TOKEN[1], chain_id=1)


@pytest.fixture()
def fake_lighter_core(web3_ethereum: Web3, mocker: MockerFixture) -> FakeLighterCore:
    """Replace Lighter's sequencer with one stateful fake behind every seam.

    Mocked because Anvil cannot run Lighter's L2: account and sub-account
    creation, transfers, equity and withdrawal proofs come from the fake,
    while all Ethereum transactions stay real.
    """
    core = FakeLighterCore(web3_ethereum, MASTER_INDEX)
    for target, replacement in core.get_patches():
        mocker.patch(target, new=replacement)
    return core


def _run(cli, *args: str) -> None:
    """Invoke a real Typer command, accepting a successful ``SystemExit``."""
    try:
        cli.main(args=list(args), standalone_mode=False)
    except SystemExit as exit_info:
        assert exit_info.code in (None, 0), f"{args} exited with {exit_info.code}"


def _positions(state: State) -> tuple[TradingPosition, TradingPosition, TradingPosition]:
    """Return the master and the two sub-account positions in creation order."""
    lighter = [position for position in state.portfolio.open_positions.values() if position.is_exchange_account()]
    master = next(position for position in lighter if not is_lighter_sub_account_pair(position.pair))
    subs = sorted((position for position in lighter if is_lighter_sub_account_pair(position.pair)), key=lambda position: -position.pair.get_exchange_account_id())
    return master, subs[0], subs[1]


def _transfer_trades(state: State, flag: TradeFlag) -> list[TradeExecution]:
    """Return all custody trades with ``flag``."""
    return [trade for trade in state.portfolio.get_all_trades() if flag in (trade.flags or set())]


@pytest.mark.timeout(900)
def test_cli_lagoon_lighter_multi_subaccount_lifecycle(
    anvil_ethereum: AnvilLaunch,
    web3_ethereum: Web3,
    usdc: TokenDetails,
    fake_lighter_core: FakeLighterCore,
    tmp_path: Path,
    capsys: CaptureFixture[str],
    caplog: LogCaptureFixture,
    mocker: MockerFixture,
) -> None:
    """Deploy, fund, rebalance, interrupt, recover and redeem a two sub-account Lighter vault.

    One Lagoon vault owns a Lighter master account and two sub-accounts, each a
    separate executor position. The executor moves capital Safe to master,
    master to sub-account, and sub-account back to the Safe without the
    sub-account keys, and the Lagoon NAV includes every account.

    1. Deploy with ``lagoon-deploy-vault`` and two sub-accounts; check keys, hand-over files and secrets.
    2. Initialise state and account for the bootstrap subscription with one position per account; flag an untracked sub-account.
    3. Subscribe 200 USDC and run ``start``: Lagoon settles it and the idle Safe cash is deposited to the master.
    4. Run ``start`` with 50/50 weights and a transfer fee: the master funds both sub-accounts.
    5. Run ``start`` with 75/25 weights: sub-account 2 is withdrawn to the Safe with the key-free L1 path, then redeposited.
    6. Interrupt the next funding transfer, check ``start`` refuses to continue, and reconcile it with ``correct-accounts``.
    7. Lock margin on sub-account 1 and check a withdrawal is capped to its free collateral; withdraw from sub-account 2 with ``lighter-move-funds``.
    8. Request a redemption larger than the Safe while the master is empty and check it is funded from a sub-account.
    9. Drain every account for a full redemption, including a master API withdrawal, and redeem with ``lagoon-redeem``.
    10. Sweep stray USDC and ETH from the emptied Safe back to the deployer, as the mainnet tutorial does.
    """
    deployer = Account.from_key(DEPLOYER_PRIVATE_KEY)
    web3_ethereum.provider.make_request("anvil_setBalance", [deployer.address, hex(100 * 10**18)])
    funding_tx = usdc.contract.functions.transfer(
        deployer.address,
        usdc.convert_to_raw(LIGHTER_BOOTSTRAP_SUBSCRIPTION + SUBSCRIPTION_USDC),
    ).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)

    strategies = Path(__file__).resolve().parents[2] / "strategies" / "test_only"
    vault_record_file = tmp_path / "lighter-vault-record.txt"
    operator_record_file = vault_record_file.with_suffix(".json")
    state_file = tmp_path / "lighter-multi.json"
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-multi",
        "NAME": "lighter-multi",
        "STRATEGY_FILE": (strategies / "lighter_multi_subaccount_strategy.py").as_posix(),
        "JSON_RPC_ETHEREUM": anvil_ethereum.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "info",
        "FILE_LOG_LEVEL": "NONE",
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_RECORD_FILE": str(vault_record_file),
        "FUND_NAME": "Lighter Multi",
        "FUND_SYMBOL": "LGM",
        "MULTISIG_OWNERS": deployer.address,
        "ANY_ASSET": "true",
        "PERFORMANCE_FEE": "0",
        "MANAGEMENT_FEE": "0",
        "GENERATE_LIGHTER_API_KEY": "true",
        "LIGHTER_API_KEY_INDEX": str(MIN_API_KEY_INDEX),
        "SAFE_SALT_NONCE": "43",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "CACHE_PATH": (tmp_path / "cache").as_posix(),
        "MIN_GAS_BALANCE": "0",
    }
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)

    # 1. Deploy with lagoon-deploy-vault and two sub-accounts; check keys, hand-over files and secrets.
    with caplog.at_level(logging.INFO):
        _run(cli, "lagoon-deploy-vault")
    sub_1, sub_2 = fake_lighter_core.get_sub_account_indices()
    operator_payload = json.loads(operator_record_file.read_text())
    deployment = operator_payload["deployments"]["ethereum"]
    setup = deployment["lighter_account_setup"]
    master_key = setup["private_key"]
    assert setup["account_index"] == MASTER_INDEX
    assert [sub["account_index"] for sub in setup["sub_accounts"]] == [sub_1, sub_2]
    assert all("private_key" not in sub for sub in setup["sub_accounts"])
    assert [sub["label"] for sub in setup["sub_accounts"]] == ["trend", "arb"]
    handover = {
        index: json.loads(operator_record_file.with_name(f"{operator_record_file.stem}.nautilus.{account_id}.json").read_text())
        for account_id, index in (("trend", sub_1), ("arb", sub_2))
    }
    for (account_id, index), record in zip((("trend", sub_1), ("arb", sub_2)), handover.values()):
        path = operator_record_file.with_name(f"{operator_record_file.stem}.nautilus.{account_id}.json")
        assert stat.S_IMODE(path.stat().st_mode) == PRIVATE_RECORD_MODE
        assert record["account_index"] == index and record["label"] == account_id
        assert record["safe_address"] == deployment["safe_address"]
        assert fake_lighter_core.get_account(index).api_keys[MIN_API_KEY_INDEX] == record["public_key"]
        assert record["private_key"] not in operator_record_file.read_text()
        receipt = web3_ethereum.eth.get_transaction_receipt(record["change_pubkey_tx_hash"])
        assert receipt["status"] == 1
    public_text = vault_record_file.read_text() + state_file.with_name("lighter-multi.deployment.json").read_text()
    captured = capsys.readouterr()
    public_text += captured.out + captured.err + caplog.text
    for secret in (master_key, handover[sub_1]["private_key"], handover[sub_2]["private_key"]):
        assert secret not in public_text
    assert str(sub_1) in public_text and str(sub_2) in public_text
    safe_address = deployment["safe_address"]
    assert usdc.fetch_balance_of(safe_address) == LIGHTER_BOOTSTRAP_SUBSCRIPTION - LIGHTER_MIN_MAINNET_USDC

    # 2. Initialise state and account for the bootstrap subscription with one position per account; flag an untracked sub-account.
    os.environ.update({
        "LIGHTER_SUB_ACCOUNT_WEIGHTS": "",
        "LIGHTER_TRANSFER_FEE_USD": "0",
        "LIGHTER_OPERATOR_RECORD_FILE": operator_record_file.as_posix(),
        "LIGHTER_TRANSFER_TIMEOUT": "5",
        "LIGHTER_WITHDRAWAL_TIMEOUT": "30",
        "VAULT_ADDRESS": deployment["vault_address"],
        "VAULT_ADAPTER_ADDRESS": deployment["module_address"],
        "RUN_SINGLE_CYCLE": "true",
        "LOG_LEVEL": "disabled",
    })
    _run(cli, "init")
    _run(cli, "correct-accounts", "--process-redemption")
    state = State.read_json_file(state_file)
    master, sub_1_position, sub_2_position = _positions(state)
    assert len({master.pair.get_identifier(), sub_1_position.pair.get_identifier(), sub_2_position.pair.get_identifier()}) == 3
    assert len({master.pair.internal_id, sub_1_position.pair.internal_id, sub_2_position.pair.internal_id}) == 3
    assert master.pair.get_ticker() == "LIGHTER-ACCOUNT-USDC"
    assert (sub_1_position.pair.get_ticker(), sub_2_position.pair.get_ticker()) == ("LIGHTER-trend-USDC", "LIGHTER-arb-USDC")
    assert (master.get_quantity(), sub_1_position.get_quantity(), sub_2_position.get_quantity()) == (LIGHTER_MIN_MAINNET_USDC, 0, 0)
    assert state.portfolio.get_default_reserve_position().quantity == LIGHTER_BOOTSTRAP_SUBSCRIPTION - LIGHTER_MIN_MAINNET_USDC
    _run(cli, "check-accounts")
    # A sub-account the strategy does not track would be missing from NAV.
    untracked = fake_lighter_core.create_sub_account()
    with pytest.raises(AssertionError, match="exited with 1"):
        _run(cli, "check-accounts")
    del fake_lighter_core.accounts[untracked]

    # 3. Subscribe 200 USDC and run start: Lagoon settles it and the idle Safe cash is deposited to the master.
    vault = LagoonVault(web3_ethereum, VaultSpec(1, deployment["vault_address"]), trading_strategy_module_address=deployment["module_address"])
    assert_transaction_success_with_explanation(web3_ethereum, usdc.contract.functions.approve(vault.address, usdc.convert_to_raw(SUBSCRIPTION_USDC)).transact({"from": deployer.address}))
    assert_transaction_success_with_explanation(web3_ethereum, vault.request_deposit(deployer.address, usdc.convert_to_raw(SUBSCRIPTION_USDC)).transact({"from": deployer.address}))
    _run(cli, "start")
    assert_transaction_success_with_explanation(web3_ethereum, vault.finalise_deposit(deployer.address).transact({"from": deployer.address}))
    deposited = LIGHTER_BOOTSTRAP_SUBSCRIPTION - LIGHTER_MIN_MAINNET_USDC + SUBSCRIPTION_USDC - SAFE_BUFFER_USDC
    master_equity = LIGHTER_MIN_MAINNET_USDC + deposited
    assert fake_lighter_core.get_account(MASTER_INDEX).collateral == master_equity
    assert usdc.fetch_balance_of(safe_address) == SAFE_BUFFER_USDC
    state = State.read_json_file(state_file)
    assert _positions(state)[0].get_quantity() == master_equity
    assert len(_transfer_trades(state, TradeFlag.external_account_transfer)) == 1

    # 4. Run start with 50/50 weights and a transfer fee: the master funds both sub-accounts.
    fake_lighter_core.transfer_fee = FUNDING_FEE_USDC
    os.environ.update({"LIGHTER_SUB_ACCOUNT_WEIGHTS": "0.5,0.5", "LIGHTER_TRANSFER_FEE_USD": str(FUNDING_FEE_USDC)})
    _run(cli, "start")
    first_funding = master_equity / 2
    second_funding = master_equity - first_funding - 2 * FUNDING_FEE_USDC
    assert [fake_lighter_core.get_account(index).collateral for index in (MASTER_INDEX, sub_1, sub_2)] == [0, first_funding, second_funding]
    state = State.read_json_file(state_file)
    master, sub_1_position, sub_2_position = _positions(state)
    assert (master.get_quantity(), sub_1_position.get_quantity(), sub_2_position.get_quantity()) == (0, first_funding, second_funding)
    fundings = _transfer_trades(state, TradeFlag.internal_account_transfer)
    assert len(fundings) == 2 and all(trade.is_success() for trade in fundings)
    assert all(trade.other_data["funding_fee_usdc"] == str(FUNDING_FEE_USDC) for trade in fundings)
    assert state.portfolio.get_net_asset_value() == pytest.approx(float(SAFE_BUFFER_USDC + first_funding + second_funding))
    _run(cli, "check-accounts")

    # 5. Run start with 75/25 weights: sub-account 2 is withdrawn to the Safe with the key-free L1 path, then redeposited.
    fake_lighter_core.transfer_fee = Decimal(0)
    os.environ.update({"LIGHTER_SUB_ACCOUNT_WEIGHTS": "0.75,0.25", "LIGHTER_TRANSFER_FEE_USD": "0"})
    investable = first_funding + second_funding
    excess = second_funding - investable / 4
    _run(cli, "start")
    assert fake_lighter_core.withdrawals[-1] == ("l1_withdraw", sub_2, excess)
    assert usdc.fetch_balance_of(safe_address) == SAFE_BUFFER_USDC + excess
    _run(cli, "start")
    assert fake_lighter_core.get_account(MASTER_INDEX).collateral == excess

    # 6. Interrupt the next funding transfer, check start refuses to continue, and reconcile it with correct-accounts.
    fake_lighter_core.execute_transfers = False
    with pytest.raises(TimeoutError):
        _run(cli, "start")
    state = State.read_json_file(state_file)
    pending = [trade for trade in _transfer_trades(state, TradeFlag.internal_account_transfer) if trade.is_internal_account_transfer_pending()]
    assert len(pending) == 1 and pending[0].other_data["lighter_transfer_tx_hash"]
    with pytest.raises(UncleanState):
        _run(cli, "start")
    assert len(fake_lighter_core.transfers) == 3
    fake_lighter_core.execute_pending_transfers()
    fake_lighter_core.execute_transfers = True
    _run(cli, "correct-accounts")
    state = State.read_json_file(state_file)
    master, sub_1_position, sub_2_position = _positions(state)
    assert not any(trade.is_exchange_account_transfer_pending() for trade in state.portfolio.get_all_trades())
    assert (master.get_quantity(), sub_1_position.get_quantity(), sub_2_position.get_quantity()) == (0, investable * 3 / 4, investable / 4)
    _run(cli, "check-accounts")

    # 7. Lock margin on sub-account 1 and check a withdrawal is capped to its free collateral; withdraw from sub-account 2 with lighter-move-funds.
    free_collateral = Decimal(5)
    fake_lighter_core.get_account(sub_1).margin_in_use = investable * 3 / 4 - free_collateral
    os.environ["LIGHTER_SUB_ACCOUNT_WEIGHTS"] = "0.25,0.75"
    _run(cli, "start")
    assert fake_lighter_core.withdrawals[-1] == ("l1_withdraw", sub_1, free_collateral)
    fake_lighter_core.get_account(sub_1).margin_in_use = Decimal(0)
    manual_amount = Decimal(2)
    safe_before_manual = usdc.fetch_balance_of(safe_address)
    mocker.patch("tradeexecutor.cli.commands.lighter_move_funds.typer.prompt", side_effect=["w", str(manual_amount)])
    mocker.patch("tradeexecutor.cli.commands.lighter_move_funds.typer.confirm", return_value=True)
    _run(cli, "lighter-move-funds", "--lighter-account-index", str(sub_2))
    assert fake_lighter_core.withdrawals[-1] == ("l1_withdraw", sub_2, manual_amount)
    assert usdc.fetch_balance_of(safe_address) == safe_before_manual + manual_amount
    _run(cli, "check-accounts")

    # 8. Request a redemption larger than the Safe while the master is empty and check it is funded from a sub-account.
    os.environ["LIGHTER_SUB_ACCOUNT_WEIGHTS"] = ""
    safe_before = usdc.fetch_balance_of(safe_address)
    shares = vault.share_token.fetch_raw_balance_of(deployer.address)
    assert_transaction_success_with_explanation(web3_ethereum, vault.request_redeem(deployer.address, shares // 2).transact({"from": deployer.address}))
    _run(cli, "start")
    method, account_index, amount = fake_lighter_core.withdrawals[-1]
    assert (method, account_index) == ("l1_withdraw", sub_1)
    assert usdc.fetch_balance_of(safe_address) == safe_before + amount
    lighter_total = sum((fake_lighter_core.get_account(index).collateral for index in (MASTER_INDEX, sub_1, sub_2)), Decimal(0))
    nav_before_settlement = usdc.fetch_balance_of(safe_address) + lighter_total
    _run(cli, "start")
    assert vault.vault_contract.functions.maxRedeem(deployer.address).call() == shares // 2
    # The settlement NAV included every Lighter account, so half of it remains after redeeming half the shares.
    assert vault.fetch_nav() == pytest.approx(nav_before_settlement / 2, abs=Decimal("0.000002"))

    # 9. Drain every account for a full redemption, including a master API withdrawal, and redeem with lagoon-redeem.
    os.environ["LIGHTER_SAFE_CASH_BUFFER_USD"] = "0"
    assert_transaction_success_with_explanation(web3_ethereum, vault.finalise_redeem(deployer.address).transact({"from": deployer.address}))
    safe_cash = usdc.fetch_balance_of(safe_address)
    _run(cli, "start")
    assert fake_lighter_core.get_account(MASTER_INDEX).collateral == safe_cash
    sub_balances = {index: fake_lighter_core.get_account(index).collateral for index in (sub_1, sub_2)}
    assert_transaction_success_with_explanation(web3_ethereum, vault.request_redeem(deployer.address, vault.share_token.fetch_raw_balance_of(deployer.address)).transact({"from": deployer.address}))
    _run(cli, "start")
    # The runner orders the withdrawals by its sort keys, so compare them as a set.
    assert sorted(fake_lighter_core.withdrawals[-3:]) == sorted([
        ("api_secure_withdrawal", MASTER_INDEX, safe_cash),
        ("l1_withdraw", sub_1, sub_balances[sub_1]),
        ("l1_withdraw", sub_2, sub_balances[sub_2]),
    ])
    assert all(fake_lighter_core.get_account(index).collateral == 0 for index in (MASTER_INDEX, sub_1, sub_2))
    _run(cli, "start")
    _run(cli, "lagoon-redeem")
    assert vault.share_token.fetch_raw_balance_of(deployer.address) == 0
    assert usdc.fetch_raw_balance_of(safe_address) <= 2
    state = State.read_json_file(state_file)
    assert not any(trade.is_exchange_account_transfer_pending() for trade in state.portfolio.get_all_trades())

    # 10. Sweep stray USDC and ETH from the emptied Safe back to the deployer, as the mainnet tutorial does.
    stray_tx = usdc.contract.functions.transfer(safe_address, usdc.convert_to_raw(Decimal(3))).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, stray_tx)
    web3_ethereum.provider.make_request("anvil_setBalance", [safe_address, hex(10**16)])
    owner = HotWallet(deployer)
    usdc_before_sweep = usdc.fetch_raw_balance_of(deployer.address)
    swept = sweep_safe_to_owner(web3_ethereum, safe_address, owner, [usdc])
    assert [transfer.symbol for transfer in swept] == ["USDC", "ETH"]
    assert usdc.fetch_raw_balance_of(safe_address) == 0
    assert web3_ethereum.eth.get_balance(safe_address) == 0
    assert usdc.fetch_raw_balance_of(deployer.address) - usdc_before_sweep >= usdc.convert_to_raw(Decimal(3))
