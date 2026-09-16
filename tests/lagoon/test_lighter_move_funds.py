"""Typer black-box coverage for manual Lagoon/Lighter custody movement."""

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.compat import native_datetime_utc_now
from eth_defi.erc_4626.vault_protocol.lagoon.config import get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
    deploy_automated_lagoon_vault,
)
from eth_defi.erc_4626.vault_protocol.lagoon.funding import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.valuation import LighterEquity
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, TokenDetails, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.commands.lagoon_lighter_test_trade import LighterWithdrawalRejected
from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account.lighter import (
    LIGHTER_PROTOCOL,
    create_lighter_exchange_account_pair,
)
from tradeexecutor.exchange_account.state import open_exchange_account_position
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag


#: Optional upstream RPC used to enable the fixed-block Ethereum fork.
JSON_RPC_ETHEREUM = os.environ.get("JSON_RPC_ETHEREUM")
#: Public deterministic Anvil account zero key; never a production secret.
DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
#: Sequencer-only Lighter account represented by API mocks.
LIGHTER_ACCOUNT_INDEX = 223
#: Public delegated API key slot represented by the operator record.
LIGHTER_API_KEY_INDEX = 4
#: Safe USDC used to fund the real Lagoon vault.
SAFE_USDC = Decimal("10")
#: Amount moved in each direction by the command.
TRANSFER_USDC = Decimal("2")

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
    """Reset the fixed Ethereum fork around the custody command test."""
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
    """Build one public Lighter equity response for the sequencer mock."""
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


def _transfer_from_safe(
    web3: Web3,
    usdc: TokenDetails,
    safe_address: str,
    amount: Decimal,
) -> None:
    """Move real forked USDC out of a Safe for the Lighter deposit mock."""
    web3.provider.make_request("anvil_setBalance", [safe_address, hex(10**18)])
    web3.provider.make_request("anvil_impersonateAccount", [safe_address])
    try:
        transaction_hash = usdc.transfer(USDC_WHALE[1], amount).transact({"from": safe_address})
        assert_transaction_success_with_explanation(web3, transaction_hash)
    finally:
        web3.provider.make_request("anvil_stopImpersonatingAccount", [safe_address])


@pytest.mark.timeout(600)
def test_cli_lighter_move_funds_records_safe_and_lighter_custody(
    web3_ethereum: Web3,
    deployer: HotWallet,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Record deposit and withdrawal custody movements through the Typer CLI.

    1. Deploy and fund a real Lagoon Safe on the fixed Ethereum Anvil block.
    2. Mock only Lighter sequencer and withdrawal behaviour while moving real Safe USDC.
    3. Run completed, interrupted and definitively rejected transfers through Typer.
    4. Verify check/correct accounts against completed and out-of-band custody changes.
    5. Verify transfer trades, Safe custody, Lighter value and secret-free state.
    """
    # 1. Deploy and fund a real Lagoon Safe on the fixed Ethereum Anvil block.
    usdc = fetch_erc20_details(web3_ethereum, USDC_NATIVE_TOKEN[1], chain_id=1)
    funding_tx = usdc.transfer(deployer.address, SAFE_USDC + Decimal("2")).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)
    chain_config = get_lagoon_chain_config(1)
    deployment = deploy_automated_lagoon_vault(
        web3=web3_ethereum,
        deployer=deployer,
        asset_manager=deployer.address,
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[1],
            name="Lighter move funds test",
            symbol="LMF",
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
    state_file = tmp_path / "lighter-move-funds.json"
    deployment_artifact_file = tmp_path / "lighter-move-funds.deployment.json"
    deployment_artifact_file.write_text(json.dumps({
        "deployments": {
            "ethereum": {
                "vault_address": deployment.vault.address,
                "module_address": deployment.trading_strategy_module.address,
            },
        },
        "deployment_record": {
            "lighter_account_setup": {
                "account_index": LIGHTER_ACCOUNT_INDEX,
                "api_key_index": LIGHTER_API_KEY_INDEX,
            },
        },
    }))
    reserve_asset = AssetIdentifier(
        chain_id=1,
        address=usdc.address,
        token_symbol=usdc.symbol,
        decimals=usdc.decimals,
    )
    state = State()
    state.portfolio.initialise_reserves(reserve_asset, reserve_token_price=1.0)
    state.portfolio.get_reserve_position(reserve_asset).quantity = SAFE_USDC
    pair = create_lighter_exchange_account_pair(reserve_asset, LIGHTER_ACCOUNT_INDEX)
    open_exchange_account_position(
        state=state,
        strategy_cycle_at=native_datetime_utc_now(),
        pair=pair,
        reserve_currency=reserve_asset,
        notes="Initial empty Lighter account",
    )
    state_file.write_text(state.to_json_safe())

    # 2. Mock only Lighter sequencer and withdrawal behaviour while moving real Safe USDC.
    command_module = "tradeexecutor.cli.commands.lighter_move_funds"
    lighter_state = {"equity": Decimal(0)}
    lighter_control = {"reject_next_withdrawal": False}
    observed = {"deposits": 0, "requests": 0, "withdrawals": 0, "history_checks": 0, "claims": 0}

    def fetch_equity(*_args: object, **_kwargs: object) -> LighterEquity:
        return _lighter_equity(lighter_state["equity"])

    def fake_deposit(*_args: object, **kwargs: object) -> str:
        amount = kwargs["deposit_usdc"]
        assert amount == TRANSFER_USDC
        vault = kwargs["vault"]
        _transfer_from_safe(web3_ethereum, usdc, vault.safe_address, amount)
        lighter_state["equity"] += amount
        observed["deposits"] += 1
        return "0x" + "12" * 32

    def fake_wait_for_collateral(*_args: object, **_kwargs: object) -> Decimal:
        return lighter_state["equity"]

    async def fake_request_withdrawal(_operator: object, amount: Decimal) -> str:
        assert amount == TRANSFER_USDC
        observed["requests"] += 1
        if lighter_control["reject_next_withdrawal"]:
            lighter_control["reject_next_withdrawal"] = False
            raise LighterWithdrawalRejected("Lighter secure withdrawal was rejected")
        lighter_state["equity"] -= amount
        observed["withdrawals"] += 1
        return "0x" + "34" * 32

    async def fake_wait_for_claimable(
        _operator: object,
        amount: Decimal,
        _timeout: int,
        requested_at: int,
    ) -> Decimal:
        assert requested_at > 0
        observed["history_checks"] += 1
        if observed["history_checks"] == 1:
            raise TimeoutError("Test interruption after withdrawal submission")
        return amount

    def fake_claim(*_args: object, **kwargs: object) -> str:
        amount = kwargs["claimable_usdc"]
        vault = kwargs["vault"]
        claim_tx = usdc.transfer(vault.safe_address, amount).transact({"from": USDC_WHALE[1]})
        assert_transaction_success_with_explanation(web3_ethereum, claim_tx)
        observed["claims"] += 1
        return Web3.to_hex(claim_tx)

    session = mocker.Mock()
    mocker.patch(f"{command_module}.create_lighter_session", return_value=session)
    mocker.patch(f"{command_module}.fetch_lighter_total_equity", side_effect=fetch_equity)
    mocker.patch(f"{command_module}.fetch_lighter_account_by_index", return_value={"positions": []})
    mocker.patch(f"{command_module}.deposit_usdc_from_lagoon_safe_into_lighter", side_effect=fake_deposit)
    mocker.patch(f"{command_module}.wait_for_lighter_collateral", side_effect=fake_wait_for_collateral)
    mocker.patch(f"{command_module}.fetch_lighter_withdrawal_delay", return_value=1_800)
    mocker.patch(f"{command_module}.request_lighter_withdrawal", side_effect=fake_request_withdrawal)
    mocker.patch(f"{command_module}.wait_for_lighter_withdrawal_claimable", side_effect=fake_wait_for_claimable)
    mocker.patch(f"{command_module}.claim_usdc_to_lagoon_safe_from_lighter", side_effect=fake_claim)
    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        side_effect=fetch_equity,
    )
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-move-funds",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "JSON_RPC_ETHEREUM": web3_ethereum.provider.endpoint_uri,
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": deployment.vault.address,
        "VAULT_ADAPTER_ADDRESS": deployment.trading_strategy_module.address,
        "LIGHTER_ACCOUNT_INDEX": str(LIGHTER_ACCOUNT_INDEX),
        "LIGHTER_OPERATOR_RECORD_FILE": operator_record_file.as_posix(),
        "LOG_LEVEL": "disabled",
        "UNIT_TESTING": "true",
    }
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)

    # 3. Run completed, interrupted and definitively rejected transfers through Typer.
    mocker.patch(f"{command_module}.typer.prompt", side_effect=["d", str(TRANSFER_USDC)])
    mocker.patch(f"{command_module}.typer.confirm", return_value=True)
    cli.main(args=["lighter-move-funds"], standalone_mode=False)
    state_before_check = state_file.read_text()
    with pytest.raises(SystemExit) as check_after_deposit:
        cli.main(args=["check-accounts"], standalone_mode=False)
    assert check_after_deposit.value.code == 0
    assert state_file.read_text() == state_before_check
    mocker.patch(f"{command_module}.typer.prompt", side_effect=["w", str(TRANSFER_USDC)])
    mocker.patch(f"{command_module}.typer.confirm", return_value=True)
    with pytest.raises(TimeoutError, match="Test interruption after withdrawal submission"):
        cli.main(args=["lighter-move-funds"], standalone_mode=False)
    cli.main(args=["lighter-move-funds"], standalone_mode=False)
    state_before_check = state_file.read_text()
    with pytest.raises(SystemExit) as check_after_withdrawal:
        cli.main(args=["check-accounts"], standalone_mode=False)
    assert check_after_withdrawal.value.code == 0
    assert state_file.read_text() == state_before_check

    mocker.patch(f"{command_module}.typer.prompt", side_effect=["d", str(TRANSFER_USDC)])
    cli.main(args=["lighter-move-funds"], standalone_mode=False)
    lighter_control["reject_next_withdrawal"] = True
    mocker.patch(f"{command_module}.typer.prompt", side_effect=["w", str(TRANSFER_USDC)])
    with pytest.raises(LighterWithdrawalRejected, match="secure withdrawal was rejected"):
        cli.main(args=["lighter-move-funds"], standalone_mode=False)
    rejected_state = State.read_json_file(state_file)
    rejected_state.check_if_clean()
    rejected_position = next(iter(rejected_state.portfolio.open_positions.values()))
    rejected_transfers = [
        trade
        for trade in rejected_position.trades.values()
        if TradeFlag.external_account_transfer in (trade.flags or set()) and trade.is_failed()
    ]
    assert len(rejected_transfers) == 1
    assert rejected_transfers[0].other_data["outcome"] == "withdrawal_rejected"
    mocker.patch(f"{command_module}.typer.prompt", side_effect=["w", str(TRANSFER_USDC)])
    cli.main(args=["lighter-move-funds"], standalone_mode=False)

    # 4. Add an out-of-band Safe balance and verify correction preserves transfer trades.
    out_of_band_usdc = Decimal("1")
    out_of_band_tx = usdc.transfer(deployment.safe_address, out_of_band_usdc).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, out_of_band_tx)
    dry_run_state = state_file.read_text()
    cli.main(args=["correct-accounts", "--dry-run"], standalone_mode=False)
    assert state_file.read_text() == dry_run_state
    with pytest.raises(SystemExit) as corrected:
        cli.main(args=["correct-accounts"], standalone_mode=False)
    assert corrected.value.code == 0

    # 5. Verify transfer trades, Safe custody, Lighter value and secret-free state.
    final_state = State.read_json_file(state_file)
    position = next(iter(final_state.portfolio.open_positions.values()))
    transfers = [
        trade
        for trade in position.trades.values()
        if TradeFlag.external_account_transfer in (trade.flags or set())
    ]
    assert usdc.fetch_balance_of(deployment.safe_address) == SAFE_USDC + out_of_band_usdc
    assert position.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
    assert position.get_quantity() == Decimal(0)
    assert final_state.portfolio.get_default_reserve_position().quantity == SAFE_USDC + out_of_band_usdc
    assert len(transfers) == 5
    assert sum(trade.is_success() for trade in transfers) == 4
    assert sum(trade.is_failed() for trade in transfers) == 1
    assert {trade.other_data["direction"] for trade in transfers} == {"deposit", "withdraw"}
    assert observed == {"deposits": 2, "requests": 3, "withdrawals": 2, "history_checks": 3, "claims": 2}
    assert delegated_private_key not in state_file.read_text()
