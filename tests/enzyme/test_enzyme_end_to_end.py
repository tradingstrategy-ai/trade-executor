"""Eznyme end-to-end test."""
import json
import os
import secrets
import tempfile
import logging
from _decimal import Decimal

from pathlib import Path
from unittest.mock import patch

import flaky
import pytest
from IPython.terminal.ipapp import flags
from click.testing import Result
from eth_account import Account
from typer.main import get_command
from web3 import Web3

from eth_defi.abi import get_deployed_contract
from eth_defi.provider.anvil import AnvilLaunch
from hexbytes import HexBytes
from typer.testing import CliRunner
from web3.contract import Contract
from eth_typing import HexAddress

from eth_defi.enzyme.deployment import EnzymeDeployment
from eth_defi.enzyme.vault import Vault
from eth_defi.hotwallet import HotWallet
from eth_defi.trace import assert_transaction_success_with_explanation
from eth_defi.uniswap_v2.deployment import UniswapV2Deployment
from tradeexecutor.monkeypatch.web3 import construct_sign_and_send_raw_middleware

from tradingstrategy.pair import PandasPairUniverse

from eth_defi.uniswap_v2.swap import swap_with_slippage_protection
from tradeexecutor.cli.main import app
from tradeexecutor.state.blockhain_transaction import BlockchainTransactionType
from tradeexecutor.state.trade import TradeType
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State


logger = logging.getLogger(__name__)


def transfer_vault_ownership(
    vault: Vault,
    new_owner: str,
):
    """Transfer the vault ownership to a new owner.

    Used in testing.
    """
    web3 = vault.web3
    current_owner = vault.get_owner()
    logger.info("transfer_ownership(): %s -> %s", vault.get_owner(), new_owner),
    assert vault.get_owner() != new_owner

    tx_hash = vault.vault.functions.setNominatedOwner(new_owner).transact({"from": current_owner})
    assert_transaction_success_with_explanation(web3, tx_hash)
    logger.info("New vault owner nominated to be %s", new_owner)

    # Accept transfer
    tx_hash = vault.vault.functions.claimOwnership().transact({"from": new_owner})
    assert_transaction_success_with_explanation(web3, tx_hash)
    logger.info("New vault owner claimed ownership to be %s", new_owner)

    assert vault.get_owner() == new_owner


@pytest.fixture
def hot_wallet(web3, deployer, user_1, usdc: Contract) -> HotWallet:
    """Create hot wallet for the signing tests.

    Top is up with some gas money and 500 USDC.
    """
    private_key = HexBytes(secrets.token_bytes(32))
    account = Account.from_key(private_key)
    wallet = HotWallet(account)
    wallet.sync_nonce(web3)
    tx_hash = web3.eth.send_transaction({"to": wallet.address, "from": user_1, "value": 15 * 10**18})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = usdc.functions.transfer(wallet.address, 500 * 10**6).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Add to the local signer chain
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))

    return wallet


@pytest.fixture()
def enzyme_vault_contract(
    web3,
    deployer,
    usdc,
    user_1,
    hot_wallet: HotWallet,
    enzyme_deployment,
) -> Contract:
    """Create an example vault.

    - USDC nominatead

    - user_1 is the owner
    """
    comptroller_contract, vault_contract = enzyme_deployment.create_new_vault(
        hot_wallet.address,
        usdc,
    )

    # Promote the hot wallet to the asset manager
    tx_hash = vault_contract.functions.addAssetManagers([hot_wallet.address]).transact({"from": hot_wallet.address})
    assert_transaction_success_with_explanation(web3, tx_hash)

    return vault_contract


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / "../../strategies/test_only" / "enzyme_end_to_end.py"


@pytest.fixture()
def state_file() -> Path:
    """The state file for the tests.

    Always start with an empty file.
    """
    path = Path("/tmp/test_enzyme_end_to_end.json")
    if path.exists():
        os.remove(path)
    return path


@pytest.fixture()
def environment(
    anvil: AnvilLaunch,
    deployer: HexAddress,
    vault: Vault,
    usdc: Contract,
    weth: Contract,
    usdc_asset: AssetIdentifier,
    weth_asset: AssetIdentifier,
    user_1: HexAddress,
    uniswap_v2: UniswapV2Deployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    pair_universe: PandasPairUniverse,
    hot_wallet: HotWallet,
    state_file: Path,
    strategy_file: Path,
    ) -> dict:
    """Passed to init and start commands as environment variables"""
    # Set up the configuration for the live trader
    environment = {
        "EXECUTOR_ID": "test_enzyme_live_trading_init",
        "NAME": "test_enzyme_live_trading_init",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "PRIVATE_KEY": hot_wallet.account.key.hex(),
        "JSON_RPC_ANVIL": anvil.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "enzyme",
        "UNIT_TESTING": "true",
        # "LOG_LEVEL": "info",  # Set to info to get debug data for the test run
        "LOG_LEVEL": "disabled",
        "VAULT_ADDRESS": vault.address,
        "VAULT_ADAPTER_ADDRESS": vault.generic_adapter.address,
        "TEST_EVM_UNISWAP_V2_ROUTER": uniswap_v2.router.address,
        "TEST_EVM_UNISWAP_V2_FACTORY": uniswap_v2.factory.address,
        "TEST_EVM_UNISWAP_V2_INIT_CODE_HASH": uniswap_v2.init_code_hash,
        "CONFIRMATION_BLOCK_COUNT": "0",  # Needed for test backend, Anvil
        "MAX_CYCLES": "5",  # Run decide_trades() 5 times
        "PATH": os.environ["PATH"],
    }
    return environment


def run_init(environment: dict) -> Result:
    """Run vault init command"""

    # https://typer.tiangolo.com/tutorial/testing/
    runner = CliRunner()

    # Need to use patch here, or parent shell env vars will leak in and cause random test failres
    with patch.dict(os.environ, environment, clear=True):
        result = runner.invoke(app, "init", env=environment)

    if result.exception:
        raise result.exception

    return result


def test_enzyme_live_trading_init(
    environment: dict,
    state_file: Path,
):
    """Initialize Enzyme vault for live trading.

    Provide faux chain using Anvil with one pool that a sample strategy is trading.
    """

    result = run_init(environment)
    assert result.exit_code == 0

    # Check the initial state sync set some of the variables
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())
        assert state.sync.deployment.vault_token_name is not None
        assert state.sync.deployment.vault_token_symbol is not None
        assert state.sync.deployment.block_number > 1


def test_enzyme_live_trading_start(
    environment: dict,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
):
    """Run Enzyme vaulted strategy for few cycles.

    - Set up local Anvil testnet with Uniswap v2 and Enzyme

    - Create a strategy that trade ETH-USDC pair and does few buys and sells

    - Run cycles of this strategy

    - Check that the state file output looks good

    - Check that the chain output looks good

    At the end of 5th cycle we should have

    - 1 open position, id 2

    - 1 closed position, id 1
    """


    # Need to be initialised first
    result = run_init(environment)
    assert result.exit_code == 0

    # Deposit some money in the vault
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    # Run strategy for few cycles.
    # Manually call the main() function so that Typer's CliRunner.invoke() does not steal
    # stdin and we can still set breakpoints
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # Check that trades completed
    with state_file.open("rt") as inp:
        state = State.from_json(inp.read())

        # Show tx revert reason if possible
        if len(state.portfolio.frozen_positions) > 0:
            for p in state.portfolio.frozen_positions.values():
                raise AssertionError(f"Frozen position {p}: {p.get_freeze_reason()}")

        assert len(state.portfolio.closed_positions) == 1
        assert len(state.portfolio.open_positions) == 1

        # Pick an example trade to examine
        p = state.portfolio.open_positions[2]
        t = p.trades[3]
        assert t.is_success()
        assert t.lp_fees_estimated == pytest.approx(0.14991015720000014)
        assert t.lp_fees_paid == pytest.approx(0.14991015600000002)
        assert t.trade_type == TradeType.rebalance
        assert t.slippage_tolerance == 0.02  # Set in enzyme_end_to_end.py strategy module

        tx = t.blockchain_transactions[0]
        assert tx.type == BlockchainTransactionType.enzyme_vault

    # Check on-chain balances
    usdc_balance = usdc.functions.balanceOf(vault.vault.address).call()
    weth_balance = weth.functions.balanceOf(vault.vault.address).call()

    assert usdc_balance == pytest.approx(10**6 * 449.730472)
    assert weth_balance == pytest.approx(10**18 * 0.03112978758721282)


# This is slow for some reason as the test itself executes fast, but pytest takes forever to shutdown
@pytest.mark.slow_test_group
def test_enzyme_live_trading_run_single_cycle(
    environment: dict,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
):
    """Run Enzyme vaulted strategy using --run_single_cycle"""

    result = run_init(environment)
    assert result.exit_code == 0

    # Deposit some money in the vault
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    # Run strategy for a single cycle
    env = environment.copy()
    env["RUN_SINGLE_CYCLE"] = "true"
    del env["MAX_CYCLES"]
    cli = get_command(app)
    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["start"], standalone_mode=False)


@pytest.mark.slow_test_group
def test_enzyme_deploy_vault(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
    enzyme_deployment: EnzymeDeployment,
):
    """Deploy Enzymeß vault via CLI.

    - Set up local Anvil testnet with Uniswap v2 and Enzyme

    - Deploy a new vault using CLI

    """

    vault_record_file = os.path.join(tempfile.mkdtemp(), 'vault_record.json')
    env = environment.copy()
    env["FUND_NAME"] = "Toholampi Capital"
    env["FUND_SYMBOL"] = "COW"
    env["VAULT_RECORD_FILE"] = vault_record_file
    env["COMPTROLLER_LIB"] = enzyme_deployment.contracts.comptroller_lib.address
    env["DENOMINATION_ASSET"] = usdc.address
    env["WHITELISTED_ASSETS"] = " ".join([usdc.address, weth.address])

    # Run strategy for few cycles.
    # Manually call the main() function so that Typer's CliRunner.invoke() does not steal
    # stdin and we can still set breakpoints
    cli = get_command(app)
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["enzyme-deploy-vault"])
        assert e.value.code == 0

    # Check tat the vault was created
    with open(vault_record_file, "rt") as inp:
        vault_record = json.load(inp)
        comptroller_contract, vault_contract = EnzymeDeployment.fetch_vault(enzyme_deployment, vault_record["vault"])
        generic_adapter_contract = get_deployed_contract(web3, f"VaultSpecificGenericAdapter.json", vault_record["generic_adapter"])
        payment_forwarder_contract = get_deployed_contract(web3, f"VaultUSDCPaymentForwarder.json", vault_record["usdc_payment_forwarder"])

        vault = Vault(
            vault=vault_contract,
            comptroller=comptroller_contract,
            deployment=enzyme_deployment,
            generic_adapter=generic_adapter_contract,
            payment_forwarder=payment_forwarder_contract,
        )

        assert vault.get_name() == "Toholampi Capital"
        assert vault.get_symbol() == "COW"
        assert vault.payment_forwarder.functions.amountProxied().call() == 0


@flaky.flaky
def test_enzyme_perform_test_trade(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
    enzyme_deployment: EnzymeDeployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
):
    """Perform a test trade on Enzymy vault via CLI.

    - Use a vault deployed by the test fixtures

    - Initialise the strategy to use this vault

    - Perform a test trade on this fault
    """

    env = environment.copy()
    env["VAULT_ADDRESS"] = vault.address
    env["VAULT_ADAPTER_ADDRESS"] = vault.generic_adapter.address

    cli = get_command(app)

    # Deposit some USDC to start
    deposit_amount = 500 * 10**6
    tx_hash = usdc.functions.approve(vault.comptroller.address, deposit_amount).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.functions.balanceOf(vault.address).call() == deposit_amount
    logger.info("Deposited %d %s at block %d", deposit_amount, usdc.address, web3.eth.block_number)

    # Check we have a deposit event
    logs = vault.comptroller.events.SharesBought.get_logs()
    logger.info("Got logs %s", logs)
    assert len(logs) == 1

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["init"])
        assert e.value.code == 0

    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade"])
        assert e.value.code == 0

    assert usdc.functions.balanceOf(vault.address).call() < deposit_amount, "No deposits where spent; trades likely did not happen"

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 2
        reserve_value = state.portfolio.get_default_reserve_position().get_value()
        assert reserve_value == pytest.approx(499.994009)


def test_enzyme_live_trading_reinit(
    environment: dict,
    state_file: Path,
    vault,
    deployer,
    usdc,
):
    """Reinitialize Enzyme vault for live trading.

    Check that reinitialise works and accounting information is read from the chain state.
    """

    if os.path.exists("/tmp/test_enzyme_end_to_end.reinit-backup-1.json"):
        os.remove("/tmp/test_enzyme_end_to_end.reinit-backup-1.json")

    result = run_init(environment)
    assert result.exit_code == 0

    assert os.path.exists("/tmp/test_enzyme_end_to_end.json")

    cli = get_command(app)

    # Deposit some money in the vault
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["reset"])
        assert e.value.code == 0

    assert os.path.exists("/tmp/test_enzyme_end_to_end.reinit-backup-1.json")

    # See that the reinitialised state looks correct
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        reserve_position = state.portfolio.get_default_reserve_position()
        assert reserve_position.quantity == 500

        treasury = state.sync.treasury
        deployment = state.sync.deployment
        assert deployment.initialised_at
        assert treasury.last_block_scanned > 1
        assert treasury.last_updated_at
        assert len(treasury.balance_update_refs) == 1
        assert len(reserve_position.balance_updates) == 1


def test_enzyme_correct_accounts(
    environment: dict,
    state_file: Path,
    vault,
    deployer,
    usdc,
):
    """Run the correct-accounts command.

    """

    # Avoid dirty fs
    if os.path.exists("/tmp/test_enzyme_end_to_end.backup-1.json"):
        os.remove("/tmp/test_enzyme_end_to_end.backup-1.json")

    # Deposit some money in the vault which should be picked up by correct accounts
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0

    # We created a backup of the old state
    assert os.path.exists("/tmp/test_enzyme_end_to_end.backup-1.json")

    # See that the corrected reverse balance looks ok
    state: State = State.read_json_file(state_file)
    reserve_position = state.portfolio.get_default_reserve_position()
    assert reserve_position.quantity == 500
    assert state.sync.accounting.last_updated_at is not None


def test_enzyme_check_accounts(
    environment: dict,
    state_file: Path,
    vault,
    deployer,
    usdc,
):
    """Run the check-accounts command.

    """

    # Deposit some money in the vault which should be picked up by check-accounts
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})

    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)

    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["check-accounts"])
        assert e.value.code == 1


def test_enzyme_live_trading_reset_deposits(
    environment: dict,
    state_file: Path,
    vault,
    deployer,
    usdc,
):
    """Reinitialise position balances from on-chain.

    - Start with 500 USDC

    - Run strat

    - End up with 449 USDC, and open position

    - Do unsynced redemption and mess up the open position balance,
      this will remove WETH from the strategy

    - Get rid of unhandled redemption with reset-deposits

    - Correct wrong balance with correct-accounts

    - Run the strat again to see it starts correctly after reset-deposits
    """

    if os.path.exists("/tmp/test_enzyme_end_to_end.reinit-backup-1.json"):
        os.remove("/tmp/test_enzyme_end_to_end.reinit-backup-1.json")

    # Buy and redeem some tokens in order to mess up the sync queue
    web3 = vault.web3
    usdc.functions.approve(vault.comptroller.address, 500 * 10**6).transact({"from": deployer})
    tx_hash = vault.comptroller.functions.buyShares(500 * 10**6, 1).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0

    # See that the initial deposit looks correct
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        reserve_position = state.portfolio.get_default_reserve_position()
        assert reserve_position.quantity == pytest.approx(Decimal('449.7304715999999998521161615'))
        assert len(state.portfolio.open_positions) == 1

    # Mess up by doing unsynced redemption,
    # this will remove half of ETH
    tx_hash = vault.comptroller.functions.redeemSharesInKind(deployer, 250 * 10**18, [], []).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Reset deposits from the on-chain state
    # with patch.dict(os.environ, environment, clear=True):
    #     with pytest.raises(SystemExit) as e:
    #         cli = get_command(app)
    #         cli.main(args=["reset-deposits"])
    #     assert e.value.code == 0
    #
    # assert os.path.exists("/tmp/test_enzyme_end_to_end.reinit-backup-1.json")

    # Correct wrong WETH balance
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli = get_command(app)
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0

    # See that the reinitialised state looks correct
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        reserve_position = state.portfolio.get_default_reserve_position()
        assert reserve_position.quantity == pytest.approx(Decimal('224.8652360000000000000000000'))

        treasury = state.sync.treasury
        deployment = state.sync.deployment
        assert deployment.initialised_at
        assert treasury.last_block_scanned > 1
        assert treasury.last_updated_at
        assert len(treasury.balance_update_refs) == 1
        assert len(reserve_position.balance_updates) == 2

    # Run strategy for few cycles to see it still starts after reset-deposits
    cli = get_command(app)
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["start"])

        assert e.value.code == 0


def test_enzyme_correct_accounts_for_closed_position_transfer_away(
    environment: dict,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
    enzyme_deployment: EnzymeDeployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    uniswap_v2: UniswapV2Deployment,
    web3: Web3,
):
    """Correct a closed position.

    - Open position

    - Close position

    - Drop some tokens to the closed positions

    - These tokens should be transferred away
    """

    # Set up executor
    result = run_init(environment)
    assert result.exit_code == 0

    cli = get_command(app)

    # Deposit some USDC to perform the test trade
    deposit_amount = 500 * 10**6
    tx_hash = usdc.functions.approve(vault.comptroller.address, deposit_amount).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.functions.balanceOf(vault.address).call() == deposit_amount

    # Open and close position
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["perform-test-trade"])
        assert e.value.code == 0

    # TODO: Not sure what test fixture drops WETH on deployer
    weth_balance = weth.functions.balanceOf(deployer).call()
    assert weth_balance == 375000000000000000000

    # tx_hash = usdc.functions.approve(uniswap_v2.router.address, 1000 * 10**18).transact({"from": deployer})
    # assert_transaction_success_with_explanation(web3, tx_hash)
    # prepared_swap_call = swap_with_slippage_protection(
    #     uniswap_v2_deployment=uniswap_v2,
    #     recipient_address=deployer,
    #     quote_token=usdc,
    #     base_token=weth,
    #     amount_in=100 * 10**6,
    #     max_slippage=10_000,
    # )
    # tx_hash = prepared_swap_call.transact({"from": deployer})
    # assert_transaction_success_with_explanation(web3, tx_hash)
    # weth_balance = weth.functions.balanceOf(deployer).call()
    # assert weth_balance == 900 * 10**6

    # Drop 10 WETH in vault, not associated with any open position
    tx_hash = weth.functions.transfer(vault.address, 10*10**18).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)

    # Correct accounts,
    # we have one closed position for WETH
    environment = environment.copy()
    environment["TRANSFER_AWAY"] = "true"
    with patch.dict(os.environ, environment, clear=True):
        with pytest.raises(SystemExit) as e:
            cli.main(args=["correct-accounts"])
        assert e.value.code == 0


def test_deploy_guard_standalone(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    deployer: HexAddress,
):
    """Deploy GuardV0 smart contract standalone.

    - Don't associate with any vault
    """

    env = environment.copy()
    env["WHITELISTED_ASSETS"] = " ".join([usdc.address, weth.address])

    with patch.dict(os.environ, env, clear=True):
        app(["deploy-guard"], standalone_mode=False)


def test_deploy_guard_for_vault(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
):
    """Deploy GuardV0 smart contract for an existing vault.

    - Update guard to use the vault, and vault generic adapter use the guard
    """

    env = environment.copy()
    env.update({
        "VAULT_ADDRESS": vault.address,
        "VAULT_ADAPTER_ADDRESS": vault.generic_adapter.address,
        "WHITELISTED_ASSETS": " ".join([usdc.address, weth.address]),
    })

    with patch.dict(os.environ, env, clear=True):
        app(["deploy-guard"], standalone_mode=False)


@pytest.mark.skip(reason="Currently Enzyme does not way to update AdapterPolicy. Instead, the whole vault needs to be reconfigured with 7 days delay.")
def test_enzyme_perform_test_trade_with_redeployed_guard(
    environment: dict,
    web3: Web3,
    state_file: Path,
    usdc: Contract,
    weth: Contract,
    vault: Vault,
    deployer: HexAddress,
    enzyme_deployment: EnzymeDeployment,
    weth_usdc_trading_pair: TradingPairIdentifier,
    tmp_path,
    user_1,
    hot_wallet: HotWallet,
):
    """Perform a test trade on Enzymy vault via CLI.

    - Use deploy-guard to deploy a new guard for the vault

    - If the guard deployment is broken, the test trade should not success
    """


    assert vault.get_owner() == hot_wallet.address
    report_file = tmp_path / "guard.json"
    env = environment.copy()
    env["VAULT_ADDRESS"] = vault.address
    env["VAULT_ADAPTER_ADDRESS"] = vault.generic_adapter.address
    env["REPORT_FILE"] = str(report_file)
    env["UPDATE_GENERIC_ADAPTER"] = "true"
    env["WHITELISTED_ASSETS"] = " ".join([usdc.address, weth.address])
    env["COMPTROLLER_LIB"] = enzyme_deployment.contracts.comptroller_lib.address
    env["ALLOWED_ADAPTERS_POLICY"] = enzyme_deployment.contracts.allowed_adapters_policy.address

    # We need to be vault owner to update the generic adapter policy
    # transfer_vault_ownership(vault, hot_wallet.address)

    cli = get_command(app)

    # Deposit some USDC to start
    deposit_amount = 500 * 10**6
    tx_hash = usdc.functions.approve(vault.comptroller.address, deposit_amount).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    tx_hash = vault.comptroller.functions.buyShares(deposit_amount, 1).transact({"from": deployer})
    assert_transaction_success_with_explanation(web3, tx_hash)
    assert usdc.functions.balanceOf(vault.address).call() == deposit_amount
    logger.info("Deposited %d %s at block %d", deposit_amount, usdc.address, web3.eth.block_number)

    # Check we have a deposit event
    logs = vault.comptroller.events.SharesBought.get_logs()
    logger.info("Got logs %s", logs)
    assert len(logs) == 1

    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["init"], standalone_mode=False)

    # We (may) have alreaddy guard deployed earlier,
    # but we will now redeploy it
    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["deploy-guard"], standalone_mode=False)

    # Check the deployed address look somewhat correct
    report_data = json.load(report_file.open("rt"))
    assert report_data["generic_adapter"]
    assert report_data["guard"]
    vault = Vault.fetch(
        web3,
        vault_address=vault.address,
        generic_adapter_address=vault.generic_adapter.address,
    )

    with patch.dict(os.environ, env, clear=True):
        cli.main(args=["perform-test-trade"], standalone_mode=False)

    assert usdc.functions.balanceOf(vault.address).call() < deposit_amount, "No deposits where spent; trades likely did not happen"

    # Check the resulting state and see we made some trade for trading fee losses
    with state_file.open("rt") as inp:
        state: State = State.from_json(inp.read())
        assert len(list(state.portfolio.get_all_trades())) == 2
