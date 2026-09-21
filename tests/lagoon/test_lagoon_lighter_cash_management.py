"""Black-box automatic Lagoon/Lighter cash-management coverage.

The fixed Ethereum fork keeps Lagoon, Safe, USDC, the guard and the Lighter
L1 deposit call real. Only Lighter's sequencer-owned public account read and
collateral observation are mocked.
"""

import json
import os
from decimal import Decimal
from pathlib import Path

import pytest
from eth_account import Account
from eth_defi.erc_4626.vault_protocol.lagoon.config import get_lagoon_chain_config
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    LagoonDeploymentParameters,
)
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import (
    deploy_automated_lagoon_vault,
)
from eth_defi.erc_4626.vault_protocol.lagoon.funding import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.lighter.deployment import LighterDeployment
from eth_defi.lighter.testing import register_lighter_account_on_anvil
from eth_defi.lighter.valuation import LighterEquity
from eth_defi.provider.anvil import AnvilLaunch
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.testing.anvil_fork_pool import AnvilForkPool
from eth_defi.testing.evm_snapshot_fixture import evm_snapshot_revert
from eth_defi.testing.fork_blocks import ETHEREUM_MIDNIGHT_BLOCK
from eth_defi.token import USDC_NATIVE_TOKEN, USDC_WHALE, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from pytest_mock import MockerFixture
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.exchange_account.lighter import LIGHTER_PROTOCOL
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeFlag

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
#: Synthetic account returned by the mocked Lighter sequencer registration.
LIGHTER_ACCOUNT_INDEX = 126
#: Initial real Lagoon subscription used by the strategy lifecycle.
INITIAL_SUBSCRIPTION_USDC = Decimal("100")
#: Safe cash retained by the automatic cash-management policy.
SAFE_CASH_BUFFER_USDC = Decimal("20")
#: First automatic deposit selected by the policy.
EXPECTED_LIGHTER_DEPOSIT_USDC = INITIAL_SUBSCRIPTION_USDC - SAFE_CASH_BUFFER_USDC
#: Owner-only record mode for the mocked withdrawal operator record.
PRIVATE_RECORD_MODE = 0o600


@pytest.fixture()
def anvil_ethereum(anvil_fork_pool: AnvilForkPool) -> AnvilLaunch:
    """Reset the shared fixed Ethereum fork around the cash-management test."""
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
    """Fund the deterministic deployer used by Lagoon transactions."""
    account = Account.from_key(DEPLOYER_PRIVATE_KEY)
    web3_ethereum.provider.make_request(
        "anvil_setBalance",
        [account.address, hex(100 * 10**18)],
    )
    return HotWallet(account)


@pytest.fixture()
def strategy_file() -> Path:
    """Return the automatic Lighter cash-management strategy."""
    return Path(__file__).resolve().parents[2] / "strategies" / "test_only" / "lighter_cash_management_strategy.py"


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


@pytest.mark.timeout(600)
def test_cli_lagoon_lighter_cash_management_uses_normal_pipeline(
    anvil_ethereum: AnvilLaunch,
    web3_ethereum: Web3,
    deployer: HotWallet,
    strategy_file: Path,
    mocker: MockerFixture,
    tmp_path: Path,
) -> None:
    """Move automatic Safe cash to Lighter through real Typer lifecycle commands.

    1. Deploy and fund a real Lighter-enabled Lagoon vault on the fixed fork.
    2. Mock only public Lighter account observations and initialise executor state.
    3. Run real Typer ``start`` cycles and verify the automatic deposit pipeline.
    4. Run real account diagnostics and assert non-negative conserved balances.
    """
    # 1. Deploy and fund a real Lighter-enabled Lagoon vault on the fixed fork.
    usdc = fetch_erc20_details(web3_ethereum, USDC_NATIVE_TOKEN[1], chain_id=1)
    funding_tx = usdc.transfer(
        deployer.address,
        INITIAL_SUBSCRIPTION_USDC + Decimal("2"),
    ).transact({"from": USDC_WHALE[1]})
    assert_transaction_success_with_explanation(web3_ethereum, funding_tx)
    chain_config = get_lagoon_chain_config(1)
    lighter_deployment = LighterDeployment.create_ethereum()
    deployment = deploy_automated_lagoon_vault(
        web3=web3_ethereum,
        deployer=deployer,
        asset_manager=deployer.address,
        parameters=LagoonDeploymentParameters(
            underlying=USDC_NATIVE_TOKEN[1],
            name="Lighter cash management test",
            symbol="LCM",
        ),
        safe_owners=[deployer.address],
        safe_threshold=1,
        assets=[USDC_NATIVE_TOKEN[1]],
        lighter_deployment=lighter_deployment,
        max_settlement_amount=Decimal("5000"),
        settlement_window=86400,
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
        amount=INITIAL_SUBSCRIPTION_USDC,
        hot_wallet=deployer,
    )
    register_lighter_account_on_anvil(
        web3_ethereum,
        deployment.safe_address,
        LIGHTER_ACCOUNT_INDEX,
    )

    module = deployment.trading_strategy_module
    approve_selector = Web3.keccak(text="approve(address,uint256)")[:4]
    deposit_selector = Web3.keccak(text="deposit(address,uint16,uint8,uint256)")[:4]
    assert module.functions.isAllowedCallSite(USDC_NATIVE_TOKEN[1], approve_selector).call()
    assert module.functions.isAllowedCallSite(lighter_deployment.zk_lighter, deposit_selector).call()
    assert module.functions.isAllowedApprovalDestination(lighter_deployment.zk_lighter).call()
    assert module.functions.isAllowedAsset(USDC_NATIVE_TOKEN[1]).call()
    assert module.functions.isAllowedLighter(lighter_deployment.zk_lighter).call()

    operator_record_file = tmp_path / "lighter-operator.json"
    operator_record_file.write_text(json.dumps({
        "deployments": {
            "ethereum": {
                "vault_address": deployment.vault.address,
                "safe_address": deployment.safe_address,
                "module_address": deployment.trading_strategy_module.address,
                "lighter_account_setup": {
                    "account_index": LIGHTER_ACCOUNT_INDEX,
                    "api_key_index": 4,
                    "private_key": "0x" + "11" * 32,
                },
            },
        },
    }))
    operator_record_file.chmod(PRIVATE_RECORD_MODE)
    state_file = tmp_path / "lighter-cash-management.json"

    # 2. Mock only public Lighter account observations and initialise executor state.
    lighter_state = {"equity": Decimal("0")}

    def fetch_equity(*_args: object, **_kwargs: object) -> LighterEquity:
        return _lighter_equity(lighter_state["equity"])

    def wait_for_collateral(
        _session: object,
        _account_index: int,
        expected_collateral: Decimal,
        **_kwargs: object,
    ) -> Decimal:
        lighter_state["equity"] = expected_collateral
        return expected_collateral

    mocker.patch(
        "tradeexecutor.exchange_account.lighter.fetch_lighter_total_equity",
        side_effect=fetch_equity,
    )
    mocker.patch(
        "tradeexecutor.ethereum.lighter.lighter_routing.fetch_lighter_total_equity",
        side_effect=fetch_equity,
    )
    mocker.patch(
        "tradeexecutor.ethereum.lighter.transfer_verification.fetch_lighter_total_equity",
        side_effect=fetch_equity,
    )
    mocker.patch(
        "tradeexecutor.ethereum.lighter.transfer_verification.wait_for_lighter_collateral",
        side_effect=wait_for_collateral,
    )
    environment = {
        "PATH": os.environ["PATH"],
        "EXECUTOR_ID": "lighter-cash-management",
        "NAME": "lighter-cash-management",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ETHEREUM": anvil_ethereum.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "FILE_LOG_LEVEL": "NONE",
        "RUN_SINGLE_CYCLE": "true",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "PRIVATE_KEY": DEPLOYER_PRIVATE_KEY,
        "VAULT_ADDRESS": deployment.vault.address,
        "VAULT_ADAPTER_ADDRESS": deployment.trading_strategy_module.address,
        "LIGHTER_OPERATOR_RECORD_FILE": operator_record_file.as_posix(),
        "CACHE_PATH": (tmp_path / "cache").as_posix(),
        "MIN_GAS_BALANCE": "0",
    }
    mocker.patch.dict("os.environ", environment, clear=True)
    cli = get_command(app)
    cli.main(args=["init"], standalone_mode=False)

    # 3. Run real Typer ``start`` cycles and verify the automatic deposit pipeline.
    cli.main(args=["start"], standalone_mode=False)
    cli.main(args=["start"], standalone_mode=False)
    state = State.read_json_file(state_file)
    reserve = state.portfolio.get_default_reserve_position()
    position = next(iter(state.portfolio.open_positions.values()))
    transfers = [
        trade
        for trade in position.trades.values()
        if trade.pair.get_exchange_account_protocol() == LIGHTER_PROTOCOL
        and TradeFlag.external_account_transfer in (trade.flags or set())
    ]

    # 4. Run real account diagnostics and assert non-negative conserved balances.
    assert len(transfers) == 1
    assert transfers[0].is_success()
    assert transfers[0].planned_reserve == EXPECTED_LIGHTER_DEPOSIT_USDC
    assert reserve.quantity == SAFE_CASH_BUFFER_USDC
    assert position.get_quantity() == EXPECTED_LIGHTER_DEPOSIT_USDC
    assert lighter_state["equity"] == EXPECTED_LIGHTER_DEPOSIT_USDC
    assert usdc.fetch_balance_of(deployment.safe_address) == SAFE_CASH_BUFFER_USDC
    assert reserve.quantity >= 0
    assert position.get_value() >= 0
    assert position.get_total_profit_usd() == 0

    with pytest.raises(SystemExit) as check_result:
        cli.main(args=["check-accounts"], standalone_mode=False)
    assert check_result.value.code == 0
    cli.main(args=["correct-accounts", "--dry-run"], standalone_mode=False)
    assert "11" * 32 not in state_file.read_text()
