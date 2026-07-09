"""Live CLI blackbox test for phase-aware alpha model with Lagoon vaults."""

import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from eth_defi import token as token_module
from eth_defi.deploy import deploy_contract
from eth_defi.erc_4626.vault_protocol.lagoon import deployment as lagoon_deployment
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig, LagoonAutomatedDeployment, LagoonDeploymentParameters, deploy_automated_lagoon_vault
from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.safe import deployment as safe_deployment
from eth_defi.token import create_token, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from safe_eth.eth.contracts import get_proxy_factory_V1_4_1_contract, get_safe_V1_4_1_contract
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.vault.testing import PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus
from tradeexecutor.strategy.phase_aware import EVENT_PARK, EVENT_PROMOTE, QUEUE_VAULT_EVENT_LOG_KEY
from tradeexecutor.utils.hex import hexbytes_to_hex_str


TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

ANVIL_DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
TARGET_VAULT_SCENARIO = tuple(
    {
        "deposit_open_cycle": deposit_open_cycle,
        "deposit_close_cycle": deposit_open_cycle + 2,
        "deposit_settle_cycle": deposit_open_cycle + 1,
        "redemption_open_cycle": deposit_open_cycle + 7,
        "redemption_close_cycle": deposit_open_cycle + 9,
        "redemption_settle_cycle": deposit_open_cycle + 8,
    }
    for deposit_open_cycle in (1, 3, 5)
)


pytestmark = pytest.mark.skipif(
    not TRADING_STRATEGY_API_KEY,
    reason="Set TRADING_STRATEGY_API_KEY for the live Lagoon e2e test",
)


@pytest.fixture()
def strategy_file() -> Path:
    """Where do we load our strategy file."""
    return Path(os.path.dirname(__file__)) / ".." / ".." / "strategies" / "test_only" / "phase-aware-lagoon-live.py"


@pytest.fixture()
def state_file(tmp_path: Path) -> Path:
    """Where the CLI persists the state JSON."""
    return tmp_path / "phase-aware-lagoon-live.json"


SIMPLE_ERC4626_SOURCE = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

interface IERC20Like {
    function transfer(address to, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

contract SimpleERC4626 {
    string public name;
    string public symbol;
    uint8 public constant decimals = 6;
    IERC20Like public immutable assetToken;
    uint256 public totalSupply;
    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Deposit(address indexed sender, address indexed owner, uint256 assets, uint256 shares);
    event Withdraw(address indexed sender, address indexed receiver, address indexed owner, uint256 assets, uint256 shares);

    constructor(address asset_, string memory name_, string memory symbol_) {
        assetToken = IERC20Like(asset_);
        name = name_;
        symbol = symbol_;
    }

    function asset() external view returns (address) {
        return address(assetToken);
    }

    function totalAssets() external view returns (uint256) {
        return totalSupply;
    }

    function convertToShares(uint256 assets) external pure returns (uint256) {
        return assets;
    }

    function convertToAssets(uint256 shares) external pure returns (uint256) {
        return shares;
    }

    function maxDeposit(address) external pure returns (uint256) {
        return type(uint256).max;
    }

    function maxMint(address) external pure returns (uint256) {
        return type(uint256).max;
    }

    function maxWithdraw(address owner) external view returns (uint256) {
        return balanceOf[owner];
    }

    function maxRedeem(address owner) external view returns (uint256) {
        return balanceOf[owner];
    }

    function previewDeposit(uint256 assets) external pure returns (uint256) {
        return assets;
    }

    function previewMint(uint256 shares) external pure returns (uint256) {
        return shares;
    }

    function previewWithdraw(uint256 assets) external pure returns (uint256) {
        return assets;
    }

    function previewRedeem(uint256 shares) external pure returns (uint256) {
        return shares;
    }

    function approve(address spender, uint256 value) external returns (bool) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }

    function transfer(address to, uint256 value) external returns (bool) {
        _transfer(msg.sender, to, value);
        return true;
    }

    function transferFrom(address from, address to, uint256 value) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            require(allowed >= value, "allowance");
            allowance[from][msg.sender] = allowed - value;
        }
        _transfer(from, to, value);
        return true;
    }

    function deposit(uint256 assets, address receiver) external returns (uint256 shares) {
        shares = assets;
        require(assetToken.transferFrom(msg.sender, address(this), assets), "asset transfer");
        _mint(receiver, shares);
        emit Deposit(msg.sender, receiver, assets, shares);
    }

    function mint(uint256 shares, address receiver) external returns (uint256 assets) {
        assets = shares;
        require(assetToken.transferFrom(msg.sender, address(this), assets), "asset transfer");
        _mint(receiver, shares);
        emit Deposit(msg.sender, receiver, assets, shares);
    }

    function withdraw(uint256 assets, address receiver, address owner) external returns (uint256 shares) {
        shares = assets;
        _spend(owner, shares);
        _burn(owner, shares);
        require(assetToken.transfer(receiver, assets), "asset transfer");
        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }

    function redeem(uint256 shares, address receiver, address owner) external returns (uint256 assets) {
        assets = shares;
        _spend(owner, shares);
        _burn(owner, shares);
        require(assetToken.transfer(receiver, assets), "asset transfer");
        emit Withdraw(msg.sender, receiver, owner, assets, shares);
    }

    function _spend(address owner, uint256 shares) internal {
        if (owner != msg.sender) {
            uint256 allowed = allowance[owner][msg.sender];
            if (allowed != type(uint256).max) {
                require(allowed >= shares, "allowance");
                allowance[owner][msg.sender] = allowed - shares;
            }
        }
    }

    function _transfer(address from, address to, uint256 value) internal {
        require(balanceOf[from] >= value, "balance");
        balanceOf[from] -= value;
        balanceOf[to] += value;
        emit Transfer(from, to, value);
    }

    function _mint(address to, uint256 value) internal {
        totalSupply += value;
        balanceOf[to] += value;
        emit Transfer(address(0), to, value);
    }

    function _burn(address from, uint256 value) internal {
        require(balanceOf[from] >= value, "balance");
        balanceOf[from] -= value;
        totalSupply -= value;
        emit Transfer(from, address(0), value);
    }
}
"""


def _compile_simple_erc4626(tmp_path: Path) -> Path:
    project_dir = tmp_path / "simple-erc4626"
    source_dir = project_dir / "src"
    source_dir.mkdir(parents=True)
    (project_dir / "foundry.toml").write_text(
        '[profile.default]\nsrc = "src"\nout = "out"\nsolc_version = "0.8.26"\nevm_version = "cancun"\nbytecode_hash = "none"\n',
        encoding="utf-8",
    )
    (source_dir / "SimpleERC4626.sol").write_text(SIMPLE_ERC4626_SOURCE, encoding="utf-8")
    result = subprocess.run(
        ["forge", "build", "--root", project_dir.as_posix()],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return project_dir / "out" / "SimpleERC4626.sol" / "SimpleERC4626.json"


def _seed_safe_contracts(web3) -> None:
    deployer = web3.eth.accounts[0]
    safe_contract = get_safe_V1_4_1_contract(web3)
    tx_hash = safe_contract.constructor().transact({"from": deployer, "gas": 8_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    safe_impl = web3.eth.get_transaction_receipt(tx_hash)["contractAddress"]

    proxy_factory_contract = get_proxy_factory_V1_4_1_contract(web3)
    tx_hash = proxy_factory_contract.constructor().transact({"from": deployer, "gas": 3_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    proxy_factory_impl = web3.eth.get_transaction_receipt(tx_hash)["contractAddress"]

    web3.provider.make_request("anvil_setCode", [safe_deployment.SAFE_L2_MASTER_COPY_ADDRESS, web3.eth.get_code(safe_impl).hex()])
    web3.provider.make_request("anvil_setCode", [safe_deployment.SAFE_PROXY_FACTORY_ADDRESS, web3.eth.get_code(proxy_factory_impl).hex()])


@pytest.fixture()
def anvil_lagoon_chain() -> AnvilLaunch:
    """Launch a fresh Anvil chain with no parent RPC."""
    anvil = launch_anvil(gas_limit=100_000_000)
    try:
        yield anvil
    finally:
        anvil.close(log_level=None)


@pytest.fixture()
def web3(anvil_lagoon_chain: AnvilLaunch) -> Web3:
    """Web3 connection to the fresh Anvil chain."""
    web3 = create_multi_provider_web3(
        anvil_lagoon_chain.json_rpc_url,
        default_http_timeout=(3, 250.0),
        retries=1,
    )
    assert web3.eth.chain_id == 31337
    assert web3.eth.block_number <= 1
    return web3


@pytest.fixture()
def lagoon_controller(web3: Web3) -> HotWallet:
    """Anvil-controlled key used to deploy and settle all target Lagoon vaults."""
    controller = HotWallet.from_private_key(ANVIL_DEPLOYER_PRIVATE_KEY)
    controller.sync_nonce(web3)
    return controller


@pytest.fixture()
def deployed_lagoon_vaults(
    web3: Web3,
    lagoon_controller: HotWallet,
    tmp_path: Path,
) -> dict:
    """Deploy faux tokens, one sync queue vault and three async Lagoon target vaults."""
    deployer = web3.eth.accounts[0]
    chain_id = web3.eth.chain_id
    _seed_safe_contracts(web3)

    usdc_contract = create_token(web3, deployer, "Faux USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    weth_contract = create_token(web3, deployer, "Faux Wrapped Ether", "WETH", 100_000_000 * 10**18, decimals=18)
    assert web3.eth.get_code(usdc_contract.address)
    assert web3.eth.get_code(weth_contract.address)
    original_token_wrapped_native = token_module.WRAPPED_NATIVE_TOKEN.get(chain_id)
    original_lagoon_wrapped_native = lagoon_deployment.WRAPPED_NATIVE_TOKEN.get(chain_id)
    original_lagoon_factory = lagoon_deployment.LAGOON_BEACON_PROXY_FACTORIES.get(chain_id)
    token_module.WRAPPED_NATIVE_TOKEN[chain_id] = weth_contract.address
    lagoon_deployment.WRAPPED_NATIVE_TOKEN[chain_id] = weth_contract.address

    try:
        usdc = fetch_erc20_details(web3, usdc_contract.address)
        queue_vault = deploy_contract(
            web3,
            _compile_simple_erc4626(tmp_path).as_posix(),
            deployer,
            usdc.address,
            "Cash Allocation Vault",
            "caUSDC",
            gas=4_000_000,
        )

        tx_hash = usdc.transfer(lagoon_controller.address, Decimal(50)).transact({"from": deployer, "gas": 100_000})
        assert_transaction_success_with_explanation(web3, tx_hash)
        lagoon_controller.sync_nonce(web3)

        target_vaults = []
        for index in range(3):
            lagoon_controller.sync_nonce(web3)
            parameters = LagoonDeploymentParameters(
                underlying=usdc.address,
                name=f"PhaseAwareTestLagoon{index + 1}",
                symbol=f"PATL{index + 1}",
            )
            from_the_scratch = index == 0
            deploy_info = deploy_automated_lagoon_vault(
                web3=web3,
                deployer=lagoon_controller,
                config=LagoonConfig(
                    parameters=parameters,
                    safe_owners=[lagoon_controller.address],
                    safe_threshold=1,
                    asset_manager=lagoon_controller.address,
                    any_asset=True,
                    factory_contract=True,
                    from_the_scratch=from_the_scratch,
                    use_forge=False,
                    between_contracts_delay_seconds=0,
                    safe_salt_nonce=754_000 + index,
                ),
            )
            assert isinstance(deploy_info, LagoonAutomatedDeployment)
            if index == 0:
                lagoon_deployment.LAGOON_BEACON_PROXY_FACTORIES[chain_id] = {
                    "address": deploy_info.beacon_proxy_factory,
                    "abi": "lagoon/BeaconProxyFactory.json",
                }
            fund_lagoon_vault(
                web3,
                deploy_info.vault.vault_address,
                lagoon_controller.address,
                lagoon_controller.address,
                deploy_info.trading_strategy_module.address,
                amount=Decimal(1),
                nav=Decimal(0),
            )
            target_vaults.append({
                "address": deploy_info.vault.vault_address,
                "asset_manager": lagoon_controller.address,
                "trading_strategy_module": deploy_info.trading_strategy_module.address,
                "safe": deploy_info.safe_address,
                "deployment_block": int(deploy_info.block_number),
            })

        deployment_file = tmp_path / "phase-aware-lagoon-deployments.json"
        payload = {
            "target_vaults": target_vaults,
            "queue_vault": queue_vault.address,
            "controller": lagoon_controller.address,
            "usdc": usdc.address,
            "weth": weth_contract.address,
            "target_vault_scenario": TARGET_VAULT_SCENARIO,
        }
        deployment_file.write_text(json.dumps(payload, indent=2))
        yield {
            "file": deployment_file,
            "target_vaults": target_vaults,
            "queue_vault": queue_vault.address,
            "controller": lagoon_controller.address,
            "usdc": usdc.address,
            "weth": weth_contract.address,
        }
    finally:
        if original_token_wrapped_native is None:
            token_module.WRAPPED_NATIVE_TOKEN.pop(chain_id, None)
        else:
            token_module.WRAPPED_NATIVE_TOKEN[chain_id] = original_token_wrapped_native
        if original_lagoon_wrapped_native is None:
            lagoon_deployment.WRAPPED_NATIVE_TOKEN.pop(chain_id, None)
        else:
            lagoon_deployment.WRAPPED_NATIVE_TOKEN[chain_id] = original_lagoon_wrapped_native
        if original_lagoon_factory is None:
            lagoon_deployment.LAGOON_BEACON_PROXY_FACTORIES.pop(chain_id, None)
        else:
            lagoon_deployment.LAGOON_BEACON_PROXY_FACTORIES[chain_id] = original_lagoon_factory


@pytest.fixture()
def hot_wallet(web3: Web3, deployed_lagoon_vaults: dict) -> HotWallet:
    """Create and fund the live strategy hot wallet on the fresh Anvil chain."""
    wallet = HotWallet.create_for_testing(web3)
    wallet.sync_nonce(web3)
    usdc = fetch_erc20_details(web3, deployed_lagoon_vaults["usdc"])
    tx_hash = usdc.transfer(wallet.address, Decimal(150)).transact({"from": web3.eth.accounts[0], "gas": 100_000})
    assert_transaction_success_with_explanation(web3, tx_hash)
    return wallet


@pytest.fixture()
def environment(
    strategy_file: Path,
    anvil_lagoon_chain: AnvilLaunch,
    state_file: Path,
    hot_wallet: HotWallet,
    deployed_lagoon_vaults: dict,
    persistent_test_client: object,
) -> dict:
    """CLI environment for the fresh-Anvil live Lagoon run."""
    cache_path = persistent_test_client.transport.cache_path
    return {
        "EXECUTOR_ID": "test_phase_aware_lagoon_live_e2e",
        "NAME": "test_phase_aware_lagoon_live_e2e",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "JSON_RPC_ANVIL": anvil_lagoon_chain.json_rpc_url,
        "STATE_FILE": state_file.as_posix(),
        "ASSET_MANAGEMENT_MODE": "hot_wallet",
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "MAX_CYCLES": "15",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),
        "MIN_GAS_BALANCE": "0",
        "GAS_BALANCE_WARNING_LEVEL": "0.0",
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.private_key),
        "CACHE_PATH": cache_path,
        "CYCLE_DURATION": "1s",
        "TRADE_IMMEDIATELY": "true",
        "CONFIRMATION_BLOCK_COUNT": "0",
        "DIRECT_ANVIL_BROADCAST": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "STATS_REFRESH_MINUTES": "0",
        "CHECK_ACCOUNTS": "false",
        "VISUALISATION": "false",
        "SYNC_TREASURY_ON_STARTUP": "true",
        "PHASE_AWARE_LAGOON_DEPLOYMENTS_FILE": deployed_lagoon_vaults["file"].as_posix(),
        "PATH": os.environ["PATH"],
    }


def _collect_observations(state: State) -> list[dict]:
    return [
        cycle_data[PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY]
        for _, cycle_data in sorted(state.other_data.data.items(), key=lambda item: int(item[0]))
        if PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY in cycle_data
    ]


def _collect_phase_events(state: State) -> list[dict]:
    return [
        event
        for _, cycle_data in sorted(state.other_data.data.items(), key=lambda item: int(item[0]))
        for event in cycle_data.get(QUEUE_VAULT_EVENT_LOG_KEY, [])
    ]


def _positive_decimal(value: str | None) -> bool:
    if value is None:
        return False
    return abs(Decimal(value)) > 0


@pytest.mark.slow_test_group
def test_phase_aware_lagoon_live_e2e(
    environment: dict,
    monkeypatch: pytest.MonkeyPatch,
    state_file: Path,
    deployed_lagoon_vaults: dict,
) -> None:
    """Run the phase-aware alpha model through the live CLI and Lagoon contracts.

    1. Replace the CLI environment with three async Lagoon target vaults and one sync queue vault.
    2. Start the normal live trade-executor loop with one-second cycles on a fresh Anvil chain.
    3. Let the strategy tick hooks toggle and settle target vaults to simulate staggered vault epochs.
    4. Read the persisted state JSON and assert queued deposits, queued redemptions, processed claims and queue-vault utilisation.
    """
    cli = get_command(app)
    target_vaults = [vault["address"].lower() for vault in deployed_lagoon_vaults["target_vaults"]]
    assert len(target_vaults) == len(TARGET_VAULT_SCENARIO)

    # 1. Replace the CLI environment with three async Lagoon target vaults and one sync queue vault.
    for key in list(os.environ.keys()):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    # 2. Start the normal live trade-executor loop with one-second cycles on a fresh Anvil chain.
    cli.main(args=["init"], standalone_mode=False)

    # 3. Let the strategy tick hooks toggle and settle target vaults to simulate staggered vault epochs.
    cli.main(args=["start"], standalone_mode=False)

    # 4. Read the persisted state JSON and assert queued deposits, queued redemptions, processed claims and queue-vault utilisation.
    state = State.read_json_file(state_file)
    observations = _collect_observations(state)
    assert observations
    assert all(observation["core_vaults"] for observation in observations)

    processed_redemption_cycles_by_vault = {}
    scenario_by_vault = {
        vault: TARGET_VAULT_SCENARIO[index]
        for index, vault in enumerate(target_vaults)
    }
    for vault in target_vaults:
        schedule = scenario_by_vault[vault]
        pending_deposit_cycles = [
            observation["cycle"]
            for observation in observations
            if any(trade["vault"] == vault for trade in observation["pending_deposits"])
        ]
        processed_deposit_cycles = [
            observation["cycle"]
            for observation in observations
            if any(trade["vault"] == vault for trade in observation["processed_deposits"])
        ]
        pending_redemption_cycles = [
            observation["cycle"]
            for observation in observations
            if any(trade["vault"] == vault for trade in observation["pending_redemptions"])
        ]
        processed_redemption_cycles = [
            observation["cycle"]
            for observation in observations
            if any(trade["vault"] == vault for trade in observation["processed_redemptions"])
        ]
        assert pending_deposit_cycles
        assert processed_deposit_cycles
        assert pending_redemption_cycles
        assert processed_redemption_cycles
        assert min(pending_deposit_cycles) == schedule["deposit_open_cycle"]
        assert all(cycle >= schedule["deposit_open_cycle"] for cycle in pending_deposit_cycles)
        assert min(processed_deposit_cycles) == schedule["deposit_settle_cycle"] + 1
        assert all(cycle >= schedule["deposit_settle_cycle"] + 1 for cycle in processed_deposit_cycles)
        assert min(pending_redemption_cycles) == schedule["redemption_open_cycle"]
        assert all(cycle >= schedule["redemption_open_cycle"] for cycle in pending_redemption_cycles)
        assert min(processed_redemption_cycles) == schedule["redemption_settle_cycle"] + 1
        assert all(cycle >= schedule["redemption_settle_cycle"] + 1 for cycle in processed_redemption_cycles)
        processed_redemption_cycles_by_vault[vault] = min(processed_redemption_cycles)
        assert any(
            observation["core_vaults"][vault]["can_deposit"] is False
            for observation in observations
        )
        assert any(
            observation["core_vaults"][vault]["can_deposit"] is True
            for observation in observations
        )
        assert any(
            observation["core_vaults"][vault]["can_redeem"] is False
            and int(observation["pre_tick_onchain_vaults"][vault]["share_balance_raw"]) > 0
            for observation in observations
        )
        assert any(
            observation["core_vaults"][vault]["can_redeem"] is True
            for observation in observations
        )
        assert all(
            observation["core_vaults"][vault]["can_deposit"] is False
            and observation["core_vaults"][vault]["can_redeem"] is False
            for observation in observations
            if observation["pre_tick_onchain_vaults"][vault]["paused"] is True
        )
        for observation in observations:
            cycle = observation["cycle"]
            deposit_window = schedule["deposit_open_cycle"] <= cycle <= schedule["deposit_close_cycle"]
            redemption_window = schedule["redemption_open_cycle"] <= cycle <= schedule["redemption_close_cycle"]
            expected_open = deposit_window or redemption_window
            assert observation["pre_tick_onchain_vaults"][vault]["paused"] is (not expected_open)
            if not expected_open:
                assert observation["core_vaults"][vault]["can_deposit"] is False
                assert observation["core_vaults"][vault]["can_redeem"] is False
        assert any(
            observation["pre_tick_onchain_vaults"][vault]["paused"] is False
            for observation in observations
        )
        assert any(
            observation["pre_tick_onchain_vaults"][vault]["paused"] is True
            and int(observation["pre_tick_onchain_vaults"][vault]["share_balance_raw"]) > 0
            for observation in observations
        )
        assert any(int(observation["pre_tick_onchain_vaults"][vault]["max_deposit_raw"]) > 0 for observation in observations)
        assert any(int(observation["onchain_vaults"][vault]["share_balance_raw"]) > 0 for observation in observations)
        assert any(int(observation["pre_tick_onchain_vaults"][vault]["max_redeem_raw"]) > 0 for observation in observations)
        assert any(int(observation["onchain_vaults"][vault]["pending_deposit_raw"]) > 0 for observation in observations)
        assert any(int(observation["onchain_vaults"][vault]["pending_redeem_raw"]) > 0 for observation in observations)
        assert int(observations[-1]["onchain_vaults"][vault]["pending_deposit_raw"]) == 0
        assert int(observations[-1]["onchain_vaults"][vault]["pending_redeem_raw"]) == 0

    assert any(observation["queue_deposits"] for observation in observations)
    assert any(observation["queue_redemptions"] for observation in observations)
    assert any(int(observation["queue_vault"]["share_balance_raw"]) > 0 for observation in observations)
    assert any(int(observation["queue_vault"]["total_assets_raw"]) > 0 for observation in observations)
    first_processed_redemption_cycle = min(processed_redemption_cycles_by_vault.values())
    assert any(
        observation["cycle"] >= first_processed_redemption_cycle
        and any(
            trade["opened_at"] == observation["timestamp"]
            for trade in observation["queue_deposits"]
        )
        for observation in observations
    )
    assert int(observations[-1]["queue_vault"]["share_balance_raw"]) > 0
    assert any(
        _positive_decimal(trade["executed_reserve"]) or _positive_decimal(trade["planned_reserve"])
        for observation in observations
        for trade in observation["queue_deposits"]
    )
    assert any(
        _positive_decimal(trade["executed_quantity"]) or _positive_decimal(trade["planned_quantity"])
        for observation in observations
        for trade in observation["queue_redemptions"]
    )
    assert any(
        trade["yield_decision"]
        for observation in observations
        for trade in observation["queue_trades"]
    )

    events = _collect_phase_events(state)
    event_kinds = {event["kind"] for event in events}
    assert EVENT_PARK in event_kinds
    assert EVENT_PROMOTE in event_kinds
    park_events = [event for event in events if event["kind"] == EVENT_PARK]
    promote_events = [event for event in events if event["kind"] == EVENT_PROMOTE]
    parked_vault_ids = {event["vault_internal_id"] for event in park_events}
    promoted_vault_ids = {event["vault_internal_id"] for event in promote_events}
    target_vault_ids = {
        trade["pair_internal_id"]
        for observation in observations
        for trade in observation["target_trades"]
    }
    target_vault_by_id = {
        trade["pair_internal_id"]: trade["vault"]
        for observation in observations
        for trade in observation["target_trades"]
    }
    observation_by_cycle = {observation["cycle"]: observation for observation in observations}
    assert parked_vault_ids
    assert parked_vault_ids == promoted_vault_ids
    assert parked_vault_ids <= target_vault_ids
    assert all(event["usd"] > 0 for event in park_events + promote_events)
    for promote_event in promote_events:
        vault_id = promote_event["vault_internal_id"]
        vault = target_vault_by_id[vault_id]
        expected_cycle = scenario_by_vault[vault]["deposit_open_cycle"]
        promote_cycle = promote_event["cycle"]
        promote_timestamp = promote_event["timestamp"]
        observation = observation_by_cycle[promote_cycle]
        assert promote_cycle == expected_cycle
        assert any(
            event["vault_internal_id"] == vault_id and event["cycle"] < expected_cycle
            for event in park_events
        )
        assert any(
            trade["pair_internal_id"] == vault_id
            and trade["direction"] == "deposit"
            and trade["opened_at"] == promote_timestamp
            and (_positive_decimal(trade["planned_reserve"]) or _positive_decimal(trade["executed_reserve"]))
            for trade in observation["target_trades"]
        )
        assert any(
            trade["direction"] == "redeem"
            and trade["yield_decision"]
            and trade["opened_at"] == promote_timestamp
            and (_positive_decimal(trade["planned_quantity"]) or _positive_decimal(trade["executed_quantity"]))
            for trade in observation["queue_trades"]
        )

    trades = list(state.portfolio.get_all_trades())
    assert len(state.portfolio.frozen_positions) == 0
    assert all(trade.is_success() for trade in trades)
    assert not any(trade.get_status() == TradeStatus.vault_settlement_pending for trade in trades)
