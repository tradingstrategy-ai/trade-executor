"""Live CLI blackbox test for phase-aware alpha model with Lagoon vaults."""

import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path

import pytest
from pytest_mock import MockerFixture
from eth_defi import token as token_module
from eth_defi.deploy import deploy_contract
from eth_defi.erc_4626.classification import ERC4626Feature, create_vault_instance
from eth_defi.erc_4626.vault_protocol.lagoon import deployment as lagoon_deployment
from eth_defi.erc_4626.vault_protocol.lagoon.deployment import LagoonConfig, LagoonAutomatedDeployment, LagoonDeploymentParameters, deploy_automated_lagoon_vault
from eth_defi.erc_4626.vault_protocol.lagoon.testing import fund_lagoon_vault
from eth_defi.event_reader.conversion import convert_uin256_to_bytes
from eth_defi.event_reader.multicall_batcher import EncodedCall
from eth_defi.hotwallet import HotWallet
from eth_defi.provider.anvil import AnvilLaunch, launch_anvil
from eth_defi.provider.multi_provider import create_multi_provider_web3
from eth_defi.provider.receipt import wait_for_transaction_receipt_robust
from eth_defi.safe import deployment as safe_deployment
from eth_defi.token import create_token, fetch_erc20_details
from eth_defi.trace import assert_transaction_success_with_explanation
from safe_eth.eth.contracts import get_proxy_factory_V1_4_1_contract, get_safe_V1_4_1_contract
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.loop import ExecutionLoop
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.testing import set_lagoon_vault_open_for_testing
from tradeexecutor.ethereum.vault import settlement_retry
from tradeexecutor.ethereum.vault.testing import PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution, TradeStatus
from tradeexecutor.strategy.phase_aware import EVENT_PARK, EVENT_PROMOTE, QUEUE_VAULT_EVENT_LOG_KEY
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverseModel
from tradeexecutor.utils.hex import hexbytes_to_hex_str


TRADING_STRATEGY_API_KEY = os.environ.get("TRADING_STRATEGY_API_KEY")

ANVIL_DEPLOYER_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
TARGET_VAULT_SCENARIO = tuple(
    {
        "deposit_open_cycle": deposit_open_cycle,
        "deposit_close_cycle": deposit_open_cycle + 2,
        "deposit_settle_cycle": deposit_open_cycle + 1,
        "redemption_open_cycle": deposit_open_cycle + 4,
        "redemption_close_cycle": deposit_open_cycle + 6,
        "redemption_settle_cycle": deposit_open_cycle + 5,
    }
    for deposit_open_cycle in range(1, 4)
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


def _deploy_fresh_lagoon_protocol_from_abi(web3, deployer: HotWallet, safe, broadcast_func, **kwargs):
    wrapped_native_token_address = lagoon_deployment.WRAPPED_NATIVE_TOKEN[web3.eth.chain_id]
    fee_registry = deploy_contract(web3, "lagoon/ProtocolRegistry.json", deployer, False, gas=4_000_000)
    tx_hash = broadcast_func(fee_registry.functions.initialize(safe.address, safe.address))
    assert_transaction_success_with_explanation(web3, tx_hash)
    implementation = deploy_contract(web3, "lagoon/v0.5.0/Vault.json", deployer, True, gas=9_000_000)
    return deploy_contract(
        web3,
        "lagoon/BeaconProxyFactory.json",
        deployer,
        fee_registry.address,
        implementation.address,
        safe.address,
        wrapped_native_token_address,
        gas=6_000_000,
    )


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
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    """Deploy faux tokens, one sync queue vault and three async Lagoon target vaults."""
    deployer = web3.eth.accounts[0]
    _seed_safe_contracts(web3)
    monkeypatch.setattr(lagoon_deployment, "deploy_fresh_lagoon_protocol", _deploy_fresh_lagoon_protocol_from_abi)

    usdc_contract = create_token(web3, deployer, "Faux USD Coin", "USDC", 100_000_000 * 10**6, decimals=6)
    weth_contract = create_token(web3, deployer, "Faux Wrapped Ether", "WETH", 100_000_000 * 10**18, decimals=18)
    assert web3.eth.get_code(usdc_contract.address)
    assert web3.eth.get_code(weth_contract.address)
    monkeypatch.setitem(token_module.WRAPPED_NATIVE_TOKEN, web3.eth.chain_id, weth_contract.address)
    monkeypatch.setitem(lagoon_deployment.WRAPPED_NATIVE_TOKEN, web3.eth.chain_id, weth_contract.address)

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
                use_forge=from_the_scratch,
                between_contracts_delay_seconds=0,
                safe_salt_nonce=754_000 + index,
            ),
        )
        assert isinstance(deploy_info, LagoonAutomatedDeployment)
        if index == 0:
            monkeypatch.setitem(
                lagoon_deployment.LAGOON_BEACON_PROXY_FACTORIES,
                web3.eth.chain_id,
                {
                    "address": deploy_info.beacon_proxy_factory,
                    "abi": "lagoon/BeaconProxyFactory.json",
                },
            )
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
    }
    deployment_file.write_text(json.dumps(payload, indent=2))
    return {
        "file": deployment_file,
        "target_vaults": target_vaults,
        "queue_vault": queue_vault.address,
        "controller": lagoon_controller.address,
        "usdc": usdc.address,
        "weth": weth_contract.address,
    }


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
        "MAX_CYCLES": "10",
        "TRADING_STRATEGY_API_KEY": TRADING_STRATEGY_API_KEY,
        "MAX_DATA_DELAY_MINUTES": str(10 * 60 * 24 * 365),
        "MIN_GAS_BALANCE": "0.01",
        "GAS_BALANCE_WARNING_LEVEL": "0.0",
        "PRIVATE_KEY": hexbytes_to_hex_str(hot_wallet.private_key),
        "CACHE_PATH": cache_path,
        "CYCLE_DURATION": "1s",
        "TRADE_IMMEDIATELY": "true",
        "POSITION_TRIGGER_CHECK_MINUTES": "0",
        "STATS_REFRESH_MINUTES": "0",
        "CHECK_ACCOUNTS": "false",
        "VISUALISATION": "false",
        "SYNC_TREASURY_ON_STARTUP": "true",
        "PHASE_AWARE_LAGOON_DEPLOYMENTS_FILE": deployed_lagoon_vaults["file"].as_posix(),
        "PATH": os.environ["PATH"],
    }


def _trade_vault_address(trade: TradeExecution) -> str:
    return trade.pair.pool_address.lower()


def _trade_direction(trade: TradeExecution) -> str:
    return "deposit" if trade.is_buy() else "redeem"


def _has_claim_transaction(trade: TradeExecution) -> bool:
    return any(tx.other.get("vault_settlement_action") == "claim" for tx in trade.blockchain_transactions)


def _decimal_str(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _datetime_str(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _raw_str(value: int) -> str:
    return str(int(value))


def _uint256_arg(value: int) -> str:
    return f"{int(value):064x}"


def _address_arg(address: str) -> str:
    return address.removeprefix("0x").lower().zfill(64)


def _eth_call_uint256(receipt_web3, address: str, signature: str, args: str = "") -> int:
    data = "0x" + Web3.keccak(text=signature)[0:4].hex() + args
    result = receipt_web3.eth.call(
        {
            "to": Web3.to_checksum_address(address),
            "data": data,
        },
        block_identifier="latest",
    )
    if len(result) == 0:
        return 0
    return int.from_bytes(result, byteorder="big")


def _trade_evidence(trade: TradeExecution) -> dict:
    return {
        "trade_id": trade.trade_id,
        "vault": _trade_vault_address(trade),
        "pair_internal_id": trade.pair.internal_id,
        "direction": _trade_direction(trade),
        "opened_at": _datetime_str(trade.opened_at),
        "status": trade.get_status().value,
        "pending": trade.get_status() == TradeStatus.vault_settlement_pending,
        "processed": trade.is_success() and _has_claim_transaction(trade),
        "claim_transaction": _has_claim_transaction(trade),
        "planned_reserve": _decimal_str(trade.planned_reserve),
        "planned_quantity": _decimal_str(trade.planned_quantity),
        "executed_reserve": _decimal_str(trade.executed_reserve),
        "executed_quantity": _decimal_str(trade.executed_quantity),
    }


def _load_lagoon_vault(receipt_web3, target_vault: str, vault_cache: dict[str, object]):
    vault = vault_cache.get(target_vault)
    if vault is None:
        vault = create_vault_instance(
            receipt_web3,
            target_vault,
            features={ERC4626Feature.lagoon_like, ERC4626Feature.erc_7540_like},
        )
        vault_cache[target_vault] = vault
    return vault


def _collect_onchain_vault_observations(
    receipt_web3,
    target_vaults: list[str],
    owner_address: str,
) -> dict[str, dict]:
    onchain = {}
    owner_arg = _address_arg(owner_address)
    for target_vault in target_vaults:
        request_args = _uint256_arg(0) + owner_arg
        onchain[target_vault] = {
            "paused": bool(_eth_call_uint256(receipt_web3, target_vault, "paused()")),
            "pending_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "pendingDepositRequest(uint256,address)", request_args)),
            "pending_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "pendingRedeemRequest(uint256,address)", request_args)),
            "max_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "maxDeposit(address)", owner_arg)),
            "max_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "maxRedeem(address)", owner_arg)),
            "share_balance_raw": _raw_str(_eth_call_uint256(receipt_web3, target_vault, "balanceOf(address)", owner_arg)),
        }
    return onchain


def _collect_queue_vault_observation(
    receipt_web3,
    queue_vault: str,
    owner_address: str,
) -> dict:
    owner_arg = _address_arg(owner_address)
    return {
        "share_balance_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "balanceOf(address)", owner_arg)),
        "max_deposit_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "maxDeposit(address)", owner_arg)),
        "max_redeem_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "maxRedeem(address)", owner_arg)),
        "total_assets_raw": _raw_str(_eth_call_uint256(receipt_web3, queue_vault, "totalAssets()")),
    }


def _record_observation(
    state: State,
    cycle: int,
    cycle_timestamp,
    target_vaults: list[str],
    queue_vault: str,
    receipt_web3,
    owner_address: str,
    pre_tick_onchain_vaults: dict[str, dict],
) -> None:
    target_set = set(target_vaults)
    existing_observation = state.other_data.data.get(cycle, {}).get(PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY, {})
    target_trades = [
        _trade_evidence(trade)
        for trade in state.portfolio.get_all_trades()
        if _trade_vault_address(trade) in target_set
    ]
    queue_trades = [
        {
            "trade_id": trade.trade_id,
            "vault": _trade_vault_address(trade),
            "direction": _trade_direction(trade),
            "opened_at": _datetime_str(trade.opened_at),
            "status": trade.get_status().value,
            "yield_decision": trade.get_yield_decision() is not None,
            "planned_reserve": _decimal_str(trade.planned_reserve),
            "planned_quantity": _decimal_str(trade.planned_quantity),
            "executed_reserve": _decimal_str(trade.executed_reserve),
            "executed_quantity": _decimal_str(trade.executed_quantity),
        }
        for trade in state.portfolio.get_all_trades()
        if _trade_vault_address(trade) == queue_vault
    ]
    observation = {
        **existing_observation,
        "cycle": cycle,
        "timestamp": _datetime_str(cycle_timestamp),
        "target_trades": target_trades,
        "pending_deposits": [trade for trade in target_trades if trade["direction"] == "deposit" and trade["pending"]],
        "processed_deposits": [trade for trade in target_trades if trade["direction"] == "deposit" and trade["processed"]],
        "pending_redemptions": [trade for trade in target_trades if trade["direction"] == "redeem" and trade["pending"]],
        "processed_redemptions": [trade for trade in target_trades if trade["direction"] == "redeem" and trade["processed"]],
        "queue_trades": queue_trades,
        "queue_deposits": [trade for trade in queue_trades if trade["direction"] == "deposit" and trade["status"] == TradeStatus.success.value],
        "queue_redemptions": [trade for trade in queue_trades if trade["direction"] == "redeem" and trade["status"] == TradeStatus.success.value],
        "core_vaults": existing_observation.get("core_vaults", {}),
        "pre_tick_onchain_vaults": pre_tick_onchain_vaults,
        "onchain_vaults": _collect_onchain_vault_observations(receipt_web3, target_vaults, owner_address),
        "queue_vault": _collect_queue_vault_observation(receipt_web3, queue_vault, owner_address),
    }
    state.other_data.save(cycle, PHASE_AWARE_LIVE_E2E_OBSERVATION_KEY, observation)


def _controller_should_open_target_vault(index: int, cycle: int) -> bool:
    """Private Anvil scenario controller for Lagoon open windows."""
    schedule = TARGET_VAULT_SCENARIO[index]
    deposit_open = schedule["deposit_open_cycle"] <= cycle <= schedule["deposit_close_cycle"]
    redemption_open = schedule["redemption_open_cycle"] <= cycle <= schedule["redemption_close_cycle"]
    return deposit_open or redemption_open


def _set_controller_lagoon_open_states(
    receipt_web3,
    cycle: int,
    target_vaults: list[str],
    safe_addresses: list[str],
    vault_cache: dict[str, object],
) -> None:
    for index, target_vault in enumerate(target_vaults):
        _set_lagoon_vault_open_state(
            receipt_web3,
            target_vault,
            safe_addresses[index],
            _controller_should_open_target_vault(index, cycle),
            vault_cache,
        )


def _set_lagoon_vault_open_state(
    receipt_web3,
    target_vault: str,
    safe_address: str,
    open_: bool,
    vault_cache: dict[str, object],
) -> None:
    vault = _load_lagoon_vault(receipt_web3, target_vault, vault_cache)
    tx_hash = set_lagoon_vault_open_for_testing(
        receipt_web3,
        vault.vault_contract,
        safe_address,
        open_,
    )
    if tx_hash is not None:
        receipt = _wait_for_local_anvil_receipt(receipt_web3, tx_hash)
        _assert_local_receipt_success(receipt_web3, tx_hash, receipt)


def _force_settle_controller_vaults(
    receipt_web3,
    cycle: int,
    target_vaults: list[str],
    asset_managers: list[str],
    safe_addresses: list[str],
    vault_cache: dict[str, object],
) -> None:
    for index, target_vault in enumerate(target_vaults):
        schedule = TARGET_VAULT_SCENARIO[index]
        deposit_settle_cycle = schedule["deposit_settle_cycle"]
        redemption_settle_cycle = schedule["redemption_settle_cycle"]
        if cycle not in (deposit_settle_cycle, redemption_settle_cycle):
            continue
        vault = _load_lagoon_vault(receipt_web3, target_vault, vault_cache)
        safe_address = safe_addresses[index]
        settle_open_tx_hash = set_lagoon_vault_open_for_testing(
            receipt_web3,
            vault.vault_contract,
            safe_address,
            True,
        )
        if settle_open_tx_hash is not None:
            settle_open_receipt = _wait_for_local_anvil_receipt(receipt_web3, settle_open_tx_hash)
            _assert_local_receipt_success(receipt_web3, settle_open_tx_hash, settle_open_receipt)
        valuation = vault.underlying_token.fetch_balance_of(vault.safe_address)
        raw_valuation = vault.denomination_token.convert_to_raw(valuation)
        valuation_tx_hash = vault.post_new_valuation(valuation).transact({"from": asset_managers[index], "gas": 1_000_000})
        valuation_receipt = _wait_for_local_anvil_receipt(receipt_web3, valuation_tx_hash)
        _assert_local_receipt_success(receipt_web3, valuation_tx_hash, valuation_receipt)
        # Do not use force_lagoon_settle() here: this scenario needs the
        # valuation tx from the asset manager and settleDeposit() from the Safe.
        settle_call = EncodedCall.from_keccak_signature(
            address=vault.address,
            function="settleDeposit()",
            signature=Web3.keccak(text="settleDeposit(uint256)")[0:4],
            data=convert_uin256_to_bytes(raw_valuation),
            extra_data=None,
        )
        receipt_web3.provider.make_request("anvil_setBalance", [safe_address, hex(5 * 10**18)])
        receipt_web3.provider.make_request("anvil_impersonateAccount", [safe_address])
        tx_hash = receipt_web3.eth.send_transaction(settle_call.transact(from_=safe_address, gas_limit=1_000_000))
        receipt = _wait_for_local_anvil_receipt(receipt_web3, tx_hash)
        _assert_local_receipt_success(receipt_web3, tx_hash, receipt)


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


def _wait_for_local_anvil_receipt(web3, tx_hash):
    return wait_for_transaction_receipt_robust(web3, tx_hash, timeout=15, poll_delay=0.1)


def _assert_local_receipt_success(web3, tx_hash, receipt, func=None) -> None:
    if receipt["status"] != 1:
        assert_transaction_success_with_explanation(web3, tx_hash, func=func, tracing=True)
    assert receipt["status"] == 1


@pytest.mark.slow_test_group
def test_phase_aware_lagoon_live_e2e(
    environment: dict,
    mocker: MockerFixture,
    state_file: Path,
    web3: Web3,
    deployed_lagoon_vaults: dict,
    hot_wallet: HotWallet,
) -> None:
    """Run the phase-aware alpha model through the live CLI and Lagoon contracts.

    1. Patch the CLI environment with three async Lagoon target vaults and one sync queue vault.
    2. Start the normal live trade-executor loop with one-second cycles on a fresh Anvil chain.
    3. Force-settle target vaults from the private Anvil scenario controller to simulate staggered vault epochs.
    4. Read the persisted state JSON and assert queued deposits, queued redemptions, processed claims and queue-vault utilisation.
    """
    cli = get_command(app)
    target_vaults = [vault["address"].lower() for vault in deployed_lagoon_vaults["target_vaults"]]
    asset_managers = [vault["asset_manager"] for vault in deployed_lagoon_vaults["target_vaults"]]
    safe_addresses = [vault["safe"] for vault in deployed_lagoon_vaults["target_vaults"]]
    queue_vault = deployed_lagoon_vaults["queue_vault"].lower()
    assert len(target_vaults) == len(TARGET_VAULT_SCENARIO)
    original_tick = ExecutionLoop.tick
    settlement_vault_cache = {}
    settlement_receipt_web3 = create_multi_provider_web3(
        environment["JSON_RPC_ANVIL"],
        default_http_timeout=(3, 30.0),
        retries=1,
    )
    def wait_and_broadcast_on_local_anvil(web3, txs, **kwargs):
        receipts = {}
        for tx in txs:
            try:
                web3.eth.send_raw_transaction(tx.rawTransaction)
            except ValueError as e:
                if "already known" not in str(e):
                    raise
        for tx in txs:
            receipt = _wait_for_local_anvil_receipt(web3, tx.hash)
            receipts[tx.hash] = receipt
        return receipts

    def tick_wrapper(self, *args, **kwargs):
        state = kwargs.get("state") if "state" in kwargs else args[2]
        cycle = kwargs.get("cycle") if "cycle" in kwargs else args[3]
        if "strategy_cycle_timestamp" in kwargs:
            cycle_timestamp = kwargs["strategy_cycle_timestamp"]
        elif len(args) > 6:
            cycle_timestamp = args[6]
        else:
            cycle_timestamp = None
        _set_controller_lagoon_open_states(
            settlement_receipt_web3,
            cycle,
            target_vaults,
            safe_addresses,
            settlement_vault_cache,
        )
        pre_tick_onchain_vaults = _collect_onchain_vault_observations(
            settlement_receipt_web3,
            target_vaults,
            hot_wallet.address,
        )
        universe = original_tick(self, *args, **kwargs)
        _record_observation(
            state,
            cycle,
            cycle_timestamp,
            target_vaults,
            queue_vault,
            settlement_receipt_web3,
            hot_wallet.address,
            pre_tick_onchain_vaults,
        )
        # 3. Force-settle target vaults from the private Anvil scenario controller to simulate staggered vault epochs.
        _force_settle_controller_vaults(
            settlement_receipt_web3,
            cycle,
            target_vaults,
            asset_managers,
            safe_addresses,
            settlement_vault_cache,
        )
        self.store.sync(state)
        return universe

    # 1. Patch the CLI environment with three async Lagoon target vaults and one sync queue vault.
    mocker.patch.dict("os.environ", environment, clear=True)
    # The synthetic in-memory universe has fresh Anvil deployments whose data age is controlled by this test.
    mocker.patch.object(TradingStrategyUniverseModel, "check_data_age", return_value=None)
    mocker.patch.object(ExecutionLoop, "tick", tick_wrapper)
    # This blackbox test restarts from local state only; live-run metadata refresh would call external services.
    mocker.patch.object(ExecutionLoop, "refresh_live_run_state", return_value=None)
    mocker.patch("tradeexecutor.ethereum.execution.wait_and_broadcast_multiple_nodes", side_effect=wait_and_broadcast_on_local_anvil)
    mocker.patch.object(settlement_retry, "_wait_for_settlement_tx_receipt", side_effect=_wait_for_local_anvil_receipt)
    # The target vaults are freshly deployed on Anvil and have no historical profitability data.
    mocker.patch("tradeexecutor.ethereum.vault.vault_routing.estimate_4626_recent_profitability", return_value=None)
    # Fresh Anvil automines transactions; the runner's extra manual mining can block the one-second test loop.
    mocker.patch("tradeexecutor.strategy.runner.mine", return_value=None)

    # 2. Start the normal live trade-executor loop with one-second cycles on a fresh Anvil chain.
    cli.main(args=["init"], standalone_mode=False)
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
