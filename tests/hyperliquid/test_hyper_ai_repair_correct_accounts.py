"""Test HyperAI frozen-position repair and account correction against live data.

This test exercises the exact HyperAI damage pattern captured in the
downloaded state fixture.
"""

import datetime
import logging
import os
import shutil
import warnings
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest
from eth_defi.compat import native_datetime_utc_now
from eth_defi.hyperliquid.api import fetch_user_vault_equity
from eth_defi.hyperliquid.session import (HYPERLIQUID_API_URL,
                                          create_hyperliquid_session)
from typer.main import get_command
from web3 import Web3

from tradeexecutor.cli.bootstrap import (create_client,
                                         create_execution_and_sync_model,
                                         create_web3_config)
from tradeexecutor.cli.main import app
from tradeexecutor.state.state import State
from tradeexecutor.statistics.core import update_statistics
from tradeexecutor.strategy.approval import UncheckedApprovalModel
from tradeexecutor.strategy.bootstrap import make_factory_from_strategy_mod
from tradeexecutor.strategy.description import StrategyExecutionDescription
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.execution_model import AssetManagementMode
from tradeexecutor.strategy.run_state import RunState
from tradeexecutor.strategy.strategy_module import read_strategy_module
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverseModel
from tradeexecutor.strategy.universe_model import UniverseOptions
from tradeexecutor.strategy.valuation import revalue_state
from tradeexecutor.utils.timer import timed_task

warnings.filterwarnings(
    "ignore",
    message="`NoneType` object value of non-optional type exchange_address detected when decoding TradingPairIdentifier.",
    category=RuntimeWarning,
)


pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        not os.environ.get("TRADING_STRATEGY_API_KEY"),
        reason="TRADING_STRATEGY_API_KEY not set",
    ),
]


HYPER_AI_LAGOON_VAULT = "0x282cB588099844Dc93C0B7bd6701298666Ee76bE"
HYPER_AI_MODULE = "0xAf4e8d50dA5Aa49Eee8cf04fc4682d5c090902E7"
HYPER_AI_VAULT_POSITION = "0x07fd993f0fa3a185f7207adccd29f7a87404689d"
HYPER_EVM_RPC = "https://rpc.hyperliquid.xyz/evm"
USDC_ADDRESS = "0xb88339CB7199b77E23DB6E890353E22632Ba630f"
DUMMY_PRIVATE_KEY = "0x111e53aed5e777996f26b4bdb89300bbc05b84743f32028c41be7193c0fe0b83"


@pytest.fixture()
def strategy_file() -> Path:
    """Return the live HyperAI strategy module path."""
    return Path(__file__).resolve().parents[2] / "strategies" / "hyper-ai.py"


@pytest.fixture()
def state_fixture_file() -> Path:
    """Return the committed broken HyperAI state fixture."""
    return Path(__file__).resolve().parent / "state" / "hyperai-frozen-vault.json"


@pytest.fixture()
def state_file(tmp_path: Path, state_fixture_file: Path) -> Path:
    """Copy the committed fixture to a temporary mutable test state file."""
    state_file = tmp_path / "hyperai-frozen-vault.json"
    shutil.copy(state_fixture_file, state_file)
    return state_file


@pytest.fixture()
def environment(state_file: Path, strategy_file: Path) -> dict[str, str]:
    """Build the minimal single-chain CLI environment for the HyperAI test."""
    return {
        "EXECUTOR_ID": "test_hyper_ai_repair_correct_accounts",
        "STRATEGY_FILE": strategy_file.as_posix(),
        "STATE_FILE": state_file.as_posix(),
        "PRIVATE_KEY": DUMMY_PRIVATE_KEY,
        "JSON_RPC_HYPERLIQUID": HYPER_EVM_RPC,
        "ASSET_MANAGEMENT_MODE": "lagoon",
        "VAULT_ADDRESS": HYPER_AI_LAGOON_VAULT,
        "VAULT_ADAPTER_ADDRESS": HYPER_AI_MODULE,
        "UNIT_TESTING": "true",
        "LOG_LEVEL": "disabled",
        "TRADING_STRATEGY_API_KEY": os.environ["TRADING_STRATEGY_API_KEY"],
        "PATH": os.environ.get("PATH", ""),
    }


def _run_cli(environment: dict[str, str], args: list[str]) -> int | None:
    """Run one CLI command with a clean environment and capture the exit code."""
    cli = get_command(app)
    previous_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with patch.dict(os.environ, environment, clear=True):
            try:
                cli.main(args=args, standalone_mode=False)
            except SystemExit as exc:
                return exc.code
    finally:
        logging.disable(previous_disable)
    return None


def _get_live_hyper_ai_reality() -> tuple[str, Decimal, Decimal, Decimal]:
    """Read the current HyperAI Safe balance, vault equity, and share supply."""
    web3 = Web3(Web3.HTTPProvider(HYPER_EVM_RPC, request_kwargs={"timeout": 15}))

    lagoon_vault = web3.eth.contract(
        address=Web3.to_checksum_address(HYPER_AI_LAGOON_VAULT),
        abi=[
            {
                "inputs": [],
                "name": "safe",
                "outputs": [{"internalType": "address", "name": "", "type": "address"}],
                "stateMutability": "view",
                "type": "function",
            },
            {
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function",
            },
        ],
    )
    safe_address = lagoon_vault.functions.safe().call()
    total_supply = Decimal(lagoon_vault.functions.totalSupply().call()) / Decimal(10**18)

    usdc = web3.eth.contract(
        address=Web3.to_checksum_address(USDC_ADDRESS),
        abi=[
            {
                "constant": True,
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "type": "function",
            },
        ],
    )
    safe_usdc_balance = Decimal(usdc.functions.balanceOf(safe_address).call()) / Decimal(10**6)

    session = create_hyperliquid_session(api_url=HYPERLIQUID_API_URL)
    vault_equity = fetch_user_vault_equity(
        session,
        user=safe_address,
        vault_address=HYPER_AI_VAULT_POSITION,
        bypass_cache=True,
    )
    assert vault_equity is not None

    return safe_address, safe_usdc_balance, vault_equity.equity, total_supply


def _revalue_and_refresh_statistics(state: State, environment: dict[str, str]) -> None:
    """Revalue the repaired state and append one fresh portfolio statistics sample."""
    mod = read_strategy_module(Path(environment["STRATEGY_FILE"]))

    web3config = create_web3_config(
        json_rpc_binance=None,
        json_rpc_polygon=None,
        json_rpc_avalanche=None,
        json_rpc_ethereum=None,
        json_rpc_arbitrum=None,
        json_rpc_base=None,
        json_rpc_anvil=None,
        json_rpc_derive=None,
        json_rpc_arbitrum_sepolia=None,
        json_rpc_base_sepolia=None,
        json_rpc_hyperliquid=environment["JSON_RPC_HYPERLIQUID"],
        json_rpc_hyperliquid_testnet=None,
        json_rpc_monad=None,
        unit_testing=True,
    )

    try:
        web3config.set_default_chain(mod.get_default_chain_id())
        web3config.choose_single_chain()

        execution_context = ExecutionContext(
            mode=ExecutionMode.one_off,
            timed_task_context_manager=timed_task,
            engine_version=mod.trading_strategy_engine_version,
        )

        client, routing_model = create_client(
            mod=mod,
            web3config=web3config,
            trading_strategy_api_key=environment["TRADING_STRATEGY_API_KEY"],
            cache_path=None,
            test_evm_uniswap_v2_factory=None,
            test_evm_uniswap_v2_router=None,
            test_evm_uniswap_v2_init_code_hash=None,
            clear_caches=False,
            asset_management_mode=AssetManagementMode.lagoon,
        )
        assert client is not None

        execution_model, sync_model, valuation_model_factory, pricing_model_factory = create_execution_and_sync_model(
            asset_management_mode=AssetManagementMode.lagoon,
            private_key=environment["PRIVATE_KEY"],
            web3config=web3config,
            confirmation_timeout=datetime.timedelta(seconds=60),
            confirmation_block_count=0,
            min_gas_balance=Decimal(0),
            max_slippage=0.006,
            vault_address=environment["VAULT_ADDRESS"],
            vault_adapter_address=environment["VAULT_ADAPTER_ADDRESS"],
            vault_payment_forwarder_address=None,
            routing_hint=mod.trade_routing,
        )

        factory = make_factory_from_strategy_mod(mod)
        run_description: StrategyExecutionDescription = factory(
            execution_model=execution_model,
            execution_context=execution_context,
            timed_task_context_manager=execution_context.timed_task_context_manager,
            sync_model=sync_model,
            valuation_model_factory=valuation_model_factory,
            pricing_model_factory=pricing_model_factory,
            approval_model=UncheckedApprovalModel(),
            client=client,
            routing_model=routing_model,
            run_state=RunState(),
        )

        universe_model: TradingStrategyUniverseModel = run_description.universe_model
        universe = universe_model.construct_universe(
            native_datetime_utc_now(),
            execution_context.mode,
            UniverseOptions(history_period=mod.get_live_trading_history_period()),
            execution_model=run_description.runner.execution_model,
            strategy_parameters=mod.parameters,
        )

        runner = run_description.runner
        _, _, valuation_method = runner.setup_routing(universe)

        # 1. Revalue the repaired position state from live Hyperliquid equity.
        revalue_state(state, native_datetime_utc_now(), valuation_method)

        # 2. Refresh portfolio statistics so share price is recalculated from truth.
        update_statistics(
            native_datetime_utc_now(),
            state.stats,
            state.portfolio,
            ExecutionMode.unit_testing_trading,
            treasury=state.sync.treasury,
        )
    finally:
        web3config.close()


def test_hyper_ai_repair_and_correct_accounts_repair_live_state(
    environment: dict[str, str],
    state_file: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test HyperAI repair and correct-accounts restore the state to live reality.

    1. Load the committed broken state fixture and verify it contains the frozen vault position.
    2. Run `repair` and verify the failed close trade is repaired and the position is unfrozen.
    3. Run `correct-accounts` and verify the reserve and Hyperliquid vault quantity match live reality.
    4. Revalue the corrected state and refresh statistics so share price returns to the truthful live level.
    5. Run `repair`, `correct-accounts`, and `check-accounts` again to verify the repaired state stays clean.
    """
    # 1. Load the committed broken state fixture and verify it contains the frozen vault position.
    broken_state = State.read_json_file(state_file)
    assert len(broken_state.portfolio.open_positions) == 0
    assert len(broken_state.portfolio.frozen_positions) == 1
    assert len(broken_state.portfolio.closed_positions) == 0
    assert broken_state.portfolio.get_default_reserve_position().quantity == Decimal("1")
    assert broken_state.portfolio.frozen_positions[1].trades[2].is_failed()
    assert broken_state.stats.portfolio[-1].share_price_usd == pytest.approx(0.1)
    assert broken_state.portfolio.get_net_asset_value() > broken_state.portfolio.calculate_total_equity()

    # 2. Run `repair` and verify the failed close trade is repaired and the position is unfrozen.
    repair_exit_code = _run_cli(environment, ["repair", "--auto-approve"])
    assert repair_exit_code is None

    repaired_state = State.read_json_file(state_file)
    assert len(repaired_state.portfolio.open_positions) == 1
    assert len(repaired_state.portfolio.frozen_positions) == 0
    repaired_position = repaired_state.portfolio.open_positions[1]
    assert repaired_position.trades[2].is_repaired()
    assert any(trade.is_repair_trade() for trade in repaired_position.trades.values())
    assert len(repaired_position.trades) == 3

    # 3. Run `correct-accounts` and verify the reserve and Hyperliquid vault quantity match live reality.
    correct_accounts_exit_code = _run_cli(environment, ["correct-accounts"])
    assert correct_accounts_exit_code == 0

    corrected_state = State.read_json_file(state_file)
    safe_address, safe_usdc_balance, live_vault_equity, total_supply = _get_live_hyper_ai_reality()
    assert safe_address == "0xB136581dFB3efA76Ae71293C1A70942f0726E8fD"
    assert len(corrected_state.portfolio.open_positions) == 1
    assert len(corrected_state.portfolio.frozen_positions) == 0
    assert len(corrected_state.sync.accounting.balance_update_refs) >= 1

    corrected_position = corrected_state.portfolio.open_positions[1]
    corrected_reserve = corrected_state.portfolio.get_default_reserve_position()
    assert float(corrected_reserve.quantity) == pytest.approx(float(safe_usdc_balance), abs=0.000001)
    assert float(corrected_position.get_quantity()) == pytest.approx(float(live_vault_equity), rel=0.03, abs=0.05)

    # 4. Revalue the corrected state and refresh statistics so share price returns to the truthful live level.
    _revalue_and_refresh_statistics(corrected_state, environment)

    expected_total_equity = float(safe_usdc_balance + live_vault_equity)
    expected_share_price = float((safe_usdc_balance + live_vault_equity) / total_supply)
    latest_stats = corrected_state.stats.portfolio[-1]

    assert corrected_state.portfolio.calculate_total_equity() == pytest.approx(expected_total_equity, rel=0.03, abs=0.05)
    assert corrected_state.portfolio.get_net_asset_value() == pytest.approx(expected_total_equity, rel=0.03, abs=0.05)
    assert latest_stats.share_price_usd == pytest.approx(expected_share_price, rel=0.03, abs=0.02)
    assert latest_stats.share_price_usd > 0.7
    assert latest_stats.share_price_usd < 0.9
    assert latest_stats.share_price_usd > broken_state.stats.portfolio[-1].share_price_usd

    first_accounting_ref_count = len(corrected_state.sync.accounting.balance_update_refs)
    first_trade_count = len(corrected_position.trades)
    first_next_trade_id = corrected_state.portfolio.next_trade_id
    first_next_balance_update_id = corrected_state.portfolio.next_balance_update_id
    first_corrected_quantity = corrected_position.get_quantity()

    # 5. Run `repair`, `correct-accounts`, and `check-accounts` again to verify the repaired state stays clean.
    second_repair_exit_code = _run_cli(environment, ["repair", "--auto-approve"])
    assert second_repair_exit_code is None

    after_second_repair = State.read_json_file(state_file)
    assert len(after_second_repair.portfolio.frozen_positions) == 0
    assert len(after_second_repair.sync.accounting.balance_update_refs) == first_accounting_ref_count
    assert after_second_repair.portfolio.next_trade_id == first_next_trade_id

    # 5. Freeze the vault-equity snapshot for the second correction pass so the
    # idempotence assertion measures state repair, not normal live vault PnL drift.
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_vault.create_hypercore_vault_value_func",
        lambda *args, **kwargs: (
            lambda pair: first_corrected_quantity
            if pair.pool_address.lower() == HYPER_AI_VAULT_POSITION
            else Decimal(0)
        ),
    )
    second_correct_accounts_exit_code = _run_cli(environment, ["correct-accounts"])
    assert second_correct_accounts_exit_code == 0

    final_state = State.read_json_file(state_file)
    final_position = final_state.portfolio.open_positions[1]
    assert len(final_state.portfolio.frozen_positions) == 0
    assert len(final_state.sync.accounting.balance_update_refs) == first_accounting_ref_count
    assert final_state.portfolio.next_balance_update_id == first_next_balance_update_id
    assert final_state.portfolio.next_trade_id == first_next_trade_id
    assert len(final_position.trades) == first_trade_count
    assert final_position.get_quantity() == first_corrected_quantity

    final_check_accounts_exit_code = _run_cli(environment, ["check-accounts"])
    assert final_check_accounts_exit_code == 0
