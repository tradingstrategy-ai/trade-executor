"""Hyper-ai Hypercore replay integration tests."""

import datetime
from decimal import Decimal
from types import ModuleType
from typing import Protocol

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.token import TokenDetails
from web3 import Web3

from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.exchange_account.allocation import get_redeemable_capital
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.position_manager import \
    PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.redemption import (
    RedemptionBlockReason,
    RedemptionCheckResult,
    RedemptionCheckStage,
)
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverse
from tradeexecutor.testing.hypercore import (
    assert_hyper_ai_buy_cycle_reaches_target,
    assert_phased_withdrawal_trade,
    assert_hyper_ai_sell_cycle_reaches_target,
    assert_single_multicall_trade,
    create_hyper_ai_test_parameters,
    ensure_hypercore_routing_state,
    finalise_lagoon_deposit,
    get_lagoon_pending_redemptions_underlying,
    install_hypercore_live_withdrawal_mocks,
    install_hypercore_live_deposit_mocks,
    install_hypercore_wait_failures,
    make_hyper_ai_strategy_input,
    mirror_lagoon_safe_reserve_balance,
    request_lagoon_deposit,
    request_lagoon_redemption,
    request_lagoon_redemption_all,
    request_lagoon_redemption_fraction,
    run_hyper_ai_cycle,
    settle_lagoon_redemption_queue,
)


class IndicatorFactory(Protocol):
    def __call__(self, values: dict[tuple[str, int | None], object]) -> object:
        ...


@pytest.mark.timeout(300)
def test_hyper_ai_live_loop_hypercore_replay_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    deposited_hypercore_vault_state: tuple[LagoonVault, State],
    depositor: str,
    web3_hyperevm: Web3,
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_sync_model: LagoonVaultSyncModel,
    hypercore_valuation_model: GenericValuation,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Exercise the full Hypercore replay lifecycle on Anvil.

    1. Use the real Hyper-ai strategy to open and then close one replay-backed Hypercore position.
    2. Revalue the position and verify the stored BlockchainTransaction data for the Hypercore multicalls.
    3. Finalise a Lagoon deposit, request a redemption and reconcile treasury state at the end.
    """
    # 1. Use the real Hyper-ai strategy to open and then close one replay-backed Hypercore position.
    # We mock the live Hypercore wait helpers because Anvil simulate mode settles immediately and
    # the production polling path would only add brittle network waits to this test.
    vault, state = deposited_hypercore_vault_state
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    # Keep quarantine disabled so this test stays focused on execution, valuation and accounting flow.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)
    install_hypercore_wait_failures(monkeypatch)

    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=100.0,
        allocation=0.50,
    )

    execution_context = ExecutionContext(
        mode=ExecutionMode.unit_testing,
        parameters=parameters,
    )

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )
    reserve_asset = hypercore_strategy_universe.get_reserve_asset()

    open_ts = datetime.datetime(2026, 1, 21)
    open_input = StrategyInput(
        cycle=1,
        timestamp=open_ts,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        parameters=parameters,
        indicators=make_fake_indicators(
            {
                ("tvl_included_pair_count", None): 1,
                ("inclusion_criteria", None): [pair.internal_id],
                ("age_ramp_weight", pair.internal_id): 1.0,
            }
        ),
        pricing_model=hypercore_pricing_model,
        execution_context=execution_context,
        other_data={},
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
    )

    open_trades = hyper_ai_strategy_module.decide_trades(open_input)
    assert len(open_trades) == 1
    open_trade = open_trades[0]
    assert open_trade.is_buy()

    hypercore_execution_model.execute_trades(
        open_ts,
        state,
        open_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=True,
    )

    assert_single_multicall_trade(open_trade, note_substring="Hypercore deposit (simulate)")

    # 2. Revalue the position and verify the stored BlockchainTransaction data for the Hypercore multicalls.
    position = next(iter(state.portfolio.open_positions.values()))
    valuation_ts = datetime.datetime(2026, 2, 3)
    valuation_update = hypercore_valuation_model(valuation_ts, position)
    assert valuation_update.new_price > 1.0
    assert valuation_update.new_value == pytest.approx(float(position.get_quantity()) * position.last_token_price)

    phased_withdrawal_router = install_hypercore_live_withdrawal_mocks(
        monkeypatch,
        hypercore_routing_model,
        pair,
    )
    phased_withdrawal_router.simulate = False

    close_input = StrategyInput(
        cycle=2,
        timestamp=valuation_ts,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        parameters=parameters,
        indicators=make_fake_indicators(
            {
                ("tvl_included_pair_count", None): 0,
                ("inclusion_criteria", None): [],
            }
        ),
        pricing_model=hypercore_pricing_model,
        execution_context=execution_context,
        other_data={},
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
    )

    close_trades = hyper_ai_strategy_module.decide_trades(close_input)
    assert len(close_trades) == 1
    close_trade = close_trades[0]
    assert close_trade.is_sell()

    hypercore_execution_model.execute_trades(
        valuation_ts,
        state,
        close_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=True,
    )

    assert_phased_withdrawal_trade(close_trade)

    state_trades = list(state.portfolio.get_all_trades())
    assert len(state_trades) == 2
    for trade in state_trades:
        assert trade.blockchain_transactions, f"Trade {trade.trade_id} has no BlockchainTransaction objects"
        for tx in trade.blockchain_transactions:
            assert tx.chain_id == 999
            assert tx.tx_hash is not None
            assert tx.signed_bytes is not None
            assert tx.signed_tx_object is not None

    # 3. Finalise a Lagoon deposit, request a redemption and reconcile treasury state at the end.
    finalise_lagoon_deposit(web3_hyperevm, vault, depositor)
    request_lagoon_redemption(web3_hyperevm, vault, depositor, Decimal("100"))

    hypercore_sync_model.sync_treasury(
        datetime.datetime(2026, 2, 4),
        state,
        post_valuation=True,
    )

    final_position_manager = PositionManager(
        datetime.datetime(2026, 2, 4),
        universe=hypercore_strategy_universe,
        state=state,
        pricing_model=hypercore_pricing_model,
        default_slippage_tolerance=0.20,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
    )

    assert final_position_manager.get_pending_redemptions() == pytest.approx(0, abs=1e-6)
    assert all(trade.is_success() for trade in state.portfolio.get_all_trades())
    assert all(len(trade.blockchain_transactions) >= 1 for trade in state.portfolio.get_all_trades())
    assert not any(position.is_frozen() for position in state.portfolio.get_all_positions())

    json_blob = state.to_json_safe()
    restored_state = State.read_json_blob(json_blob)
    assert len(list(restored_state.portfolio.get_all_trades())) == len(list(state.portfolio.get_all_trades()))


@pytest.mark.skip(reason="TODO: add CLI Hypercore replay coverage after the loop-level harness is stable")
def test_hyper_ai_cli_hypercore_replay_todo():
    """Document the future CLI replay coverage.

    1. Reuse the same replay fixture and Hypercore mock module from the loop-level tests.
    2. Run the Hyper-ai strategy through the CLI entry point on a HyperEVM Anvil fork.
    3. Verify the CLI path produces the same open, close and accounting behaviour.
    """
    # 1. Reuse the same replay fixture and Hypercore mock module from the loop-level tests.
    # We will keep the same mock because Hypercore does not offer historical execution simulation.
    # 2. Run the Hyper-ai strategy through the CLI entry point on a HyperEVM Anvil fork.
    # 3. Verify the CLI path produces the same open, close and accounting behaviour.
    pass
@pytest.mark.timeout(300)
def test_hyper_ai_hypercore_open_cycle(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    hypercore_state_with_safe_reserves: tuple[LagoonVault, State],
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Exercise one isolated Hypercore open cycle.

    1. Prepare a Lagoon Safe with reserve USDC already mirrored into state.
    2. Run one Hyper-ai decision cycle that includes the replayed Hypercore vault.
    3. Execute the buy trade and verify the stored Hypercore multicall transaction data.
    """
    # 1. Prepare a Lagoon Safe with reserve USDC already mirrored into state.
    # We mock the live Hypercore wait helpers because simulate mode should not depend on real bridge delays.
    install_hypercore_wait_failures(monkeypatch)
    # Keep quarantine disabled so this test isolates the Hypercore open-path mechanics.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = hypercore_state_with_safe_reserves
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    # 2. Run one Hyper-ai decision cycle that includes the replayed Hypercore vault.
    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )

    open_ts = datetime.datetime(2026, 1, 21)
    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=100.0,
        allocation=0.50,
    )
    open_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        cycle=1,
        include_pair=True,
        parameters=parameters,
    )

    trades = hyper_ai_strategy_module.decide_trades(open_input)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.is_buy()

    # 3. Execute the buy trade and verify the stored Hypercore multicall transaction data.
    hypercore_execution_model.execute_trades(
        open_ts,
        state,
        trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    assert_single_multicall_trade(trade, note_substring="Hypercore deposit (simulate)")
    assert len(state.portfolio.open_positions) == 1


@pytest.mark.timeout(300)
def test_hyper_ai_live_cycle_persists_blocked_redemption_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    deposited_hypercore_vault_state: tuple[LagoonVault, State],
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_sync_model: LagoonVaultSyncModel,
    hypercore_valuation_model: GenericValuation,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Persist blocked redemption diagnostics only for live-style blocked cycles.

    1. Run one live-style open cycle before any lockup block is expected and verify no noisy calculations are stored.
    2. Run a second live-style decision cycle with a deterministic blocked redemption result and the vault excluded from inclusion criteria.
    3. Verify the blocked cycle stores compact calculations and still persists the latest alpha-model snapshot.
    """
    # 1. Run one live-style open cycle before any lockup block is expected and verify no noisy calculations are stored.
    install_hypercore_wait_failures(monkeypatch)
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = deposited_hypercore_vault_state
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)
    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=100.0,
        allocation=0.50,
    )

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )

    open_cycle = run_hyper_ai_cycle(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        cycle=1,
        timestamp=datetime.datetime(2026, 1, 21),
        include_pair=True,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        sync_model=hypercore_sync_model,
        execution_model=hypercore_execution_model,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        valuation_model=hypercore_valuation_model,
        pair=pair,
        parameters=parameters,
        execution_mode=ExecutionMode.unit_testing_trading,
    )

    assert len(open_cycle.trades) == 1
    assert state.visualisation.calculations == {}

    # 2. Run a second live-style cycle with a deterministic blocked redemption result and the vault excluded from inclusion criteria.
    monkeypatch.setattr(
        hyper_ai_strategy_module,
        "collect_blocked_redemption_results",
        lambda signals: [RedemptionCheckResult(
            timestamp=datetime.datetime(2026, 1, 28),
            stage=RedemptionCheckStage.carry_forward,
            can_redeem=False,
            reason_code=RedemptionBlockReason.user_lockup_not_expired,
            pair_ticker=pair.get_ticker(),
            vault_address=pair.pool_address,
            safe_address="0x0000000000000000000000000000000000000abc",
            position_recorded_lockup_expires_at=datetime.datetime(2026, 2, 1),
            user_lockup_expires_at=datetime.datetime(2026, 2, 1),
            message="Vault user lockup has not expired yet",
            max_redemption=Decimal(0),
        )],
    )
    monkeypatch.setattr(
        hyper_ai_strategy_module,
        "group_blocked_redemption_reasons",
        lambda blocked_results: {"user_lockup_not_expired": len(blocked_results)},
    )
    monkeypatch.setattr(
        hypercore_pricing_model,
        "check_redemption",
        lambda ts, checked_pair, **kwargs: RedemptionCheckResult(
            timestamp=ts,
            stage=kwargs["stage"],
            can_redeem=False,
            reason_code=RedemptionBlockReason.user_lockup_not_expired,
            pair_ticker=checked_pair.get_ticker(),
            vault_address=checked_pair.pool_address,
            safe_address="0x0000000000000000000000000000000000000abc",
            position_recorded_lockup_expires_at=datetime.datetime(2026, 2, 1),
            user_lockup_expires_at=datetime.datetime(2026, 2, 1),
            message="Hypercore user lockup has not expired yet",
            max_redemption=Decimal(0),
        ),
    )
    blocked_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=datetime.datetime(2026, 1, 28),
        cycle=2,
        include_pair=False,
        parameters=parameters,
        execution_mode=ExecutionMode.unit_testing_trading,
    )
    blocked_trades = hyper_ai_strategy_module.decide_trades(blocked_input)

    assert blocked_trades == []
    assert len(state.visualisation.calculations) == 1

    # 3. Verify the blocked cycle stores compact calculations and the latest alpha-model snapshot still persists.
    calculations = next(iter(state.visualisation.calculations.values()))
    assert calculations["blocked_signal_count"] == 1
    assert calculations["reason_counts"]["user_lockup_not_expired"] == 1
    assert calculations["blocked_redemptions"][0]["reason_code"] == "user_lockup_not_expired"
    assert "lockup" in calculations["blocked_redemptions"][0]["message"].lower()

    alpha_model = state.visualisation.discardable_data["alpha_model"]
    assert alpha_model is not None

    restored_state = State.read_json_blob(state.to_json_safe())
    restored_calculations = next(iter(restored_state.visualisation.calculations.values()))
    assert restored_calculations["blocked_redemptions"][0]["reason_code"] == "user_lockup_not_expired"
    restored_alpha_model = restored_state.visualisation.discardable_data["alpha_model"]
    assert restored_alpha_model is not None


@pytest.mark.timeout(300)
def test_hyper_ai_hypercore_close_cycle(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    hypercore_state_with_safe_reserves: tuple[LagoonVault, State],
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_valuation_model: GenericValuation,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Exercise one isolated Hypercore open and close path.

    1. Open one replay-backed Hypercore position from a funded Lagoon Safe.
    2. Revalue the position at a later replay timestamp and generate the closing decision.
    3. Execute the sell trade and verify the withdrawal multicall state is stored correctly.
    """
    # 1. Open one replay-backed Hypercore position from a funded Lagoon Safe.
    # We mock the live Hypercore wait helpers because Anvil simulate mode shortcuts the production delay logic.
    install_hypercore_wait_failures(monkeypatch)
    # Keep quarantine disabled so this test isolates the open-then-close execution path.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = hypercore_state_with_safe_reserves
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )

    open_ts = datetime.datetime(2026, 1, 21)
    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=100.0,
        allocation=0.50,
    )
    open_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        cycle=1,
        include_pair=True,
        parameters=parameters,
    )
    open_trades = hyper_ai_strategy_module.decide_trades(open_input)
    hypercore_execution_model.execute_trades(
        open_ts,
        state,
        open_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    # 2. Revalue the position at a later replay timestamp and generate the closing decision.
    position = next(iter(state.portfolio.open_positions.values()))
    valuation_ts = datetime.datetime(2026, 2, 3)
    valuation_update = hypercore_valuation_model(valuation_ts, position)
    assert valuation_update.new_price > 1.0

    phased_withdrawal_router = install_hypercore_live_withdrawal_mocks(
        monkeypatch,
        hypercore_routing_model,
        pair,
    )
    phased_withdrawal_router.simulate = False

    close_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=valuation_ts,
        cycle=2,
        include_pair=False,
        parameters=parameters,
    )
    close_trades = hyper_ai_strategy_module.decide_trades(close_input)
    assert len(close_trades) == 1
    close_trade = close_trades[0]
    assert close_trade.is_sell()

    # 3. Execute the sell trade and verify the withdrawal multicall state is stored correctly.
    hypercore_execution_model.execute_trades(
        valuation_ts,
        state,
        close_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    assert_phased_withdrawal_trade(close_trade)
    assert all(t.blockchain_transactions for t in state.portfolio.get_all_trades())
    assert not any(position.is_frozen() for position in state.portfolio.get_all_positions())


@pytest.mark.timeout(300)
def test_hyper_ai_hypercore_increase_then_decrease_position(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    hypercore_state_with_safe_reserves: tuple[LagoonVault, State],
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_valuation_model: GenericValuation,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Exercise a live-style increase and partial decrease on one Hypercore position.

    1. Open one Hypercore position through the existing split-scenario Anvil harness.
    2. Revalue the position, then increase it with an additional buy and verify quantity grows.
    3. Partially decrease the position with a sell, revalue again and verify the position stays open.
    """
    # 1. Open one Hypercore position through the existing split-scenario Anvil harness.
    # We mock the live Hypercore wait helpers because Anvil simulate mode shortcuts the production delay logic.
    install_hypercore_wait_failures(monkeypatch)
    # Keep quarantine disabled so this test isolates the increase/decrease execution path.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = hypercore_state_with_safe_reserves
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )

    open_ts = datetime.datetime(2026, 1, 21)
    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=100.0,
        allocation=0.50,
    )
    open_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        cycle=1,
        include_pair=True,
        parameters=parameters,
    )
    open_trades = hyper_ai_strategy_module.decide_trades(open_input)
    assert len(open_trades) == 1
    hypercore_execution_model.execute_trades(
        open_ts,
        state,
        open_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    position = next(iter(state.portfolio.open_positions.values()))
    open_quantity = position.get_quantity()
    assert open_quantity > 0

    # 2. Revalue the position, then increase it with an additional buy and verify quantity grows.
    first_valuation_ts = datetime.datetime(2026, 2, 3)
    first_valuation = hypercore_valuation_model(first_valuation_ts, position)
    assert first_valuation.new_price > 1.0

    increase_position_manager = PositionManager(
        datetime.datetime(2026, 2, 4),
        universe=hypercore_strategy_universe,
        state=state,
        pricing_model=hypercore_pricing_model,
        default_slippage_tolerance=0.20,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
    )
    increase_trades = increase_position_manager.adjust_position(
        pair=pair,
        dollar_delta=Decimal("50"),
        quantity_delta=Decimal("50"),
        weight=1.0,
    )
    assert len(increase_trades) == 1
    increase_trade = increase_trades[0]
    assert increase_trade.is_buy()

    hypercore_execution_model.execute_trades(
        datetime.datetime(2026, 2, 4),
        state,
        increase_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    increased_quantity = position.get_quantity()
    assert increase_trade.is_success()
    assert increased_quantity > open_quantity
    assert_single_multicall_trade(increase_trade, note_substring="Hypercore deposit (simulate)")

    second_valuation = hypercore_valuation_model(datetime.datetime(2026, 2, 4), position)
    assert second_valuation.new_value > first_valuation.new_value

    # 3. Partially decrease the position with a sell, revalue again and verify the position stays open.
    decrease_position_manager = PositionManager(
        datetime.datetime(2026, 2, 5),
        universe=hypercore_strategy_universe,
        state=state,
        pricing_model=hypercore_pricing_model,
        default_slippage_tolerance=0.20,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
    )
    decrease_trades = decrease_position_manager.adjust_position(
        pair=pair,
        dollar_delta=Decimal("-40"),
        quantity_delta=Decimal("-40"),
        weight=1.0,
    )
    assert len(decrease_trades) == 1
    decrease_trade = decrease_trades[0]
    assert decrease_trade.is_sell()

    phased_withdrawal_router = install_hypercore_live_withdrawal_mocks(
        monkeypatch,
        hypercore_routing_model,
        pair,
    )
    phased_withdrawal_router.simulate = False

    hypercore_execution_model.execute_trades(
        datetime.datetime(2026, 2, 5),
        state,
        decrease_trades,
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    decreased_quantity = position.get_quantity()
    assert decrease_trade.is_success()
    assert decreased_quantity < increased_quantity
    assert decreased_quantity > 0
    assert_phased_withdrawal_trade(decrease_trade)

    final_valuation = hypercore_valuation_model(datetime.datetime(2026, 2, 5), position)
    assert final_valuation.new_price > 1.0
    assert final_valuation.quantity == pytest.approx(decreased_quantity)
    assert len(state.portfolio.open_positions) == 1
    assert not any(position.is_frozen() for position in state.portfolio.get_all_positions())


@pytest.mark.timeout(300)
def test_hyper_ai_lagoon_redeem_accounting(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    deposited_hypercore_vault_state: tuple[LagoonVault, State],
    depositor: str,
    secondary_depositor: str,
    web3_hyperevm: Web3,
    hypercore_usdc_token: TokenDetails,
    hypercore_execution_model: LagoonExecution,
    hypercore_pricing_model: GenericPricing,
    hypercore_routing_model: GenericRouting,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_sync_model: LagoonVaultSyncModel,
    hypercore_valuation_model: GenericValuation,
    hypercore_vault_pair: TradingPairIdentifier,
) -> None:
    """Exercise Hyper AI redemption accounting across repeated live-style cycles.

    1. Finalise one initial deposit, open the first Hypercore position and record the deployable target.
    2. Queue a 50% redemption, run a rebalance cycle and verify pending redemptions reduce the target invested value.
    3. Queue a further 25% redemption, run another rebalance and verify the strategy sells down to the new target.
    4. Queue the remaining redemption, run another rebalance and verify the position fully exits.
    5. Add a new deposit from another account, run one more cycle and verify the strategy reinvests only the newly available capital.
    """
    # 1. Finalise one initial deposit, open the first Hypercore position and record the deployable target.
    install_hypercore_wait_failures(monkeypatch)
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = deposited_hypercore_vault_state
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)
    parameters = create_hyper_ai_test_parameters(
        hyper_ai_strategy_module,
        initial_cash=399.0,
        allocation=0.98,
    )

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )
    ensure_hypercore_routing_state(
        hypercore_routing_model,
        routing_state,
        pair,
    )
    install_hypercore_live_withdrawal_mocks(
        monkeypatch,
        hypercore_routing_model,
        pair,
    )
    install_hypercore_live_deposit_mocks(monkeypatch)
    reserve_asset = hypercore_strategy_universe.get_reserve_asset()
    cycle_specs = [
        {
            "cycle": 1,
            "timestamp": datetime.datetime(2026, 1, 21),
            "pre_action": lambda: finalise_lagoon_deposit(web3_hyperevm, vault, depositor),
            "fully_exit": False,
        },
        {
            "cycle": 2,
            "timestamp": datetime.datetime(2026, 2, 3),
            "pre_action": lambda: request_lagoon_redemption_fraction(
                web3_hyperevm,
                vault,
                depositor,
                Decimal("0.5"),
            ),
            "fully_exit": False,
        },
        {
            "cycle": 3,
            "timestamp": datetime.datetime(2026, 2, 10),
            "pre_action": lambda: request_lagoon_redemption_fraction(
                web3_hyperevm,
                vault,
                depositor,
                Decimal("0.5"),
            ),
            "fully_exit": False,
        },
        {
            "cycle": 4,
            "timestamp": datetime.datetime(2026, 2, 17),
            "pre_action": lambda: request_lagoon_redemption_all(
                web3_hyperevm,
                vault,
                depositor,
            ),
            "fully_exit": True,
        },
        {
            "cycle": 5,
            "timestamp": datetime.datetime(2026, 2, 24),
            "pre_action": lambda: request_lagoon_deposit(
                web3_hyperevm,
                vault,
                hypercore_usdc_token,
                secondary_depositor,
                Decimal("250"),
            ),
            "fully_exit": False,
        },
    ]

    cycle_results = []
    for spec in cycle_specs:
        current_router, _ = hypercore_routing_model.get_router(pair)
        current_router.simulate = False
        spec["pre_action"]()
        cycle_result = run_hyper_ai_cycle(
            hyper_ai_strategy_module=hyper_ai_strategy_module,
            make_fake_indicators=make_fake_indicators,
            cycle=spec["cycle"],
            timestamp=spec["timestamp"],
            include_pair=True,
            state=state,
            strategy_universe=hypercore_strategy_universe,
            sync_model=hypercore_sync_model,
            execution_model=hypercore_execution_model,
            pricing_model=hypercore_pricing_model,
            routing_model=hypercore_routing_model,
            routing_state=routing_state,
            valuation_model=hypercore_valuation_model,
            pair=pair,
            parameters=parameters,
        )

        target_delta = cycle_result.snapshot.deployable_equity - cycle_result.snapshot.open_position_value

        if target_delta > parameters.individual_rebalance_min_threshold_usd:
            trade = assert_hyper_ai_buy_cycle_reaches_target(cycle_result)
        else:
            trade = assert_hyper_ai_sell_cycle_reaches_target(
                cycle_result,
                fully_exit=spec["fully_exit"],
            )

        if trade.is_buy():
            assert len(trade.blockchain_transactions) == 3
            assert all(tx.tx_hash is not None for tx in trade.blockchain_transactions)
        else:
            assert_phased_withdrawal_trade(trade)
        mirror_lagoon_safe_reserve_balance(
            web3_hyperevm,
            vault,
            hypercore_usdc_token,
            state,
        )
        if trade.is_sell():
            settle_lagoon_redemption_queue(
                hypercore_sync_model,
                state,
                reserve_asset,
                spec["timestamp"],
            )
        cycle_results.append(cycle_result)

    cycle_1, cycle_2, cycle_3, cycle_4, cycle_5 = cycle_results

    initial_shares = vault.share_token.fetch_balance_of(depositor)
    assert initial_shares == pytest.approx(Decimal("0"))
    assert cycle_1.snapshot.pending_redemptions == pytest.approx(0.0, abs=1e-6)
    assert cycle_2.snapshot.pending_redemptions > 0
    assert cycle_5.snapshot.pending_redemptions == pytest.approx(0.0, abs=1e-6)
    assert cycle_2.snapshot.deployable_equity < cycle_1.snapshot.deployable_equity
    assert cycle_5.snapshot.open_position_value == pytest.approx(0.0, abs=1e-6)
    assert cycle_5.snapshot.deployable_equity > 0
    assert len(state.portfolio.open_positions) == 1
    assert get_lagoon_pending_redemptions_underlying(vault, web3_hyperevm.eth.block_number) == pytest.approx(Decimal("0"))
    assert get_redeemable_capital(
        next(iter(state.portfolio.open_positions.values())),
        timestamp=datetime.datetime(2026, 2, 24),
    ) == pytest.approx(0.0, abs=1e-6)
