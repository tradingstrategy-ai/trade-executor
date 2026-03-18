"""Hyper-ai Hypercore replay integration tests."""

import datetime
from decimal import Decimal
from types import ModuleType
from typing import Protocol
from unittest import mock

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.trace import assert_transaction_success_with_explanation
from web3 import Web3

from tradeexecutor.cli.commands import start as start_command
from tradeexecutor.cli.main import app
from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeType
from tradeexecutor.strategy.execution_context import (ExecutionContext,
                                                      ExecutionMode)
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.pandas_trader.position_manager import \
    PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import \
    TradingStrategyUniverse


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

    def _unexpected_wait(*args, **kwargs):
        raise AssertionError("Live Hypercore wait logic must be short-circuited in Anvil simulate tests")

    # Keep quarantine disabled so this test stays focused on execution, valuation and accounting flow.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.activate_account", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation", _unexpected_wait)
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting._wait_for_usdc_arrival",
        _unexpected_wait,
    )

    parameters = StrategyParameters.from_class(hyper_ai_strategy_module.Parameters)
    parameters.initial_cash = 100
    parameters.max_assets_in_portfolio = 1
    parameters.allocation = 0.50
    parameters.max_concentration = 1.0
    parameters.per_position_cap_of_pool = 1.0
    parameters.individual_rebalance_min_threshold_usd = 5.0
    parameters.sell_rebalance_min_threshold = 1.0
    parameters.min_portfolio_weight = 0.0

    execution_context = ExecutionContext(
        mode=ExecutionMode.unit_testing,
        parameters=parameters,
    )

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )

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

    assert open_trade.is_success()
    assert len(open_trade.blockchain_transactions) == 1
    open_tx = open_trade.blockchain_transactions[0]
    assert open_tx.function_selector == "multicall"
    assert open_tx.tx_hash is not None
    assert open_tx.signed_bytes is not None
    assert open_tx.signed_tx_object is not None
    assert "Hypercore deposit (simulate)" in (open_tx.notes or "")

    # 2. Revalue the position and verify the stored BlockchainTransaction data for the Hypercore multicalls.
    position = next(iter(state.portfolio.open_positions.values()))
    valuation_ts = datetime.datetime(2026, 2, 3)
    valuation_update = hypercore_valuation_model(valuation_ts, position)
    assert valuation_update.new_price > 1.0
    assert valuation_update.new_value == pytest.approx(float(position.get_quantity()) * position.last_token_price)

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

    assert close_trade.is_success()
    assert len(close_trade.blockchain_transactions) == 1
    close_tx = close_trade.blockchain_transactions[0]
    assert close_tx.function_selector == "multicall"
    assert close_tx.tx_hash is not None
    assert close_tx.signed_bytes is not None
    assert close_tx.signed_tx_object is not None
    assert "Hypercore withdrawal" in (close_tx.notes or "")

    state_trades = list(state.portfolio.get_all_trades())
    assert len(state_trades) == 2
    for trade in state_trades:
        assert trade.blockchain_transactions, f"Trade {trade.trade_id} has no BlockchainTransaction objects"
        for tx in trade.blockchain_transactions:
            assert tx.chain_id == 999
            assert tx.function_selector == "multicall"
            assert tx.tx_hash is not None
            assert tx.signed_bytes is not None
            assert tx.signed_tx_object is not None

    # 3. Finalise a Lagoon deposit, request a redemption and reconcile treasury state at the end.
    tx_hash = vault.finalise_deposit(depositor).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    shares_to_redeem_raw = vault.share_token.convert_to_raw(Decimal("100"))
    tx_hash = vault.request_redeem(depositor, shares_to_redeem_raw).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

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


@pytest.mark.timeout(300)
def test_hypercore_cli_start_single_cycle(
    monkeypatch: pytest.MonkeyPatch,
    hypercore_cli_environment: dict[str, str],
    hypercore_replay_source,
) -> None:
    """Exercise Hypercore writer through the CLI start path.

    1. Patch the CLI execution bootstrap so the CLI test injects replay-backed Hypercore market data into Lagoon execution.
    2. Run one CLI start cycle against the HyperEVM Anvil fork using the pre-seeded Lagoon state file.
    3. Load the resulting state file and verify one successful Hypercore multicall trade was stored.
    """
    # 1. Patch the CLI execution bootstrap so the CLI test injects replay-backed Hypercore market data into Lagoon execution.
    # We mock execution-model construction here because the production CLI has no environment flag yet for
    # injecting the replay data source, and this test needs deterministic Hypercore valuation on Anvil.
    original_create_execution_and_sync_model = start_command.create_execution_and_sync_model

    def _patched_create_execution_and_sync_model(*args, **kwargs):
        execution_model, sync_model, valuation_model_factory, pricing_model_factory = original_create_execution_and_sync_model(*args, **kwargs)
        if isinstance(execution_model, LagoonExecution):
            execution_model.hypercore_market_data_source = hypercore_replay_source
        return execution_model, sync_model, valuation_model_factory, pricing_model_factory

    monkeypatch.setattr(
        start_command,
        "create_execution_and_sync_model",
        _patched_create_execution_and_sync_model,
    )
    # We also mock the live Hypercore waits to prove the CLI start path uses the Anvil shortcut logic.
    _install_wait_failures(monkeypatch)

    # 2. Run one CLI start cycle against the HyperEVM Anvil fork using the pre-seeded Lagoon state file.
    with mock.patch.dict("os.environ", hypercore_cli_environment, clear=True):
        app(["start"], standalone_mode=False)

    # 3. Load the resulting state file and verify one successful Hypercore multicall trade was stored.
    state = State.read_json_file(hypercore_cli_environment["STATE_FILE"])
    assert len(state.portfolio.open_positions) == 1

    trades = list(state.portfolio.get_all_trades())
    assert len(trades) == 1
    trade = trades[0]
    assert trade.is_success()
    assert len(trade.blockchain_transactions) == 1
    tx = trade.blockchain_transactions[0]
    assert tx.chain_id == 999
    assert tx.function_selector == "multicall"
    assert tx.tx_hash is not None
    assert "Hypercore deposit (simulate)" in (tx.notes or "")


def _make_test_input(
    *,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators: IndicatorFactory,
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model: GenericPricing,
    routing_model: GenericRouting,
    routing_state: object,
    pair: TradingPairIdentifier,
    timestamp: datetime.datetime,
    include_pair: bool,
) -> StrategyInput:
    parameters = StrategyParameters.from_class(hyper_ai_strategy_module.Parameters)
    parameters.initial_cash = 100
    parameters.max_assets_in_portfolio = 1
    parameters.allocation = 0.50
    parameters.max_concentration = 1.0
    parameters.per_position_cap_of_pool = 1.0
    parameters.individual_rebalance_min_threshold_usd = 5.0
    parameters.sell_rebalance_min_threshold = 1.0
    parameters.min_portfolio_weight = 0.0

    execution_context = ExecutionContext(
        mode=ExecutionMode.unit_testing,
        parameters=parameters,
    )

    indicator_values = {
        ("tvl_included_pair_count", None): 1 if include_pair else 0,
        ("inclusion_criteria", None): [pair.internal_id] if include_pair else [],
    }
    if include_pair:
        indicator_values[("age_ramp_weight", pair.internal_id)] = 1.0

    return StrategyInput(
        cycle=1,
        timestamp=timestamp,
        state=state,
        strategy_universe=strategy_universe,
        parameters=parameters,
        indicators=make_fake_indicators(indicator_values),
        pricing_model=pricing_model,
        execution_context=execution_context,
        other_data={},
        routing_model=routing_model,
        routing_state=routing_state,
    )


def _install_wait_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def _unexpected_wait(*args, **kwargs):
        raise AssertionError("Live Hypercore wait logic must be short-circuited in Anvil simulate tests")

    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.activate_account", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation", _unexpected_wait)
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting._wait_for_usdc_arrival",
        _unexpected_wait,
    )


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
    _install_wait_failures(monkeypatch)
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

    open_ts = datetime.datetime(2026, 1, 21)
    open_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        include_pair=True,
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

    assert trade.is_success()
    assert len(trade.blockchain_transactions) == 1
    tx = trade.blockchain_transactions[0]
    assert tx.chain_id == 999
    assert tx.function_selector == "multicall"
    assert tx.tx_hash is not None
    assert tx.signed_bytes is not None
    assert tx.signed_tx_object is not None
    assert "Hypercore deposit (simulate)" in (tx.notes or "")
    assert len(state.portfolio.open_positions) == 1


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
    _install_wait_failures(monkeypatch)
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

    open_ts = datetime.datetime(2026, 1, 21)
    open_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        include_pair=True,
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

    close_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=valuation_ts,
        include_pair=False,
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

    assert close_trade.is_success()
    assert len(close_trade.blockchain_transactions) == 1
    tx = close_trade.blockchain_transactions[0]
    assert tx.chain_id == 999
    assert tx.function_selector == "multicall"
    assert tx.tx_hash is not None
    assert tx.signed_bytes is not None
    assert tx.signed_tx_object is not None
    assert "Hypercore withdrawal" in (tx.notes or "")
    assert all(t.blockchain_transactions for t in state.portfolio.get_all_trades())
    assert not any(position.is_frozen() for position in state.portfolio.get_all_positions())


def test_hyper_ai_hypercore_multi_trade_same_cycle(
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
    """Exercise multiple Hypercore trades executed in the same strategy cycle.

    1. Open one Hypercore position so the later same-cycle trades have a live position to adjust.
    2. Create one increase trade and one decrease trade for the same cycle directly in state.
    3. Execute both trades together and verify both Hypercore multicalls are stored and the position remains open.
    """
    # 1. Open one Hypercore position so the later same-cycle trades have a live position to adjust.
    # We mock the live Hypercore wait helpers because Anvil simulate mode shortcuts the production delay logic.
    _install_wait_failures(monkeypatch)
    # Keep quarantine disabled so this test isolates the same-cycle Hypercore execution path.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = hypercore_state_with_safe_reserves
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )

    open_ts = datetime.datetime(2026, 1, 21)
    open_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        include_pair=True,
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

    position = next(iter(state.portfolio.open_positions.values()))
    initial_quantity = position.get_quantity()
    reserve_asset = hypercore_strategy_universe.get_reserve_asset()

    # 2. Create one increase trade and one decrease trade for the same cycle directly in state.
    same_cycle_ts = datetime.datetime(2026, 2, 4)
    _, increase_trade, _ = state.create_trade(
        strategy_cycle_at=same_cycle_ts,
        pair=pair,
        quantity=None,
        reserve=Decimal("30"),
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Same-cycle Hypercore increase",
        pair_fee=0.0,
        lp_fees_estimated=0,
        position=position,
    )
    _, decrease_trade, _ = state.create_trade(
        strategy_cycle_at=same_cycle_ts,
        pair=pair,
        quantity=Decimal("-20"),
        reserve=None,
        assumed_price=1.0,
        trade_type=TradeType.rebalance,
        reserve_currency=reserve_asset,
        reserve_currency_price=1.0,
        notes="Same-cycle Hypercore decrease",
        pair_fee=0.0,
        lp_fees_estimated=0,
        position=position,
    )

    assert increase_trade.is_buy()
    assert decrease_trade.is_sell()

    # 3. Execute both trades together and verify both Hypercore multicalls are stored and the position remains open.
    hypercore_execution_model.execute_trades(
        same_cycle_ts,
        state,
        [increase_trade, decrease_trade],
        hypercore_routing_model,
        routing_state,
        check_balances=False,
    )

    final_quantity = position.get_quantity()
    assert increase_trade.is_success()
    assert decrease_trade.is_success()
    assert len(increase_trade.blockchain_transactions) == 1
    assert len(decrease_trade.blockchain_transactions) == 1
    assert "Hypercore deposit (simulate)" in (increase_trade.blockchain_transactions[0].notes or "")
    assert "Hypercore withdrawal" in (decrease_trade.blockchain_transactions[0].notes or "")
    assert float(final_quantity) == pytest.approx(float(initial_quantity + Decimal("10")))
    assert len(state.portfolio.open_positions) == 1


def test_hyper_ai_hypercore_revaluation_in_three_steps(
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
    """Exercise a Hypercore position whose replay valuation rises across three timestamps.

    1. Open one Hypercore position at the earliest replay timestamp.
    2. Revalue the same open position at three replay timestamps in chronological order.
    3. Confirm each valuation step increases the position price and value.
    """
    # 1. Open one Hypercore position at the earliest replay timestamp.
    # We mock the live Hypercore wait helpers because Anvil simulate mode shortcuts the production delay logic.
    _install_wait_failures(monkeypatch)
    # Keep quarantine disabled so this test isolates the replay valuation path.
    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", lambda pool_address, timestamp: False)

    vault, state = hypercore_state_with_safe_reserves
    del vault
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    hypercore_execution_model.initialize()
    routing_state = hypercore_routing_model.create_routing_state(
        hypercore_strategy_universe,
        hypercore_execution_model.get_routing_state_details(),
    )

    open_ts = datetime.datetime(2026, 1, 7)
    open_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        include_pair=True,
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

    position = next(iter(state.portfolio.open_positions.values()))

    # 2. Revalue the same open position at three replay timestamps in chronological order.
    valuation_1 = hypercore_valuation_model(datetime.datetime(2026, 1, 7), position)
    valuation_2 = hypercore_valuation_model(datetime.datetime(2026, 1, 21), position)
    valuation_3 = hypercore_valuation_model(datetime.datetime(2026, 2, 3), position)

    # 3. Confirm each valuation step increases the position price and value.
    assert valuation_1.new_price > 1.0
    assert valuation_2.new_price > valuation_1.new_price
    assert valuation_3.new_price > valuation_2.new_price
    assert valuation_1.new_value < valuation_2.new_value
    assert valuation_2.new_value < valuation_3.new_value
    assert valuation_3.new_value == pytest.approx(float(position.get_quantity()) * valuation_3.new_price)


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
    _install_wait_failures(monkeypatch)
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

    open_ts = datetime.datetime(2026, 1, 21)
    open_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=open_ts,
        include_pair=True,
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
    assert len(increase_trade.blockchain_transactions) == 1
    assert "Hypercore deposit (simulate)" in (increase_trade.blockchain_transactions[0].notes or "")

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
    assert len(decrease_trade.blockchain_transactions) == 1
    assert "Hypercore withdrawal" in (decrease_trade.blockchain_transactions[0].notes or "")

    final_valuation = hypercore_valuation_model(datetime.datetime(2026, 2, 5), position)
    assert final_valuation.new_price > 1.0
    assert final_valuation.quantity == pytest.approx(decreased_quantity)
    assert len(state.portfolio.open_positions) == 1
    assert not any(position.is_frozen() for position in state.portfolio.get_all_positions())


def test_hyper_ai_lagoon_redeem_accounting(
    deposited_hypercore_vault_state: tuple[LagoonVault, State],
    depositor: str,
    web3_hyperevm: Web3,
    hypercore_sync_model: LagoonVaultSyncModel,
    hypercore_strategy_universe: TradingStrategyUniverse,
    hypercore_pricing_model: GenericPricing,
) -> None:
    """Exercise Lagoon redemption accounting without any Hypercore trade execution.

    1. Start from a Lagoon state that already contains a completed treasury deposit sync.
    2. Finalise the deposit, request a redemption, and settle Lagoon treasury again.
    3. Verify reserve cash drops by the redeemed amount and pending redemptions are cleared.
    """
    # 1. Start from a Lagoon state that already contains a completed treasury deposit sync.
    vault, state = deposited_hypercore_vault_state
    initial_cash = float(state.portfolio.get_default_reserve_position().quantity)
    redeem_amount = Decimal("150")

    # 2. Finalise the deposit, request a redemption, and settle Lagoon treasury again.
    tx_hash = vault.finalise_deposit(depositor).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    shares_to_redeem_raw = vault.share_token.convert_to_raw(redeem_amount)
    tx_hash = vault.request_redeem(depositor, shares_to_redeem_raw).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3_hyperevm, tx_hash)

    events = hypercore_sync_model.sync_treasury(
        datetime.datetime(2026, 2, 4),
        state,
        post_valuation=True,
    )

    position_manager = PositionManager(
        datetime.datetime(2026, 2, 4),
        universe=hypercore_strategy_universe,
        state=state,
        pricing_model=hypercore_pricing_model,
        default_slippage_tolerance=0.20,
    )

    # 3. Verify reserve cash drops by the redeemed amount and pending redemptions are cleared.
    assert len(events) == 1
    event = events[0]
    assert float(event.quantity) == pytest.approx(-float(redeem_amount), abs=1e-6)
    assert position_manager.get_pending_redemptions() == pytest.approx(0, abs=1e-6)
    assert float(position_manager.get_current_cash()) == pytest.approx(initial_cash - float(redeem_amount), abs=1e-6)
    assert state.sync.treasury.pending_redemptions == pytest.approx(0, abs=1e-6)
    assert len(list(state.portfolio.get_all_trades())) == 0
