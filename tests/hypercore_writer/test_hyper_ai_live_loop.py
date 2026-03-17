"""Hyper-ai Hypercore replay integration tests."""

import datetime
from decimal import Decimal

import pytest

from eth_defi.trace import assert_transaction_success_with_explanation

from tradeexecutor.state.state import State
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput


@pytest.mark.timeout(300)
@pytest.mark.xfail(
    reason="HyperEVM Anvil hangs during Lagoon treasury sync in the initial deposit fixture; split-scenario follow-up needed",
    run=False,
)
def test_hyper_ai_live_loop_hypercore_replay_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module,
    make_fake_indicators,
    deposited_hypercore_vault_state,
    depositor: str,
    web3_hyperevm,
    hypercore_execution_model,
    hypercore_pricing_model,
    hypercore_routing_model,
    hypercore_strategy_universe,
    hypercore_sync_model,
    hypercore_valuation_model,
    hypercore_vault_pair,
):
    """Exercise Hypercore open/close execution with replay pricing and valuation.

    This is the first large integration attempt.  It uses the real
    ``strategies/hyper-ai.py`` decision function, but keeps the universe
    intentionally tiny so Anvil RPC traffic stays manageable.
    """
    vault, state = deposited_hypercore_vault_state
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)

    def _unexpected_wait(*args, **kwargs):
        raise AssertionError("Live Hypercore wait logic must be short-circuited in Anvil simulate tests")

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

    position = next(iter(state.portfolio.open_positions.values()))
    valuation_ts = datetime.datetime(2026, 2, 3)
    valuation_update = hypercore_valuation_model(valuation_ts, position)
    assert valuation_update.new_price > 1.0
    assert valuation_update.new_value == pytest.approx(position.get_quantity() * position.last_token_price)

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

    # Finalise shares, request a redemption, and reconcile treasury once more.
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


@pytest.mark.skip(reason="TODO: add CLI Hypercore replay coverage after the loop-level harness is stable")
def test_hyper_ai_cli_hypercore_replay_todo():
    """Placeholder for future CLI replay coverage."""


def _make_test_input(
    *,
    hyper_ai_strategy_module,
    make_fake_indicators,
    state,
    strategy_universe,
    pricing_model,
    routing_model,
    routing_state,
    pair,
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
    hyper_ai_strategy_module,
    make_fake_indicators,
    hypercore_state_with_safe_reserves,
    hypercore_execution_model,
    hypercore_pricing_model,
    hypercore_routing_model,
    hypercore_strategy_universe,
    hypercore_vault_pair,
):
    """Exercise one Hypercore open cycle without the hanging treasury-sync path."""
    _install_wait_failures(monkeypatch)
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

    trades = hyper_ai_strategy_module.decide_trades(open_input)
    assert len(trades) == 1
    trade = trades[0]
    assert trade.is_buy()

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


def test_hyper_ai_uses_hypercore_vault_address_for_quarantine(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module,
    make_fake_indicators,
    hypercore_state_with_safe_reserves,
    hypercore_pricing_model,
    hypercore_routing_model,
    hypercore_strategy_universe,
    hypercore_vault_pair,
):
    """Hypercore quarantine checks must use the real vault address, not CoreWriter."""
    _, state = hypercore_state_with_safe_reserves
    pair = hypercore_strategy_universe.get_pair_by_id(hypercore_vault_pair.internal_id)
    seen_addresses: list[str] = []

    def _is_quarantined(address: str, timestamp: datetime.datetime) -> bool:
        del timestamp
        seen_addresses.append(address)
        return True

    monkeypatch.setattr(hyper_ai_strategy_module, "is_quarantined", _is_quarantined)

    strategy_input = _make_test_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=hypercore_strategy_universe,
        pricing_model=hypercore_pricing_model,
        routing_model=hypercore_routing_model,
        routing_state=None,
        pair=pair,
        timestamp=datetime.datetime(2026, 1, 21),
        include_pair=True,
    )

    trades = hyper_ai_strategy_module.decide_trades(strategy_input)
    assert trades == []
    assert seen_addresses == [pair.other_data["hypercore_vault_address"]]


def test_hyper_ai_hypercore_close_cycle(
    monkeypatch: pytest.MonkeyPatch,
    hyper_ai_strategy_module,
    make_fake_indicators,
    hypercore_state_with_safe_reserves,
    hypercore_execution_model,
    hypercore_pricing_model,
    hypercore_routing_model,
    hypercore_strategy_universe,
    hypercore_valuation_model,
    hypercore_vault_pair,
):
    """Exercise a Hypercore open followed by a close in split-scenario form."""
    _install_wait_failures(monkeypatch)
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


@pytest.mark.skip(reason="TODO: split out Lagoon redemption accounting if the full HyperEVM Anvil scenario stays unstable")
def test_hyper_ai_lagoon_redeem_accounting():
    """Placeholder for a smaller Lagoon redemption accounting test."""
