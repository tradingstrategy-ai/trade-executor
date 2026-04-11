"""Reusable Hypercore strategy test helpers."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from decimal import Decimal
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import pytest
from eth_defi.erc_4626.vault_protocol.lagoon.vault import LagoonVault
from eth_defi.provider.anvil import fund_erc20_on_anvil
from eth_defi.token import TokenDetails
from eth_defi.trace import assert_transaction_success_with_explanation
from web3 import Web3

from tradeexecutor.ethereum.lagoon.execution import LagoonExecution
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.vault.hypercore_routing import (
    HypercoreVaultRouting,
    raw_to_usdc,
)
from tradeexecutor.exchange_account.allocation import calculate_portfolio_target_value
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.generic.generic_pricing_model import GenericPricing
from tradeexecutor.strategy.generic.generic_router import GenericRouting, GenericRoutingState
from tradeexecutor.strategy.generic.generic_valuation import GenericValuation
from tradeexecutor.strategy.parameters import StrategyParameters
from tradeexecutor.strategy.pandas_trader.position_manager import PositionManager
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.strategy.valuation import revalue_state


@dataclass(frozen=True, slots=True)
class HyperAiCycleSnapshot:
    """One pre-trade Hyper AI strategy snapshot for assertions."""

    cycle: int
    timestamp: datetime.datetime
    total_equity: float
    current_cash: float
    open_position_value: float
    pending_redemptions: float
    deployable_equity: float


@dataclass(frozen=True, slots=True)
class HyperAiCycleResult:
    """One executed Hyper AI cycle with pre/post state snapshots."""

    snapshot: HyperAiCycleSnapshot
    trades: list[TradeExecution]
    post_trade_cash: float
    post_trade_open_position_value: float


def install_hypercore_wait_failures(monkeypatch: Any) -> None:
    """Fail fast if live Hypercore polling leaks into Anvil simulate tests."""

    def _unexpected_wait(*args, **kwargs):
        raise AssertionError("Live Hypercore wait logic must be short-circuited in Anvil simulate tests")

    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.activate_account", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear", _unexpected_wait)
    monkeypatch.setattr("tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation", _unexpected_wait)
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.HypercoreVaultRouting._wait_for_usdc_arrival",
        _unexpected_wait,
    )


def install_hypercore_live_withdrawal_mocks(
    monkeypatch: Any,
    routing_model: GenericRouting,
    pair: TradingPairIdentifier,
) -> HypercoreVaultRouting:
    """Switch one Hypercore router to phased withdrawal mode on Anvil.

    1. Resolve the Hypercore router for the tested vault pair and disable
       its simulate flag so sell trades take the live phased path.
    2. Replace Hyperliquid API reads with deterministic local values so the
       test does not depend on real off-chain state or bridge latency.
    3. Return the router so the caller can toggle ``simulate`` between buy
       and sell cycles within the same replay test.
    """

    # 1. Resolve the Hypercore router for the tested vault pair and disable
    # its simulate flag so sell trades take the live phased path.
    router, _ = routing_model.get_router(pair)
    assert isinstance(router, HypercoreVaultRouting)
    router.simulate = False

    # 2. Replace Hyperliquid API reads with deterministic local values so the
    # test does not depend on real off-chain state or bridge latency.
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.is_account_activated",
        lambda web3, user: True,
    )
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_abstraction_mode",
        lambda session, user: "default",
    )
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.fetch_user_vault_equity",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(router, "_fetch_safe_evm_usdc_balance", lambda: 0)
    monkeypatch.setattr(router, "_fetch_safe_perp_withdrawable_balance", lambda: Decimal("0"))
    monkeypatch.setattr(router, "_fetch_safe_spot_free_usdc_balance", lambda: Decimal("0"))
    monkeypatch.setattr(
        router,
        "_wait_for_perp_withdrawable_balance",
        lambda baseline_balance, expected_increase_raw, timeout=30.0, poll_interval=2.0: (
            baseline_balance + raw_to_usdc(expected_increase_raw)
        ),
    )
    monkeypatch.setattr(
        router,
        "_wait_for_spot_free_usdc_balance",
        lambda baseline_balance, expected_increase_raw, timeout=30.0, poll_interval=2.0: (
            baseline_balance + raw_to_usdc(expected_increase_raw)
        ),
    )
    monkeypatch.setattr(
        router,
        "_wait_for_usdc_arrival",
        lambda baseline_balance_raw, expected_increase_raw, timeout=30.0, poll_interval=2.0: expected_increase_raw,
    )

    # 3. Return the router so the caller can toggle ``simulate`` between buy
    # and sell cycles within the same replay test.
    return router


def install_hypercore_live_deposit_mocks(monkeypatch: Any) -> None:
    """Short-circuit live Hypercore deposit polling for deterministic Anvil tests.

    1. Replace the escrow-clear wait with a no-op because the mock Hypercore
       contracts settle immediately on the fork.
    2. Replace the vault-deposit confirmation poll with a deterministic
       synthetic equity response based on the expected deposit delta.
    """

    # 1. Replace the escrow-clear wait with a no-op because the mock Hypercore
    # contracts settle immediately on the fork.
    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.wait_for_evm_escrow_clear",
        lambda *args, **kwargs: None,
    )

    # 2. Replace the vault-deposit confirmation poll with a deterministic
    # synthetic equity response based on the expected deposit delta.
    def _mock_wait_for_vault_deposit_confirmation(
        session,
        user: str,
        vault_address: str,
        expected_deposit: Decimal,
        existing_equity: Decimal | None,
        timeout: float = 60.0,
        poll_interval: float = 2.0,
    ) -> SimpleNamespace:
        del session, user, vault_address, timeout, poll_interval
        return SimpleNamespace(equity=(existing_equity or Decimal("0")) + expected_deposit)

    monkeypatch.setattr(
        "tradeexecutor.ethereum.vault.hypercore_routing.wait_for_vault_deposit_confirmation",
        _mock_wait_for_vault_deposit_confirmation,
    )


def ensure_hypercore_routing_state(
    routing_model: GenericRouting,
    routing_state: GenericRoutingState,
    pair: TradingPairIdentifier,
) -> None:
    """Ensure one replay routing state contains the Hypercore router substate.

    Some Anvil replay fixtures only seed the generic ``vault`` substate up
    front. Hypercore trades still route through ``hypercore_vault``, so these
    tests need the corresponding router state injected before execution.
    """

    router, protocol_config = routing_model.get_router(pair)
    router_name = protocol_config.routing_id.router_name

    if router_name in routing_state.state_map:
        return

    existing_state = next(iter(routing_state.state_map.values()))
    routing_state.state_map[router_name] = router.create_routing_state(
        routing_state.strategy_universe,
        {"tx_builder": existing_state.tx_builder},
    )


def create_hyper_ai_test_parameters(
    hyper_ai_strategy_module: ModuleType,
    **overrides,
) -> StrategyParameters:
    """Create a Hyper AI parameter set tuned for deterministic integration tests."""

    parameters = StrategyParameters.from_class(hyper_ai_strategy_module.Parameters)
    parameters.initial_cash = 100.0
    parameters.max_assets_in_portfolio = 1
    parameters.allocation = 0.98
    parameters.max_concentration = 1.0
    parameters.per_position_cap_of_pool = 1.0
    parameters.individual_rebalance_min_threshold_usd = 5.0
    parameters.sell_rebalance_min_threshold = 1.0
    parameters.min_portfolio_weight = 0.0

    for key, value in overrides.items():
        setattr(parameters, key, value)

    return parameters


def make_hyper_ai_strategy_input(
    *,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators,
    state: State,
    strategy_universe: TradingStrategyUniverse,
    pricing_model: GenericPricing,
    routing_model: GenericRouting,
    routing_state: object,
    pair: TradingPairIdentifier,
    timestamp: datetime.datetime,
    cycle: int,
    include_pair: bool,
    parameters: StrategyParameters,
    execution_mode: ExecutionMode = ExecutionMode.unit_testing,
) -> StrategyInput:
    """Build one Hyper AI strategy input payload for a deterministic cycle."""

    execution_context = ExecutionContext(
        mode=execution_mode,
        parameters=parameters,
    )

    indicator_values = {
        ("tvl_included_pair_count", None): 1 if include_pair else 0,
        ("inclusion_criteria", None): [pair.internal_id] if include_pair else [],
    }
    if include_pair:
        indicator_values[("age_ramp_weight", pair.internal_id)] = 1.0

    return StrategyInput(
        cycle=cycle,
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


def prepare_hyper_ai_cycle_snapshot(
    *,
    hyper_ai_strategy_module: ModuleType,
    timestamp: datetime.datetime,
    cycle: int,
    state: State,
    strategy_universe: TradingStrategyUniverse,
    sync_model: LagoonVaultSyncModel,
    pricing_model: GenericPricing,
    valuation_model: GenericValuation,
    routing_model: GenericRouting,
    routing_state: object,
    parameters: StrategyParameters,
) -> HyperAiCycleSnapshot:
    """Revalue and sync treasury, then calculate deployable capital for one cycle."""

    revalue_state(state, timestamp, valuation_model)
    sync_model.sync_treasury(
        timestamp,
        state,
        supported_reserves=[strategy_universe.get_reserve_asset()],
        post_valuation=True,
    )

    position_manager = PositionManager(
        timestamp,
        universe=strategy_universe,
        state=state,
        pricing_model=pricing_model,
        default_slippage_tolerance=0.20,
        routing_model=routing_model,
        routing_state=routing_state,
    )
    portfolio = position_manager.get_current_portfolio()
    total_equity = float(portfolio.calculate_total_equity())
    current_cash = float(position_manager.get_current_cash())
    open_position_value = float(sum(position.get_value() for position in state.portfolio.open_positions.values()))
    pending_redemptions = float(position_manager.get_pending_redemptions())
    deployable_equity = float(
        calculate_portfolio_target_value(
            position_manager,
            float(parameters.allocation),
        )
    )

    return HyperAiCycleSnapshot(
        cycle=cycle,
        timestamp=timestamp,
        total_equity=total_equity,
        current_cash=current_cash,
        open_position_value=open_position_value,
        pending_redemptions=pending_redemptions,
        deployable_equity=deployable_equity,
    )


def run_hyper_ai_cycle(
    *,
    hyper_ai_strategy_module: ModuleType,
    make_fake_indicators,
    cycle: int,
    timestamp: datetime.datetime,
    include_pair: bool,
    state: State,
    strategy_universe: TradingStrategyUniverse,
    sync_model: LagoonVaultSyncModel,
    execution_model: LagoonExecution,
    pricing_model: GenericPricing,
    routing_model: GenericRouting,
    routing_state: object,
    valuation_model: GenericValuation,
    pair: TradingPairIdentifier,
    parameters: StrategyParameters,
    execution_mode: ExecutionMode = ExecutionMode.unit_testing,
) -> HyperAiCycleResult:
    """Run one revalue + treasury-sync + Hyper AI rebalance cycle."""

    snapshot = prepare_hyper_ai_cycle_snapshot(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        timestamp=timestamp,
        cycle=cycle,
        state=state,
        strategy_universe=strategy_universe,
        sync_model=sync_model,
        pricing_model=pricing_model,
        valuation_model=valuation_model,
        routing_model=routing_model,
        routing_state=routing_state,
        parameters=parameters,
    )
    strategy_input = make_hyper_ai_strategy_input(
        hyper_ai_strategy_module=hyper_ai_strategy_module,
        make_fake_indicators=make_fake_indicators,
        state=state,
        strategy_universe=strategy_universe,
        pricing_model=pricing_model,
        routing_model=routing_model,
        routing_state=routing_state,
        pair=pair,
        timestamp=timestamp,
        cycle=cycle,
        include_pair=include_pair,
        parameters=parameters,
        execution_mode=execution_mode,
    )
    trades = hyper_ai_strategy_module.decide_trades(strategy_input)

    if trades:
        execution_model.execute_trades(
            timestamp,
            state,
            trades,
            routing_model,
            routing_state,
            check_balances=False,
        )

        # Hypercore simulate-mode execution settles against the live fork block
        # timestamp, but these replay tests move through historical strategy
        # cycle timestamps. Normalise the executed timestamp back to the replay
        # cycle so redemption-delay accounting follows the intended test
        # timeline instead of the wall-clock Anvil fork time.
        for trade in trades:
            if trade.executed_at is not None:
                trade.executed_at = timestamp

        revalue_state(state, timestamp, valuation_model)

    post_trade_open_position_value = float(sum(position.get_value() for position in state.portfolio.open_positions.values()))
    post_trade_cash = float(state.portfolio.get_cash())

    return HyperAiCycleResult(
        snapshot=snapshot,
        trades=trades,
        post_trade_cash=post_trade_cash,
        post_trade_open_position_value=post_trade_open_position_value,
    )


def assert_single_multicall_trade(
    trade: TradeExecution,
    *,
    note_substring: str,
    chain_id: int = 999,
) -> None:
    """Assert that one Hypercore trade stored the expected multicall metadata."""

    assert trade.is_success()
    assert len(trade.blockchain_transactions) == 1
    tx = trade.blockchain_transactions[0]
    assert tx.chain_id == chain_id
    assert tx.function_selector == "multicall"
    assert tx.tx_hash is not None
    assert tx.signed_bytes is not None
    assert tx.signed_tx_object is not None
    assert note_substring in (tx.notes or "")


def assert_phased_withdrawal_trade(
    trade: TradeExecution,
    *,
    chain_id: int = 999,
) -> None:
    """Assert that one Hypercore sell stored all three phased withdrawal txs."""

    assert trade.is_success()
    assert len(trade.blockchain_transactions) == 3

    expected_notes = (
        "Hypercore withdrawal phase 1",
        "Hypercore withdrawal phase 2",
        "Hypercore withdrawal phase 3",
    )

    for tx, note_substring in zip(trade.blockchain_transactions, expected_notes, strict=True):
        assert tx.chain_id == chain_id
        assert tx.function_selector == "performCall"
        assert tx.details["function"] == "sendRawAction"
        assert tx.tx_hash is not None
        assert tx.signed_bytes is not None
        assert tx.signed_tx_object is not None
        assert note_substring in (tx.notes or "")


def assert_hyper_ai_buy_cycle_reaches_target(
    cycle_result: HyperAiCycleResult,
    *,
    trade_tolerance: float = 2.0,
    cash_tolerance: float = 2.0,
) -> TradeExecution:
    """Assert that one Hyper AI buy cycle deployed the expected capital.

    Hypercore positions are marked to the replayed vault share value straight
    after execution, so we validate the deployed cash amount and the remaining
    reserve cash instead of assuming the marked position value equals the
    reserve capital just invested.
    """

    assert len(cycle_result.trades) == 1, cycle_result
    trade = cycle_result.trades[0]
    assert trade.is_buy(), trade
    expected_buy_value = cycle_result.snapshot.deployable_equity - cycle_result.snapshot.open_position_value
    assert trade.get_planned_value() == pytest.approx(expected_buy_value, abs=trade_tolerance)
    assert cycle_result.post_trade_cash == pytest.approx(
        max(cycle_result.snapshot.current_cash - trade.get_planned_value(), 0.0),
        abs=cash_tolerance,
    )
    assert cycle_result.post_trade_open_position_value > cycle_result.snapshot.open_position_value
    return trade


def assert_hyper_ai_sell_cycle_reaches_target(
    cycle_result: HyperAiCycleResult,
    *,
    fully_exit: bool = False,
    trade_tolerance: float = 2.0,
    position_tolerance: float = 5.0,
) -> TradeExecution:
    """Assert that one Hyper AI sell cycle reduced the position to its target.

    When the strategy keeps the position open, the post-trade marked value
    should land near the computed deployable target. When the target is zero,
    the cycle should fully exit the position.
    """

    assert len(cycle_result.trades) == 1, cycle_result
    trade = cycle_result.trades[0]
    assert trade.is_sell(), trade

    expected_sell_value = cycle_result.snapshot.open_position_value - cycle_result.snapshot.deployable_equity
    assert trade.get_planned_value() == pytest.approx(expected_sell_value, abs=trade_tolerance)

    if fully_exit:
        assert cycle_result.post_trade_open_position_value == pytest.approx(0.0, abs=1e-6)
    else:
        assert cycle_result.post_trade_open_position_value < cycle_result.snapshot.open_position_value
        assert cycle_result.post_trade_open_position_value <= cycle_result.snapshot.deployable_equity + position_tolerance, (
            cycle_result.post_trade_open_position_value,
            cycle_result.snapshot.deployable_equity,
            cycle_result.snapshot.open_position_value,
            trade.get_planned_value(),
        )

    return trade


def request_lagoon_deposit(
    web3: Web3,
    vault: LagoonVault,
    token: TokenDetails,
    depositor: str,
    amount: Decimal,
) -> None:
    """Open one Lagoon deposit request and assert that the tx succeeded."""

    tx_hash = token.approve(vault.address, amount).transact({"from": depositor})
    assert_transaction_success_with_explanation(web3, tx_hash)

    tx_hash = vault.request_deposit(depositor, token.convert_to_raw(amount)).transact(
        {"from": depositor, "gas": 1_000_000}
    )
    assert_transaction_success_with_explanation(web3, tx_hash)


def finalise_lagoon_deposit(
    web3: Web3,
    vault: LagoonVault,
    depositor: str,
) -> None:
    """Finalise one already-settled Lagoon deposit request."""

    tx_hash = vault.finalise_deposit(depositor).transact({"from": depositor, "gas": 1_000_000})
    assert_transaction_success_with_explanation(web3, tx_hash)


def request_lagoon_redemption(
    web3: Web3,
    vault: LagoonVault,
    redeemer: str,
    shares: Decimal,
) -> None:
    """Open one Lagoon redemption request in share units."""

    tx_hash = vault.request_redeem(redeemer, vault.share_token.convert_to_raw(shares)).transact(
        {"from": redeemer, "gas": 1_000_000}
    )
    assert_transaction_success_with_explanation(web3, tx_hash)


def request_lagoon_redemption_fraction(
    web3: Web3,
    vault: LagoonVault,
    redeemer: str,
    fraction: Decimal,
) -> Decimal:
    """Redeem a fraction of the caller's currently available Lagoon shares."""

    assert Decimal("0") < fraction <= Decimal("1")
    shares = vault.share_token.fetch_balance_of(redeemer)
    requested_shares = shares * fraction
    request_lagoon_redemption(web3, vault, redeemer, requested_shares)
    return requested_shares


def request_lagoon_redemption_all(
    web3: Web3,
    vault: LagoonVault,
    redeemer: str,
) -> Decimal:
    """Redeem all currently available Lagoon shares for the caller."""

    shares = vault.share_token.fetch_balance_of(redeemer)
    request_lagoon_redemption(web3, vault, redeemer, shares)
    return shares


def get_lagoon_pending_redemptions_underlying(
    vault: LagoonVault,
    block_number: int,
) -> Decimal:
    """Read the current Lagoon redemption queue in underlying denomination units."""

    return vault.get_flow_manager().calculate_underlying_needed_for_redemptions(block_number)


def mirror_lagoon_safe_reserve_balance(
    web3: Web3,
    vault: LagoonVault,
    reserve_token: TokenDetails,
    state: State,
) -> None:
    """Mirror simulated reserve cash back to the Lagoon Safe on Anvil.

    Hypercore simulate-mode settlement updates strategy state, but it does not
    consistently move the Safe's ERC-20 balance on the fork. Tests that run
    repeated Lagoon treasury sync cycles need the on-chain Safe balance to
    match ``state.portfolio.get_cash()`` or Lagoon will reconcile reserves back
    to stale on-chain cash and distort deployable-capital accounting.
    """

    cash = Decimal(str(state.portfolio.get_cash()))
    fund_erc20_on_anvil(
        web3,
        reserve_token.address,
        vault.safe_address,
        reserve_token.convert_to_raw(cash),
    )


def settle_lagoon_redemption_queue(
    sync_model: LagoonVaultSyncModel,
    state: State,
    reserve_asset: AssetIdentifier,
    timestamp: datetime.datetime,
) -> None:
    """Run one post-trade Lagoon settlement pass for queued redemptions.

    The live strategy loop first rebalances positions and only then has enough
    reserve cash for Lagoon to settle pending withdrawals. Replay tests that
    model several consecutive redemption requests need this extra settlement
    step between requests so each finished rebalance can clear the previous
    queue before the next redemption is opened.
    """

    sync_model.sync_treasury(
        timestamp,
        state,
        supported_reserves=[reserve_asset],
        post_valuation=True,
    )
