"""Idle Lighter USDC belongs to investor NAV before a pool position exists."""

import datetime
import json
from contextlib import ExitStack
from decimal import Decimal
from unittest import mock
from types import SimpleNamespace

import pytest
from eth_defi.lighter.execution import LighterOperationStage, prepare_pool_deposit, fetch_execution_account
from eth_defi.token import USDC_NATIVE_TOKEN
from eth_defi.vault.deposit_redeem import VaultRequestInvariantViolation

from tradeexecutor.ethereum.lighter.idle_collateral import refresh_lighter_idle_collateral
from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.lighter.lighter_pool import create_lighter_pool_pair
from tradeexecutor.ethereum.lighter.lighter_pool_routing import LighterPoolContext, LighterPoolRouting
from tradeexecutor.ethereum.lighter.lighter_pool_routing import LighterPoolPhase
from tradeexecutor.ethereum.lagoon.vault import LagoonVaultSyncModel
from tradeexecutor.ethereum.vault.settlement_retry import check_and_resolve_vault_settlements
from tradeexecutor.exchange_account.lighter_operator import LighterOperatorRecord
from tradeexecutor.state.identifier import AssetIdentifier
from tradeexecutor.state.lighter import LighterIdleCollateral
from tradeexecutor.state.state import State
from tradeexecutor.state.trade import TradeStatus, TradeType
from tradeexecutor.strategy.runner import StrategyRunner
from tradeexecutor.testing.lighter_core import FakeLighterAccount, FakeLighterCore, FakeLighterPool


def test_pending_lighter_batch_expires_unfunded_pool_buys() -> None:
    """A delayed mint must not crash or fund later pool orders on the same account.

    1. Plan three buys against the dedicated account, including a fresh pool.
    2. Delay the first buy and run the normal sequential execution loop.
    3. Verify later orders expire without allocation and trigger checks defer.

    Routing is replaced at the submission boundary to leave a real pending
    trade without EVM or sequencer traffic; allocation and batching stay real.
    """
    # 1. Plan three buys against the dedicated account, including a fresh pool.
    safe = "0x" + "22" * 20
    ts = datetime.datetime(2026, 11, 4)
    state = State()
    reserve = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve, Decimal(10), "Safe funding")
    observation = LighterIdleCollateral(1, safe, 127, Decimal(1), ts)
    state.portfolio.lighter_idle_collateral[observation.get_identifier()] = observation
    pair = create_lighter_pool_pair(99, internal_id=1, internal_exchange_id=1)
    other_pair = create_lighter_pool_pair(100, internal_id=2, internal_exchange_id=1)
    trades = [state.create_trade(ts, target, Decimal(1), None, 1.0, TradeType.rebalance, reserve, 1.0)[1] for target in (pair, pair, other_pair)]
    state.portfolio.plan_lighter_idle_funding(trades)

    # 2. Delay the first buy and run the normal sequential execution loop.
    executor = mock.MagicMock(spec=EthereumExecution)
    executor.max_slippage = None
    routing = mock.Mock()
    routing.check_trade_before_execution.return_value = None

    def submit_pending(**kwargs) -> None:
        trade = kwargs["trades"][0]
        state.mark_broadcasted(ts, trade)
        state.mark_vault_settlement_pending(ts, trade, {})
        observation.valid = False

    routing.setup_trades.side_effect = submit_pending
    EthereumExecution._execute_trades_sequentially(executor, ts, state, trades, routing, mock.Mock(), False, False, False)

    # 3. Verify later orders expire without allocation and trigger checks defer.
    assert trades[0].is_vault_settlement_in_flight()
    assert all(t.get_status() == TradeStatus.expired for t in trades[1:])
    assert all(t.reserve_currency_allocated is None for t in trades[1:])
    assert trades[2].position_id in state.portfolio.expired_positions
    assert state.portfolio.get_cash() == pytest.approx(10)
    routing.setup_trades.assert_called_once()
    assert StrategyRunner.check_position_triggers(SimpleNamespace(), ts, state, None, None, None) == []


@pytest.mark.parametrize("budget", [Decimal(5.1234567), Decimal("5.1234567")])
def test_lighter_fractional_budget_and_sell_batch(budget: Decimal) -> None:
    """Fractional budgets must fit USDC units and never double-allocate swept idle cash.

    1. Round a strategy budget and allocate its idle and Safe contributions.
    2. Check a sell/buy batch does not promise idle cash that the sell sweeps.
    3. Credit the raw deposit and mint within the available account balance.

    Only the sequencer is simulated; the deposit is credited directly so this
    unit regression isolates token rounding from the guarded fork coverage.
    """
    # 1. Round a strategy budget and allocate its idle and Safe contributions.
    safe = "0x" + "22" * 20
    ts = datetime.datetime(2026, 11, 4)
    state = State()
    reserve = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve, Decimal(10), "Safe funding")
    observation = LighterIdleCollateral(1, safe, 127, Decimal(1), ts)
    state.portfolio.lighter_idle_collateral[observation.get_identifier()] = observation
    pair = create_lighter_pool_pair(99, internal_id=1, internal_exchange_id=1)
    position, trade, _ = state.create_trade(ts, pair, None, budget, 1.0, TradeType.rebalance, reserve, 1.0)
    state.portfolio.plan_lighter_idle_funding([trade])
    assert trade.planned_reserve == Decimal("5.123456")
    assert state.portfolio.get_lighter_buy_funding(trade) == Decimal("4.123456")

    # 2. Check a sell/buy batch does not promise idle cash that the sell sweeps.
    batch = State.from_json(state.to_json())
    batch_buy = batch.portfolio.get_trade_by_id(trade.trade_id)
    _, sell, _ = batch.create_trade(ts, pair, -Decimal(1), None, 1.0, TradeType.rebalance, reserve, 1.0)
    batch.portfolio.plan_lighter_idle_funding([sell, batch_buy])
    assert batch_buy.other_data["lighter_idle_allocated"] == "0"
    assert batch.portfolio.get_lighter_buy_funding(batch_buy) == Decimal("5.123456")

    # 3. Credit the raw deposit and mint within the available account balance.
    state.start_execution(ts, trade)
    state.mark_broadcasted(ts, trade)
    state.mark_vault_settlement_pending(ts, trade, {})
    core = FakeLighterCore(None, master_index=127)
    core.owner = safe
    core.accounts[127] = FakeLighterAccount(index=127, role="master", collateral=Decimal("5.123456"))
    core.add_pool(FakeLighterPool(index=99, equity=Decimal(100), total_shares=100, operator_shares=90))
    session = core.create_api_session()
    operator = LighterOperatorRecord(vault_address=safe, safe_address=safe, module_address=safe, account_index=127, api_key_index=4, api_private_key="0x00")
    router = LighterPoolRouting(session=session, poll_interval=0.0)
    ctx = LighterPoolContext(None, None, safe, None, None, session, operator)
    data = {"phase": LighterPoolPhase.collateral_credited.value, "pool_index": 99, "account_index": 127, "safe_address": safe, "shares_before": "0", "collateral_before": "1", "deposit_usdc": "4.123456", "operations": {}}
    trade.other_data["lighter_pool"] = data
    with ExitStack() as patches:
        for target, replacement in core.get_pool_patches():
            patches.enter_context(mock.patch(target, new=replacement))
        router._advance_buy(state, trade, ctx, data, core.create_pool_signer(operator))
        router.finalise_trade(state, trade)
    assert trade.is_success()
    assert trade.executed_reserve <= trade.planned_reserve
    assert position.get_unrealised_profit_usd() == pytest.approx(0)
    assert state.portfolio.get_net_asset_value() == pytest.approx(11)


def test_lighter_idle_only_buy_does_not_sign_safe_transactions(monkeypatch: pytest.MonkeyPatch) -> None:
    """An idle-funded buy must not consume Safe nonces for discarded transactions.

    1. Allocate an investment entirely from saved Lighter collateral.
    2. Complete the pool purchase through the fake sequencer without EVM calls.
    3. Check acquisition cost, NAV and the untouched transaction builder.

    The fake supplies the sequencer; a mocked token conversion avoids an RPC
    lookup for a zero deposit. The transaction builder must remain unused.
    """
    # 1. Allocate an investment entirely from saved Lighter collateral.
    safe = "0x" + "22" * 20
    ts = datetime.datetime(2026, 11, 4)
    state = State()
    reserve = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    observation = LighterIdleCollateral(1, safe, 127, Decimal(1), ts)
    state.portfolio.lighter_idle_collateral[observation.get_identifier()] = observation
    pair = create_lighter_pool_pair(99, internal_id=1, internal_exchange_id=1)
    position, trade, _ = state.create_trade(ts, pair, Decimal(1), None, 1.0, TradeType.rebalance, reserve, 1.0)
    state.start_execution(ts, trade)

    # 2. Complete the pool purchase through the fake sequencer without EVM calls.
    core = FakeLighterCore(None, master_index=127)
    core.owner = safe
    core.accounts[127] = FakeLighterAccount(index=127, role="master", collateral=Decimal(1))
    core.add_pool(FakeLighterPool(index=99, equity=Decimal(100), total_shares=100, operator_shares=90))
    session = core.create_api_session()
    operator = LighterOperatorRecord(vault_address=safe, safe_address=safe, module_address=safe, account_index=127, api_key_index=4, api_private_key="0x00")
    builder = mock.Mock()
    usdc = SimpleNamespace(convert_to_raw=mock.Mock(return_value=0), convert_to_decimals=reserve.convert_to_decimal)
    router = LighterPoolRouting(session=session, poll_interval=0.0)
    monkeypatch.setattr(router, "create_context", lambda tx_builder: LighterPoolContext(None, builder, safe, usdc, None, session, operator))
    with ExitStack() as patches:
        for target, replacement in core.get_pool_patches():
            patches.enter_context(mock.patch(target, new=replacement))
        assert router.advance_trade(state, trade, builder)
        router.finalise_trade(state, trade)

    # 3. Check acquisition cost, NAV and the untouched transaction builder.
    usdc.convert_to_raw.assert_called_once_with(Decimal(0))
    assert builder.mock_calls == []
    assert not trade.blockchain_transactions
    assert trade.executed_reserve == Decimal(1)
    assert position.get_unrealised_profit_usd() == pytest.approx(0)
    assert state.portfolio.get_net_asset_value() == pytest.approx(1)


@pytest.mark.parametrize("refresh_error", [TimeoutError, VaultRequestInvariantViolation])
def test_lighter_bootstrap_nav_and_pool_cost(monkeypatch: pytest.MonkeyPatch, refresh_error: type[Exception]) -> None:
    """Import idle cash once and include it in acquisition cost without spending it twice.

    1. Observe one activation USDC alongside the remaining bootstrap Safe cash.
    2. Round-trip state and reserve a five-USDC buy funded partly by idle cash.
    3. Submit a mint with the fake venue, then resume an unbooked submission.
    4. Recover a failed post-completion read from its durable successful trade.
    5. Recover an exit which swept idle cash without booking it as pool profit.

    The fake replaces only the sequencer. This unit test credits the deposit
    directly; guarded EVM deposits are exercised by the dual-fork tests.
    """
    # 1. Observe one activation USDC alongside the remaining bootstrap Safe cash.
    safe = "0x" + "22" * 20
    core = FakeLighterCore(None, master_index=127)
    core.owner = safe
    core.accounts[127] = FakeLighterAccount(index=127, role="master", collateral=Decimal(1))
    core.add_pool(FakeLighterPool(index=99, equity=Decimal(100), total_shares=100, operator_shares=90))
    session = core.create_api_session()
    state = State()
    reserve = AssetIdentifier(1, USDC_NATIVE_TOKEN[1], "USDC", 6)
    state.portfolio.initialise_reserves(reserve, reserve_token_price=1.0)
    state.portfolio.adjust_reserves(reserve, Decimal("18.99"), "Bootstrap net of bridge fee")
    refresh_lighter_idle_collateral(state, session, 127, safe)
    refresh_lighter_idle_collateral(state, session, 127, safe)
    assert state.portfolio.get_net_asset_value() == pytest.approx(19.99)
    assert state.portfolio.get_cash() == pytest.approx(18.99)
    assert state.portfolio.calculate_total_equity_chain()[1] == pytest.approx(19.99)

    # 2. Round-trip state and reserve a five-USDC buy funded partly by idle cash.
    state = State.from_json(state.to_json())
    observation = next(iter(state.portfolio.lighter_idle_collateral.values()))
    assert observation.amount == Decimal(1) and observation.observed_at.tzinfo is None
    legacy = json.loads(state.to_json())
    legacy["portfolio"].pop("lighter_idle_collateral")
    assert State.from_json(json.dumps(legacy)).portfolio.lighter_idle_collateral == {}
    pair = create_lighter_pool_pair(99, internal_id=1, internal_exchange_id=1)
    ts = datetime.datetime(2026, 11, 4)
    position, trade, _ = state.create_trade(ts, pair, Decimal(5), None, 1.0, TradeType.rebalance, reserve, 1.0)
    state.start_execution(ts, trade)
    assert trade.reserve_currency_allocated == Decimal(4)
    assert observation.amount == 0
    state.mark_broadcasted(ts, trade)
    state.mark_vault_settlement_pending(ts, trade, {})
    assert state.portfolio.get_net_asset_value() == pytest.approx(19.99)

    # 3. Submit a mint with the fake venue, then resume an unbooked submission.
    operator = LighterOperatorRecord(vault_address=safe, safe_address=safe, module_address=safe, account_index=127, api_key_index=4, api_private_key="0x00")
    router = LighterPoolRouting(session=session, poll_interval=0.0)
    ctx = LighterPoolContext(None, None, safe, None, None, session, operator)
    signer = core.create_pool_signer(operator)
    core.accounts[127].collateral += Decimal(4)
    data = {"operations": {}}
    with ExitStack() as patches:
        for target, replacement in core.get_pool_patches():
            patches.enter_context(mock.patch(target, new=replacement))
        operation = router._run_operation(ctx, data, "mint", signer, lambda executor: prepare_pool_deposit(executor, 99, Decimal(5)))
        saved = json.loads(data["operations"]["mint"])
        saved["stage"] = LighterOperationStage.unknown.value
        data["operations"]["mint"] = json.dumps(saved)
        resumed = router._run_operation(ctx, data, "mint", signer, lambda executor: pytest.fail("Must reconcile the submitted identity"))
    assert resumed.tx_hash == operation.tx_hash and len(core.api_transactions) == 1

    # 4. Recover a failed post-completion read from its durable successful trade.
    trade.other_data["lighter_pool"] = {"phase": "complete", "account_index": 127, "safe_address": safe, "executed_shares": str(operation.amount), "invested_usdc": "5"}
    saved: list[str] = []
    execution = SimpleNamespace(sync_state_before_broadcast=lambda: saved.append(state.to_json()))
    # Replace only chain-dependent dispatch. Actual pool finalisation books
    # success before the injected public transport outage, just as a mined
    # claim can complete before its post-completion valuation fails.
    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.vault.settlement_retry._resolve_lighter_pool_trade", lambda state, trade, *args: router.finalise_trade(state, trade))
        patch.setattr("tradeexecutor.ethereum.vault.settlement_retry.create_executor_lighter_session", lambda: session)
        patch.setattr("tradeexecutor.ethereum.lighter.idle_collateral.fetch_execution_account", mock.Mock(side_effect=refresh_error("Post-completion read unavailable")))
        with pytest.raises(refresh_error):
            check_and_resolve_vault_settlements(state, execution_model=execution)
    assert len(saved) == 1
    state = State.from_json(saved[0])
    trade = state.portfolio.get_trade_by_id(trade.trade_id)
    position = state.portfolio.get_position_by_id(trade.position_id)
    assert trade.is_success()
    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.vault.settlement_retry.create_executor_lighter_session", lambda: session)
        assert check_and_resolve_vault_settlements(state, execution_model=execution) == []
    assert trade.executed_reserve == Decimal(5)
    assert position.get_unrealised_profit_usd() == pytest.approx(0)
    assert state.portfolio.get_net_asset_value() == pytest.approx(19.99)
    assert state.portfolio.get_cash() == pytest.approx(14.99)
    assert state.portfolio.get_lighter_idle_collateral_value() == 0

    # An expired top-up never moved capital and must not block future NAV.
    _, expired, _ = state.create_trade(ts, pair, Decimal(1), None, 1.0, TradeType.rebalance, reserve, 1.0)
    expired.mark_expired(ts)
    refresh_lighter_idle_collateral(state, session, 127, safe)
    assert state.portfolio.get_net_asset_value() == pytest.approx(19.99)

    # 5. Recover a terminal withdrawal which swept idle cash as well as pool
    # proceeds. The saved terminal evidence stands in for an already-mined
    # claim; real guarded claims are exercised by the fork scenario.
    core.accounts[127].collateral = Decimal("0.25")
    refresh_lighter_idle_collateral(state, session, 127, safe)
    _, sell, _ = state.create_trade(ts, pair, -position.get_quantity(), None, 1.0, TradeType.rebalance, reserve, 1.0)
    state.start_execution(ts, sell)
    state.mark_broadcasted(ts, sell)
    state.mark_vault_settlement_pending(ts, sell, {})
    core.accounts[127].pool_shares.clear()
    core.accounts[127].collateral = Decimal(0)
    sell.other_data["lighter_pool"] = {"phase": "complete", "account_index": 127, "safe_address": safe, "burned_shares": "5", "claimed_usdc": "5.25", "idle_swept": "0.25"}
    router.finalise_trade(state, sell)
    assert sell.executed_reserve == Decimal(5)
    assert position.get_realised_profit_usd() == pytest.approx(0)
    assert state.portfolio.get_cash() == pytest.approx(20.24)


def test_lighter_snapshot_refuses_unverified_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unverified ownership, provider failures and torn observations cannot publish NAV.

    1. Save a valid observation of the dedicated Safe's activation cash.
    2. Reject another owner and mark the existing valuation incomplete.
    3. Reject an account change between reads, then recover on a stable retry.
    4. Recover a persisted invalid observation through an empty settlement sweep.
    5. Refuse unsupported custom NAV callbacks before invoking them.
    6. Refuse pool routing without public account metadata, even in legacy state.

    Only the public API transport is replaced to expose a changing snapshot.
    """
    # 1. Save a valid observation of the dedicated Safe's activation cash.
    safe = "0x" + "22" * 20
    core = FakeLighterCore(None, master_index=127)
    core.owner = safe
    core.accounts[127] = FakeLighterAccount(index=127, role="master", collateral=Decimal(1))
    session = core.create_api_session()
    state = State()
    refresh_lighter_idle_collateral(state, session, 127, safe)

    # 2. Reject another owner and mark the existing valuation incomplete.
    core.owner = "0x" + "33" * 20
    with pytest.raises(VaultRequestInvariantViolation, match="does not belong"):
        refresh_lighter_idle_collateral(state, session, 127, safe)
    with pytest.raises(RuntimeError, match="incomplete"):
        state.portfolio.get_net_asset_value()
    core.owner = safe

    # 3. Reject an account change between reads, then recover on a stable retry.
    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.lighter.idle_collateral.fetch_lighter_account_by_index", lambda *args: {})
        with pytest.raises(VaultRequestInvariantViolation, match="changed during"):
            refresh_lighter_idle_collateral(state, session, 127, safe)
    refresh_lighter_idle_collateral(state, session, 127, safe)
    assert state.portfolio.get_net_asset_value() == 1

    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.lighter.idle_collateral.fetch_execution_account", mock.Mock(side_effect=TimeoutError("Unavailable public API")))
        with pytest.raises(TimeoutError):
            refresh_lighter_idle_collateral(state, session, 127, safe)
    with pytest.raises(RuntimeError, match="incomplete"):
        state.portfolio.get_net_asset_value()

    # 4. Recover a persisted invalid observation through an empty settlement sweep.
    state = State.from_json(state.to_json())
    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.vault.settlement_retry.create_executor_lighter_session", lambda: session)
        assert check_and_resolve_vault_settlements(state, execution_model=None) == []
    assert state.portfolio.get_net_asset_value() == 1
    account, _ = fetch_execution_account(session, 127)
    account["total_asset_value"] = "12345"  # Aggregate equity is not a collateral input.
    with monkeypatch.context() as patch:
        patch.setattr("tradeexecutor.ethereum.lighter.idle_collateral.fetch_lighter_account_by_index", lambda *args: account)
        refresh_lighter_idle_collateral(state, session, 127, safe)
    assert state.portfolio.get_net_asset_value() == 1

    # 5. Custom callbacks have an explicit refusal before either callback or
    # account transport can produce an incomplete combined valuation.
    sync = object.__new__(LagoonVaultSyncModel)
    sync.lighter_pool_account = (127, safe)
    sync.calculate_valuation_func = lambda *args, **kwargs: pytest.fail("Unsupported callback must not run")
    with pytest.raises(RuntimeError, match="Custom NAV"):
        sync.calculate_valuation(state)

    # 6. Refuse pool routing without public account metadata, even in legacy state.
    sync.lighter_pool_account = None
    runner = SimpleNamespace(routing_model=None, sync_model=sync)
    pair = create_lighter_pool_pair(99, internal_id=1, internal_exchange_id=1)
    universe = SimpleNamespace(iterate_pairs=lambda: iter([pair]))
    with pytest.raises(RuntimeError, match="public account metadata"):
        StrategyRunner.setup_routing(runner, universe)
