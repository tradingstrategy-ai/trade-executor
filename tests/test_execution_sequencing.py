"""Tests for sequential trade execution orchestration.

The Hypercore regression happened because some routes need each trade to be
fully settled before the next trade is prepared and broadcast.
"""

import datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tradingstrategy.chain import ChainId

from tradeexecutor.ethereum.execution import EthereumExecution
from tradeexecutor.ethereum.tx import TransactionBuilder
from tradeexecutor.strategy.execution_model import ExecutionHaltableIssue
from tradeexecutor.strategy.generic.generic_router import GenericRouting


class DummyTransactionBuilder(TransactionBuilder):
    """Tiny transaction builder for executor orchestration tests."""

    def init(self):
        """Initialise the builder."""

    def sign_transaction(
        self,
        contract,
        args_bound_func,
        gas_limit: int | None = None,
        gas_price_suggestion=None,
        asset_deltas=None,
        notes: str = "",
    ):
        """Unused in these tests."""
        raise NotImplementedError()

    def get_token_delivery_address(self) -> str:
        """Return a dummy token delivery address."""
        return "0x0000000000000000000000000000000000000001"

    def get_erc_20_balance_address(self) -> str:
        """Return a dummy ERC-20 balance address."""
        return "0x0000000000000000000000000000000000000002"

    def get_gas_wallet_address(self) -> str:
        """Return a dummy gas wallet address."""
        return "0x0000000000000000000000000000000000000003"

    def get_gas_wallet_balance(self) -> Decimal:
        """Return a dummy gas balance."""
        return Decimal("10")


class RecordingExecution(EthereumExecution):
    """Execution model that records orchestration instead of broadcasting."""

    def __init__(self):
        web3 = SimpleNamespace(
            eth=SimpleNamespace(chain_id=31337),
            provider=SimpleNamespace(),
        )
        super().__init__(
            DummyTransactionBuilder(web3),
            confirmation_block_count=1,
        )
        self.executed_batches: list[list[int]] = []
        self.logged_outcomes: list[int] = []

    def _execute_trade_batch(self, routing_model, state, trades, rebroadcast: bool) -> None:
        """Record the batch and mark each trade as succeeded."""
        del routing_model
        del state
        del rebroadcast

        self.executed_batches.append([trade.trade_id for trade in trades])
        for trade in trades:
            trade.get_status.return_value = SimpleNamespace(value="success")

    def _log_trade_outcome(self, trade) -> None:
        """Record the trade ids whose outcomes were logged."""
        self.logged_outcomes.append(trade.trade_id)


def _make_trade(trade_id: int) -> MagicMock:
    """Create a tiny trade stub for execution orchestration tests."""
    trade = MagicMock()
    trade.trade_id = trade_id
    trade.route = None
    trade.pair = MagicMock()
    trade.pair.is_exchange_account.return_value = False
    trade.pair.get_ticker.return_value = f"PAIR-{trade_id}"
    trade.get_planned_value.return_value = float(trade_id)
    trade.get_status.return_value = SimpleNamespace(value="planned")
    trade.is_failed.return_value = False
    trade.get_revert_reason.return_value = None
    return trade


def test_execute_trades_runs_sequential_router_one_trade_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run sequential routes one trade at a time.

    1. Build an execution model and a routing model that declares sequential settlement.
    2. Execute two trades and verify start/setup/broadcast happen per trade, not per batch.
    3. Verify slippage, freeze handling, and final outcome logging also happen per trade.
    """
    # 1. Build an execution model and a routing model that declares sequential settlement.
    execution = RecordingExecution()
    state = MagicMock()
    routing_model = MagicMock()
    routing_model.needs_sequential_trade_execution.return_value = True
    routing_model.get_sequential_trade_execution_reason.return_value = "router needs settlement sequencing"
    routing_state = SimpleNamespace()
    trades = [_make_trade(1), _make_trade(2)]

    frozen_batches: list[list[int]] = []

    def record_freeze(ts, state_arg, trade_batch):
        del ts
        del state_arg
        frozen_batches.append([trade.trade_id for trade in trade_batch])

    monkeypatch.setattr(
        "tradeexecutor.ethereum.execution.freeze_position_on_failed_trade",
        record_freeze,
    )

    # 2. Execute two trades and verify start/setup/broadcast happen per trade, not per batch.
    execution.execute_trades(
        ts=datetime.datetime(2026, 4, 13, 12, 0, 0),
        state=state,
        trades=trades,
        routing_model=routing_model,
        routing_state=routing_state,
        check_balances=True,
    )

    # 3. Verify slippage, freeze handling, and final outcome logging also happen per trade.
    state.start_execution_all.assert_not_called()
    assert [call.args[1].trade_id for call in state.start_execution.call_args_list] == [1, 2]
    assert all(call.kwargs["underflow_check"] is True for call in state.start_execution.call_args_list)
    assert [trade.planned_max_slippage for trade in trades] == [execution.max_slippage, execution.max_slippage]
    assert [call.kwargs["trades"][0].trade_id for call in routing_model.setup_trades.call_args_list] == [1, 2]
    assert execution.executed_batches == [[1], [2]]
    assert frozen_batches == [[1], [2]]
    assert execution.logged_outcomes == [1, 2]


def test_execute_trades_keeps_batch_mode_for_normal_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the existing batch behaviour for normal routes.

    1. Build an execution model and a routing model that does not need sequential settlement.
    2. Execute two trades and verify setup and broadcast still happen as a single batch.
    3. Verify freeze handling and outcome logging also stay batched.
    """
    # 1. Build an execution model and a routing model that does not need sequential settlement.
    execution = RecordingExecution()
    state = MagicMock()
    routing_model = MagicMock()
    routing_model.needs_sequential_trade_execution.return_value = False
    routing_state = SimpleNamespace()
    trades = [_make_trade(10), _make_trade(11)]

    frozen_batches: list[list[int]] = []

    def record_freeze(ts, state_arg, trade_batch):
        del ts
        del state_arg
        frozen_batches.append([trade.trade_id for trade in trade_batch])

    monkeypatch.setattr(
        "tradeexecutor.ethereum.execution.freeze_position_on_failed_trade",
        record_freeze,
    )

    # 2. Execute two trades and verify setup and broadcast still happen as a single batch.
    execution.execute_trades(
        ts=datetime.datetime(2026, 4, 13, 12, 5, 0),
        state=state,
        trades=trades,
        routing_model=routing_model,
        routing_state=routing_state,
    )

    # 3. Verify freeze handling and outcome logging also stay batched.
    state.start_execution_all.assert_called_once()
    state.start_execution.assert_not_called()
    assert len(routing_model.setup_trades.call_args_list) == 1
    assert routing_model.setup_trades.call_args.kwargs["trades"] == trades
    assert execution.executed_batches == [[10, 11]]
    assert frozen_batches == [[10, 11]]
    assert execution.logged_outcomes == [10, 11]


def test_execute_trades_stops_sequential_batch_after_failed_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop the remaining sequential batch after a failed trade.

    1. Build a sequential execution run where the first trade resolves as failed.
    2. Execute the batch and verify the executor raises a haltable issue immediately.
    3. Verify the second trade is never started or prepared.
    """
    # 1. Build a sequential execution run where the first trade resolves as failed.
    execution = RecordingExecution()
    state = MagicMock()
    routing_model = MagicMock()
    routing_model.needs_sequential_trade_execution.return_value = True
    routing_model.get_sequential_trade_execution_reason.return_value = "router needs settlement sequencing"
    routing_state = SimpleNamespace()
    trades = [_make_trade(21), _make_trade(22)]

    frozen_batches: list[list[int]] = []

    def record_freeze(ts, state_arg, trade_batch):
        del ts
        del state_arg
        frozen_batches.append([trade.trade_id for trade in trade_batch])

    monkeypatch.setattr(
        "tradeexecutor.ethereum.execution.freeze_position_on_failed_trade",
        record_freeze,
    )

    def fail_first_trade(routing_model_arg, state_arg, trade_batch, rebroadcast: bool) -> None:
        del routing_model_arg
        del state_arg
        del rebroadcast

        execution.executed_batches.append([trade.trade_id for trade in trade_batch])
        trade = trade_batch[0]
        trade.get_status.return_value = SimpleNamespace(value="failed")
        trade.is_failed.return_value = True
        trade.get_revert_reason.return_value = "insufficient settled capital"

    execution._execute_trade_batch = fail_first_trade

    # 2. Execute the batch and verify the executor raises a haltable issue immediately.
    with pytest.raises(ExecutionHaltableIssue, match="Failed trade_id=21"):
        execution.execute_trades(
            ts=datetime.datetime(2026, 4, 13, 12, 10, 0),
            state=state,
            trades=trades,
            routing_model=routing_model,
            routing_state=routing_state,
        )

    # 3. Verify the second trade is never started or prepared.
    assert [call.args[1].trade_id for call in state.start_execution.call_args_list] == [21]
    assert [call.kwargs["trades"][0].trade_id for call in routing_model.setup_trades.call_args_list] == [21]
    assert execution.executed_batches == [[21]]
    assert frozen_batches == [[21]]
    assert execution.logged_outcomes == [21]


def _make_spot_buy_trade(
    *,
    chain_id: int,
    value: float,
    is_buy: bool = True,
) -> MagicMock:
    """Create a spot buy/sell trade stub for the treasury trip-wire tests."""
    trade = MagicMock()
    trade.trade_id = 1
    trade.pair = MagicMock()
    trade.pair.chain_id = chain_id
    trade.is_spot.return_value = True
    trade.is_buy.return_value = is_buy
    trade.is_sell.return_value = not is_buy
    trade.is_failed.return_value = False
    trade.get_value.return_value = value
    trade.get_failed_transaction.return_value = None
    return trade


def test_buy_treasury_tripwire_raises_haltable_issue_when_underfunded() -> None:
    """Abort a same-chain spot buy that cannot be funded from the on-chain treasury.

    This regression-guards an ``UnboundLocalError`` that previously masked the real error:
    when ``completed_trades`` was empty (the buy was the first trade in the batch), the
    diagnostics loop never ran and ``failed_tx`` was referenced before assignment, crashing
    with ``UnboundLocalError`` instead of the intended ``ExecutionHaltableIssue``.

    1. Build an execution model (home chain 31337) and a same-chain spot buy of value 5.
    2. Call the trip-wire helper with an empty completed_trades list and a zero balance.
    3. Verify it raises a clean ExecutionHaltableIssue, not an UnboundLocalError.
    """
    # 1. Build an execution model (home chain 31337) and a same-chain spot buy of value 5.
    execution = RecordingExecution()
    trade = _make_spot_buy_trade(chain_id=31337, value=5.0)

    # 2 & 3. Empty completed_trades must still produce a clean ExecutionHaltableIssue.
    with pytest.raises(ExecutionHaltableIssue, match="Not enough treasury to buy token"):
        execution._raise_if_insufficient_buy_treasury(
            trade,
            onchain_treasury_balance=0.0,
            needed_usd=5.0,
            completed_trades=[],
            total_trade_count=1,
        )


def test_buy_treasury_tripwire_skips_cross_chain_satellite_trade() -> None:
    """Do not trip the treasury wire for cross-chain (satellite) buys.

    A cross-chain buy spends reserve that has been bridged via CCTP to another chain, so the
    home-chain treasury balance is legitimately empty. The trip-wire must skip these trades
    instead of falsely aborting (the production bug that crashed ``trade-ui`` test trades).

    1. Build an execution model on home chain 31337.
    2. Call the trip-wire helper with a buy whose pair lives on a different chain and a zero
       home-chain balance.
    3. Verify it returns without raising.
    """
    # 1. Build an execution model on home chain 31337.
    execution = RecordingExecution()

    # 2. A cross-chain buy (pair chain 8453 != home 31337) with empty home-chain treasury.
    trade = _make_spot_buy_trade(chain_id=8453, value=5.0)

    # 3. Must not raise even though the home-chain balance (0) is below the needed value.
    execution._raise_if_insufficient_buy_treasury(
        trade,
        onchain_treasury_balance=0.0,
        needed_usd=5.0,
        completed_trades=[],
        total_trade_count=1,
    )


def test_buy_treasury_tripwire_allows_funded_same_chain_buy() -> None:
    """Allow a same-chain spot buy that is fully funded.

    1. Build an execution model on home chain 31337 and a same-chain spot buy of value 5.
    2. Call the trip-wire helper with a sufficient on-chain balance.
    3. Verify it returns without raising.
    """
    # 1. Build an execution model on home chain 31337 and a same-chain spot buy of value 5.
    execution = RecordingExecution()
    trade = _make_spot_buy_trade(chain_id=31337, value=5.0)

    # 2 & 3. Sufficient balance must not trip the wire.
    execution._raise_if_insufficient_buy_treasury(
        trade,
        onchain_treasury_balance=10.0,
        needed_usd=5.0,
        completed_trades=[],
        total_trade_count=1,
    )


def _make_broadcast_trade(trade_id: int, pair_chain_id: int, tx_chain_id: int) -> MagicMock:
    """Create a fully-funded spot buy with one signed tx on the given chain."""
    trade = MagicMock()
    trade.trade_id = trade_id
    trade.pair.chain_id = pair_chain_id
    trade.is_spot.return_value = True
    trade.is_buy.return_value = True
    trade.get_value.return_value = 5.0

    tx = SimpleNamespace(
        chain_id=tx_chain_id,
        tx_hash="0x" + f"{trade_id:064x}",
        signed_bytes=bytes([2, trade_id]),
        from_address="0x" + "22" * 20,
        nonce=trade_id,
        details={},
    )
    trade.blockchain_transactions = [tx]
    return trade


def test_mev_blocker_broadcast_uses_per_chain_provider_for_satellite_tx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Broadcast satellite-chain txs through their own provider, not the home-chain one.

    The sequential / MEV-blocker path selects a provider once from the home-chain web3. A
    cross-chain (satellite) buy carries a tx signed for another chain, which must be submitted
    via that chain's provider — otherwise a Base tx would be sent to the Arbitrum provider and
    fail one step after the treasury trip-wire is skipped.

    1. Build an execution model on home chain 31337 with a web3config that hands out a
       distinct Base (8453) connection.
    2. Broadcast a home-chain buy followed by a satellite-chain buy through
       broadcast_and_resolve_mev_blocker().
    3. Verify each tx was broadcast through its own chain's provider and the satellite
       connection was resolved via web3config.
    """
    # 1. Build an execution model on home chain 31337 and a satellite (Base) connection.
    execution = RecordingExecution()

    home_web3 = execution.web3
    home_web3.eth.get_balance = lambda address: 10 ** 30
    home_provider = home_web3.provider

    satellite_provider = SimpleNamespace()
    satellite_web3 = SimpleNamespace(
        eth=SimpleNamespace(chain_id=8453, get_balance=lambda address: 10 ** 30),
        provider=satellite_provider,
    )

    get_connection_calls: list[ChainId] = []

    def get_connection(chain_id: ChainId):
        get_connection_calls.append(chain_id)
        return satellite_web3

    execution.web3config = SimpleNamespace(get_connection=get_connection)

    # Treasury is well funded so the buy-side trip-wire never fires.
    monkeypatch.setattr(
        "tradeexecutor.ethereum.execution.fetch_erc20_details",
        lambda *args, **kwargs: SimpleNamespace(fetch_balance_of=lambda address: Decimal("1000000")),
    )

    broadcast_providers: list = []

    def fake_broadcast(provider, signed_txs, max_timeout=None, **kwargs):
        broadcast_providers.append(provider)
        return {}

    monkeypatch.setattr(
        "tradeexecutor.ethereum.execution.wait_and_broadcast_multiple_nodes_mev_blocker",
        fake_broadcast,
    )

    # resolve_trades touches state/receipts we do not model here.
    execution.resolve_trades = lambda *args, **kwargs: None

    state = MagicMock()
    reserve_asset = SimpleNamespace(address="0x" + "11" * 20, chain_id=31337)
    state.portfolio.get_default_reserve_asset.return_value = (reserve_asset, None)

    trades = [
        _make_broadcast_trade(1, pair_chain_id=31337, tx_chain_id=31337),
        _make_broadcast_trade(2, pair_chain_id=8453, tx_chain_id=8453),
    ]

    # 2. Broadcast a home-chain buy followed by a satellite-chain buy.
    execution.broadcast_and_resolve_mev_blocker(
        routing_model=MagicMock(),
        state=state,
        trades=trades,
    )

    # 3. Each tx was broadcast through its own chain's provider.
    assert broadcast_providers == [home_provider, satellite_provider]
    assert get_connection_calls == [ChainId(8453)]


def test_generic_routing_detects_underlying_sequential_router() -> None:
    """Surface sequential requirements from the matched underlying router.

    1. Build a GenericRouting with one normal router and one sequential router.
    2. Ask GenericRouting about a mixed batch of trades.
    3. Verify it reports sequential execution and forwards the underlying reason.
    """
    # 1. Build a GenericRouting with one normal router and one sequential router.
    sequential_router = MagicMock()
    sequential_router.needs_sequential_trade_execution.return_value = True
    sequential_router.get_sequential_trade_execution_reason.return_value = "hypercore settlement appends follow-up txs"

    normal_router = MagicMock()
    normal_router.needs_sequential_trade_execution.return_value = False
    normal_router.get_sequential_trade_execution_reason.return_value = None

    pair_a = object()
    pair_b = object()
    routing_id_a = object()
    routing_id_b = object()
    configs = {
        routing_id_a: SimpleNamespace(routing_model=normal_router),
        routing_id_b: SimpleNamespace(routing_model=sequential_router),
    }
    pair_configurator = SimpleNamespace(
        match_router=lambda pair: routing_id_a if pair is pair_a else routing_id_b,
        get_config=lambda routing_id, three_leg_resolution=True: configs[routing_id],
    )
    routing = GenericRouting(pair_configurator)

    trades = [
        SimpleNamespace(pair=pair_a),
        SimpleNamespace(pair=pair_b),
    ]

    # 2. Ask GenericRouting about a mixed batch of trades.
    needs_sequential = routing.needs_sequential_trade_execution(trades)
    reason = routing.get_sequential_trade_execution_reason(trades)

    # 3. Verify it reports sequential execution and forwards the underlying reason.
    assert needs_sequential is True
    assert reason == "hypercore settlement appends follow-up txs"
