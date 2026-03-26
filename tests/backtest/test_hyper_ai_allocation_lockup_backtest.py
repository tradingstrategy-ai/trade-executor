"""Backtest Hyper AI-style allocation with Hypercore redemption lock-ups.

This covers the specific case where one Hyperliquid vault stays locked while
another shorter-lock vault can be rebalanced away on a later cycle.
"""

import datetime

import pytest

from tradeexecutor.backtest.backtest_runner import BacktestResult, run_backtest_inline
from tradeexecutor.ethereum.vault.hypercore_vault import HLP_VAULT_ADDRESS
from tradeexecutor.exchange_account.allocation import (
    calculate_portfolio_target_value,
    get_redeemable_capital,
)
from tradeexecutor.state.identifier import AssetIdentifier, TradingPairIdentifier, TradingPairKind
from tradeexecutor.state.trade import TradeExecution
from tradeexecutor.strategy.alpha_model import AlphaModel
from tradeexecutor.strategy.cycle import CycleDuration
from tradeexecutor.strategy.default_routing_options import TradeRouting
from tradeexecutor.strategy.execution_context import ExecutionContext, ExecutionMode
from tradeexecutor.strategy.pandas_trader.indicator import IndicatorSet
from tradeexecutor.strategy.pandas_trader.strategy_input import StrategyInput
from tradeexecutor.strategy.reserve_currency import ReserveCurrency
from tradeexecutor.strategy.strategy_module import StrategyParameters
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse, create_pair_universe_from_code
from tradeexecutor.strategy.weighting import weight_passthrouh
from tradeexecutor.testing.synthetic_ethereum_data import generate_random_ethereum_address
from tradeexecutor.testing.synthetic_exchange_data import generate_exchange, generate_simple_routing_model
from tradeexecutor.testing.synthetic_price_data import generate_fixed_price_candles
from tradingstrategy.candle import GroupedCandleUniverse
from tradingstrategy.chain import ChainId
from tradingstrategy.timebucket import TimeBucket
from tradingstrategy.universe import Universe


BACKTEST_START_AT = datetime.datetime(2026, 1, 1)
BACKTEST_END_AT = datetime.datetime(2026, 1, 5)


class HyperAiAllocationLockupParameters:
    """Parameters for the synthetic Hyper AI allocation-lockup backtest."""

    initial_cash = 10_000
    cycle_duration = CycleDuration.cycle_1d
    allocation = 1.0
    min_portfolio_weight = 0.001
    individual_rebalance_min_threshold_usd = 0.01
    sell_rebalance_min_threshold = 0.01


def _create_hypercore_pair(
    *,
    exchange,
    reserve_asset: AssetIdentifier,
    internal_id: int,
    symbol: str,
    pool_address: str,
) -> TradingPairIdentifier:
    """Create one synthetic Hypercore vault pair for the backtest."""

    base = AssetIdentifier(
        ChainId.hypercore.value,
        generate_random_ethereum_address(),
        symbol,
        18,
        internal_id,
    )
    pair = TradingPairIdentifier(
        base,
        reserve_asset,
        pool_address,
        exchange.address,
        internal_id=internal_id,
        internal_exchange_id=exchange.exchange_id,
        fee=0.0001,
        kind=TradingPairKind.vault,
    )
    pair.other_data["vault_protocol"] = "hypercore"
    return pair


def _get_cycle_configuration(
    timestamp: datetime.datetime,
) -> tuple[float, dict[str, float]] | None:
    """Return the scheduled allocation plan for one strategy cycle."""

    if timestamp == BACKTEST_START_AT:
        return 0.75, {"HLP": 2.0, "A": 1.0}

    if timestamp == BACKTEST_START_AT + datetime.timedelta(days=2):
        return 1.0, {"B": 1.0}

    return None


def decide_trades(input: StrategyInput) -> list[TradeExecution]:
    """Run a scheduled Hyper AI-style rebalance with locked-capital carry-forward.

    1. Build AlphaModel targets from a fixed cycle schedule.
    2. Carry any non-redeemable Hypercore positions forward at their current marked value.
    3. Allocate only the remaining deployable target capital across the currently selected vaults.
    """

    cycle_configuration = _get_cycle_configuration(input.timestamp)
    if cycle_configuration is None:
        return []

    # 1. Build AlphaModel targets from a fixed cycle schedule.
    allocation, free_weight_schedule = cycle_configuration
    alpha_model = AlphaModel(
        input.timestamp,
        close_position_weight_epsilon=input.parameters.min_portfolio_weight,
    )
    position_manager = input.get_position_manager()
    pair_by_symbol = {
        pair.base.token_symbol: pair
        for pair in input.strategy_universe.iterate_pairs()
    }

    # 2. Carry any non-redeemable Hypercore positions forward at their current marked value.
    locked_position_value = alpha_model.carry_forward_non_redeemable_positions(
        position_manager,
        can_redeem=lambda position: get_redeemable_capital(
            position,
            timestamp=input.timestamp,
        ) > 0,
    )

    portfolio_target_value = calculate_portfolio_target_value(
        position_manager,
        allocation,
    )

    # 3. Allocate only the remaining deployable target capital across the currently selected vaults.
    deployable_target_value = max(portfolio_target_value - locked_position_value, 0.0)
    schedule_weight_sum = sum(free_weight_schedule.values())
    for symbol, raw_weight in free_weight_schedule.items():
        pair = pair_by_symbol[symbol]
        target_value = deployable_target_value * raw_weight / schedule_weight_sum
        alpha_model.set_signal(pair, target_value)

    alpha_model.select_top_signals(count=3)
    alpha_model.assign_weights(method=weight_passthrouh)
    alpha_model.normalise_weights(max_weight=1.0)
    alpha_model.update_old_weights(
        input.state.portfolio,
        ignore_credit=False,
    )
    alpha_model.calculate_target_positions(
        position_manager,
        investable_equity=deployable_target_value,
    )

    return alpha_model.generate_rebalance_trades_and_triggers(
        position_manager,
        min_trade_threshold=input.parameters.individual_rebalance_min_threshold_usd,
        individual_rebalance_min_threshold=input.parameters.individual_rebalance_min_threshold_usd,
        sell_rebalance_min_threshold=input.parameters.sell_rebalance_min_threshold,
        execution_context=input.execution_context,
    )


def create_indicators(
    parameters: StrategyParameters,
    indicators: IndicatorSet,
    strategy_universe: TradingStrategyUniverse,
    execution_context: ExecutionContext,
) -> None:
    """Use the modern backtest API without defining any custom indicators."""
    del parameters
    del indicators
    del strategy_universe
    del execution_context


@pytest.fixture()
def hyper_ai_allocation_lockup_universe() -> TradingStrategyUniverse:
    """Create a three-vault synthetic universe for the allocation-lockup backtest."""

    chain_id = ChainId.hypercore
    exchange = generate_exchange(
        exchange_id=1,
        chain_id=chain_id,
        address=generate_random_ethereum_address(),
    )
    reserve_asset = AssetIdentifier(
        chain_id.value,
        generate_random_ethereum_address(),
        "USDC",
        6,
        1,
    )

    hlp_pair = _create_hypercore_pair(
        exchange=exchange,
        reserve_asset=reserve_asset,
        internal_id=101,
        symbol="HLP",
        pool_address=HLP_VAULT_ADDRESS["mainnet"].lower(),
    )
    vault_a_pair = _create_hypercore_pair(
        exchange=exchange,
        reserve_asset=reserve_asset,
        internal_id=102,
        symbol="A",
        pool_address="0x00000000000000000000000000000000000000aa",
    )
    vault_b_pair = _create_hypercore_pair(
        exchange=exchange,
        reserve_asset=reserve_asset,
        internal_id=103,
        symbol="B",
        pool_address="0x00000000000000000000000000000000000000bb",
    )

    pair_universe = create_pair_universe_from_code(
        chain_id,
        [hlp_pair, vault_a_pair, vault_b_pair],
    )
    candles = generate_fixed_price_candles(
        TimeBucket.d1,
        BACKTEST_START_AT,
        BACKTEST_END_AT,
        {
            hlp_pair: 1.0,
            vault_a_pair: 1.0,
            vault_b_pair: 1.0,
        },
    )
    candle_universe = GroupedCandleUniverse(candles)

    universe = Universe(
        time_bucket=TimeBucket.d1,
        chains={chain_id},
        exchanges={exchange},
        pairs=pair_universe,
        candles=candle_universe,
        liquidity=None,
    )
    strategy_universe = TradingStrategyUniverse(
        data_universe=universe,
        reserve_assets=[reserve_asset],
    )
    strategy_universe.data_universe.pairs.exchange_universe = strategy_universe.data_universe.exchange_universe
    return strategy_universe


@pytest.fixture()
def allocation_lockup_backtest_result(
    hyper_ai_allocation_lockup_universe: TradingStrategyUniverse,
) -> BacktestResult:
    """Run the dedicated allocation-lockup backtest once for the module."""

    routing_model = generate_simple_routing_model(hyper_ai_allocation_lockup_universe)
    return run_backtest_inline(
        start_at=BACKTEST_START_AT,
        end_at=BACKTEST_END_AT,
        client=None,
        decide_trades=decide_trades,
        create_indicators=create_indicators,
        universe=hyper_ai_allocation_lockup_universe,
        cycle_duration=CycleDuration.cycle_1d,
        initial_deposit=HyperAiAllocationLockupParameters.initial_cash,
        reserve_currency=ReserveCurrency.usdc,
        trade_routing=TradeRouting.user_supplied_routing_model,
        routing_model=routing_model,
        allow_missing_fees=True,
        engine_version="0.5",
        parameters=HyperAiAllocationLockupParameters,
        mode=ExecutionMode.unit_testing,
        name="hyper-ai-allocation-lockup-backtest",
    )


@pytest.mark.timeout(300)
def test_hyper_ai_backtest_only_rebalances_redeemable_capital(
    allocation_lockup_backtest_result: BacktestResult,
) -> None:
    """Check that only redeemable Hypercore capital can be reallocated.

    1. Run a synthetic three-vault backtest with one long-lock HLP vault and two short-lock vaults.
    2. Open HLP and vault A on the first cycle, then switch the target allocation to vault B on the later rebalance cycle.
    3. Verify HLP stays in place while only vault A's redeemable capital plus idle cash moves into vault B.
    """

    # 1. Run a synthetic three-vault backtest with one long-lock HLP vault and two short-lock vaults.
    state, _, debug_dump = allocation_lockup_backtest_result

    # 2. Open HLP and vault A on the first cycle, then switch the target allocation to vault B on the later rebalance cycle.
    all_trades = list(state.portfolio.get_all_trades())
    assert len(all_trades) == 4

    hlp_trades = [trade for trade in all_trades if trade.pair.base.token_symbol == "HLP"]
    vault_a_trades = [trade for trade in all_trades if trade.pair.base.token_symbol == "A"]
    vault_b_trades = [trade for trade in all_trades if trade.pair.base.token_symbol == "B"]

    assert len(hlp_trades) == 1
    assert hlp_trades[0].is_buy()
    assert len(vault_a_trades) == 2
    assert vault_a_trades[0].is_buy()
    assert vault_a_trades[1].is_sell()
    assert len(vault_b_trades) == 1
    assert vault_b_trades[0].is_buy()

    # 3. Verify HLP stays in place while only vault A's redeemable capital plus idle cash moves into vault B.
    assert hlp_trades[0].get_value() == pytest.approx(5_000.0, abs=1e-6)
    assert vault_a_trades[0].get_value() == pytest.approx(2_500.0, abs=1e-6)
    assert vault_a_trades[1].get_value() == pytest.approx(2_500.0, abs=1.0)
    assert vault_b_trades[0].get_value() == pytest.approx(
        2_500.0 + vault_a_trades[1].get_value(),
        abs=1e-6,
    )
    assert vault_b_trades[0].get_value() < 7_500.0

    open_positions = {
        position.pair.base.token_symbol: position
        for position in state.portfolio.open_positions.values()
    }
    assert set(open_positions.keys()) == {"HLP", "B"}
    assert open_positions["HLP"].get_value() > 4_998.0
    assert open_positions["B"].get_value() > 4_998.0
    assert state.portfolio.get_cash() == pytest.approx(0.0, abs=1e-6)
    assert len(debug_dump) == 6
