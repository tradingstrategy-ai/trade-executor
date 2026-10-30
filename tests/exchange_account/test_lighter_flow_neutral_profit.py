"""Flow-neutral profit of Lighter account positions.

Each Lighter account's profit must not depend on the capital moved in and out
of it. The timelines in :py:mod:`tradeexecutor.testing.lighter_share_price_timelines`
drive deposits, withdrawals, transfers between accounts, gains, losses,
write-offs and closes through the executor's own state helpers, and check
four invariants after every event. These tests replay them, and cover what
the timelines do not: states saved before the withdrawn total was tracked,
transfer recovery, the stored statistics and the chart.
"""

import json
from decimal import Decimal

import pytest

from tradeexecutor.exchange_account.state import (
    ExchangeAccountTransferError,
    complete_exchange_account_funding,
    create_exchange_account_funding,
)
from tradeexecutor.state.state import State
from tradeexecutor.strategy.chart.definition import ChartInput
from tradeexecutor.strategy.chart.standard.exchange_account import exchange_account_share_price
from tradeexecutor.strategy.execution_context import unit_test_execution_context
from tradeexecutor.strategy.position_internal_share_price import calculate_flow_neutral_profit_usd
from tradeexecutor.testing.lighter_share_price_timelines import (
    MASTER,
    TIMELINES,
    Timeline,
    TimelineReplay,
    replay_timeline,
)
from tradeexecutor.visual.image_output import render_plotly_figure_as_image_file


@pytest.mark.parametrize("timeline", TIMELINES, ids=[timeline.timeline_id for timeline in TIMELINES])
def test_lighter_timeline_profit_is_flow_neutral(timeline: Timeline) -> None:
    """Every timeline keeps each account's return and USD profit independent of its flows.

    The timelines cover a gain before an inflow, a loss after inflows with a
    partial exit, a full exit and re-entry, a funding chain with fees, a
    valuation-seeded opening, a withdrawal fee, a write-off and close, a
    total loss followed by re-funding, and a repeated allocation.

    1. Replay the timeline, checking the four invariants after every event
    2. Check each account's final shares, share price, return and USD profit
    """
    # 1. Replay the timeline, checking the four invariants after every event
    # 2. Check each account's final shares, share price, return and USD profit
    replay = replay_timeline(timeline)
    assert all(position.share_price_state.total_withdrawn is not None for name, position in replay.positions.items() if name in timeline.expected)


def test_lighter_flow_neutral_profit_old_state_recovery_and_chart() -> None:
    """States saved before the withdrawn total existed stay on the old figures, recovery counts once, and the outputs carry the new figures.

    1. Book a deposit and a transfer, then save and reload the state without the new key
    2. Apply the rest of the funding chain: the reloaded accounts stay unknown on the old figures, an account first funded after the reload is tracked
    3. Recover a failed transfer once, then check a repeated completion is refused and changes nothing
    4. Check the stored statistics and the chart of a fresh funding chain carry the flow-neutral figures
    """
    funding_chain = next(timeline for timeline in TIMELINES if timeline.timeline_id == "funding-chain")

    # 1. Book a deposit and a transfer, then save and reload the state without the new key
    replay = TimelineReplay(funding_chain.accounts)
    for event in funding_chain.events[:2]:
        replay.apply(event)
    serialised = json.loads(replay.state.to_json_safe())
    for position in serialised["portfolio"]["open_positions"].values():
        if position.get("share_price_state"):
            del position["share_price_state"]["total_withdrawn"]
    replay.state = State.from_json(json.dumps(serialised))
    replay.allow_unknown_withdrawn = True
    replay.positions = {name: replay.state.portfolio.get_position_by_id(position.position_id) for name, position in replay.positions.items()}
    assert replay.positions[MASTER].share_price_state.total_withdrawn is None
    assert calculate_flow_neutral_profit_usd(replay.positions[MASTER]) is None

    # 2. Apply the rest of the funding chain: the reloaded accounts stay unknown on the old figures, an account first funded after the reload is tracked
    for event in funding_chain.events[2:]:
        replay.apply(event)
    assert replay.positions[MASTER].share_price_state.total_withdrawn is None
    assert replay.positions["chain-trend"].share_price_state.total_withdrawn is None
    # The old closed formula: sells 5.5 minus invested 10, missing the 4 USDC that funded the sub-account.
    assert replay.positions[MASTER].get_total_profit_usd() == pytest.approx(-4.5)
    # chain-arb got its share-price state from its first transfer, after the reload.
    assert calculate_flow_neutral_profit_usd(replay.positions["chain-arb"]) == pytest.approx(-0.22)

    # 3. Recover a failed transfer once, then check a repeated completion is refused and changes nothing
    recovery = TimelineReplay((MASTER, "chain-trend"))
    recovery.apply(funding_chain.events[0])
    master, trend = recovery.positions[MASTER], recovery.positions["chain-trend"]
    trade = create_exchange_account_funding(state=recovery.state, source_position=master, destination_position=trend, strategy_cycle_at=recovery.clock, reserve_currency=recovery.usdc, amount=Decimal(4), notes="Interrupted transfer")
    recovery.state.start_execution(recovery.clock, trade)
    trade.mark_failed(recovery.clock)
    complete_exchange_account_funding(state=recovery.state, trade=trade, executed_amount=Decimal(4), fee=Decimal("0.5"), executed_at=recovery.clock, recovery=True)
    recovered = (master.share_price_state.total_withdrawn, master.share_price_state.total_supply, trend.share_price_state.total_supply)
    assert recovered[0] == pytest.approx(4)
    with pytest.raises(ExchangeAccountTransferError):
        complete_exchange_account_funding(state=recovery.state, trade=trade, executed_amount=Decimal(4), fee=Decimal("0.5"), executed_at=recovery.clock)
    assert (master.share_price_state.total_withdrawn, master.share_price_state.total_supply, trend.share_price_state.total_supply) == recovered
    assert calculate_flow_neutral_profit_usd(master) == pytest.approx(-0.5)

    # 4. Check the stored statistics and the chart of a fresh funding chain carry the flow-neutral figures
    replay = replay_timeline(funding_chain)
    master = replay.positions[MASTER]
    last_stats = replay.state.stats.positions[master.position_id][-1]
    assert last_stats.profit_usd == pytest.approx(-0.5)
    assert last_stats.internal_profit_usd == pytest.approx(-0.5)
    assert last_stats.profitability == pytest.approx(-0.05)
    figure = exchange_account_share_price(ChartInput(execution_context=unit_test_execution_context, state=replay.state, pairs=[master.pair]))
    assert len(render_plotly_figure_as_image_file(figure, width=800, height=600)) > 0
