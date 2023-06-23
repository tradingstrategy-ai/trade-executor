"""Key metric calculations.


"""
import datetime
from typing import List, Iterable

from tradeexecutor.state.state import State
from tradeexecutor.strategy.summary import KeyMetric, KeyMetricKind, KeyMetricSource


def calculate_key_metrics(
        live_state: State,
        backtested_state: State,
        required_history = datetime.timedelta(days=90),
        now_ = datetime.datetime | None,
) -> Iterable[KeyMetric]:
    """- Calculate summary metrics to be displayed on the web frontend

    - Metrics are calcualted either based live trading data or backtested data,
      whichever makes more sense

    - Live execution state is used if it has enough history

    :param live_state:
        The current live execution state

    :param backtested_state:
        The backtested state

    :param required_history:
        How long history we need before using live execution
        as the basis for the key metric calculations

    :param now_:
        Override the current timestamp for testing

    :return:
        Key metrics
    """

    if not now_:
        now_ = datetime.datetime.utcnow()

    # Live history is calculated from the
    live_history = now_ - live_state.created_at
    if live_history >= required_history:
        state = live_state
        source = KeyMetricSource.live_trading
    else:
        state = backtested_state
        source = KeyMetricSource.backtesting



    yield KeyMetric(
        KeyMetricKind.started_at,
        KeyMetricSource.live_trading,
        live_state.created_at,
    )