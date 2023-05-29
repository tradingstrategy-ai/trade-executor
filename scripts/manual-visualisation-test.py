"""Run the manual strategy visualisation output test.

How to run:

.. code-block:: shell

    export WEBHOOK_URL=https://enzyme-polygon-eth-usdc.tradingstrategy.ai/
    python scripts/manual-visualisation-test.py


See also: https://tradingstrategy.ai/docs/programming/strategy-examples/examine-live-strategy.html
"""

import datetime
import os

import pandas as pd
import requests
from IPython.core.display_functions import display

from tradeexecutor.strategy.reverse_universe import reverse_trading_universe_from_state
from tradeexecutor.strategy.trading_strategy_universe import TradingStrategyUniverse
from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state
from tradingstrategy.client import Client
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.state import State
from tradeexecutor.state.identifier import TradingPairIdentifier
from tradingstrategy.timebucket import TimeBucket
from tradeexecutor.visual.single_pair import visualise_single_pair_positions_with_duration_and_slippage

# Currently needed because unpatched dataclasses_json package issues
patch_dataclasses_json()

client = Client.create_jupyter_client()

# Public internet endpoint as exposed by the trade executor Docker
webbhook_url = os.environ["WEBHOOK_URL"]

state_api = f"{webbhook_url}/state"
resp = requests.get(state_api)
state_blob = resp.content

print(f"Downloaded {len(state_blob):,} bytes state data")

state = State.from_json(state_blob)
pair = state.portfolio.get_single_pair()

# Add some data margin around our
# trade timeline visualisation
first_trade, last_trade = state.portfolio.get_first_and_last_executed_trade()
feed_start_at = first_trade.started_at - datetime.timedelta(days=2)
feed_end_at = last_trade.executed_at + datetime.timedelta(days=2)

candles: pd.DataFrame = client.fetch_candles_by_pair_ids(
    {pair.internal_id},
    TimeBucket.m15,
    progress_bar_description=f"Download data for {pair.base.token_symbol} - {pair.quote.token_symbol}",
    start_time=feed_start_at,
    end_time=feed_end_at,
)

universe = reverse_trading_universe_from_state(
    state,
    client,
    TimeBucket.h1,
)

print(f"Loaded {len(candles)} candles, {feed_start_at} - {feed_end_at}")

small_figure = draw_single_pair_strategy_state(
    state,
    universe,
    height=512,
    candle_count=64,
)

# Open a web browser pop up
small_figure.show()