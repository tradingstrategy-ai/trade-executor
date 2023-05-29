"""Run the manual strategy visualisation output test.

How to run:

.. code-block:: shell

    export WEBHOOK_URL=https://enzyme-polygon-eth-usdc.tradingstrategy.ai/
    python scripts/manual-visualisation-test.py

It will open the visualisation in a web browser pop up.

See also: https://tradingstrategy.ai/docs/programming/strategy-examples/examine-live-strategy.html
"""

import datetime
import os

import requests

from tradeexecutor.strategy.reverse_universe import reverse_trading_universe_from_state
from tradeexecutor.visual.strategy_state import draw_single_pair_strategy_state
from tradingstrategy.charting.candle_chart import VolumeBarMode
from tradingstrategy.client import Client
from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.state import State
from tradingstrategy.timebucket import TimeBucket

# Currently needed because unpatched dataclasses_json package issues
patch_dataclasses_json()

client = Client.create_jupyter_client()

# Public internet endpoint as exposed by the trade executor Docker
webbhook_url = os.environ["WEBHOOK_URL"]

state_api = f"{webbhook_url}/state"
resp = requests.get(state_api)
state_blob = resp.content

print(f"Downloaded {len(state_blob):,} bytes state data")

state: State = State.from_json(state_blob)
pair = state.portfolio.get_single_pair()

# Add some data margin around our
# trade timeline visualisation
first_trade, last_trade = state.portfolio.get_first_and_last_executed_trade()
feed_start_at = first_trade.started_at - datetime.timedelta(days=2)
feed_end_at = last_trade.executed_at + datetime.timedelta(days=2)

print("Loading data the strategy has been executing on")
universe = reverse_trading_universe_from_state(
    state,
    client,
    TimeBucket.h1,
)

print("Drawing an example visualisation")
small_figure = draw_single_pair_strategy_state(
    state,
    universe,
    height=512,
)

# Open a web browser pop up
#small_figure.show()

for p in state.portfolio.get_all_positions():
    print(f"Position #{p.position_id}: {p.get_opening_price()} {p.pair.get_ticker()}")

for t in state.portfolio.get_all_trades():
    print(f"Trade {t.trade_id}, assumed price: {t.planned_price}, executed price: {t.executed_price}")
