"""Examine externally created trade executor state.

Load JSON dump and then reflect on it.
"""

import sys

from tradeexecutor.monkeypatch.dataclasses_json import patch_dataclasses_json
from tradeexecutor.state.state import State

patch_dataclasses_json()

file = sys.argv[1]
json_text = open(file, "rt").read()
state = State.from_json(json_text)