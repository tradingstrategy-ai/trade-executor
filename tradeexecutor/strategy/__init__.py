"""Strategy core - the internal state data structures.

This is the code of the strategy execution engine.
We define how the data structures look like and what actions are possible.

All data structures are based on :py:mod:`dataclass` DLS and `dataclasses_json <https://github.com/lidatong/dataclasses-json>`_ serialisation.

The main :py:class:`tradeexecutor.state.State` class serialises the whole strategy execution state
in one go and passes it to the web client as one JSON download.
"""