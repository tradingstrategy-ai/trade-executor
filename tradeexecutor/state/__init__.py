"""Strategy execution state.

This module defines data structures used to manage the strategy execution.

- The internal data is a nested tree structure starting with
  :py:class:`tradeexecutor.state.state.State` root class.

- The state includes portfolios, open and closed positions, trades being currently
  executed, deposits and withdraws, portfolio valuation events and such.

- The whole state must be :ref:`serialisable <serialisation>` as JSON,
  so that the JavaScript clients can read it.

- The same state structure is used for both backtesting (simulation)
  and live trading

- The application also internally stores its state as a flat file on the disk,
  see :py:mod:`tradeexecutor.state.store`

For an overview, see :ref:`architecture documentation <architecture>`.
"""