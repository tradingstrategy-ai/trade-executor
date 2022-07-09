"""Execution mode modes."""
import enum


class ExecutionMode(enum.Enum):
    """Different execution modes the strategy engine can hvae."""

    #: We are live trading with real assets
    real_trading = "real_trading"

    #: We are live trading with mock assets
    #: TODO: This mode is not yet supported
    paper_trading = "paper_trading"

    #: We are backtesting
    backtesting = "backtesting"

    #: We are loading and caching datasets before a backtesting session can begin.
    #: We call create_trading_universe() and assume :py:class:`tradingstrategy.client.Client`
    #: class is set to a such state it can display nice progress bar when loading
    #: data in a Jupyter notebook.
    data_preload = "data_preload"

    def is_live_trading(self) -> bool:
        """Are we trading with real money or paper money real time?"""
        return self.value in (self.real_trading, self.paper_trading)