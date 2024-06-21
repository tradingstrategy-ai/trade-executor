"""Market regime analysis helpers."""
import enum
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from tradeexecutor.state.types import Percent


class Regime(enum.Enum):
    """Possible market regimes.

    The content of signal for :py:func:`visualise_market_regime_filter`.

    Any incoming `nan` or `None` is aliased to crab region.
    """

    #: We are in a bull market (trends up)
    bull = 1

    #: We are in a bear market (trends down)
    bear = -1

    #: We are in a sideways market (no trends)
    #:
    #: `None` is alias for `0` value here.
    crab = 0

    @classmethod
    def _missing_(cls, val):
        """Construct enum based on a raw int value.

        None (lack of data) is automatically converted to crab market.


        """
        if not val or pd.isna(val):
            return Regime.crab
        return super()._missing_(val)


@dataclass(frozen=True, slots=True)
class RegimeRegion:
    """One regime colouring region for the charts."""
    start: pd.Timestamp
    end: pd.Timestamp
    regime: Regime

    def __repr__(self):
        return f"{self.regime.name} {self.start} - {self.end}"

    def get_duration(self) -> pd.Timedelta:
        return self.end - self.start

    def is_predition_correct(
        self,
        close: pd.Series,
        crab_region_tolerance: Percent = 0.02,
    ) -> tuple[bool, Percent]:
        """Check if the region matches actual data.

        - Bull region is correct if we closed higher since start

        - Bear region is correct if we closed lowed since start

        - Crab

        :param close:
            Close price series

        :return:
            Tuple (match, diff).

            True if the underlying market data matches the region,
            and what was % observed price movement within the region.
        """
        diff = (close[self.end] - close[self.start]) / close[self.start]
        if abs(diff) < crab_region_tolerance:
            return self.regime == Regime.crab, diff

        if diff > 0:
            return self.regime == Regime.bull, diff
        else:
            return self.regime == Regime.bear, diff


def get_regime_signal_regions(signal: pd.Series) -> Iterable[RegimeRegion]:
    """Get regions of the regime signal.

    Split the signal to continous blocks for coloring.

    :return:
        Iterable of market regimes for colouring
    """

    # https://stackoverflow.com/a/69222703/315168
    edges = signal.diff(periods=1)

    # edge_mask = edges.loc[edges != 0]

    current_signal = Regime(signal.iloc[0])
    current_start = edges.index[0]

    regime_change_timestamps = edges.index[edges != 0]

    if len(regime_change_timestamps) > 0:
        # Skip the start region
        for idx in regime_change_timestamps[1:]:
            yield RegimeRegion(
                current_start,
                idx,
                current_signal
            )
            current_start = idx
            current_signal = Regime(signal[idx])

    # The closing region
    yield RegimeRegion(
        current_start,
        signal.index[-1],
        current_signal,
    )
